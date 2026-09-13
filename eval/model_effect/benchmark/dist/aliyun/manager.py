#!/usr/bin/env python3
"""Viewer-facing lifecycle manager for remote Aliyun Benchmark jobs."""
from __future__ import annotations

import json
import re
import threading
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from .config import AliyunConfig, load_defaults
from .dlc import DlcClient, public_job_detail, submission_name


_TERMINAL_DLC = {"Succeeded", "Failed", "Stopped", "Deleted"}


def _slug(value: str, limit: int = 80) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (clean or "model")[:limit]


def _read_json(path: Path) -> dict | None:
    try:
        with path.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _progress_with_timing(progress: dict, state: dict) -> dict:
    """Expose submit-to-finish wall time and a rolling whole-suite ETA."""
    value = dict(progress or {})
    started_at = state.get("started_at")
    if started_at is None:
        return value
    end = time.time() if state.get("running") or state.get("finished_at") is None else float(state["finished_at"])
    elapsed = max(0.0, end - float(started_at))
    frac = value.get("suite_frac", value.get("frac"))
    value["elapsed_s"] = elapsed
    if not state.get("running"):
        value.update(estimated_total_s=elapsed, remaining_s=0.0)
    elif isinstance(frac, (int, float)) and 0.005 <= float(frac) <= 1.0:
        estimated = elapsed / float(frac)
        value.update(estimated_total_s=estimated,
                     remaining_s=max(0.0, estimated - elapsed))
    return value


class AliyunBenchmarkManager:
    """Submit one DLC job and expose a local-Viewer-compatible status snapshot."""

    def __init__(self, repo_dir: Path | str, *, credentials: Path | str | None = None):
        self.repo_dir = Path(repo_dir).resolve()
        self.credentials = Path(credentials) if credentials else (
            self.repo_dir / "model_train" / "configs" / "cloud_credentials" / "aliyun.md"
        )
        self._lock = threading.RLock()
        self._state = self._empty_state()
        self._config: AliyunConfig | None = None
        self._client: DlcClient | None = None
        self._request_path: Path | None = None
        self._remote_state_path: Path | None = None
        self._cancel_requested = False

    @staticmethod
    def _empty_state() -> dict:
        return {
            "backend": "aliyun", "running": False, "phase": "待运行", "count": 0,
            "log": [], "report": None, "live_report": None, "error": None,
            "out": None, "ckpt_tag": None, "models": [], "active_model": None,
            "selection": {}, "progress": {}, "job_id": None, "job_status": None,
            "job": {}, "aliyun": None,
            "started_at": None, "finished_at": None,
        }

    def defaults(self) -> dict:
        config = load_defaults()
        return {"aliyun": config.to_dict(), "world_size": config.world_size}

    def is_running(self) -> bool:
        return bool(self.status().get("running"))

    def start(self, *, models: list[dict], datasets: str, heads: str,
              max_seqs=None, max_frames=None, seq_start=0, seq_end=None,
              dataset_selection=None, reuse_cache=True, config: AliyunConfig) -> dict:
        with self._lock:
            if self._state.get("running"):
                return {"ok": False, "error": "已有 Aliyun 远程测评正在运行"}

            remote_repo = Path(config.repo_dir)
            if not remote_repo.is_dir():
                return {"ok": False, "error": f"Aliyun repo_dir 在当前 CPFS 不存在: {remote_repo}"}
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_dir = remote_repo / "output" / "eval" / "benchmark" / "aliyun" / stamp
            request_path = output_dir / "request.json"
            display_name = submission_name(config, stamp[:15])
            request_models = []
            for index, model in enumerate(models, 1):
                entry = dict(model)
                entry["status"] = "completed" if entry.get("cache_hit") else "pending"
                entry["report"] = None
                entry["error"] = None
                entry["out_name"] = f"model_{index:02d}_{_slug(entry.get('tag') or entry.get('step'))}"
                request_models.append(entry)
            selection = {
                "datasets": str(datasets), "heads": str(heads),
                "max_seqs": max_seqs, "max_frames": max_frames,
                "seq_start": int(seq_start), "seq_end": seq_end,
                "dataset_selection": dataset_selection or {},
                "reuse_cache": bool(reuse_cache),
                "auto_ukf_best": any(
                    model.get("variant") == "ukf" or model.get("auto_select_ukf")
                    for model in request_models
                ),
            }
            request = {
                "version": 1,
                "created_at": datetime.now().astimezone().isoformat(),
                "display_name": display_name,
                "output_dir": str(output_dir),
                "aliyun": config.to_dict(),
                "selection": selection,
                "models": request_models,
                "barrier_timeout_seconds": 7 * 24 * 60 * 60,
            }
            try:
                _atomic_json(request_path, request)
            except OSError as exc:
                return {"ok": False, "error": f"写入远程测评请求失败: {exc}"}

            cached_count = sum(bool(model.get("cache_hit")) for model in request_models)
            if cached_count == len(request_models):
                self._config = config
                self._client = None
                self._request_path = request_path
                self._remote_state_path = None
                self._cancel_requested = False
                hydrated, last_report = self._hydrate_reports(request_models)
                try:
                    from benchmark.cache import publish_step_benchmark_logs

                    publish_step_benchmark_logs(hydrated, selection)
                except Exception:  # noqa: BLE001
                    pass
                comparison = {
                    "selection": selection,
                    "models": [
                        {key: value for key, value in model.items() if key != "report"}
                        for model in hydrated
                    ],
                }
                _atomic_json(output_dir / "comparison.json", comparison)
                self._state = {
                    **self._empty_state(),
                    "running": False, "phase": "全部模型复用历史测评结果",
                    "out": str(output_dir), "models": hydrated,
                    "ckpt_tag": request_models[0].get("tag") if len(request_models) == 1 else f"{len(request_models)} models",
                    "report": last_report, "live_report": last_report,
                    "selection": selection,
                    "progress": {
                        "frac": 1.0, "suite_frac": 1.0,
                        "model_index": len(hydrated), "model_total": len(hydrated),
                        "done": 0, "total": 0, "gpus": [], "cache_hit": True,
                    },
                    "aliyun": config.to_dict(),
                    "started_at": time.time(), "finished_at": time.time(),
                    "log": [f"[cache] 复用 {cached_count}/{len(hydrated)} 个模型，未提交 DLC 作业"],
                }
                return {
                    "ok": True, "backend": "aliyun", "models": len(hydrated),
                    "cached": cached_count, "submitted": False, "out": str(output_dir),
                }

            try:
                client = DlcClient(config, credentials=self.credentials)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": f"初始化 DLC 客户端失败: {exc}"}
            self._config = config
            self._client = client
            self._request_path = request_path
            self._remote_state_path = output_dir / "remote_state.json"
            self._cancel_requested = False
            self._state = {
                **self._empty_state(),
                "running": True,
                "started_at": time.time(), "finished_at": None,
                "phase": "正在提交 Aliyun PAI-DLC 作业…",
                "out": str(output_dir),
                "ckpt_tag": request_models[0]["tag"] if len(request_models) == 1 else f"{len(request_models)} models",
                "models": request_models,
                "selection": selection,
                "progress": {
                    "frac": 0.0, "suite_frac": 0.0,
                    "model_index": 0, "model_total": len(request_models),
                    "done": 0, "total": 0, "gpus": [],
                },
                "aliyun": config.to_dict(),
                "display_name": display_name,
                "log": [
                    f"[aliyun] 提交 {config.nnodes} nodes × {config.gpus_per_node} GPUs = {config.world_size} shards",
                    f"[aliyun] request={request_path}",
                ],
            }
        threading.Thread(target=self._submit_and_monitor, daemon=True,
                         name="aliyun-benchmark-submit").start()
        return {
            "ok": True, "backend": "aliyun", "models": len(models),
            "cached": cached_count, "submitted": True,
            "nodes": config.nnodes, "world_size": config.world_size,
            "out": str(output_dir),
        }

    def _append_log(self, line: str) -> None:
        with self._lock:
            logs = list(self._state.get("log") or [])
            logs.append(str(line))
            self._state["log"] = logs[-48:]

    def _submit_and_monitor(self) -> None:
        with self._lock:
            client, request_path = self._client, self._request_path
            display_name = self._state.get("display_name")
        if client is None or request_path is None:
            return
        try:
            job_id, submit_output = client.submit(request_path, display_name=display_name)
            self._append_log(f"[aliyun] DLC JobId={job_id} name={display_name}")
            for line in submit_output.splitlines()[-4:]:
                if line.strip():
                    self._append_log("[dlc] " + line.strip())
            with self._lock:
                self._state.update(job_id=job_id, job_status="Submitted",
                                   phase="Aliyun 作业已提交，等待调度…")
                cancel_now = self._cancel_requested
            if cancel_now:
                self._stop_job(job_id)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._state.update(running=False, phase="Aliyun 提交失败", error=str(exc),
                                   finished_at=time.time())
            return

        consecutive_errors = 0
        while True:
            try:
                detail = client.get_job(job_id)
                public = public_job_detail(detail)
                status = str(public.get("job_status") or "Unknown")
                consecutive_errors = 0
                with self._lock:
                    self._state.update(job_id=job_id, job_status=status, job=public)
                if status in _TERMINAL_DLC:
                    self._apply_terminal_dlc(status, public)
                    return
            except Exception as exc:  # noqa: BLE001
                consecutive_errors += 1
                if consecutive_errors in {1, 6}:
                    self._append_log(f"[aliyun] 查询 DLC 状态失败: {exc}")
            time.sleep(8)

    def _apply_terminal_dlc(self, status: str, public: dict) -> None:
        remote = _read_json(self._remote_state_path) if self._remote_state_path else None
        if remote and not remote.get("running", True):
            return
        reason = public.get("reason_message") or public.get("reason_code")
        with self._lock:
            if status == "Succeeded":
                self._state.update(running=False, phase="Aliyun 作业完成",
                                   finished_at=time.time())
                if not remote:
                    self._state["error"] = "DLC 已成功结束，但未找到 remote_state.json"
            else:
                phase = "Aliyun 作业已停止" if status == "Stopped" else "Aliyun 作业失败"
                self._state.update(running=False, phase=phase, error=reason or status,
                                   finished_at=time.time())

    def _stop_job(self, job_id: str) -> None:
        try:
            client = self._client
            if client is None:
                raise RuntimeError("DLC 客户端尚未初始化")
            client.stop(job_id)
            self._append_log(f"[aliyun] 已请求停止 DLC JobId={job_id}")
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"[aliyun] 停止作业失败: {exc}")
            with self._lock:
                self._state["error"] = str(exc)

    def cancel(self) -> dict:
        with self._lock:
            if not self._state.get("running"):
                return {"ok": False, "error": "当前无运行中的 Aliyun 测评"}
            self._cancel_requested = True
            self._state["phase"] = "正在停止 Aliyun DLC 作业…"
            job_id = self._state.get("job_id")
        if job_id:
            threading.Thread(target=self._stop_job, args=(job_id,), daemon=True,
                             name="aliyun-benchmark-stop").start()
        return {"ok": True, "job_id": job_id}

    def status(self) -> dict:
        with self._lock:
            base = deepcopy(self._state)
            remote_path = self._remote_state_path
        remote = _read_json(remote_path) if remote_path else None
        if remote:
            # Submission-side fields remain authoritative because the remote worker does not
            # know the JobId and should never receive local credentials.
            for key in (
                "running", "phase", "count", "report", "live_report", "error", "out",
                "ckpt_tag", "models", "active_model", "selection", "progress",
            ):
                if key in remote:
                    base[key] = remote[key]
            base["remote_updated_at"] = remote.get("updated_at")
        official_status = str(base.get("job_status") or "")
        if official_status in _TERMINAL_DLC and base.get("running"):
            base["running"] = False
            if official_status == "Succeeded":
                base["phase"] = "Aliyun 作业完成"
                base["error"] = base.get("error") or "DLC 已结束，但远程状态未正常收尾"
            else:
                base["phase"] = "Aliyun 作业已停止" if official_status == "Stopped" else "Aliyun 作业失败"
                job = base.get("job") or {}
                base["error"] = job.get("reason_message") or job.get("reason_code") or official_status
        if not base.get("running") and base.get("started_at") is not None and base.get("finished_at") is None:
            with self._lock:
                if self._state.get("finished_at") is None:
                    self._state["finished_at"] = time.time()
                base["finished_at"] = self._state["finished_at"]
        base["progress"] = self._progress_snapshot(base)
        base["progress"] = _progress_with_timing(base["progress"], base)
        live_result_rows = base["progress"].pop("_live_result_rows", {})
        base["count"] = int((base.get("progress") or {}).get("done") or 0)
        base["models"], last_report = self._hydrate_reports(base.get("models") or [])
        if base.get("running") and live_result_rows:
            from benchmark.dist.aggregate import merge_result_rows

            model_index = int((base.get("progress") or {}).get("model_index") or 0)
            active = base["models"][model_index - 1] if 1 <= model_index <= len(base["models"]) else {}
            base["live_report"] = merge_result_rows(
                live_result_rows,
                ckpt=active.get("ckpt"),
                config=active.get("config") or "",
                selection=base.get("selection") or {},
            )
        if last_report is not None:
            base["report"] = last_report
            if not base.get("running"):
                base["live_report"] = last_report
        with self._lock:
            submit_logs = list(self._state.get("log") or [])
        remote_logs = list((remote or {}).get("log") or [])
        base["log"] = (submit_logs + remote_logs)[-48:]
        base["backend"] = "aliyun"
        return base

    @staticmethod
    def _hydrate_reports(models: list[dict]) -> tuple[list[dict], dict | None]:
        hydrated, last = [], None
        for model in models:
            item = dict(model)
            path = item.get("report_path")
            if path and not item.get("report"):
                item["report"] = _read_json(Path(path))
            if item.get("report"):
                last = item["report"]
            hydrated.append(item)
        return hydrated, last

    @staticmethod
    def _progress_snapshot(state: dict) -> dict:
        progress = dict(state.get("progress") or {})
        output = state.get("out")
        model_index = int(progress.get("model_index") or 0)
        models = state.get("models") or []
        if not output or model_index < 1:
            return progress
        progress_dir = Path(output) / "progress" / f"model_{model_index:02d}"
        shards = []
        for path in sorted(progress_dir.glob("shard_*.json")):
            row = _read_json(path)
            if row:
                shards.append(row)
        if not shards:
            return progress

        aliyun = state.get("aliyun") or {}
        nodes = int(aliyun.get("nnodes") or 0)
        gpus_per_node = int(aliyun.get("gpus_per_node") or 0)
        expected_world = nodes * gpus_per_node
        if expected_world:
            by_index = {int(row.get("index", -1)): row for row in shards}
            shards = [
                by_index.get(index, {
                    "index": index,
                    "node": index // gpus_per_node,
                    "local_gpu": index % gpus_per_node,
                    "done": 0, "total": 0, "live": {}, "datasets": {},
                    "status": "pending",
                })
                for index in range(expected_world)
            ]

        live_result_rows = {
            str(int(row.get("index", index))): dict(row.get("results") or {})
            for index, row in enumerate(shards)
        }
        public_shards = [
            {key: value for key, value in row.items() if key != "results"}
            for row in shards
        ]

        done = sum(int(row.get("done") or 0) for row in shards)
        total = sum(int(row.get("total") or 0) for row in shards)
        model_total = int(progress.get("model_total") or len(models) or 1)
        frac = done / total if total else 0.0
        order = []
        for row in shards:
            for name in row.get("ds_order") or []:
                if name not in order:
                    order.append(name)
        datasets = {}
        for name in order:
            parts = [(row.get("datasets") or {}).get(name) or {} for row in shards]
            ds_done = sum(int(part.get("done") or 0) for part in parts)
            ds_total = sum(int(part.get("total") or 0) for part in parts)
            all_finished = bool(parts) and all(part.get("finished") for part in parts)
            datasets[name] = {
                "done": ds_done, "total": ds_total,
                "status": "done" if all_finished else ("running" if any(parts) else "pending"),
                "report": None,
            }
        return {
            **progress,
            "frac": frac,
            "suite_frac": ((model_index - 1) + frac) / model_total,
            "done": done, "total": total,
            "gpus": public_shards,
            "ds_order": order, "datasets": datasets,
            "_live_result_rows": live_result_rows,
        }
