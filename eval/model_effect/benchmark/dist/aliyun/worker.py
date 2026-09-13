#!/usr/bin/env python3
"""PAI-DLC node entrypoint: fan out one Benchmark shard per local GPU."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


_REPO = Path(__file__).resolve().parents[5]
_MODEL_EFFECT = _REPO / "eval" / "model_effect"
for _path in (str(_MODEL_EFFECT), str(_REPO)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_SHARD_RE = re.compile(r"\[SHARD\]\s+done=(\d+)\s+total=(\d+)")
_DS_RE = re.compile(r"\[DS\]\s+(.+?)\|(\d+)\|(\d+)")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def node_rank(hostname: str | None = None, env: dict | None = None) -> int:
    env = os.environ if env is None else env
    for key in ("NODE_RANK",):
        value = env.get(key)
        if value is not None and str(value).isdigit():
            return int(value)
    hostname = hostname or os.uname().nodename
    if re.search(r"-master(?:-0)?$", hostname) or "master" in hostname:
        return 0
    match = re.search(r"-worker-([0-9]+)$", hostname)
    if match:
        return int(match.group(1)) + 1
    raise RuntimeError(f"无法从 hostname={hostname!r} 推断 DLC node rank")


def _protocol_only(datasets: str, heads: str) -> bool:
    selected_datasets = {item.strip() for item in datasets.split(",") if item.strip()}
    selected_heads = {item.strip() for item in heads.split(",") if item.strip()}
    return (
        bool(selected_datasets)
        and all(name.endswith("_hand_coverage") for name in selected_datasets)
        and selected_heads == {"hands_coverage"}
    )


def build_shard_command(request: dict, model: dict, *, global_rank: int,
                        world_size: int, output_dir: Path) -> list[str]:
    selection = request["selection"]
    command = [
        sys.executable,
        str(_REPO / "eval" / "model_effect" / "benchmark" / "run.py"),
        "--ckpt", str(model["ckpt"]),
        "--config", str(model["config"]),
        "--model", str(model.get("model") or "lingbotmap"),
        "--heads", str(selection["heads"]),
        "--datasets", str(selection["datasets"]),
        "--shard-index", str(global_rank),
        "--shard-count", str(world_size),
        "--out", str(output_dir),
    ]
    if not _protocol_only(str(selection["datasets"]), str(selection["heads"])):
        command.append("--windowed")
    command += ["--hand-mode", str(model.get("hand_mode") or "hard")]
    if selection.get("dataset_selection"):
        command += ["--dataset-selection-json", json.dumps(
            selection["dataset_selection"], ensure_ascii=True, separators=(",", ":")
        )]
    for option, key in (
        ("--max-seqs", "max_seqs"),
        ("--max-frames", "max_frames"),
        ("--seq-start", "seq_start"),
        ("--seq-end", "seq_end"),
    ):
        value = selection.get(key)
        if value is not None and not (key == "seq_start" and int(value) == 0):
            command += [option, str(int(value))]
    return command


def _write_progress(path: Path, state: dict, lock: threading.Lock) -> None:
    with lock:
        snapshot = json.loads(json.dumps(state, ensure_ascii=False))
    _atomic_json(path, snapshot)


def _run_shard(request: dict, model: dict, *, node: int, local_gpu: int,
               world_size: int, model_index: int, output_dir: Path,
               progress_dir: Path) -> int:
    global_rank = node * int(request["aliyun"]["gpus_per_node"]) + local_gpu
    shard_out = output_dir / f"gpu{global_rank:03d}"
    shard_out.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / f"shard_{global_rank:03d}.json"
    state = {
        "index": global_rank, "node": node, "local_gpu": local_gpu,
        "done": 0, "total": 0, "live": {}, "ds_order": [], "datasets": {},
        "results": {},
        "status": "starting", "updated_at": _now(),
    }
    lock = threading.Lock()
    _write_progress(progress_path, state, lock)
    command = build_shard_command(
        request, model, global_rank=global_rank, world_size=world_size,
        output_dir=shard_out,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(local_gpu)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path = shard_out / "process.log"
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("[aliyun-worker] " + " ".join(command) + "\n")
        log_file.flush()
        proc = subprocess.Popen(
            command, cwd=_REPO, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        with lock:
            state["status"] = "running"
            state["pid"] = proc.pid
            state["updated_at"] = _now()
        _write_progress(progress_path, state, lock)
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, ""):
            log_file.write(raw)
            log_file.flush()
            for part in re.split(r"[\r\n]+", _ANSI.sub("", raw)):
                line = part.strip()
                if not line:
                    continue
                changed = False
                with lock:
                    if line.startswith("[DSINIT] ds="):
                        state["ds_order"] = [
                            item for item in line[len("[DSINIT] ds="):].split(",") if item
                        ]
                        changed = True
                    elif line.startswith("[DSDONE] ds="):
                        name = line[len("[DSDONE] ds="):].strip()
                        state["datasets"].setdefault(name, {})["finished"] = True
                        changed = True
                    elif line.startswith("[LIVE] "):
                        try:
                            state["live"] = json.loads(line[len("[LIVE] "):])
                            changed = True
                        except json.JSONDecodeError:
                            pass
                    elif line.startswith("[RESULT] "):
                        try:
                            result = json.loads(line[len("[RESULT] "):])
                            required = {"head", "dataset", "seq_id", "status"}
                            if required.issubset(result):
                                key = "\0".join(str(result[name]) for name in ("head", "dataset", "seq_id"))
                                state["results"][key] = result
                                changed = True
                        except json.JSONDecodeError:
                            pass
                    else:
                        match = _SHARD_RE.search(line)
                        dataset_match = _DS_RE.search(line)
                        if match:
                            state["done"] = int(match.group(1))
                            state["total"] = int(match.group(2))
                            changed = True
                        elif dataset_match:
                            state["datasets"].setdefault(dataset_match.group(1), {}).update(
                                done=int(dataset_match.group(2)),
                                total=int(dataset_match.group(3)),
                            )
                            changed = True
                    if changed:
                        state["updated_at"] = _now()
                if changed:
                    _write_progress(progress_path, state, lock)
        returncode = proc.wait()
    with lock:
        state["status"] = "completed" if returncode == 0 else "failed"
        state["returncode"] = returncode
        state["updated_at"] = _now()
    _write_progress(progress_path, state, lock)
    return returncode


def _wait_for(paths: list[Path], timeout: int, description: str) -> None:
    deadline = time.monotonic() + timeout
    while True:
        if all(path.is_file() for path in paths):
            return
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.is_file()]
            raise TimeoutError(f"等待{description}超时，缺少: {missing}")
        time.sleep(3)


def _master_state(request: dict, models: list[dict], *, running: bool, phase: str,
                  model_index: int, error: str | None = None) -> dict:
    model_total = len(models)
    completed = sum(model.get("status") == "completed" for model in models)
    active = None
    if running and 1 <= model_index <= model_total:
        model = models[model_index - 1]
        active = {key: model.get(key) for key in (
            "run", "step", "ckpt", "tag", "label", "status"
        )}
    return {
        "backend": "aliyun", "running": running, "phase": phase, "count": 0,
        "report": None, "live_report": None, "error": error,
        "out": request["output_dir"],
        "ckpt_tag": models[0].get("tag") if model_total == 1 else f"{model_total} models",
        "models": models, "active_model": active,
        "selection": request["selection"],
        "progress": {
            "frac": 0.0, "suite_frac": completed / model_total if model_total else 0.0,
            "model_index": model_index, "model_total": model_total,
            "model_label": active.get("label") if active else None,
            "done": 0, "total": 0, "gpus": [],
        },
        "updated_at": _now(),
        "log": [
            f"[remote] {request['aliyun']['nnodes']} nodes × "
            f"{request['aliyun']['gpus_per_node']} GPUs",
        ],
    }


def run_request(request_path: Path, *, forced_node_rank: int | None = None) -> int:
    with request_path.open(encoding="utf-8") as handle:
        request = json.load(handle)
    if int(request.get("version") or 0) != 1:
        raise ValueError(f"不支持的 request version: {request.get('version')}")
    config = request["aliyun"]
    nnodes = int(config["nnodes"])
    gpus_per_node = int(config["gpus_per_node"])
    world_size = nnodes * gpus_per_node
    rank = node_rank() if forced_node_rank is None else int(forced_node_rank)
    if not 0 <= rank < nnodes:
        raise ValueError(f"node rank {rank} 不在 0..{nnodes - 1}")

    output_root = Path(request["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "remote_state.json"
    barrier_root = output_root / "barriers"
    progress_root = output_root / "progress"
    timeout = int(request.get("barrier_timeout_seconds") or 604800)
    models = [dict(model) for model in request["models"]]
    failures = []

    if rank == 0:
        _atomic_json(
            state_path,
            _master_state(request, models, running=True, phase="远程节点启动中…", model_index=1),
        )

    for model_index, model in enumerate(models, 1):
        if model.get("auto_select_ukf"):
            choice_marker = barrier_root / f"model_{model_index:02d}" / "ukf_choice.json"
            if rank == 0:
                try:
                    from benchmark.ranking import resolve_auto_ukf_model

                    candidates = []
                    for previous in models[:model_index - 1]:
                        candidate = dict(previous)
                        report_path = Path(candidate.get("report_path") or "")
                        if report_path.is_file():
                            candidate["report"] = json.loads(report_path.read_text(encoding="utf-8"))
                        candidates.append(candidate)
                    resolved = resolve_auto_ukf_model(
                        candidates,
                        reuse_cache=bool(request.get("selection", {}).get("reuse_cache", True)),
                        out_name=model.get("out_name"),
                    )
                    resolved.pop("report", None)
                    choice = {"ok": True, "model": resolved}
                except Exception as exc:  # noqa: BLE001
                    choice = {"ok": False, "error": str(exc)}
                _atomic_json(choice_marker, choice)
            else:
                _wait_for([choice_marker], timeout, "UKF 自动优选结果")
            with choice_marker.open(encoding="utf-8") as handle:
                choice = json.load(handle)
            if not choice.get("ok"):
                error = f"UKF 自动优选失败: {choice.get('error') or '未知错误'}"
                model.update(status="failed", error=error)
                failures.append(error)
                if rank == 0:
                    _atomic_json(
                        state_path,
                        _master_state(request, models, running=True, phase=error,
                                      model_index=model_index),
                    )
                continue
            model = dict(choice["model"])
            models[model_index - 1] = model
            if rank == 0:
                ranking = model.get("quality_ranking") or {}
                state = _master_state(
                    request, models, running=True,
                    phase=f"模型 {model_index}/{len(models)} · 已选择最佳模型，准备 UKF+RTS 测评…",
                    model_index=model_index,
                )
                state["log"].append(
                    f"[auto-ukf] {model.get('source_model_label')} · "
                    f"平均归一化损失={ranking.get('score', 0):.4f} · "
                    f"有效指标={ranking.get('metrics', 0)}"
                )
                _atomic_json(state_path, state)
        cached_report = Path(model.get("report_path") or "")
        if model.get("cache_hit") and cached_report.is_file():
            model["status"] = "completed"
            if rank == 0:
                phase = f"模型 {model_index}/{len(models)} · 复用历史测评结果"
                _atomic_json(
                    state_path,
                    _master_state(request, models, running=True, phase=phase,
                                  model_index=min(model_index + 1, len(models))),
                )
            continue
        model_dir = output_root / model["out_name"]
        progress_dir = progress_root / f"model_{model_index:02d}"
        if rank == 0:
            model["status"] = "running"
            phase = f"模型 {model_index}/{len(models)} · {model.get('label')} · 远程分片评测中…"
            _atomic_json(
                state_path,
                _master_state(request, models, running=True, phase=phase,
                              model_index=model_index),
            )

        returncodes = {}
        with ThreadPoolExecutor(max_workers=gpus_per_node) as executor:
            futures = {
                executor.submit(
                    _run_shard, request, model, node=rank, local_gpu=local_gpu,
                    world_size=world_size, model_index=model_index,
                    output_dir=model_dir, progress_dir=progress_dir,
                ): local_gpu
                for local_gpu in range(gpus_per_node)
            }
            for future in as_completed(futures):
                local_gpu = futures[future]
                try:
                    returncodes[local_gpu] = int(future.result())
                except Exception as exc:  # noqa: BLE001
                    returncodes[local_gpu] = -1
                    failures.append(f"model {model_index} node {rank} gpu {local_gpu}: {exc}")

        node_marker = barrier_root / f"model_{model_index:02d}" / f"node_{rank:03d}.json"
        _atomic_json(node_marker, {
            "node": rank, "returncodes": returncodes, "updated_at": _now(),
        })
        node_markers = [
            barrier_root / f"model_{model_index:02d}" / f"node_{node:03d}.json"
            for node in range(nnodes)
        ]
        complete_marker = barrier_root / f"model_{model_index:02d}" / "complete.json"

        if rank == 0:
            _wait_for(node_markers, timeout, f"模型 {model_index} 各节点")
            all_returncodes = {}
            for marker in node_markers:
                with marker.open(encoding="utf-8") as handle:
                    node_result = json.load(handle)
                for gpu, code in (node_result.get("returncodes") or {}).items():
                    global_rank = int(node_result["node"]) * gpus_per_node + int(gpu)
                    all_returncodes[global_rank] = int(code)
            failed_shards = {gpu: code for gpu, code in all_returncodes.items() if code != 0}
            report = None
            aggregate_error = None
            try:
                from benchmark.dist.aggregate import aggregate
                report = aggregate(model_dir)
            except Exception as exc:  # noqa: BLE001
                aggregate_error = str(exc)
            model.update(
                status="completed" if report is not None and not failed_shards else "failed",
                out=str(model_dir),
                report_path=str(model_dir / "report.json") if report is not None else None,
                error=(
                    f"分片失败: {failed_shards}" if failed_shards
                    else aggregate_error
                ),
            )
            if model["status"] == "failed":
                failures.append(f"{model.get('label')}: {model.get('error')}")
            elif model.get("benchmark_signature"):
                try:
                    from benchmark.cache import register_cached_report

                    model["cache_manifest"] = str(register_cached_report(
                        model["benchmark_signature"], model_dir / "report.json", report,
                    ))
                except Exception as exc:  # noqa: BLE001
                    model["cache_error"] = str(exc)
            _atomic_json(complete_marker, {
                "status": model["status"], "error": model.get("error"),
                "updated_at": _now(),
            })
            next_index = min(model_index + 1, len(models))
            phase = (
                f"模型 {model_index}/{len(models)} 完成，准备下一个…"
                if model_index < len(models) else "远程结果汇总中…"
            )
            _atomic_json(
                state_path,
                _master_state(request, models, running=True, phase=phase,
                              model_index=next_index),
            )
        else:
            _wait_for([complete_marker], timeout, f"模型 {model_index} 聚合结果")

    if rank == 0:
        try:
            from benchmark.cache import publish_step_benchmark_logs

            for model in models:
                report_path = Path(model.get("report_path") or "")
                if report_path.is_file():
                    model["report"] = json.loads(report_path.read_text(encoding="utf-8"))
            publish_step_benchmark_logs(models, request["selection"])
        except Exception as exc:  # noqa: BLE001
            benchmark_log_error = str(exc)
        else:
            benchmark_log_error = None
        for model in models:
            model.pop("report", None)
        comparison = {"selection": request["selection"], "models": models}
        if benchmark_log_error:
            comparison["benchmark_log_error"] = benchmark_log_error
        _atomic_json(output_root / "comparison.json", comparison)
        phase = f"完成（{sum(model.get('status') == 'failed' for model in models)} 个模型失败）" if failures else "完成"
        final_state = _master_state(
            request, models, running=False, phase=phase,
            model_index=len(models), error="；".join(failures) if failures else None,
        )
        final_state["report"] = None
        _atomic_json(state_path, final_state)
        _atomic_json(barrier_root / "all_complete.json", {
            "failed": bool(failures), "updated_at": _now(),
        })
    else:
        _wait_for([barrier_root / "all_complete.json"], timeout, "最终汇总")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aliyun distributed Benchmark worker")
    parser.add_argument("--request", required=True)
    parser.add_argument("--node-rank", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    raise SystemExit(run_request(Path(args.request), forced_node_rank=args.node_rank))


if __name__ == "__main__":
    main()
