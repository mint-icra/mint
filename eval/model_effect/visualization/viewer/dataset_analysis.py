#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Background video, intrinsic, and diversity analysis for the viewer web app."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

from . import diversity_analysis as diversity
from .const import REPO_DIR

if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from tools.video_summary import video_summary as analyzer  # noqa: E402


class DatasetAnalysisManager:
    """Own one bounded dataset-analysis task and expose polling responses."""

    def __init__(self, *, default_root: Path):
        configured = os.environ.get("VIEWER_ANALYSIS_CACHE_DIR")
        self.cache_root = (Path(configured).expanduser().resolve() if configured else
                           Path(tempfile.gettempdir()) / "mint-viewer-cache" / "dataset-analysis")
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.default_root = Path(default_root).expanduser().resolve()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._result: dict | None = None
        self._exports: dict[str, Path] = {}
        self._task_id: str | None = None
        self._status = {
            "running": False, "stage": "idle", "phase": "待分析", "input_dir": "",
            "input_dirs": [],
            "analysis_type": "video",
            "selected_datasets": [], "sample_files": 24,
            "workers": 0, "discovered": 0, "total": 0, "done": 0, "failed": 0,
            "current": "", "error": None, "cancelled": False, "cached": False,
            "result_ready": False, "cache_key": None, "elapsed_sec": 0,
        }

    @staticmethod
    def _public_report(report: dict) -> dict:
        public = {key: value for key, value in report.items()
                  if key not in {"videos", "duplicate_candidates", "failed_files"}}
        public["distributions"] = {
            name: (list(items)[:100] if isinstance(items, list) else items)
            for name, items in (report.get("distributions") or {}).items()
        }
        public["duplicate_candidates"] = [
            {**item, "paths": list(item.get("paths") or [])[:25]}
            for item in list(report.get("duplicate_candidates") or [])[:20]
        ]
        public["failed_files"] = list(report.get("failed_files") or [])[:100]
        return public

    def status(self) -> dict:
        with self._lock:
            return dict(self._status)

    def _update(self, task_id: str, **values) -> None:
        with self._lock:
            if self._task_id == task_id:
                self._status.update(values)

    def start(self, *, input_dir: str, workers: int = 32, refresh: bool = False,
              analysis_type: str = "video", selected_datasets: list[str] | None = None,
              sample_files: int = 24, input_dirs: list[str] | None = None) -> dict:
        if analysis_type not in {"video", "intrinsics", "diversity"}:
            return {"ok": False, "error": "分析类型仅支持 video、intrinsics 或 diversity"}
        if analysis_type == "diversity":
            raw_roots = input_dirs if input_dirs is not None else [input_dir or diversity.DEFAULT_ROOT]
            if not raw_roots:
                return {"ok": False, "error": "请至少添加一个数据集目录"}
            if any(not isinstance(path, str) or not path.strip() for path in raw_roots):
                return {"ok": False, "error": "数据集目录必须是非空字符串"}
            diversity_roots = list(dict.fromkeys(
                Path(path).expanduser().resolve() for path in raw_roots))
            missing = next((path for path in diversity_roots if not path.is_dir()), None)
            if missing is not None:
                return {"ok": False, "error": f"目录不存在或不可访问: {missing}"}
            root = diversity_roots[0]
        else:
            root = Path(input_dir or self.default_root).expanduser().resolve()
            diversity_roots = []
            if not root.is_dir():
                return {"ok": False, "error": f"目录不存在或不可访问: {root}"}
        if analysis_type == "video" and shutil.which("ffprobe") is None:
            return {"ok": False, "error": "找不到 ffprobe，请先安装 ffmpeg"}
        default_workers = 32 if analysis_type == "video" else 8
        workers = max(1, min(int(workers or default_workers), 32))
        if analysis_type == "diversity":
            selected_datasets = ([] if selected_datasets is None else list(selected_datasets))
            if any(not isinstance(name, str) for name in selected_datasets):
                return {"ok": False, "error": "数据集名称必须是字符串"}
            selected_datasets = list(dict.fromkeys(selected_datasets))
            unknown = [name for name in selected_datasets if name not in diversity.DATASETS]
            if unknown:
                return {"ok": False, "error": f"未知数据集: {', '.join(unknown)}"}
        else:
            selected_datasets = []
        sample_files = max(1, min(int(sample_files or 24), 96))
        stage = {"video": "scanning", "intrinsics": "discovering_intrinsics",
                 "diversity": "discovering_diversity"}[analysis_type]
        phase = {"video": "递归扫描视频…", "intrinsics": "查找 LeRobot 数据集…",
                 "diversity": "检查所选 LeRobot 数据集…"}[analysis_type]
        with self._lock:
            if self._status["running"]:
                return {"ok": False, "error": "已有数据集分析任务正在运行"}
            task_id = uuid.uuid4().hex
            self._task_id = task_id
            self._cancel.clear()
            self._result = None
            self._exports = {}
            self._status = {
                "running": True,
                "stage": stage, "phase": phase,
                "input_dir": str(root), "workers": workers, "discovered": 0,
                "input_dirs": [str(path) for path in diversity_roots],
                "analysis_type": analysis_type, "selected_datasets": selected_datasets,
                "sample_files": sample_files,
                "total": 0, "done": 0, "failed": 0, "current": "",
                "error": None, "cancelled": False, "cached": False,
                "result_ready": False, "cache_key": None, "elapsed_sec": 0,
            }
        threading.Thread(target=self._worker,
                         args=(task_id, root, workers, bool(refresh), analysis_type,
                               selected_datasets, sample_files, diversity_roots),
                         name="dataset-analysis", daemon=True).start()
        return {"ok": True, "task_id": task_id, "input_dir": str(root), "workers": workers,
                "analysis_type": analysis_type, "selected_datasets": selected_datasets,
                "sample_files": sample_files,
                "input_dirs": [str(path) for path in diversity_roots]}

    def cancel(self) -> dict:
        with self._lock:
            if not self._status["running"]:
                return {"ok": False, "error": "当前没有运行中的数据集分析"}
            self._status.update(stage="cancelling", phase="正在取消分析任务…")
        self._cancel.set()
        return {"ok": True}

    def _on_progress(self, task_id: str, update: dict) -> None:
        labels = {
            "scanning": "递归扫描视频…", "probing": "读取视频元数据…",
            "discovering_intrinsics": "查找 LeRobot 数据集…",
            "intrinsics": "采样 LeRobot episode 首帧内参…",
            "aggregating_intrinsics": "汇总 FOV 与焦距分布…",
            "discovering_diversity": "检查所选 LeRobot 数据集…",
            "diversity_labels": "统计自然语言场景与动作…",
            "diversity_motion": "采样相机与手部活动范围…",
            "aggregating": "汇总分布与质量诊断…", "done": "分析完成",
        }
        stage = update.get("stage") or "probing"
        values = dict(update)
        values["phase"] = labels.get(stage, stage)
        self._update(task_id, **values)

    def _worker(self, task_id: str, root: Path, workers: int, refresh: bool,
                analysis_type: str, selected_datasets: list[str], sample_files: int,
                diversity_roots: list[Path]) -> None:
        started = time.monotonic()
        try:
            def progress(update):
                self._on_progress(task_id, update)
            datasets = None
            if analysis_type == "video":
                roots, records = analyzer.discover_videos(
                    [root], cancel_event=self._cancel, progress_callback=progress)
                if not records:
                    raise ValueError("该目录下没有找到支持的视频文件")
                key = analyzer.analysis_cache_key(
                    roots, records, include_intrinsics=False, analysis_type="video")
                total_items = len(records)
                discovered_items = len(records)
            elif analysis_type == "intrinsics":
                roots, records = [root], []
                datasets = analyzer.discover_lerobot_datasets(
                    roots, cancel_event=self._cancel, progress_callback=progress)
                key = analyzer.analysis_cache_key(
                    roots, records, include_intrinsics=True,
                    intrinsic_datasets=datasets, analysis_type="intrinsics")
                total_items = sum(len(item.get("parquet_files") or []) for item in datasets)
                discovered_items = len(datasets)
            else:
                roots, records = diversity_roots, []
                datasets = (diversity.discover_datasets(root, selected_datasets)
                            if selected_datasets else
                            diversity.discover_input_paths(diversity_roots))
                selected_datasets = [dataset["name"] for dataset in datasets]
                key = diversity.analysis_cache_key(datasets, sample_files)
                total_items = len(datasets) * 2
                discovered_items = len(datasets)
                self._update(task_id, selected_datasets=selected_datasets)
            output_dir = self.cache_root / key
            json_path = output_dir / "result.json"
            self._update(task_id, total=total_items, discovered=discovered_items, cache_key=key,
                         stage="cache", phase="检查分析缓存…")

            report = None
            cached = False
            if not refresh and json_path.is_file():
                try:
                    candidate = json.loads(json_path.read_text(encoding="utf-8"))
                    schema_version = (diversity.SCHEMA_VERSION if analysis_type == "diversity"
                                      else analyzer.SCHEMA_VERSION)
                    if candidate.get("schema_version") == schema_version \
                            and candidate.get("cache_key") == key \
                            and candidate.get("analysis_type") == analysis_type:
                        report, cached = candidate, True
                except (OSError, json.JSONDecodeError):
                    report = None
            if report is None:
                if analysis_type == "video":
                    report = analyzer.analyze_dataset(
                        [root], workers=workers, cancel_event=self._cancel,
                        progress_callback=progress, roots=roots, records=records,
                        cache_key=key, include_intrinsics=False)
                elif analysis_type == "intrinsics":
                    report = analyzer.analyze_intrinsic_dataset(
                        [root], workers=workers, cancel_event=self._cancel,
                        progress_callback=progress, roots=roots, datasets=datasets,
                        cache_key=key)
                else:
                    report = diversity.analyze(
                        root, selected_datasets, sample_files,
                        component_cache_root=self.cache_root / "diversity-components",
                        cancel_event=self._cancel, progress_callback=progress,
                        datasets=datasets, cache_key=key, refresh=refresh)
                if self._cancel.is_set():
                    raise diversity.AnalysisCancelled("数据集分析已取消")
                paths = (diversity.write_report(report, output_dir)
                         if analysis_type == "diversity"
                         else analyzer.write_report(report, output_dir))
            else:
                csv_name = "datasets.csv" if analysis_type == "diversity" else "videos.csv"
                paths = {"json": json_path, "csv": output_dir / csv_name,
                         "summary": output_dir / "summary.txt"}
                if not paths["csv"].is_file() or not paths["summary"].is_file():
                    paths = (diversity.write_report(report, output_dir)
                             if analysis_type == "diversity"
                             else analyzer.write_report(report, output_dir))

            with self._lock:
                if self._task_id != task_id:
                    return
                self._result = report
                self._exports = paths
                overview = report.get("overview") or {}
                intrinsic = report.get("intrinsics") or {}
                if analysis_type == "intrinsics":
                    result_total = intrinsic.get("total_parquet_files", 0)
                    result_failed = intrinsic.get("failed_parquet_files", 0)
                elif analysis_type == "diversity":
                    result_total = len(report.get("datasets") or []) * 2
                    result_failed = 0
                else:
                    result_total = len(records)
                    result_failed = overview.get("parsed_failed", 0)
                self._status.update(
                    running=False, stage="done", phase="已从缓存读取" if cached else "分析完成",
                    discovered=(len(intrinsic.get("dataset_roots") or [])
                                if analysis_type == "intrinsics" else
                                len(report.get("datasets") or [])
                                if analysis_type == "diversity" else
                                overview.get("total_files", len(records))),
                    total=result_total, done=result_total, failed=result_failed, current="",
                    error=None, cancelled=False, cached=cached, result_ready=True,
                    cache_key=key, elapsed_sec=round(time.monotonic() - started, 3),
                    summary=self._public_report(report),
                )
        except (analyzer.AnalysisCancelled, diversity.AnalysisCancelled):
            self._update(task_id, running=False, stage="cancelled", phase="已取消",
                         cancelled=True, result_ready=False,
                         elapsed_sec=round(time.monotonic() - started, 3))
        except Exception as exc:  # noqa: BLE001
            self._update(task_id, running=False, stage="error", phase="分析失败",
                         error=str(exc), result_ready=False,
                         elapsed_sec=round(time.monotonic() - started, 3))

    def result_page(self, *, page: int = 1, page_size: int = 100, search: str = "",
                    anomaly: str = "", codec: str = "", orientation: str = "",
                    resolution: str = "", sort: str = "relative_path", order: str = "asc") -> dict:
        with self._lock:
            report = self._result
            if report is None:
                raise LookupError("尚无可用的数据集分析结果")
            if report.get("analysis_type") in {"intrinsics", "diversity"}:
                raise LookupError("该分析类型没有逐视频分页明细")
        rows = report.get("videos") or []
        needle = search.strip().casefold()
        if needle:
            rows = [row for row in rows if needle in " ".join(str(row.get(field) or "") for field in
                    ("relative_path", "path", "folder", "source", "codec", "resolution")).casefold()]
        if anomaly == "failed":
            rows = [row for row in rows if not row.get("ok")]
        elif anomaly == "clean":
            rows = [row for row in rows if row.get("ok") and not row.get("anomalies")]
        elif anomaly == "any":
            rows = [row for row in rows if not row.get("ok") or row.get("anomalies")]
        elif anomaly:
            rows = [row for row in rows if anomaly in (row.get("anomalies") or [])]
        if codec:
            rows = [row for row in rows if row.get("codec") == codec]
        if orientation:
            rows = [row for row in rows if row.get("orientation") == orientation]
        if resolution:
            rows = [row for row in rows if row.get("resolution") == resolution]

        allowed_sort = {"relative_path", "duration_sec", "fps", "frame_count", "width", "height",
                        "resolution", "codec", "bitrate_bps", "size_bytes", "orientation"}
        sort = sort if sort in allowed_sort else "relative_path"

        def sort_key(row):
            value = row.get(sort)
            if isinstance(value, str):
                value = value.casefold()
            return value is None, value

        order = "desc" if order == "desc" else "asc"
        available = [row for row in rows if row.get(sort) is not None]
        missing = [row for row in rows if row.get(sort) is None]
        rows = sorted(available, key=sort_key, reverse=(order == "desc")) + missing
        page_size = max(10, min(int(page_size or 100), 500))
        total = len(rows)
        pages = max(1, (total + page_size - 1) // page_size)
        page = max(1, min(int(page or 1), pages))
        offset = (page - 1) * page_size
        distributions = report.get("distributions") or {}
        return {
            "rows": rows[offset:offset + page_size], "total": total, "page": page,
            "page_size": page_size, "pages": pages, "sort": sort, "order": order,
            "filters": {
                "codecs": [item["label"] for item in distributions.get("codec", [])],
                "resolutions": [item["label"] for item in distributions.get("resolution", [])],
                "orientations": [item["label"] for item in distributions.get("orientation", [])],
                "anomalies": distributions.get("anomalies", []),
            },
        }

    def export_path(self, format_name: str) -> tuple[Path, str]:
        key = {"json": "json", "csv": "csv", "txt": "summary"}.get(format_name)
        with self._lock:
            path = self._exports.get(key) if key else None
            root = Path(self._status.get("input_dir") or "dataset").name or "dataset"
            analysis_type = self._status.get("analysis_type") or "video"
        if path is None or not path.is_file():
            raise LookupError("所选导出文件尚未生成")
        suffix = ".txt" if format_name == "txt" else f".{format_name}"
        label = ({"intrinsics": "intrinsics_analysis", "diversity": "diversity_analysis"}
                 .get(analysis_type, "video_analysis"))
        return path, f"{root}_{label}{suffix}"
