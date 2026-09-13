#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""递归分析视频数据集的构成、媒体格式和数据质量。

命令行用法::

    python tools/video_summary/video_summary.py --input <dir1> [<dir2> ...]

默认产出 ``result.json``（完整统计与逐视频元数据）、``videos.csv``（明细表）和
``summary.txt``。模块中的 :func:`analyze_dataset` 同时供可视化网页调用。分析只读取
目录项和 ffprobe 元数据，不解码视频帧，也不会修改数据集。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


SCHEMA_VERSION = 8
DEFAULT_DURATION_BINS = [10, 30, 60, 120, 300, 600]
FOV_BINS_DEG = [30, 45, 60, 75, 90, 105, 120, 150]
VIDEO_EXTENSIONS = (
    ".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm", ".flv", ".wmv",
    ".mpg", ".mpeg", ".ts", ".mts", ".m2ts", ".3gp", ".ogv",
)

ANOMALY_SPECS = {
    "missing_duration": ("缺少时长", "warning"),
    "zero_duration": ("时长为零或过短", "error"),
    "very_long": ("超长视频（>4h）", "warning"),
    "missing_fps": ("缺少 FPS", "warning"),
    "suspicious_fps": ("异常 FPS（<5 或 >240）", "warning"),
    "vfr": ("疑似可变帧率 VFR", "info"),
    "missing_frame_count": ("缺少精确帧数", "info"),
    "invalid_resolution": ("分辨率无效", "error"),
    "low_resolution": ("低分辨率（短边<360）", "info"),
    "unusual_aspect_ratio": ("异常宽高比", "warning"),
    "missing_codec": ("缺少视频编码信息", "warning"),
    "missing_bitrate": ("缺少码率", "info"),
    "duration_mismatch": ("容器/视频流时长不一致", "warning"),
    "frame_count_mismatch": ("帧数与 FPS×时长不一致", "warning"),
    "no_audio": ("无音频流", "info"),
}

ProgressCallback = Callable[[dict], None]


class AnalysisCancelled(RuntimeError):
    """Raised when a web or CLI caller cancels an in-flight scan."""


def _cancelled(cancel_event) -> bool:
    return bool(cancel_event is not None and cancel_event.is_set())


def _progress(callback: ProgressCallback | None, **values) -> None:
    if callback is not None:
        callback(values)


def _as_float(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rate(rate):
    if not rate or rate in {"0/0", "N/A"}:
        return None
    try:
        if "/" in str(rate):
            numerator, denominator = str(rate).split("/", 1)
            denominator = float(denominator)
            return float(numerator) / denominator if denominator else None
        return float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _clean_number(value, digits=6):
    return None if value is None else round(float(value), digits)


def _ratio_value(value):
    if not value or value in {"0:1", "N/A"}:
        return None
    separator = ":" if ":" in str(value) else "/"
    try:
        numerator, denominator = str(value).split(separator, 1)
        denominator = float(denominator)
        return float(numerator) / denominator if denominator else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _orientation(width, height):
    if not width or not height:
        return "unknown"
    ratio = width / height
    if 0.98 <= ratio <= 1.02:
        return "square"
    return "landscape" if width > height else "portrait"


def _source_for(path: Path, roots: list[Path]) -> tuple[str, str, str]:
    for index, root in enumerate(roots):
        try:
            relative = path.relative_to(root)
            source = root.name or str(root)
            if len(roots) > 1:
                source = f"{index + 1}:{source}"
            folder = relative.parent.as_posix()
            return source, relative.as_posix(), "." if folder == "." else folder
        except ValueError:
            continue
    return "other", path.name, "."


def discover_videos(input_dirs: Iterable[str | os.PathLike], *,
                    extensions: Iterable[str] = VIDEO_EXTENSIONS,
                    cancel_event=None, progress_callback: ProgressCallback | None = None) -> tuple[list[Path], list[dict]]:
    """Find all supported videos and capture the stat snapshot used by the cache key."""
    roots = [Path(item).expanduser().resolve() for item in input_dirs]
    if not roots:
        raise ValueError("至少需要一个输入目录")
    invalid = [str(root) for root in roots if not root.is_dir()]
    if invalid:
        raise ValueError("输入不是可访问目录: " + ", ".join(invalid))

    allowed = {str(ext).lower() if str(ext).startswith(".") else "." + str(ext).lower()
               for ext in extensions}
    records, seen = [], set()
    last_update = 0.0
    for root in roots:
        for current, directories, filenames in os.walk(root, followlinks=False):
            if _cancelled(cancel_event):
                raise AnalysisCancelled("数据集扫描已取消")
            directories.sort(key=str.lower)
            filenames.sort(key=str.lower)
            for filename in filenames:
                if Path(filename).suffix.lower() not in allowed:
                    continue
                path = Path(current) / filename
                key = os.path.normcase(os.path.abspath(path))
                if key in seen:
                    continue
                seen.add(key)
                source, relative, folder = _source_for(path, roots)
                try:
                    stat = path.stat()
                    size, modified_ns = int(stat.st_size), int(stat.st_mtime_ns)
                    modified_time = datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()
                except OSError:
                    size, modified_ns, modified_time = None, None, None
                records.append({
                    "path": str(path), "relative_path": relative, "source": source,
                    "folder": folder, "extension": path.suffix.lower(),
                    "size_bytes": size, "modified_ns": modified_ns, "modified_time": modified_time,
                })
                now = time.monotonic()
                if len(records) == 1 or len(records) % 250 == 0 or now - last_update > 0.5:
                    _progress(progress_callback, stage="scanning", discovered=len(records),
                              current=str(path))
                    last_update = now
    return roots, records


def analysis_cache_key(roots: Iterable[Path], records: Iterable[dict], *,
                       duration_bins: Iterable[float] = DEFAULT_DURATION_BINS,
                       include_intrinsics: bool = True,
                       intrinsic_datasets: Iterable[dict] | None = None,
                       analysis_type: str = "combined") -> str:
    roots = list(roots)
    records = list(records)
    digest = hashlib.sha256()
    digest.update(f"video-summary-schema:{SCHEMA_VERSION}:{analysis_type}\n".encode())
    for root in roots:
        digest.update((str(root) + "\n").encode("utf-8", "surrogateescape"))
    digest.update((",".join(f"{float(value):g}" for value in duration_bins) + "\n").encode())
    for record in records:
        token = (record["path"], record.get("size_bytes"), record.get("modified_ns"))
        digest.update((json.dumps(token, ensure_ascii=False) + "\n").encode("utf-8", "surrogateescape"))
    if include_intrinsics:
        # FOV lives in Parquet, so only an explicitly requested intrinsic
        # analysis pays the recursive discovery/stat cost.
        datasets = (list(intrinsic_datasets) if intrinsic_datasets is not None
                    else _discover_lerobot_datasets(roots, records))
        for dataset in datasets:
            for path in [dataset["info_path"], *dataset["parquet_files"]]:
                try:
                    stat = path.stat()
                    token = (str(path), int(stat.st_size), int(stat.st_mtime_ns))
                except OSError:
                    token = (str(path), None, None)
                digest.update((json.dumps(token, ensure_ascii=False) + "\n").encode(
                    "utf-8", "surrogateescape"))
    return digest.hexdigest()[:24]


def _run_ffprobe(path: str, *, timeout: float, cancel_event=None) -> tuple[dict | None, str | None]:
    command = [
        "ffprobe", "-v", "error", "-show_streams", "-show_format",
        "-print_format", "json", path,
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, errors="replace")
    except FileNotFoundError:
        return None, "找不到 ffprobe，请安装 ffmpeg"
    deadline = time.monotonic() + max(1.0, float(timeout))
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if _cancelled(cancel_event):
                process.kill()
                process.communicate()
                raise AnalysisCancelled("视频解析已取消")
            if time.monotonic() >= deadline:
                process.kill()
                process.communicate()
                return None, f"ffprobe 超时（>{timeout:g}s）"
    if process.returncode != 0:
        return None, (stderr or "ffprobe failed").strip()[:500]
    try:
        return json.loads(stdout), None
    except json.JSONDecodeError:
        return None, "ffprobe 输出不是有效 JSON"


def _rotation(stream: dict) -> int:
    raw = (stream.get("tags") or {}).get("rotate")
    for item in stream.get("side_data_list") or []:
        if item.get("rotation") is not None:
            raw = item["rotation"]
            break
    value = _as_float(raw) or 0.0
    return int(round(value / 90.0) * 90) % 360


def _container_label(format_name) -> str:
    names = [name for name in str(format_name or "unknown").split(",") if name]
    preferred = next((name for name in names if name not in {"mov", "mp4", "m4a", "3gp", "3g2", "mj2"}), None)
    if preferred:
        return preferred
    return "mp4/mov" if any(name in {"mov", "mp4"} for name in names) else (names[0] if names else "unknown")


def _creation_time(format_data: dict, stream: dict):
    return ((format_data.get("tags") or {}).get("creation_time")
            or (stream.get("tags") or {}).get("creation_time"))


def _parse_probe_data(base: dict, data: dict) -> dict:
    streams = data.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("没有视频流")
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    fmt = data.get("format") or {}

    width, height = _as_int(video.get("width")), _as_int(video.get("height"))
    rotation = _rotation(video)
    display_width, display_height = (height, width) if rotation in {90, 270} else (width, height)
    avg_fps = _parse_rate(video.get("avg_frame_rate"))
    nominal_fps = _parse_rate(video.get("r_frame_rate"))
    fps = avg_fps or nominal_fps
    stream_duration = _as_float(video.get("duration"))
    format_duration = _as_float(fmt.get("duration"))
    duration = stream_duration or format_duration
    precise_frames = _as_int(video.get("nb_frames"))
    estimated_frames = precise_frames is None and duration is not None and fps is not None
    frame_count = precise_frames if precise_frames is not None else (
        int(round(duration * fps)) if estimated_frames else None)
    sample_aspect = _ratio_value(video.get("sample_aspect_ratio")) or 1.0
    display_aspect = _ratio_value(video.get("display_aspect_ratio"))
    if display_aspect is None and display_width and display_height:
        display_aspect = display_width * sample_aspect / display_height
    megapixels = (width * height / 1_000_000.0) if width and height else None
    video_bitrate = _as_int(video.get("bit_rate"))
    format_bitrate = _as_int(fmt.get("bit_rate"))
    bitrate = video_bitrate or format_bitrate
    size_bytes = base.get("size_bytes")
    if size_bytes is None:
        size_bytes = _as_int(fmt.get("size"))
    if avg_fps is None or nominal_fps is None:
        frame_rate_mode = "unknown"
    # r_frame_rate may round 29.97/59.94 to 30/60 for perfectly CFR media.
    elif abs(avg_fps - nominal_fps) > max(0.05, avg_fps * 0.005):
        frame_rate_mode = "variable"
    else:
        frame_rate_mode = "constant"

    anomalies = []
    if duration is None:
        anomalies.append("missing_duration")
    elif duration < 0.1:
        anomalies.append("zero_duration")
    elif duration > 4 * 3600:
        anomalies.append("very_long")
    if fps is None:
        anomalies.append("missing_fps")
    elif fps < 5 or fps > 240:
        anomalies.append("suspicious_fps")
    if frame_rate_mode == "variable":
        anomalies.append("vfr")
    if precise_frames is None:
        anomalies.append("missing_frame_count")
    if not width or not height:
        anomalies.append("invalid_resolution")
    else:
        if min(width, height) < 360:
            anomalies.append("low_resolution")
        ratio = display_aspect or width / height
        if ratio < 0.45 or ratio > 3.0:
            anomalies.append("unusual_aspect_ratio")
    if not video.get("codec_name"):
        anomalies.append("missing_codec")
    if not bitrate:
        anomalies.append("missing_bitrate")
    if audio is None:
        anomalies.append("no_audio")
    if stream_duration and format_duration:
        delta = abs(stream_duration - format_duration)
        if delta > max(0.5, max(stream_duration, format_duration) * 0.01):
            anomalies.append("duration_mismatch")
    if precise_frames is not None and duration and fps:
        expected = duration * fps
        if abs(precise_frames - expected) > max(2.0, expected * 0.02):
            anomalies.append("frame_count_mismatch")

    return {
        **base,
        "ok": True,
        "duration_sec": _clean_number(duration, 3),
        "stream_duration_sec": _clean_number(stream_duration, 3),
        "format_duration_sec": _clean_number(format_duration, 3),
        "start_time_sec": _clean_number(_as_float(video.get("start_time")) or _as_float(fmt.get("start_time")), 3),
        "fps": _clean_number(fps, 6),
        "avg_fps": _clean_number(avg_fps, 6),
        "nominal_fps": _clean_number(nominal_fps, 6),
        "frame_rate_mode": frame_rate_mode,
        "frame_count": frame_count,
        "frame_count_estimated": estimated_frames,
        "width": width, "height": height,
        "display_width": display_width, "display_height": display_height,
        "rotation_deg": rotation,
        "resolution": f"{display_width}x{display_height}" if display_width and display_height else "unknown",
        "megapixels": _clean_number(megapixels, 3),
        "sample_aspect_ratio": video.get("sample_aspect_ratio") or "1:1",
        "display_aspect_ratio": _clean_number(display_aspect, 4),
        "orientation": _orientation(display_width, display_height),
        "codec": video.get("codec_name") or "unknown",
        "codec_long_name": video.get("codec_long_name"),
        "profile": video.get("profile"),
        "level": _as_int(video.get("level")),
        "pixel_format": video.get("pix_fmt") or "unknown",
        "field_order": video.get("field_order") or "unknown",
        "color_space": video.get("color_space") or "unknown",
        "color_transfer": video.get("color_transfer") or "unknown",
        "color_primaries": video.get("color_primaries") or "unknown",
        "bits_per_raw_sample": _as_int(video.get("bits_per_raw_sample")),
        "bitrate_bps": bitrate,
        "video_bitrate_bps": video_bitrate,
        "container": _container_label(fmt.get("format_name")),
        "container_long_name": fmt.get("format_long_name"),
        "has_audio": audio is not None,
        "audio_codec": (audio or {}).get("codec_name"),
        "audio_channels": _as_int((audio or {}).get("channels")),
        "audio_channel_layout": (audio or {}).get("channel_layout"),
        "audio_sample_rate_hz": _as_int((audio or {}).get("sample_rate")),
        "audio_bitrate_bps": _as_int((audio or {}).get("bit_rate")),
        "creation_time": _creation_time(fmt, video),
        "anomalies": list(dict.fromkeys(anomalies)),
    }


def probe_video(record: dict, *, timeout: float = 120, cancel_event=None) -> dict:
    data, error = _run_ffprobe(record["path"], timeout=timeout, cancel_event=cancel_event)
    if error:
        return {**record, "ok": False, "error": error, "anomalies": ["probe_failed"]}
    try:
        return _parse_probe_data(record, data or {})
    except (TypeError, ValueError) as exc:
        return {**record, "ok": False, "error": str(exc), "anomalies": ["probe_failed"]}


def _quantile(sorted_values: list[float], q: float):
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def _numeric_stats(values: Iterable, *, total=True, digits=3) -> dict:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    result = {"count": len(clean), "min": None, "p25": None, "median": None,
              "p75": None, "p90": None, "p95": None, "max": None, "mean": None}
    if total:
        result["total"] = round(sum(clean), digits)
    if not clean:
        return result
    result.update({
        "min": round(clean[0], digits), "p25": round(_quantile(clean, 0.25), digits),
        "median": round(_quantile(clean, 0.5), digits), "p75": round(_quantile(clean, 0.75), digits),
        "p90": round(_quantile(clean, 0.9), digits), "p95": round(_quantile(clean, 0.95), digits),
        "max": round(clean[-1], digits), "mean": round(sum(clean) / len(clean), digits),
    })
    return result


def _fmt_seconds_short(seconds):
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:g}s"
    if seconds < 3600:
        return f"{seconds / 60:g}min"
    return f"{seconds / 3600:g}h"


def _bucket_distribution(values, bins, *, unit="") -> list[dict]:
    bins = sorted(float(value) for value in bins)
    labels = [f"<{_fmt_seconds_short(bins[0]) if unit == 'duration' else f'{bins[0]:g}{unit}'}"]
    for lower, upper in zip(bins, bins[1:]):
        left = _fmt_seconds_short(lower) if unit == "duration" else f"{lower:g}{unit}"
        right = _fmt_seconds_short(upper) if unit == "duration" else f"{upper:g}{unit}"
        labels.append(f"{left}-{right}")
    labels.append(f">={_fmt_seconds_short(bins[-1]) if unit == 'duration' else f'{bins[-1]:g}{unit}'}")
    counts = [0] * len(labels)
    for value in values:
        if value is None:
            continue
        index = next((index for index, boundary in enumerate(bins) if value < boundary), len(bins))
        counts[index] += 1
    total = sum(counts)
    return [{"label": label, "count": count, "percent": round(count / total * 100, 2) if total else 0}
            for label, count in zip(labels, counts)]


def _counter_distribution(values, *, limit=None) -> list[dict]:
    counter = Counter(str(value) if value not in {None, ""} else "unknown" for value in values)
    items = counter.most_common(limit)
    total = sum(counter.values())
    return [{"label": label, "count": count, "percent": round(count / total * 100, 2) if total else 0}
            for label, count in items]


def _fps_label(value):
    if value is None:
        return "unknown"
    common = [15, 23.976, 24, 25, 29.97, 30, 50, 59.94, 60, 90, 120, 240]
    closest = min(common, key=lambda item: abs(item - value))
    if abs(closest - value) <= max(0.02, closest * 0.001):
        return f"{closest:g}"
    return f"{value:.2f}"


def _aspect_label(value):
    if value is None:
        return "unknown"
    known = [(16 / 9, "16:9"), (4 / 3, "4:3"), (3 / 2, "3:2"), (1, "1:1"), (9 / 16, "9:16"), (3 / 4, "3:4")]
    ratio, label = min(known, key=lambda item: abs(item[0] - value))
    return label if abs(ratio - value) < 0.015 else f"{value:.2f}:1"


def _dataset_root_for_video(path: Path) -> Path | None:
    """Return the nearest LeRobot root that lexically contains ``path``."""
    current = path if path.is_dir() else path.parent
    for candidate in (current, *current.parents):
        if (candidate / "meta" / "info.json").is_file() and (candidate / "data").is_dir():
            return candidate
    return None


def _video_shape_from_info(info: dict) -> tuple[int, int] | None:
    features = info.get("features") or {}
    preferred = features.get("observation.images.ego") or {}
    candidates = [preferred] + [value for value in features.values()
                                if isinstance(value, dict) and value.get("dtype") == "video"
                                and value is not preferred]
    for feature in candidates:
        shape = feature.get("shape") or []
        if len(shape) >= 2 and _as_int(shape[0]) and _as_int(shape[1]):
            return int(shape[0]), int(shape[1])
        video_info = feature.get("info") or {}
        height = _as_int(video_info.get("video.height"))
        width = _as_int(video_info.get("video.width"))
        if height and width:
            return height, width
    return None


def _load_lerobot_datasets(candidates: Iterable[Path]) -> list[dict]:
    datasets = []
    for root in sorted(set(candidates), key=str):
        info_path = root / "meta" / "info.json"
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        shape = _video_shape_from_info(info)
        parquet_files = sorted((root / "data").rglob("*.parquet"))
        datasets.append({
            "root": root, "info_path": info_path, "shape": shape,
            "parquet_files": parquet_files,
        })
    return datasets


def discover_lerobot_datasets(input_dirs: Iterable[str | os.PathLike], *, cancel_event=None,
                              progress_callback: ProgressCallback | None = None) -> list[dict]:
    """Find LeRobot roots directly, without stat-ing every video below them."""
    roots = [Path(item).expanduser().resolve() for item in input_dirs]
    if not roots:
        raise ValueError("至少需要一个输入目录")
    invalid = [str(root) for root in roots if not root.is_dir()]
    if invalid:
        raise ValueError("输入不是可访问目录: " + ", ".join(invalid))

    candidates: set[Path] = set()
    for root in roots:
        enclosing = _dataset_root_for_video(root)
        if enclosing is not None:
            candidates.add(enclosing)
            _progress(progress_callback, stage="discovering_intrinsics", discovered=len(candidates),
                      current=str(enclosing))
            continue
        for current, directories, _ in os.walk(root, followlinks=False):
            if _cancelled(cancel_event):
                raise AnalysisCancelled("内参数据集扫描已取消")
            directories.sort(key=str.lower)
            current_path = Path(current)
            directory_names = set(directories)
            if "meta" not in directory_names or "data" not in directory_names:
                continue
            info_path = current_path / "meta" / "info.json"
            if not info_path.is_file():
                continue
            candidates.add(current_path)
            _progress(progress_callback, stage="discovering_intrinsics", discovered=len(candidates),
                      current=str(current_path))
            # A LeRobot root owns its data/videos trees; nested datasets are not valid.
            directories[:] = []
    return _load_lerobot_datasets(candidates)


def _discover_lerobot_datasets(roots: Iterable[Path], records: Iterable[dict]) -> list[dict]:
    """Compatibility wrapper for callers that already discovered video records."""
    del records
    return discover_lerobot_datasets(roots)


def _arrow_matrix(column, width: int):
    """Convert fixed/list Arrow values into a dense float matrix."""
    import numpy as np

    array = column.combine_chunks()
    try:
        flat = np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float64)
        if flat.size == len(array) * width:
            return flat.reshape(len(array), width)
    except (AttributeError, TypeError, ValueError):
        pass
    values = array.to_pylist()
    return np.asarray(values, dtype=np.float64).reshape(len(values), width)


def _read_lerobot_intrinsic_file(path: Path, *, height: int, width: int) -> dict:
    """Sample the first FOV row of every episode in one LeRobot shard."""
    import numpy as np
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    schema_names = set(parquet.schema_arrow.names)
    frame_count = int(parquet.metadata.num_rows)
    fov_column = "cam_fov" if "cam_fov" in schema_names else (
        "fov" if "fov" in schema_names else None)
    if fov_column is None:
        return {"path": str(path), "annotated": False, "frames": frame_count,
                "sampled_frames": 0, "segments": 0}

    columns = [fov_column]
    if "episode_index" in schema_names:
        columns.append("episode_index")
    if "frame_index" in schema_names:
        table = pq.read_table(path, columns=columns, filters=[("frame_index", "=", 0)],
                              use_threads=False)
        sample_strategy = "frame_index_zero"
    else:
        table = parquet.read(columns=columns, use_threads=False)
        sample_strategy = "episode_boundary"
    fov_rad = _arrow_matrix(table.column(fov_column), 2)

    if "episode_index" in columns:
        episode_ids = np.asarray(table.column("episode_index").combine_chunks().to_numpy(
            zero_copy_only=False))
        _, first_indices = np.unique(episode_ids, return_index=True)
        first_indices.sort()
        samples = fov_rad[first_indices]
    else:
        # frame_index==0 already yields one row per episode. Very old shards
        # without either index can only provide one representative per file.
        samples = fov_rad if sample_strategy == "frame_index_zero" else fov_rad[:1]
    samples = np.asarray(samples, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(samples).all(axis=1)
    valid = finite & (samples > 0).all(axis=1) & (samples < np.pi).all(axis=1)
    reps = samples[valid]
    if len(reps):
        fov_v, fov_h = reps[:, 0], reps[:, 1]
        fy = (height / 2.0) / np.tan(fov_v / 2.0)
        fx = (width / 2.0) / np.tan(fov_h / 2.0)
        derived = np.column_stack((np.degrees(fov_v), np.degrees(fov_h), fx, fy,
                                   fx / width, fy / height, fx / fy))
    else:
        derived = np.empty((0, 7), dtype=np.float64)
    return {
        "path": str(path), "annotated": True, "column": fov_column,
        "frames": frame_count, "sampled_frames": len(samples),
        # Keep the old frame fields as aliases for report consumers; they now
        # count sampled episode-first frames rather than every source frame.
        "valid_frames": int(valid.sum()), "invalid_frames": int((~valid).sum()),
        "segments": len(samples), "valid_segments": len(reps),
        "invalid_segments": int((~valid).sum()), "sample_strategy": sample_strategy,
        "derived": derived,
    }


def _intrinsic_diagnostic(code: str, label: str, severity: str, count: int,
                          denominator: int, unit: str) -> dict:
    return {"code": code, "label": label, "severity": severity, "count": int(count),
            "percent": round(count / denominator * 100, 2) if denominator else 0,
            "unit": unit}


def analyze_intrinsics(roots: list[Path], records: list[dict] | None = None, *, workers: int,
                       cancel_event=None,
                       progress_callback: ProgressCallback | None = None,
                       datasets: list[dict] | None = None) -> dict:
    """Aggregate one episode-first intrinsic sample without decoding video frames.

    LeRobot stores ``cam_fov``/``fov`` as [vertical, horizontal] radians. Focal
    lengths are derived at the dataset video resolution. The schema does not
    store principal point, skew, or distortion, so those are reported as
    assumptions/unknowns instead of inferred measurements.
    """
    datasets = (datasets if datasets is not None
                else _discover_lerobot_datasets(roots, records or []))
    jobs = [(dataset, path) for dataset in datasets for path in dataset["parquet_files"]]
    base = {
        "available": False, "source": "none", "dataset_roots": [str(item["root"]) for item in datasets],
        "convention": "fov=[vertical, horizontal] radians",
        "sampling": "first_frame_per_episode",
        "assumptions": {
            "principal_point": "图像中心（cx=W/2, cy=H/2；标注未提供）",
            "skew": "按 0 处理（标注未提供）",
            "distortion": "不可用（标注未提供畸变系数）",
        },
        "total_parquet_files": len(jobs), "annotated_parquet_files": 0,
        "failed_parquet_files": 0, "total_frames": 0, "sampled_frames": 0,
        "valid_frames": 0, "invalid_frames": 0, "total_episodes": 0, "valid_episodes": 0,
        "statistics": {}, "distributions": {}, "diagnostics": [], "failed_files": [],
    }
    if not datasets:
        base["reason"] = "未在视频所在路径发现 LeRobot meta/info.json 与 data Parquet"
        return base
    if not jobs:
        base["reason"] = "发现 LeRobot 数据集，但 data 目录下没有 Parquet"
        return base
    try:
        import numpy as np
        import pyarrow  # noqa: F401
    except ImportError as exc:
        base["reason"] = f"读取 LeRobot 内参需要 pyarrow: {exc}"
        return base

    results = []
    failures = []
    intrinsic_workers = max(1, min(int(workers or 1), 8, len(jobs)))
    completed = 0
    _progress(progress_callback, stage="intrinsics", total=len(jobs), done=0, failed=0,
              current="", item_unit="parquet")
    with ThreadPoolExecutor(max_workers=intrinsic_workers,
                            thread_name_prefix="intrinsic-scan") as executor:
        futures = {
            executor.submit(_read_lerobot_intrinsic_file, path,
                            height=dataset["shape"][0], width=dataset["shape"][1]): (dataset, path)
            for dataset, path in jobs if dataset.get("shape")
        }
        for dataset, path in jobs:
            if not dataset.get("shape"):
                failures.append({"path": str(path), "error": "meta/info.json 缺少视频分辨率"})
                completed += 1
        while futures:
            if _cancelled(cancel_event):
                for future in futures:
                    future.cancel()
                raise AnalysisCancelled("内参分析已取消")
            done, _ = wait(futures, timeout=0.4, return_when=FIRST_COMPLETED)
            for future in done:
                dataset, path = futures.pop(future)
                try:
                    result = future.result()
                    result["dataset_root"] = str(dataset["root"])
                    result["height"], result["width"] = dataset["shape"]
                    results.append(result)
                except Exception as exc:  # noqa: BLE001
                    failures.append({"path": str(path), "error": str(exc)})
                completed += 1
                _progress(progress_callback, stage="intrinsics", total=len(jobs), done=completed,
                          failed=len(failures), current=str(path), item_unit="parquet")

    annotated = [item for item in results if item.get("annotated")]
    matrices = [item["derived"] for item in annotated if len(item["derived"])]
    values = np.concatenate(matrices, axis=0) if matrices else np.empty((0, 7), dtype=np.float64)
    total_frames = sum(item.get("frames", 0) for item in annotated)
    sampled_frames = sum(item.get("sampled_frames", 0) for item in annotated)
    valid_frames = sum(item.get("valid_frames", 0) for item in annotated)
    invalid_frames = sum(item.get("invalid_frames", 0) for item in annotated)
    total_episodes = sum(item.get("segments", 0) for item in annotated)
    valid_episodes = sum(item.get("valid_segments", 0) for item in annotated)
    invalid_episodes = sum(item.get("invalid_segments", 0) for item in annotated)
    extreme = int(((values[:, 0] < 20) | (values[:, 0] > 160)
                   | (values[:, 1] < 20) | (values[:, 1] > 160)).sum()) if len(values) else 0
    non_square = int(((values[:, 6] < 0.95) | (values[:, 6] > 1.05)).sum()) if len(values) else 0

    diagnostics = []
    if failures:
        diagnostics.append(_intrinsic_diagnostic(
            "intrinsic_file_failed", "内参 Parquet 读取失败", "error", len(failures), len(jobs), "文件"))
    if invalid_frames:
        diagnostics.append(_intrinsic_diagnostic(
            "invalid_fov", "episode 首帧 FOV 非法（非有限或不在 0-180°）", "error",
            invalid_frames, sampled_frames, "episode"))
    if invalid_episodes:
        diagnostics.append(_intrinsic_diagnostic(
            "episode_missing_fov", "episode 无有效 FOV", "error",
            invalid_episodes, total_episodes, "episode"))
    if extreme:
        diagnostics.append(_intrinsic_diagnostic(
            "extreme_fov", "极端 FOV（<20° 或 >160°）", "warning",
            extreme, valid_episodes, "episode"))
    if non_square:
        diagnostics.append(_intrinsic_diagnostic(
            "focal_aspect_mismatch", "fx/fy 偏离 1 超过 5%", "warning",
            non_square, valid_episodes, "episode"))

    columns = {
        "fov_vertical_deg": values[:, 0] if len(values) else [],
        "fov_horizontal_deg": values[:, 1] if len(values) else [],
        "fx_px": values[:, 2] if len(values) else [], "fy_px": values[:, 3] if len(values) else [],
        "fx_over_width": values[:, 4] if len(values) else [],
        "fy_over_height": values[:, 5] if len(values) else [],
        "fx_over_fy": values[:, 6] if len(values) else [],
    }
    distributions = {
        "fov_vertical_deg": _bucket_distribution(columns["fov_vertical_deg"], FOV_BINS_DEG, unit="°"),
        "fov_horizontal_deg": _bucket_distribution(columns["fov_horizontal_deg"], FOV_BINS_DEG, unit="°"),
        "fx_over_width": _bucket_distribution(columns["fx_over_width"], [0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5], unit="×W"),
        "fy_over_height": _bucket_distribution(columns["fy_over_height"], [0.2, 0.3, 0.4, 0.5, 0.75, 1, 1.5], unit="×H"),
        "fx_over_fy": _bucket_distribution(columns["fx_over_fy"], [0.9, 0.95, 0.98, 1.02, 1.05, 1.1], unit=""),
        "fov_pairs": _counter_distribution(
            (f"{vertical:.1f}° × {horizontal:.1f}°" for vertical, horizontal in values[:, :2]),
            limit=20),
    }
    base.update({
        "available": bool(annotated), "source": "lerobot_parquet" if annotated else "none",
        "reason": None if annotated else "LeRobot Parquet 中未发现 cam_fov 或 fov 列",
        "annotated_parquet_files": len(annotated), "failed_parquet_files": len(failures),
        "total_frames": total_frames, "sampled_frames": sampled_frames,
        "valid_frames": valid_frames,
        "invalid_frames": invalid_frames, "total_episodes": total_episodes,
        "valid_episodes": valid_episodes,
        "valid_frame_percent": round(valid_frames / sampled_frames * 100, 2)
        if sampled_frames else 0,
        "valid_episode_percent": round(valid_episodes / total_episodes * 100, 2) if total_episodes else 0,
        "statistics": {name: _numeric_stats(series, total=False, digits=4)
                       for name, series in columns.items()},
        "distributions": distributions, "diagnostics": diagnostics,
        "failed_files": failures[:100],
    })
    return base


def analyze_intrinsic_dataset(input_dirs: Iterable[str | os.PathLike], *, workers: int | None = None,
                              cancel_event=None,
                              progress_callback: ProgressCallback | None = None,
                              roots: list[Path] | None = None,
                              datasets: list[dict] | None = None,
                              cache_key: str | None = None) -> dict:
    """Build a standalone LeRobot intrinsic report without running ffprobe."""
    started = time.monotonic()
    roots = ([Path(item).expanduser().resolve() for item in input_dirs]
             if roots is None else roots)
    datasets = (discover_lerobot_datasets(
        roots, cancel_event=cancel_event, progress_callback=progress_callback)
        if datasets is None else datasets)
    workers = max(1, min(int(workers or os.cpu_count() or 4), 32))
    intrinsic = analyze_intrinsics(
        roots, workers=workers, cancel_event=cancel_event,
        progress_callback=progress_callback, datasets=datasets,
    )
    if _cancelled(cancel_event):
        raise AnalysisCancelled("内参分析已取消")
    _progress(progress_callback, stage="aggregating_intrinsics",
              total=intrinsic.get("total_parquet_files", 0),
              done=intrinsic.get("total_parquet_files", 0),
              failed=intrinsic.get("failed_parquet_files", 0),
              current="生成 episode 首帧 FOV 与焦距统计")
    report = {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": "intrinsics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "cache_key": cache_key,
        "input_dirs": [str(root) for root in roots],
        "workers": workers,
        "intrinsics": intrinsic,
        "videos": [], "duplicate_candidates": [],
        "failed_files": intrinsic.get("failed_files", []),
    }
    _progress(progress_callback, stage="done",
              total=intrinsic.get("total_parquet_files", 0),
              done=intrinsic.get("total_parquet_files", 0),
              failed=intrinsic.get("failed_parquet_files", 0), current="")
    return report


def _folder_composition(rows: list[dict]) -> list[dict]:
    groups = defaultdict(lambda: {"files": 0, "duration_sec": 0.0, "size_bytes": 0, "failed": 0})
    for row in rows:
        # First-level folder is useful for dataset sources while retaining root-level files.
        folder = row.get("folder") or "."
        top = folder.split("/", 1)[0] if folder != "." else "."
        key = f"{row.get('source', 'dataset')}/{top}" if top != "." else row.get("source", "dataset")
        item = groups[key]
        item["files"] += 1
        item["duration_sec"] += float(row.get("duration_sec") or 0)
        item["size_bytes"] += int(row.get("size_bytes") or 0)
        item["failed"] += 0 if row.get("ok") else 1
    total_files = len(rows)
    return [{"label": label, **values,
             "percent": round(values["files"] / total_files * 100, 2) if total_files else 0,
             "duration_sec": round(values["duration_sec"], 3)}
            for label, values in sorted(groups.items(), key=lambda item: (-item[1]["files"], item[0]))]


def _duplicate_candidates(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if not row.get("ok") or not row.get("size_bytes") or row.get("duration_sec") is None:
            continue
        key = (row["size_bytes"], round(row["duration_sec"], 3), row.get("width"), row.get("height"), row.get("codec"))
        groups[key].append(row["relative_path"])
    candidates = []
    for key, paths in groups.items():
        if len(paths) > 1:
            candidates.append({"count": len(paths), "size_bytes": key[0], "duration_sec": key[1],
                               "resolution": f"{key[2]}x{key[3]}", "codec": key[4],
                               "paths": paths[:200], "paths_truncated": max(0, len(paths) - 200)})
    candidates.sort(key=lambda item: (-item["count"], -item["size_bytes"]))
    return candidates[:500]


def build_summary(roots: list[Path], rows: list[dict], *, duration_bins, workers: int,
                  elapsed_sec: float, cache_key: str | None = None,
                  intrinsic_report: dict | None = None,
                  analysis_type: str = "combined") -> dict:
    rows = sorted(rows, key=lambda row: row.get("path", ""))
    ok = [row for row in rows if row.get("ok")]
    failed = [row for row in rows if not row.get("ok")]
    durations = [row.get("duration_sec") for row in ok]
    fps_values = [row.get("fps") for row in ok]
    frame_counts = [row.get("frame_count") for row in ok]
    sizes = [row.get("size_bytes") for row in rows]
    bitrates = [row.get("bitrate_bps") for row in ok]

    anomaly_counter = Counter(code for row in ok for code in row.get("anomalies", []))
    anomaly_distribution = []
    for code, count in anomaly_counter.most_common():
        label, severity = ANOMALY_SPECS.get(code, (code, "warning"))
        anomaly_distribution.append({"code": code, "label": label, "severity": severity,
                                     "count": count, "percent": round(count / len(ok) * 100, 2) if ok else 0})
    if failed:
        anomaly_distribution.insert(0, {"code": "probe_failed", "label": "解析失败/无视频流",
                                        "severity": "error", "count": len(failed),
                                        "percent": round(len(failed) / len(rows) * 100, 2) if rows else 0})

    format_counter = Counter(
        (row.get("resolution"), _fps_label(row.get("fps")), row.get("codec"), row.get("pixel_format"))
        for row in ok
    )
    format_rows = [{"resolution": key[0], "fps": key[1], "codec": key[2], "pixel_format": key[3],
                    "count": count, "percent": round(count / len(ok) * 100, 2) if ok else 0}
                   for key, count in format_counter.most_common(50)]
    duplicates = _duplicate_candidates(ok)
    exact_frames = sum(1 for row in ok if row.get("frame_count") is not None and not row.get("frame_count_estimated"))
    with_audio = sum(1 for row in ok if row.get("has_audio"))
    without_anomalies = sum(1 for row in ok if not row.get("anomalies"))
    total_duration = sum(float(value or 0) for value in durations)
    total_frames = sum(int(value or 0) for value in frame_counts)
    total_size = sum(int(value or 0) for value in sizes)

    distributions = {
        "duration": _bucket_distribution(durations, duration_bins, unit="duration"),
        "fps": _counter_distribution(_fps_label(value) for value in fps_values),
        "resolution": _counter_distribution(row.get("resolution") for row in ok),
        "codec": _counter_distribution(row.get("codec") for row in ok),
        "pixel_format": _counter_distribution(row.get("pixel_format") for row in ok),
        "container": _counter_distribution(row.get("container") for row in ok),
        "extension": _counter_distribution(row.get("extension") for row in rows),
        "orientation": _counter_distribution(row.get("orientation") for row in ok),
        "rotation": _counter_distribution(f"{row.get('rotation_deg', 0)}°" for row in ok),
        "aspect_ratio": _counter_distribution(_aspect_label(row.get("display_aspect_ratio")) for row in ok),
        "frame_rate_mode": _counter_distribution(row.get("frame_rate_mode") for row in ok),
        "audio_codec": _counter_distribution(row.get("audio_codec") or "none" for row in ok),
        "audio_sample_rate": _counter_distribution(
            f"{row.get('audio_sample_rate_hz')} Hz" if row.get("audio_sample_rate_hz") else "none"
            for row in ok),
        "profile": _counter_distribution(row.get("profile") or "unknown" for row in ok),
        "color_space": _counter_distribution(row.get("color_space") or "unknown" for row in ok),
        "size": _bucket_distribution([None if value is None else value / (1024 ** 2) for value in sizes],
                                      [10, 50, 100, 500, 1024, 5120], unit="MB"),
        "bitrate": _bucket_distribution([None if value is None else value / 1_000_000 for value in bitrates],
                                         [0.5, 1, 2, 5, 10, 25, 50], unit="Mbps"),
        "folders": _folder_composition(rows),
        "anomalies": anomaly_distribution,
        "formats": format_rows,
    }
    overview = {
        "total_files": len(rows), "parsed_ok": len(ok), "parsed_failed": len(failed),
        "total_duration_sec": round(total_duration, 3), "total_frames": total_frames,
        "total_size_bytes": total_size,
        "mean_duration_sec": round(total_duration / len(ok), 3) if ok else None,
        "median_duration_sec": _numeric_stats(durations, total=False).get("median"),
        "mean_fps": _numeric_stats(fps_values, total=False).get("mean"),
        "median_fps": _numeric_stats(fps_values, total=False).get("median"),
        "with_audio": with_audio, "exact_frame_count": exact_frames,
        "videos_with_anomalies": sum(1 for row in ok if row.get("anomalies")),
        "clean_videos": without_anomalies,
        "duplicate_groups": len(duplicates),
        "dominant_format_coverage": format_rows[0]["percent"] if format_rows else 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_type": analysis_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_sec": round(elapsed_sec, 3),
        "cache_key": cache_key,
        "input_dirs": [str(root) for root in roots],
        "workers": workers,
        "duration_bins_sec": list(duration_bins),
        "overview": overview,
        "statistics": {
            "duration_sec": _numeric_stats(durations),
            "fps": _numeric_stats(fps_values, total=False),
            "frame_count": _numeric_stats(frame_counts),
            "file_size_bytes": _numeric_stats(sizes),
            "bitrate_bps": _numeric_stats(bitrates, total=False),
            "megapixels": _numeric_stats([row.get("megapixels") for row in ok], total=False),
        },
        "distributions": distributions,
        "metadata_coverage": {
            "duration_percent": round(sum(value is not None for value in durations) / len(ok) * 100, 2) if ok else 0,
            "fps_percent": round(sum(value is not None for value in fps_values) / len(ok) * 100, 2) if ok else 0,
            "exact_frame_count_percent": round(exact_frames / len(ok) * 100, 2) if ok else 0,
            "bitrate_percent": round(sum(value is not None for value in bitrates) / len(ok) * 100, 2) if ok else 0,
            "audio_percent": round(with_audio / len(ok) * 100, 2) if ok else 0,
            "creation_time_percent": round(sum(bool(row.get("creation_time")) for row in ok) / len(ok) * 100, 2) if ok else 0,
        },
        "intrinsics": intrinsic_report or {
            "available": False, "source": "none", "reason": "未执行内参分析",
            "statistics": {}, "distributions": {}, "diagnostics": [],
        },
        "duplicate_candidates": duplicates,
        "failed_files": [{"path": row["path"], "relative_path": row.get("relative_path"),
                          "error": row.get("error", "未知错误")} for row in failed],
        "videos": rows,
        # Compatibility fields kept for existing consumers of the original script.
        "total_files": len(rows), "parsed_ok": len(ok), "parsed_failed": len(failed),
        "total_duration_sec": round(total_duration, 3),
        "total_duration_hms": format_hms(total_duration),
        "duration_distribution": {item["label"]: item["count"] for item in distributions["duration"]},
        "fps_distribution": {item["label"]: item["count"] for item in distributions["fps"]},
        "resolution_distribution": {item["label"]: item["count"] for item in distributions["resolution"]},
    }


def analyze_dataset(input_dirs: Iterable[str | os.PathLike], *, workers: int | None = None,
                    duration_bins: Iterable[float] = DEFAULT_DURATION_BINS,
                    timeout: float = 120, cancel_event=None,
                    progress_callback: ProgressCallback | None = None,
                    records: list[dict] | None = None, roots: list[Path] | None = None,
                    cache_key: str | None = None,
                    include_intrinsics: bool = True) -> dict:
    """Analyze all videos below ``input_dirs`` and return a JSON-serializable report."""
    start = time.monotonic()
    if records is None or roots is None:
        roots, records = discover_videos(input_dirs, cancel_event=cancel_event,
                                         progress_callback=progress_callback)
    if not records:
        raise ValueError("输入目录下没有找到支持的视频文件")
    workers = max(1, min(int(workers or os.cpu_count() or 4), 32, len(records)))
    _progress(progress_callback, stage="probing", discovered=len(records), total=len(records), done=0,
              failed=0, current="")

    results, completed, failed_count = [], 0, 0
    iterator = iter(records)
    max_inflight = max(workers * 2, workers)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-probe") as executor:
        futures = {}

        def submit_more():
            while len(futures) < max_inflight and not _cancelled(cancel_event):
                try:
                    record = next(iterator)
                except StopIteration:
                    break
                futures[executor.submit(probe_video, record, timeout=timeout,
                                        cancel_event=cancel_event)] = record

        submit_more()
        while futures:
            if _cancelled(cancel_event):
                for future in futures:
                    future.cancel()
                raise AnalysisCancelled("数据集分析已取消")
            done, _ = wait(futures, timeout=0.4, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                record = futures.pop(future)
                try:
                    row = future.result()
                except AnalysisCancelled:
                    raise
                except Exception as exc:  # keep one malformed file from aborting a dataset
                    row = {**record, "ok": False, "error": str(exc), "anomalies": ["probe_failed"]}
                results.append(row)
                completed += 1
                failed_count += 0 if row.get("ok") else 1
                _progress(progress_callback, stage="probing", discovered=len(records), total=len(records),
                          done=completed, failed=failed_count, current=record["relative_path"])
            submit_more()

    if _cancelled(cancel_event):
        raise AnalysisCancelled("数据集分析已取消")
    intrinsic_report = None
    if include_intrinsics:
        intrinsic_report = analyze_intrinsics(
            roots, results, workers=workers, cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
    if _cancelled(cancel_event):
        raise AnalysisCancelled("数据集分析已取消")
    _progress(progress_callback, stage="aggregating", discovered=len(records), total=len(records),
              done=completed, failed=failed_count,
              current="生成媒体与内参统计诊断" if include_intrinsics else "生成视频统计诊断")
    report = build_summary(roots, results, duration_bins=list(duration_bins), workers=workers,
                           elapsed_sec=time.monotonic() - start, cache_key=cache_key,
                           intrinsic_report=intrinsic_report,
                           analysis_type="combined" if include_intrinsics else "video")
    _progress(progress_callback, stage="done", discovered=len(records), total=len(records),
              done=completed, failed=failed_count, current="")
    return report


def format_hms(total_seconds) -> str:
    value = max(0, int(round(float(total_seconds or 0))))
    hours, remainder = divmod(value, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


CSV_FIELDS = [
    "relative_path", "source", "folder", "extension", "ok", "duration_sec", "fps",
    "nominal_fps", "frame_rate_mode", "frame_count", "frame_count_estimated", "width",
    "height", "rotation_deg", "resolution", "megapixels", "display_aspect_ratio", "orientation",
    "codec", "profile", "pixel_format", "bitrate_bps", "size_bytes", "container", "has_audio",
    "audio_codec", "audio_channels", "audio_sample_rate_hz", "creation_time", "modified_time",
    "anomalies", "error", "path",
]


def write_report(report: dict, output_dir: str | os.PathLike) -> dict[str, Path]:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "result.json"
    csv_path = output / "videos.csv"
    summary_path = output / "summary.txt"

    temporary = output / f".result.{os.getpid()}-{threading.get_ident()}.json"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    temporary.replace(json_path)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in report.get("videos", []):
            row = dict(row)
            row["anomalies"] = ";".join(row.get("anomalies") or [])
            writer.writerow(row)
    summary_path.write_text(render_summary(report) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "summary": summary_path}


def render_summary(report: dict) -> str:
    if report.get("analysis_type") == "intrinsics":
        intrinsic = report.get("intrinsics") or {}
        lines = ["=" * 72, "LeRobot 相机内参分析", "=" * 72,
                 f"检索目录 : {', '.join(report.get('input_dirs') or [])}",
                 "采样策略 : 每个 episode 的首帧（frame_index=0）",
                 f"Parquet  : {intrinsic.get('annotated_parquet_files', 0):,} 个含 FOV / "
                 f"{intrinsic.get('total_parquet_files', 0):,} 个总计",
                 f"Episode  : {intrinsic.get('valid_episodes', 0):,} 有效 / "
                 f"{intrinsic.get('total_episodes', 0):,} 个采样"]
        if intrinsic.get("available"):
            stats = intrinsic.get("statistics") or {}
            vertical = stats.get("fov_vertical_deg") or {}
            horizontal = stats.get("fov_horizontal_deg") or {}
            fx = stats.get("fx_px") or {}
            fy = stats.get("fy_px") or {}
            ratio = stats.get("fx_over_fy") or {}
            lines.extend([
                f"垂直 FoV : median={vertical.get('median')}°  p25={vertical.get('p25')}°  "
                f"p95={vertical.get('p95')}°",
                f"水平 FoV : median={horizontal.get('median')}°  p25={horizontal.get('p25')}°  "
                f"p95={horizontal.get('p95')}°",
                f"焦距像素 : fx median={fx.get('median')}  fy median={fy.get('median')}  "
                f"fx/fy median={ratio.get('median')}",
                "主点/畸变: cx,cy 按图像中心假设；skew 按 0；畸变系数未提供",
            ])
            for item in intrinsic.get("diagnostics") or []:
                lines.append(f"! {item['label']:<34} {item['count']:>7} "
                             f"{item.get('unit', '')} ({item['percent']}%)")
        else:
            lines.append(f"不可用    : {intrinsic.get('reason') or '未发现 LeRobot FOV 标注'}")
        return "\n".join(lines)

    overview = report["overview"]
    stats = report["statistics"]
    coverage = report["metadata_coverage"]
    lines = ["=" * 72, "视频数据集构成与质量分析", "=" * 72,
             f"检索目录 : {', '.join(report['input_dirs'])}",
             f"视频总数 : {overview['total_files']}（成功 {overview['parsed_ok']} / 失败 {overview['parsed_failed']}）",
             f"总时长   : {format_hms(overview['total_duration_sec'])}（{overview['total_duration_sec'] / 3600:.3f} h）",
             f"总帧数   : {overview['total_frames']:,}",
             f"总容量   : {overview['total_size_bytes'] / 1024 ** 3:.3f} GiB",
             f"时长统计 : median={stats['duration_sec']['median']}s  p95={stats['duration_sec']['p95']}s  max={stats['duration_sec']['max']}s",
             f"FPS 统计 : median={stats['fps']['median']}  min={stats['fps']['min']}  max={stats['fps']['max']}",
             f"元数据率 : duration={coverage['duration_percent']}%  fps={coverage['fps_percent']}%  exact_frames={coverage['exact_frame_count_percent']}%",
             "", "[主要格式]"]
    for item in report["distributions"]["formats"][:15]:
        lines.append(f"  {item['resolution']:<12} {item['fps']:>7} fps  {item['codec']:<9} {item['pixel_format']:<12} {item['count']:>7} ({item['percent']}%)")
    lines.extend(["", "[目录构成]"])
    for item in report["distributions"]["folders"][:20]:
        lines.append(f"  {item['label']:<34} {item['files']:>7} files  {item['duration_sec'] / 3600:>9.3f} h  {item['size_bytes'] / 1024 ** 3:>9.3f} GiB")
    intrinsic = report.get("intrinsics") or {}
    lines.extend(["", "[相机内参]"])
    if intrinsic.get("available"):
        intrinsic_stats = intrinsic.get("statistics") or {}
        vertical = intrinsic_stats.get("fov_vertical_deg") or {}
        horizontal = intrinsic_stats.get("fov_horizontal_deg") or {}
        fx = intrinsic_stats.get("fx_px") or {}
        fy = intrinsic_stats.get("fy_px") or {}
        ratio = intrinsic_stats.get("fx_over_fy") or {}
        lines.extend([
            f"  标注来源 : LeRobot cam_fov/fov（有效 {intrinsic.get('valid_episodes', 0):,} / "
            f"{intrinsic.get('total_episodes', 0):,} episode）",
            f"  垂直 FoV : median={vertical.get('median')}°  p25={vertical.get('p25')}°  "
            f"p95={vertical.get('p95')}°",
            f"  水平 FoV : median={horizontal.get('median')}°  p25={horizontal.get('p25')}°  "
            f"p95={horizontal.get('p95')}°",
            f"  焦距像素 : fx median={fx.get('median')}  fy median={fy.get('median')}  "
            f"fx/fy median={ratio.get('median')}",
            f"  采样策略 : 每个 episode 首帧，共 {intrinsic.get('sampled_frames', 0):,} 个样本",
            "  主点/畸变: cx,cy 按图像中心假设；skew 按 0；畸变系数未提供",
        ])
        for item in intrinsic.get("diagnostics") or []:
            lines.append(f"  ! {item['label']:<30} {item['count']:>7} {item.get('unit', '')} ({item['percent']}%)")
    else:
        lines.append(f"  不可用: {intrinsic.get('reason') or '视频容器没有可读取的相机内参'}")
    lines.extend(["", "[质量诊断]"])
    if report["distributions"]["anomalies"]:
        for item in report["distributions"]["anomalies"]:
            lines.append(f"  {item['label']:<34} {item['count']:>7} ({item['percent']}%)")
    else:
        lines.append("  未发现规则可识别的异常")
    if report.get("failed_files"):
        lines.extend(["", "[解析失败（前 20 条）]"])
        for item in report["failed_files"][:20]:
            lines.append(f"  {item['relative_path']} <- {item['error']}")
    return "\n".join(lines)


def _default_output_dir() -> Path:
    repo = Path(__file__).resolve().parents[2]
    return repo / "output" / "tools" / "video_summary" / time.strftime("%Y%m%d_%H%M%S")


def main() -> int:
    parser = argparse.ArgumentParser(description="分析目录下全部视频的构成、媒体格式和数据质量")
    parser.add_argument("--input", nargs="+", required=True, help="一个或多个待扫描目录（递归）")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 4, 16),
                        help="并发 ffprobe 数（默认 min(CPU, 16)，最大 32）")
    parser.add_argument("--duration-bins", type=float, nargs="+", default=DEFAULT_DURATION_BINS,
                        help=f"时长分桶边界（秒），默认 {DEFAULT_DURATION_BINS}")
    parser.add_argument("--timeout", type=float, default=120, help="单文件 ffprobe 超时秒数")
    parser.add_argument("--out", default=None, help="输出目录")
    args = parser.parse_args()
    if shutil.which("ffprobe") is None:
        parser.error("找不到 ffprobe，请先安装 ffmpeg")
    bins = sorted(set(float(value) for value in args.duration_bins if value > 0))
    if not bins:
        parser.error("duration-bins 至少包含一个正数")

    def show_progress(update):
        if update["stage"] == "scanning":
            print(f"\r[video_summary] 扫描到 {update.get('discovered', 0)} 个视频", end="", flush=True)
        elif update["stage"] == "probing":
            print(f"\r[video_summary] 解析 {update.get('done', 0)}/{update.get('total', 0)}，失败 {update.get('failed', 0)}", end="", flush=True)
        elif update["stage"] == "intrinsics":
            print(f"\r[video_summary] 内参 {update.get('done', 0)}/{update.get('total', 0)} Parquet，失败 {update.get('failed', 0)}", end="", flush=True)

    try:
        roots, records = discover_videos(args.input, progress_callback=show_progress)
        print()
        if not records:
            raise ValueError("输入目录下没有找到支持的视频文件")
        key = analysis_cache_key(roots, records, duration_bins=bins)
        report = analyze_dataset(args.input, workers=args.workers, duration_bins=bins,
                                 timeout=args.timeout, progress_callback=show_progress,
                                 roots=roots, records=records, cache_key=key)
    except (ValueError, AnalysisCancelled) as exc:
        raise SystemExit(f"[video_summary] {exc}") from exc
    print()
    output = Path(args.out).expanduser() if args.out else _default_output_dir()
    paths = write_report(report, output)
    print(render_summary(report))
    print("[video_summary] 已写出: " + ", ".join(str(path) for path in paths.values()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
