#!/usr/bin/env python3
"""Automatic cached normalization for first-frame-rebased camera translation."""
from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from data.sampling import retained_sample_count, validate_sample_fraction


_REPO = Path(__file__).resolve().parents[2]
_ALGORITHM = "w2c_clip_rebase_position_se3_velocity_acceleration_v2"
_KINDS = ("position", "velocity", "acceleration")


def _empty_moments() -> dict:
    return {"count": 0, "mean": [0.0, 0.0, 0.0], "m2": [0.0, 0.0, 0.0]}


def _merge_moments(left: dict, right: dict) -> dict:
    if not right["count"]:
        return left
    if not left["count"]:
        return {
            "count": int(right["count"]),
            "mean": [float(value) for value in right["mean"]],
            "m2": [float(value) for value in right["m2"]],
        }
    left_count, right_count = int(left["count"]), int(right["count"])
    total = left_count + right_count
    delta = [right["mean"][i] - left["mean"][i] for i in range(3)]
    return {
        "count": total,
        "mean": [
            left["mean"][i] + delta[i] * right_count / total for i in range(3)
        ],
        "m2": [
            left["m2"][i]
            + right["m2"][i]
            + delta[i] * delta[i] * left_count * right_count / total
            for i in range(3)
        ],
    }


def _moments_std(moments: dict, min_std: float) -> tuple[list[float], list[float]]:
    count = int(moments["count"])
    if count <= 0:
        raise RuntimeError("camera normalization received no target vectors")
    raw = [math.sqrt(max(float(value) / count, 0.0)) for value in moments["m2"]]
    if not all(math.isfinite(value) for value in raw):
        raise RuntimeError(f"camera normalization contains invalid std: {raw}")
    return [max(value, min_std) for value in raw], raw


def _moments_for_selected_clips(
    moments: dict, total_clips: int, selected_clips: int
) -> dict:
    """Scale cached full-dataset moments to a uniform random retained fraction."""
    total_clips, selected_clips = int(total_clips), int(selected_clips)
    if total_clips <= 0 or not 0 <= selected_clips <= total_clips:
        raise ValueError(
            f"invalid selected clip counts: selected={selected_clips}, total={total_clips}"
        )
    count = int(moments["count"])
    if count % total_clips:
        raise RuntimeError(
            f"moment count {count} is not divisible by num_clips {total_clips}"
        )
    selected_count = selected_clips * (count // total_clips)
    scale = selected_count / count
    return {
        "count": selected_count,
        "mean": [float(value) for value in moments["mean"]],
        "m2": [float(value) * scale for value in moments["m2"]],
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_list_to_numpy(column, width: int, name: str):
    import numpy as np

    array = column.combine_chunks()
    if array.null_count:
        raise RuntimeError(f"column {name} contains {array.null_count} null values")
    if getattr(array.type, "list_size", None) != width:
        raise RuntimeError(f"column {name} must be fixed_size_list[{width}], got {array.type}")
    return np.asarray(array.values.to_numpy(zero_copy_only=False)).reshape(len(array), width)


def _quat_xyzw_to_matrix(quaternion):
    import numpy as np

    norm_squared = np.einsum("ni,ni->n", quaternion, quaternion)
    if not np.isfinite(quaternion).all() or np.any(norm_squared <= 0):
        raise RuntimeError("cam_quat contains non-finite or zero-norm values")
    x, y, z, w = quaternion.T
    two_s = 2.0 / norm_squared
    matrix = np.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        axis=-1,
    )
    return matrix.reshape(-1, 3, 3)


def _numpy_moments(values) -> dict:
    import numpy as np

    values = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    if not len(values):
        return _empty_moments()
    if not np.isfinite(values).all():
        raise RuntimeError("camera translation contains non-finite values")
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    return {
        "count": int(len(values)),
        "mean": mean.tolist(),
        "m2": np.einsum("ni,ni->i", centered, centered, dtype=np.float64).tolist(),
    }


def _episode_moments(
    translation,
    quaternion,
    clip_len: int,
    clip_stride: int,
    window_batch_size: int,
) -> dict:
    import numpy as np

    totals = {kind: _empty_moments() for kind in _KINDS}
    starts = np.arange(0, len(translation) - clip_len + 1, clip_stride, dtype=np.int64)
    if not len(starts):
        return {"num_clips": 0, "moments": totals}
    rotation = _quat_xyzw_to_matrix(quaternion)
    offsets = np.arange(clip_len, dtype=np.int64)
    for first in range(0, len(starts), window_batch_size):
        batch_starts = starts[first:first + window_batch_size]
        rows = batch_starts[:, None] + offsets[None, :]
        sampled_rotation = rotation[rows]
        first_rotation_t = np.swapaxes(rotation[batch_starts], -1, -2)[:, None]
        relative_rotation = sampled_rotation @ first_rotation_t
        rebased = translation[rows] - (
            relative_rotation @ translation[batch_starts, None, :, None]
        )[..., 0]
        step_rotation = (
            relative_rotation[:, 1:]
            @ np.swapaxes(relative_rotation[:, :-1], -1, -2)
        )
        velocity = rebased[:, 1:] - (
            step_rotation @ rebased[:, :-1, :, None]
        )[..., 0]
        values = {
            "position": rebased,
            "velocity": velocity,
            "acceleration": np.diff(rebased, n=2, axis=1),
        }
        for kind, value in values.items():
            totals[kind] = _merge_moments(totals[kind], _numpy_moments(value))
    return {
        "num_clips": int(len(starts)),
        "moments": totals,
    }


def _contiguous_slices(episode_index, frame_index):
    import numpy as np

    if not len(episode_index):
        return []
    boundaries = np.flatnonzero(
        (episode_index[1:] != episode_index[:-1])
        | (frame_index[1:] != frame_index[:-1] + 1)
    ) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(episode_index)]))
    return [slice(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _process_file_shard(args: tuple) -> dict:
    import numpy as np
    import pyarrow.parquet as pq

    file_strings, clip_len, clip_stride, window_batch_size = args
    totals = {kind: _empty_moments() for kind in _KINDS}
    num_clips = rows_read = episodes_read = 0
    columns = ["episode_index", "frame_index", "cam_trans", "cam_quat"]
    for file_string in file_strings:
        path = Path(file_string)
        parquet = pq.ParquetFile(path)
        missing = [name for name in columns if name not in parquet.schema_arrow.names]
        if missing:
            raise RuntimeError(f"{path} is missing camera columns: {missing}")
        table = parquet.read(columns=columns, use_threads=False)
        episode_index = table.column("episode_index").combine_chunks().to_numpy(
            zero_copy_only=False
        )
        frame_index = table.column("frame_index").combine_chunks().to_numpy(
            zero_copy_only=False
        )
        translation = _fixed_list_to_numpy(table.column("cam_trans"), 3, "cam_trans")
        quaternion = _fixed_list_to_numpy(table.column("cam_quat"), 4, "cam_quat")
        rows_read += int(len(episode_index))
        for part in _contiguous_slices(episode_index, frame_index):
            if int(frame_index[part.start]) != 0:
                raise RuntimeError(
                    f"{path}: episode {int(episode_index[part.start])} starts at "
                    f"frame_index={int(frame_index[part.start])}; split episodes are unsupported"
                )
            result = _episode_moments(
                translation[part],
                quaternion[part],
                clip_len,
                clip_stride,
                window_batch_size,
            )
            num_clips += result["num_clips"]
            episodes_read += 1
            for kind in _KINDS:
                totals[kind] = _merge_moments(totals[kind], result["moments"][kind])
    return {
        "num_clips": num_clips,
        "rows_read": rows_read,
        "episodes_read": episodes_read,
        "moments": totals,
    }


def _dataset_signature(root: Path, info_sha256: str,
                       clip_len: int, clip_stride: int) -> dict:
    return {
        "algorithm": _ALGORITHM,
        "dataset_root": str(root.resolve()),
        "info_sha256": info_sha256,
        "clip_len": int(clip_len),
        "clip_stride": int(clip_stride),
        "dataset_name": root.parent.name if root.name == "lerobot_v3" else root.name,
    }


def _cache_path(cache_dir: Path, signature: dict) -> Path:
    encoded = json.dumps(signature, sort_keys=True, separators=(",", ":")).encode()
    key = hashlib.sha256(encoded).hexdigest()[:24]
    name = "".join(c if c.isalnum() or c in "._-" else "_" for c in signature["dataset_name"])
    return cache_dir / f"{name}.{key}.json"


def _load_cache(path: Path, signature: dict) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if payload.get("signature") == signature else None


def _compute_dataset(root: Path, files: list[Path], signature: dict, cfg: dict) -> dict:
    files_per_task = max(1, int(cfg.get("files_per_task", 8)))
    tasks = [
        (tuple(str(path) for path in files[start:start + files_per_task]),
         signature["clip_len"], signature["clip_stride"],
         max(1, int(cfg.get("window_batch_size", 4096))))
        for start in range(0, len(files), files_per_task)
    ]
    requested_workers = int(cfg.get("workers", min(32, os.cpu_count() or 1)))
    workers = max(1, min(requested_workers, len(tasks)))
    totals = {kind: _empty_moments() for kind in _KINDS}
    num_clips = rows_read = episodes_read = 0
    started = time.perf_counter()
    def merge(result):
        nonlocal num_clips, rows_read, episodes_read
        num_clips += result["num_clips"]
        rows_read += result["rows_read"]
        episodes_read += result["episodes_read"]
        for kind in _KINDS:
            totals[kind] = _merge_moments(totals[kind], result["moments"][kind])

    if workers == 1:
        for task in tasks:
            merge(_process_file_shard(task))
    else:
        context = mp.get_context("spawn")
        ordered_results = [None] * len(tasks)
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(_process_file_shard, task): index
                for index, task in enumerate(tasks)
            }
            for future in as_completed(futures):
                ordered_results[futures[future]] = future.result()
        for result in ordered_results:
            merge(result)

    if num_clips <= 0:
        raise RuntimeError(f"no eligible clips found in {root}")
    expected = {
        "position": num_clips * signature["clip_len"],
        "velocity": num_clips * (signature["clip_len"] - 1),
        "acceleration": num_clips * (signature["clip_len"] - 2),
    }
    for kind, count in expected.items():
        if totals[kind]["count"] != count:
            raise RuntimeError(
                f"{root}: {kind} count mismatch, expected {count}, got {totals[kind]['count']}"
            )
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "signature": signature,
        "dataset_root": str(root),
        "data_file_count": len(files),
        "num_clips": num_clips,
        "rows_read": rows_read,
        "episodes_read": episodes_read,
        "elapsed_seconds": elapsed,
        "moments": totals,
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _load_or_compute(root: Path, cache_dir: Path, cfg: dict) -> tuple[dict, Path, str]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"LeRobot info.json not found: {info_path}")
    signature = _dataset_signature(
        root, _sha256(info_path), int(cfg["clip_len"]), int(cfg["clip_stride"])
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, signature)
    payload = _load_cache(cache, signature)
    if payload is not None:
        return payload, cache, "cache hit"

    lock = cache.with_suffix(cache.suffix + ".lock")
    wait_seconds = float(cfg.get("lock_wait_seconds", 12 * 60 * 60))
    poll_seconds = max(0.2, float(cfg.get("lock_poll_seconds", 2.0)))
    stale_seconds = float(cfg.get("lock_stale_seconds", 12 * 60 * 60))
    deadline = time.monotonic() + wait_seconds
    acquired = False
    while not acquired:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as handle:
                handle.write(f"host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n")
            acquired = True
        except FileExistsError:
            payload = _load_cache(cache, signature)
            if payload is not None:
                return payload, cache, "cache hit after wait"
            try:
                if time.time() - lock.stat().st_mtime > stale_seconds:
                    lock.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for normalization cache lock: {lock}")
            time.sleep(poll_seconds)

    try:
        payload = _load_cache(cache, signature)
        status = "cache hit after wait"
        if payload is None:

            files = sorted(
                (root / "data").rglob("*.parquet"), key=lambda path: path.as_posix()
            )
            if not files:
                raise RuntimeError(f"no data parquet found under {root / 'data'}")
            payload = _compute_dataset(root, files, signature, cfg)
            _write_json_atomic(cache, payload)
            status = "computed"
        return payload, cache, status
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _dataset_configs(data_cfg: dict) -> list[dict]:
    configured = data_cfg.get("root")
    items = configured if isinstance(configured, list) else [configured]
    children = []
    for item in items:
        if isinstance(item, dict) and "camera_translation_normalization" in item:
            raise RuntimeError("camera normalization cannot be overridden per data.root item")
        child = {**data_cfg, **item} if isinstance(item, dict) else {**data_cfg, "root": item}
        root = child.get("root")
        if not root:
            raise RuntimeError(f"invalid data.root item: {item!r}")
        child["root"] = Path(root).expanduser().resolve()
        name = (
            child["root"].parent.name
            if child["root"].name == "lerobot_v3"
            else child["root"].name
        )
        fraction = validate_sample_fraction(
            child.get("sample_fraction", 1.0),
            context=f"data.root[{name}].sample_fraction",
        )
        child["sample_fraction"] = fraction
        if fraction == 0.0:
            continue
        children.append(child)
    if not children:
        raise RuntimeError("all data.root sample_fraction values are 0")
    return children


def resolve_global_camera_normalization(data_cfg: dict, log=print) -> dict | None:
    """Return one global scale set for the exact datasets sampled by this training run."""
    norm_cfg = data_cfg.get("camera_translation_normalization", {}) or {}
    if not bool(norm_cfg.get("enabled", False)):
        return None
    clip_len = int(data_cfg.get("clip_len", 0))
    clip_stride = int(data_cfg.get("clip_stride", 1))
    if clip_len < 3 or clip_stride <= 0:
        raise RuntimeError("automatic camera normalization requires clip_len >= 3 and clip_stride > 0")
    cache_raw = norm_cfg.get("cache_dir", "output/cache/camera_normalization")
    cache_dir = Path(cache_raw)
    if not cache_dir.is_absolute():
        cache_dir = _REPO / cache_dir
    worker_cfg = {
        **norm_cfg,
        "clip_len": clip_len,
        "clip_stride": clip_stride,
    }
    min_std = float(norm_cfg.get("min_std", 1e-6))
    if not math.isfinite(min_std) or min_std <= 0:
        raise RuntimeError("camera normalization min_std must be a finite positive number")
    datasets = []
    for child_cfg in _dataset_configs(data_cfg):
        if int(child_cfg.get("clip_len", clip_len)) != clip_len:
            raise RuntimeError("all mixed datasets must use the same clip_len")
        child_worker_cfg = {
            **worker_cfg,
            "clip_stride": int(child_cfg.get("clip_stride", clip_stride)),
        }
        payload, cache, status = _load_or_compute(
            child_cfg["root"], cache_dir, child_worker_cfg
        )
        fraction = child_cfg["sample_fraction"]
        selected_clips = retained_sample_count(payload["num_clips"], fraction)
        datasets.append({
            "payload": payload,
            "cache": cache,
            "sample_fraction": fraction,
            "selected_clips": selected_clips,
        })
        name = payload["signature"]["dataset_name"]
        selection = (
            f"sample_fraction={fraction:.6g}, "
            f"selected={selected_clips:,}/{payload['num_clips']:,} clips"
        )
        if status == "computed":
            log(
                f"[camera-normalization] {name}: computed "
                f"{payload['num_clips']:,} clips / {payload['rows_read']:,} rows, "
                f"{payload['elapsed_seconds']:.1f}s; {selection}"
            )
        else:
            log(f"[camera-normalization] {name}: {status}; {selection}")

    global_moments = {kind: _empty_moments() for kind in _KINDS}
    for dataset in datasets:
        payload = dataset["payload"]
        for kind in _KINDS:
            global_moments[kind] = _merge_moments(
                global_moments[kind],
                _moments_for_selected_clips(
                    payload["moments"][kind],
                    payload["num_clips"],
                    dataset["selected_clips"],
                ),
            )
    scales_and_raw = {
        kind: _moments_std(global_moments[kind], min_std) for kind in _KINDS
    }
    scales = {kind: value[0] for kind, value in scales_and_raw.items()}
    raw_scales = {kind: value[1] for kind, value in scales_and_raw.items()}
    fingerprint_payload = sorted(
        ({
            "signature": dataset["payload"]["signature"],
            "sample_fraction": dataset["sample_fraction"],
            "selected_clips": dataset["selected_clips"],
        } for dataset in datasets),
        key=lambda value: json.dumps(value, sort_keys=True),
    )
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    resolved = {
        "schema_version": 1,
        "algorithm": _ALGORITHM,
        "scope": "global_across_selected_datasets",
        "subtract_mean": False,
        "clip_len": clip_len,
        "clip_stride": clip_stride,
        "dataset_count": len(datasets),
        "dataset_names": [
            dataset["payload"]["signature"]["dataset_name"] for dataset in datasets
        ],
        "dataset_sample_fractions": [dataset["sample_fraction"] for dataset in datasets],
        "dataset_total_clips": [dataset["payload"]["num_clips"] for dataset in datasets],
        "dataset_selected_clips": [dataset["selected_clips"] for dataset in datasets],
        "selected_clip_count": sum(dataset["selected_clips"] for dataset in datasets),
        "dataset_cache_files": [str(dataset["cache"]) for dataset in datasets],
        "mixture_fingerprint": fingerprint,
        "min_std": min_std,
        "position_std_m": scales["position"],
        "velocity_std_m_per_frame": scales["velocity"],
        "acceleration_std_m_per_frame2": scales["acceleration"],
        "raw_position_std_m": raw_scales["position"],
        "raw_velocity_std_m_per_frame": raw_scales["velocity"],
        "raw_acceleration_std_m_per_frame2": raw_scales["acceleration"],
        "moments": global_moments,
    }
    def compact(values):
        return "[" + ",".join(f"{value:.6g}" for value in values) + "]"

    log(
        f"[camera-normalization] global shared scale ({len(datasets)} datasets): "
        f"position={compact(resolved['position_std_m'])}m "
        f"velocity={compact(resolved['velocity_std_m_per_frame'])}m/frame "
        f"acceleration={compact(resolved['acceleration_std_m_per_frame2'])}m/frame^2"
    )
    return resolved
