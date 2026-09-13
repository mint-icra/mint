"""Persistent benchmark result reuse and lightweight per-step benchmark logs."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path


_REPO = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = _REPO / "output" / "eval" / "benchmark"
MODEL_TRAIN_ROOT = _REPO / "output" / "model_train"
CACHE_VERSION = 2
PROTOCOL_REVISION = "hawor-vidihand-camera-se3table-20260816-v2"
MANIFEST_NAME = "benchmark_cache.json"
_MAXIMIZE = {"FAcc", "Recall", "F1", "AUC"}
_PROTOCOL_ENV_DEFAULTS = {
    "ARCTIC_SPLIT": "val",
    "BENCH_DEPTH_ALIGN": "median",
    "CAMERA_TRAJECTORY_ROOT": "",
    "HAND_COVERAGE_SPLIT_SEED": "42",
    "HAND_COVERAGE_TEST_SEGMENTS": "437",
    "HOT3D_HANDS": "left,right",
    "HOT3D_LEROBOT_ROOT": "",
    "HOT3D_RECTIFY": "1",
    "HOT3D_SEQS": "",
    "HOT3D_WIDTH": "720",
}


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


def _exclusive_json(path: Path, payload: dict) -> None:
    """Create an immutable history event without ever replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(path, flags, 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _csv_values(value) -> list[str]:
    if isinstance(value, str):
        values = value.split(",")
    else:
        values = value or []
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _file_stamp(path: Path | str | None) -> dict | None:
    if not path:
        return None
    resolved = Path(path).expanduser().resolve()
    try:
        stat = resolved.stat()
    except OSError:
        return {"path": str(resolved), "missing": True}
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _checkpoint_stamp(ckpt: Path | str) -> dict:
    checkpoint = Path(ckpt).expanduser().resolve()
    if checkpoint.is_file():
        artifact = checkpoint
    else:
        artifact = checkpoint / "model.safetensors"
        if not artifact.is_file():
            candidates = sorted(checkpoint.glob("*.safetensors"))
            artifact = candidates[0] if candidates else checkpoint
    return {"path": str(checkpoint), "artifact": _file_stamp(artifact)}


def build_signature(*, ckpt, config, heads, datasets, seq_start=0,
                    seq_end=None, max_seqs=None, max_frames=None,
                    dataset_selection=None, hand_mode="hard") -> dict:
    """Describe every result-affecting input; devices/backend intentionally do not matter."""
    from benchmark.datasets.base import normalize_dataset_selection

    selected_heads = _csv_values(heads)
    selected_datasets = _csv_values(datasets)
    protocol_only = (
        bool(selected_datasets)
        and all(name.endswith("_hand_coverage") for name in selected_datasets)
        and selected_heads == ["hands_coverage"]
    )
    return {
        "cache_version": CACHE_VERSION,
        "protocol_revision": PROTOCOL_REVISION,
        "checkpoint": _checkpoint_stamp(ckpt),
        "config": _file_stamp(config),
        "selection": {
            "heads": selected_heads,
            "datasets": selected_datasets,
            "seq_start": int(seq_start or 0),
            "seq_end": None if seq_end is None else int(seq_end),
            "max_seqs": None if max_seqs is None else int(max_seqs),
            "max_frames": None if max_frames is None else int(max_frames),
            "dataset_selection": normalize_dataset_selection(dataset_selection),
        },
        "inference": {
            "base_windowed": not protocol_only,
            "coverage_forced_single_forward": True,
            "hand_mode": str(hand_mode or "hard"),
        },
        "protocol_env": {
            key: os.environ.get(key, default)
            for key, default in sorted(_PROTOCOL_ENV_DEFAULTS.items())
        },
    }


def signature_key(signature: dict) -> str:
    encoded = json.dumps(signature, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def report_is_complete(report: dict, signature: dict) -> bool:
    heads = report.get("heads") or {}
    selection = signature.get("selection") or {}
    requested_heads = selection.get("heads") or []
    requested_datasets = selection.get("datasets") or []
    if not heads or any(head not in heads for head in requested_heads):
        return False
    for dataset in requested_datasets:
        nodes = [
            (heads.get(head) or {}).get(dataset)
            for head in requested_heads
        ]
        evaluated = [node for node in nodes if node and node.get("mean")]
        if not evaluated:
            return False
        if any(int((node.get("status_counts") or {}).get("error", 0)) for node in evaluated):
            return False
    return True


def register_cached_report(signature: dict, report_path: Path | str,
                           report: dict | None = None) -> Path:
    report_path = Path(report_path).expanduser().resolve()
    root = BENCHMARK_ROOT.resolve()
    if root != report_path.parent and root not in report_path.parents:
        raise ValueError(f"benchmark cache report 必须位于 {root}: {report_path}")
    report = report or _read_json(report_path)
    if not report or not report_is_complete(report, signature):
        raise ValueError(f"benchmark report 不完整，不能缓存: {report_path}")
    manifest = {
        "cache_version": CACHE_VERSION,
        "cache_key": signature_key(signature),
        "created_at": _now(),
        "signature": signature,
        "report_path": str(report_path),
    }
    path = report_path.parent / MANIFEST_NAME
    _atomic_json(path, manifest)
    return path


def _legacy_report_matches(report: dict, signature: dict) -> bool:
    """Import reports written before cache manifests existed, using conservative fields only."""
    selection = signature.get("selection") or {}
    if str(Path(report.get("ckpt") or "").expanduser().resolve()) != signature["checkpoint"]["path"]:
        return False
    report_config = report.get("config")
    expected_config = (signature.get("config") or {}).get("path")
    if not report_config or str(Path(report_config).expanduser().resolve()) != expected_config:
        return False
    report_selection = report.get("selection") or {}
    start = int(selection.get("seq_start") or 0)
    expected_end = selection.get("seq_end")
    max_seqs = selection.get("max_seqs")
    if max_seqs is not None:
        max_end = start + int(max_seqs)
        expected_end = max_end if expected_end is None else min(int(expected_end), max_end)
    if int(report_selection.get("seq_start") or 0) != start:
        return False
    if report_selection.get("seq_end") != expected_end:
        return False
    if report_selection.get("max_frames") != selection.get("max_frames"):
        return False
    if (report_selection.get("dataset_selection") or {}) != (selection.get("dataset_selection") or {}):
        return False
    expected_hand_mode = (signature.get("inference") or {}).get("hand_mode", "hard")
    if report_selection.get("hand_mode", "hard") != expected_hand_mode:
        return False
    heads = report.get("heads") or {}
    if sorted(heads) != selection.get("heads"):
        return False
    datasets = sorted({dataset for nodes in heads.values() for dataset in (nodes or {})})
    return datasets == selection.get("datasets") and report_is_complete(report, signature)


def find_cached_report(signature: dict, root: Path | str = BENCHMARK_ROOT) -> dict | None:
    """Return the newest exact, complete hit; cancelled/partial reports are never reused."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return None
    key = signature_key(signature)
    # Benchmark reports are at root/<run>/report.json or root/<run>/<model>/report.json.
    # Do not recurse through frame caches under this root; those can contain hundreds
    # of thousands of JPEGs and made a cache lookup slower than model startup.
    manifests = list(root.glob(MANIFEST_NAME))
    manifests += list(root.glob(f"*/{MANIFEST_NAME}"))
    manifests += list(root.glob(f"*/*/{MANIFEST_NAME}"))
    manifests += list(root.glob(f"aliyun/*/*/{MANIFEST_NAME}"))
    try:
        manifests.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError:
        pass
    for path in manifests:
        manifest = _read_json(path)
        if not manifest or manifest.get("cache_key") != key or manifest.get("signature") != signature:
            continue
        report_path = Path(manifest.get("report_path") or path.parent / "report.json").expanduser().resolve()
        if root != report_path.parent and root not in report_path.parents:
            continue
        report = _read_json(report_path)
        if not report or not report_is_complete(report, signature):
            continue
        return {
            "cache_key": key,
            "manifest_path": str(path.resolve()),
            "report_path": str(report_path),
            "out": str(report_path.parent),
            "report": report,
            "created_at": manifest.get("created_at"),
        }
    legacy_reports = list(root.glob("report.json"))
    legacy_reports += list(root.glob("*/report.json"))
    legacy_reports += list(root.glob("*/*/report.json"))
    legacy_reports += list(root.glob("aliyun/*/*/report.json"))
    try:
        legacy_reports.sort(key=lambda path: path.stat().st_mtime_ns, reverse=True)
    except OSError:
        pass
    for report_path in legacy_reports:
        if report_path.parent.name.startswith("gpu"):
            continue
        report = _read_json(report_path)
        if not report or not _legacy_report_matches(report, signature):
            continue
        try:
            manifest_path = register_cached_report(signature, report_path, report)
        except (OSError, ValueError):
            manifest_path = report_path.parent / MANIFEST_NAME
        return {
            "cache_key": key,
            "manifest_path": str(manifest_path.resolve()),
            "report_path": str(report_path.resolve()),
            "out": str(report_path.parent.resolve()),
            "report": report,
            "created_at": None,
            "legacy_imported": True,
        }
    return None


def _metric_rows(report: dict) -> list[dict]:
    rows = []
    for head, datasets in (report.get("heads") or {}).items():
        for dataset, node in (datasets or {}).items():
            groups = node.get("mean_by_side") or {"overall": node.get("mean") or {}}
            for group, metrics in groups.items():
                for metric, value in (metrics or {}).items():
                    if (metric.startswith("_") or not isinstance(value, (int, float))
                            or not math.isfinite(float(value))):
                        continue
                    rows.append({
                        "head": head, "dataset": dataset, "group": group,
                        "metric": metric, "value": float(value),
                    })
    return rows


def _official_sota_rows(report: dict) -> list[dict]:
    rows = []
    for head, datasets in (report.get("heads") or {}).items():
        for dataset, node in (datasets or {}).items():
            protocol = node.get("protocol") or {}
            reference = (node.get("reference") or {}).get("metrics") or {}
            if not protocol.get("reference_same_split") or not reference:
                continue
            metrics = node.get("mean") or {}
            for metric, public in reference.items():
                value = metrics.get(metric)
                if (not isinstance(value, (int, float)) or not isinstance(public, (int, float))
                        or not math.isfinite(float(value)) or not math.isfinite(float(public))):
                    continue
                improved = value > public if metric in _MAXIMIZE else value < public
                if improved:
                    rows.append({
                        "head": head, "dataset": dataset, "group": "overall",
                        "metric": metric, "value": float(value),
                        "public_best": float(public),
                    })
    return rows


def _local_best_by_model(models: list[dict]) -> dict[str, list[dict]]:
    if len(models) < 2:
        return {}
    values: dict[tuple, list[tuple[str, float]]] = {}
    for model in models:
        key = str(model.get("benchmark_signature_key") or model.get("ckpt"))
        for row in _metric_rows(model["report"]):
            metric_key = (row["head"], row["dataset"], row["group"], row["metric"])
            values.setdefault(metric_key, []).append((key, row["value"]))
    winners: dict[str, list[dict]] = {}
    for metric_key, candidates in values.items():
        if len(candidates) < 2:
            continue
        metric = metric_key[-1]
        best = (max if metric in _MAXIMIZE else min)(value for _, value in candidates)
        for model_key, value in candidates:
            if value == best:
                winners.setdefault(model_key, []).append({
                    "head": metric_key[0], "dataset": metric_key[1],
                    "group": metric_key[2], "metric": metric,
                    "value": value,
                })
    return winners


def _report_summary(report: dict) -> dict:
    heads = {}
    for head, datasets in (report.get("heads") or {}).items():
        for dataset, node in (datasets or {}).items():
            heads.setdefault(head, {})[dataset] = {
                key: node[key]
                for key in ("mean", "mean_by_side", "protocol", "reference", "delta_vs_reference")
                if key in node
            }
    return {
        key: report.get(key)
        for key in ("ckpt", "config", "selection", "timings")
        if report.get(key) is not None
    } | {"heads": heads}


def _sampling_tier(selection: dict) -> str:
    tiers = {
        str(option.get("fixed_tier"))
        for option in (selection.get("dataset_selection") or {}).values()
        if isinstance(option, dict) and option.get("fixed_tier")
    }
    if len(tiers) == 1:
        return next(iter(tiers))
    if tiers:
        return "+".join(sorted(tiers))
    return "custom"


def _sampling_tier_slug(tier: str) -> str:
    labels = {
        "minimum": "minimum", "quarter": "25pct", "half": "50pct", "full": "100pct",
        "custom": "custom",
    }
    parts = [labels.get(part, re.sub(r"[^A-Za-z0-9.-]+", "-", part).strip("-"))
             for part in str(tier or "custom").split("+")]
    return "-".join(part for part in parts if part) or "custom"


def _benchmark_index_tiers(payload: dict) -> set[str]:
    tiers = set()
    for section in ("records", "benchmarks"):
        for entry in (payload.get(section) or {}).values():
            if not isinstance(entry, dict):
                continue
            tiers.add(str(entry.get("sampling_tier") or _sampling_tier(entry.get("selection") or {})))
    return tiers


def _history_run_dir(run: str) -> Path:
    root = MODEL_TRAIN_ROOT.resolve()
    if not run or Path(run).is_absolute():
        raise ValueError("历史测评 run 无效")
    run_dir = (root / run).resolve()
    if root not in run_dir.parents:
        raise ValueError("历史测评 run 超出 model_train")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"训练 Run 不存在: {run}")
    return run_dir


def _history_metadata(record: dict, *, run: str) -> dict:
    return {
        key: record.get(key)
        for key in (
            "record_id", "recorded_at", "step", "checkpoint", "variant", "hand_mode",
            "label", "cache_key", "protocol_revision", "sampling_tier", "selection",
            "source_report", "history_file", "source_model_label", "quality_ranking", "cache_hit",
        )
    } | {"run": run}


def _legacy_history_record(payload: dict, cache_key: str, entry: dict, *, run: str) -> dict:
    step = str(payload.get("step") or "")
    record = dict(entry)
    legacy_hint = f"{record.get('source_report') or ''} {record.get('label') or ''}".lower()
    variant = "ukf" if "ukf" in legacy_hint else "raw"
    record.setdefault("record_id", f"legacy__{step}__{cache_key}")
    record.setdefault("step", step)
    record.setdefault("checkpoint", payload.get("checkpoint"))
    record.setdefault("cache_key", cache_key)
    record.setdefault("variant", variant)
    record.setdefault("hand_mode", "smooth" if variant == "ukf" else "hard")
    record.setdefault("label", f"{run} / {step}")
    record.setdefault("protocol_revision", "legacy")
    record.setdefault("sampling_tier", _sampling_tier(record.get("selection") or {}))
    return record


def _iter_benchmark_history(run: str, step: str | None = None):
    run_dir = _history_run_dir(run)
    if step is not None and (Path(step).name != step or not step.startswith("step_")):
        raise ValueError("历史测评 step 无效")
    logs = run_dir / "logs"
    if not logs.is_dir():
        return
    seen = set()
    paths = set(logs.glob("benchmark_*.json"))
    paths.update(logs.glob("step_*_benchmark.json"))
    for path in sorted(paths, reverse=True):
        payload = _read_json(path)
        if not payload:
            continue
        if payload.get("record_id"):
            record = dict(payload)
            record.setdefault("history_file", str(path.resolve()))
            record_step = str(record.get("step") or "")
            if (step is None or record_step == step) and record["record_id"] not in seen:
                seen.add(record["record_id"])
                yield record
            continue
        payload_step = str(payload.get("step") or "")
        if step is not None and payload_step != step:
            continue
        records = payload.get("records") or {}
        for record_id, value in records.items():
            if not isinstance(value, dict) or record_id in seen:
                continue
            record = dict(value)
            record.setdefault("record_id", record_id)
            record.setdefault("step", payload_step)
            record.setdefault("checkpoint", payload.get("checkpoint"))
            seen.add(record_id)
            yield record
        for cache_key, value in (payload.get("benchmarks") or {}).items():
            if not isinstance(value, dict):
                continue
            record = _legacy_history_record(payload, cache_key, value, run=run)
            if record["record_id"] in seen:
                continue
            seen.add(record["record_id"])
            yield record


def _benchmark_history_runs() -> list[str]:
    """Discover runs from the same logs directories used by benchmark publishing."""
    root = MODEL_TRAIN_ROOT.resolve()
    if not root.is_dir():
        return []
    runs = set()
    for directory, dirnames, filenames in os.walk(root):
        path = Path(directory)
        if path.name == "logs":
            has_history = any(
                name.endswith(".json") and (
                    name.startswith("benchmark_")
                    or (name.startswith("step_") and name.endswith("_benchmark.json"))
                )
                for name in filenames
            )
            if has_history:
                runs.add(path.parent.relative_to(root).as_posix())
            dirnames[:] = []
        else:
            # Checkpoints cannot contain per-run logs; skipping them keeps discovery cheap.
            dirnames[:] = [name for name in dirnames if not name.startswith("step_")]
    return sorted(runs, reverse=True)


def list_benchmark_history(run: str | None = None, step: str | None = None) -> list[dict]:
    """Auto-discover compact benchmark events from the default per-run logs directories."""
    runs = [run] if run else _benchmark_history_runs()
    records = [
        _history_metadata(record, run=history_run)
        for history_run in runs
        for record in _iter_benchmark_history(history_run, step)
    ]
    return sorted(records, key=lambda item: str(item.get("recorded_at") or ""), reverse=True)


def load_benchmark_history(run: str, record_id: str) -> dict:
    """Load one exact compact result by its safe event identifier."""
    if not re.fullmatch(r"[A-Za-z0-9_.+:-]+", str(record_id or "")):
        raise ValueError("历史测评 record_id 无效")
    for record in _iter_benchmark_history(run):
        if record.get("record_id") != record_id:
            continue
        result = record.get("result")
        if not isinstance(result, dict) or not (result.get("heads") or {}):
            raise FileNotFoundError(f"历史测评结果不完整: {record_id}")
        result = dict(result)
        result.setdefault("ckpt", record.get("checkpoint"))
        result.setdefault("selection", record.get("selection") or {})
        metadata = _history_metadata(record, run=run)
        label = record.get("label") or f"{run} / {record.get('step') or ''}"
        return {
            **metadata,
            "model": {
                "run": run,
                "step": record.get("step"),
                "ckpt": record.get("checkpoint"),
                "label": label,
                "status": "completed",
                "variant": record.get("variant") or "raw",
                "hand_mode": record.get("hand_mode") or "hard",
                "report": result,
                "historical": True,
                "historical_record_id": record_id,
                "history_recorded_at": record.get("recorded_at"),
                "history_sampling_tier": record.get("sampling_tier") or "custom",
                "history_protocol_revision": record.get("protocol_revision") or "legacy",
                "history_selection": record.get("selection") or {},
                "history_file": record.get("history_file"),
                "cache_hit": bool(record.get("cache_hit")),
                "source_model_label": record.get("source_model_label"),
                "quality_ranking": record.get("quality_ranking"),
            },
        }


def publish_step_benchmark_logs(models: list[dict], selection: dict) -> list[str]:
    """Index every completed model and create one immutable step/timestamp event file."""
    selection = selection or {}
    completed = [
        model for model in models
        if model.get("report") and model.get("status") == "completed"
        and (
            not ((model.get("benchmark_signature") or {}).get("selection"))
            or report_is_complete(model["report"], model["benchmark_signature"])
        )
    ]
    local_winners = _local_best_by_model(completed)
    comparison_models = [model.get("label") or model.get("ckpt") for model in completed]
    written = []
    model_root = MODEL_TRAIN_ROOT.resolve()
    for model in completed:
        official = _official_sota_rows(model["report"])
        model_key = str(model.get("benchmark_signature_key") or model.get("ckpt"))
        local_best = local_winners.get(model_key, [])
        ckpt = Path(model["ckpt"]).expanduser().resolve()
        if model_root not in ckpt.parents:
            continue
        step = ckpt.name
        logs = ckpt.parent / "logs"
        sampling_tier = _sampling_tier(selection)
        tier_slug = _sampling_tier_slug(sampling_tier)
        index_path = logs / f"benchmark_{step}_{tier_slug}.json"
        untiered_index_path = logs / f"benchmark_{step}.json"
        legacy_index_path = logs / f"{step}_benchmark.json"
        payload = _read_json(index_path)
        if not payload:
            for candidate_path in (untiered_index_path, legacy_index_path):
                candidate = _read_json(candidate_path)
                tiers = _benchmark_index_tiers(candidate or {})
                if candidate and (not tiers or tiers == {sampling_tier}):
                    payload = candidate
                    break
        payload = payload or {
            "version": 2, "step": step, "checkpoint": str(ckpt), "benchmarks": {},
        }
        cache_key = str(model.get("benchmark_signature_key") or signature_key(model["benchmark_signature"]))
        report_path = model.get("report_path") or str(Path(model.get("out") or "") / "report.json")
        recorded_at = _now()
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S_%f%z")
        variant = "ukf" if model.get("variant") == "ukf" else "raw"
        prefix = (f"{step}_ukf_{tier_slug}_{timestamp}" if variant == "ukf"
                  else f"{step}_{tier_slug}_{timestamp}")
        event_id = prefix
        event_path = logs / f"benchmark_{event_id}.json"
        collision = 1
        signature = model.get("benchmark_signature") or {}
        entry = {
            "record_id": None,
            "recorded_at": recorded_at,
            "run": model.get("run") or ckpt.parent.name,
            "step": step,
            "checkpoint": str(ckpt),
            "variant": variant,
            "hand_mode": model.get("hand_mode") or ("smooth" if variant == "ukf" else "hard"),
            "label": model.get("label") or f"{ckpt.parent.name} / {step}",
            "cache_key": cache_key,
            "cache_hit": bool(model.get("cache_hit")),
            "protocol_revision": signature.get("protocol_revision") or PROTOCOL_REVISION,
            "sampling_tier": sampling_tier,
            "source_report": str(Path(report_path).expanduser().resolve()),
            "history_file": None,
            "selection": selection,
            "official_sota_metrics": official,
            "local_best_metrics": local_best,
            "comparison_models": comparison_models,
            "source_model_label": model.get("source_model_label"),
            "quality_ranking": model.get("quality_ranking"),
            "result": _report_summary(model["report"]),
        }
        while True:
            entry["record_id"] = event_id
            entry["history_file"] = str(event_path.resolve())
            try:
                _exclusive_json(event_path, {"version": 2, **entry})
                break
            except FileExistsError:
                event_id = f"{prefix}_{collision:02d}"
                event_path = logs / f"benchmark_{event_id}.json"
                collision += 1
        payload["version"] = 2
        payload["latest"] = cache_key
        payload["latest_record"] = event_id
        payload.setdefault("benchmarks", {})[cache_key] = entry
        payload.setdefault("records", {})[event_id] = entry
        _atomic_json(index_path, payload)
        model["benchmark_log"] = str(event_path)
        model["benchmark_log_index"] = str(index_path)
        model["benchmark_record_id"] = event_id
        model["official_sota_metrics"] = official
        model["local_best_metrics"] = local_best
        written.append(str(event_path))
    return written
