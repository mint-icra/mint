#!/usr/bin/env python3
"""Audit external trajectory NPZ files and evaluate them with panel metrics."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .metrics import aggregate_trajectory_metrics, trajectory_metrics
except ImportError:  # Allow direct execution as well as ``python -m``.
    MODEL_EFFECT = Path(__file__).resolve().parents[2]
    if str(MODEL_EFFECT) not in sys.path:
        sys.path.insert(0, str(MODEL_EFFECT))
    from benchmark.camera_trajectory.metrics import (  # type: ignore[no-redef]
        aggregate_trajectory_metrics,
        trajectory_metrics,
    )


CAMERA_ROOT = Path(os.environ.get("CAMERA_BASELINE_ROOT", "data/benchmark/camera_trajectory"))
DEFAULT_PRED_ROOT = CAMERA_ROOT / "results/pred"
DEFAULT_OUTPUT = CAMERA_ROOT / "results/metrics/baselines.json"
DATASET_SPECS = {
    "hot3d_val": {"panel": "camera_hot3d", "sequences": 27, "frames": 94978},
    "arctic_val": {"panel": "camera_arctic", "sequences": 34, "frames": 25883},
}
METHOD_LABELS = {
    "droid_slam_official": "DROID-SLAM",
    "megasam": "MegaSaM",
    "hawor": "HaWoR",
    "infinitevggt": "InfiniteVGGT",
    "lingbotmap": "LingBot-Map",
    "egopipeline": "EgoPipeline",
    "ours_step4500": "MINT (Stage 2 camera-trajectory fine-tuned)",
    "ours_step19000": "MINT (camera trajectory not fine-tuned)",
}
DEFAULT_METHODS = tuple(METHOD_LABELS)
EXPECTED_VARIANT_PREFIXES = {
    "droid_slam_official": "droid_slam_official:",
    "megasam": "megasam:moge2_depth+full_ba",
    "hawor": "hawor_official:",
    "infinitevggt": "infinitevggt:",
    "lingbotmap": "lingbot_map:",
    "egopipeline": "egopipeline:geocalib_intr+moge2_depth+full_ba",
    "ours_step4500": "student_split_fov_head:windowed32_se3_chain:step_00004500",
    "ours_step19000": "student_split_fov_head:windowed32_se3_chain:step_00019000",
}
EXPECTED_CHECKPOINT_NAMES = {
    "ours_step4500": "step_00004500",
    "ours_step19000": "step_00019000",
}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _scalar(archive, key, default=None):
    if key not in archive.files:
        return default
    value = archive[key]
    return value.item() if np.ndim(value) == 0 else value


def _stage_seconds(archive) -> dict[str, float]:
    raw = _scalar(archive, "stage_seconds", {})
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
    if not isinstance(raw, dict):
        return {}
    result = {}
    for key, value in raw.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _expected_sequences(data_root: Path, dataset: str) -> dict[str, dict]:
    directory = data_root / dataset
    if not directory.is_dir():
        raise FileNotFoundError(f"dataset is missing: {directory}")
    result = {}
    for sequence in sorted(directory.iterdir()):
        gt_path = sequence / "gt.npz"
        images = sequence / "images"
        if not gt_path.is_file() or not images.is_dir():
            continue
        image_count = sum(1 for _ in images.glob("*.jpg"))
        with np.load(gt_path, allow_pickle=False) as archive:
            if "c2w" not in archive.files:
                raise ValueError(f"GT has no c2w: {gt_path}")
            gt_count = len(archive["c2w"])
        if image_count != gt_count:
            raise ValueError(
                f"GT/image mismatch: {sequence} images={image_count} c2w={gt_count}"
            )
        result[sequence.name] = {
            "path": sequence, "gt": gt_path, "frames": image_count,
        }
    return result


def _audit_prediction(path: Path, expected_frames: int, method: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as archive:
        if "c2w" not in archive.files:
            raise ValueError("missing c2w")
        c2w = np.asarray(archive["c2w"], np.float64)
        if c2w.shape != (expected_frames, 4, 4):
            raise ValueError(f"c2w shape {c2w.shape}, expected ({expected_frames}, 4, 4)")
        if not np.isfinite(c2w).all():
            raise ValueError("c2w contains NaN/Inf")
        declared_frames = int(_scalar(archive, "frames", len(c2w)))
        if declared_frames != expected_frames:
            raise ValueError(
                f"declared frames={declared_frames}, expected={expected_frames}"
            )
        part_bounds = np.asarray(archive["part_bounds"]) if "part_bounds" in archive.files else None
        align_protocol = str(_scalar(archive, "align_protocol", "whole_sequence"))
        stages = _stage_seconds(archive)
        declared_method = str(_scalar(archive, "method", ""))
        if declared_method != method:
            raise ValueError(
                f"method={declared_method!r}, expected={method!r}"
            )
        variant = str(_scalar(archive, "variant", ""))
        expected_variant = EXPECTED_VARIANT_PREFIXES.get(method)
        if expected_variant and not variant.startswith(expected_variant):
            raise ValueError(
                f"variant={variant!r}, expected prefix={expected_variant!r}"
            )
        expected_checkpoint = EXPECTED_CHECKPOINT_NAMES.get(method)
        if expected_checkpoint:
            checkpoint = Path(str(_scalar(archive, "ckpt", ""))).name
            if checkpoint != expected_checkpoint:
                raise ValueError(
                    f"checkpoint={checkpoint!r}, expected={expected_checkpoint!r}"
                )
        required_stages = []
        if "moge2_depth" in variant:
            required_stages.append("moge2")
        if variant.startswith("egopipeline:"):
            required_stages.append("geocalib")
        timing_complete = (
            float(_scalar(archive, "seconds", 0.0) or 0.0) > 0
            and all(stages.get(stage, 0.0) > 0 for stage in required_stages)
        )
        metadata = {
            "seconds": float(_scalar(archive, "seconds", 0.0) or 0.0),
            "metric_scale": bool(_scalar(archive, "metric_scale", False)),
            "variant": variant,
            "align_protocol": align_protocol,
            "part_count": int(len(part_bounds)) if part_bounds is not None else 1,
            "stage_seconds": stages,
            "timing_complete": timing_complete,
        }
    return c2w, metadata


def evaluate_method_dataset(
    pred_root: Path, data_root: Path, method: str, dataset: str,
) -> dict:
    expected = _expected_sequences(data_root, dataset)
    pred_directory = pred_root / method / "camera_pose" / dataset
    metrics_by_sequence = {}
    issues = []
    variants = set()
    metric_scale_values = set()
    protocols = set()
    segmented = []
    incomplete_timing = []

    for name, record in expected.items():
        path = pred_directory / f"{name}.npz"
        if not path.is_file():
            issues.append({"sequence": name, "kind": "missing_prediction"})
            continue
        try:
            pred, metadata = _audit_prediction(path, record["frames"], method)
            with np.load(record["gt"], allow_pickle=False) as archive:
                gt = np.asarray(archive["c2w"], np.float64)
            is_segmented = (
                metadata["align_protocol"] != "whole_sequence"
                or metadata["part_count"] > 1
            )
            forward_seconds = metadata["seconds"] if metadata["timing_complete"] else None
            if is_segmented:
                # Coverage still has to be exact, but no synthetic global metric is
                # computed for independently solved parts.
                metrics = {
                    "n_frames": float(record["frames"]),
                    "FPS": (
                        record["frames"] / forward_seconds
                        if forward_seconds else float("nan")
                    ),
                    "_forward_s": float(forward_seconds or 0.0),
                }
            else:
                metrics = trajectory_metrics(
                    pred, gt, forward_seconds=forward_seconds,
                )
        except (OSError, ValueError, np.linalg.LinAlgError) as exc:
            issues.append({"sequence": name, "kind": "invalid_prediction",
                           "detail": str(exc)})
            continue
        metrics["_truncated"] = 0.0
        metrics["_full_frames"] = float(record["frames"])
        metrics_by_sequence[name] = metrics
        variants.add(metadata["variant"])
        metric_scale_values.add(metadata["metric_scale"])
        protocols.add(metadata["align_protocol"])
        if is_segmented:
            segmented.append(name)
        if not metadata["timing_complete"]:
            incomplete_timing.append(name)

    expected_frames = sum(record["frames"] for record in expected.values())
    actual_frames = int(sum(item.get("n_frames", 0) for item in metrics_by_sequence.values()))
    result = {
        "method": method,
        "label": METHOD_LABELS.get(method, method),
        "coverage": {
            "sequences": len(metrics_by_sequence),
            "expected_sequences": len(expected),
            "frames": actual_frames,
            "expected_frames": expected_frames,
        },
        "metric_scale": next(iter(metric_scale_values)) if len(metric_scale_values) == 1 else None,
        "variants": sorted(variants),
        "source_protocols": sorted(protocols),
        "timing": {
            "status": "complete" if not incomplete_timing else "incomplete",
            "incomplete_sequences": incomplete_timing,
        },
        "issues": issues,
    }
    if issues:
        result["status"] = "incomplete"
        if segmented:
            result["segmented_sequences"] = segmented
    elif segmented:
        # HaWoR's official 1000-frame parts have unrelated origins/scales.  Concatenating
        # them and reporting one global ATE would create a number the method never outputs.
        result.update({
            "status": "unsupported_global_trajectory",
            "unsupported_reason": (
                "prediction contains independently solved trajectory parts; it cannot be "
                "ranked by whole-sequence Sim(3) without inventing cross-part transforms"
            ),
            "segmented_sequences": segmented,
        })
    else:
        result["status"] = "complete"
        result["aggregate"] = aggregate_trajectory_metrics(
            metrics_by_sequence, DATASET_SPECS[dataset]["panel"],
        )
        if incomplete_timing:
            # Missing cached-stage time does not invalidate poses or ATE, but a
            # corpus FPS computed from partial timing would be misleading.
            result["aggregate"]["mean"]["FPS"] = None
    result["per_sequence"] = metrics_by_sequence
    return _json_value(result)


def build_snapshot(pred_root: Path, data_root: Path, methods: tuple[str, ...],
                   datasets: tuple[str, ...]) -> dict:
    unknown_datasets = sorted(set(datasets) - set(DATASET_SPECS))
    if unknown_datasets:
        raise ValueError(f"unknown datasets: {', '.join(unknown_datasets)}")
    snapshot = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "version": "icra_full_sequence_v1",
            "alignment": "whole-sequence Umeyama Sim(3)",
            "metrics_implementation": "benchmark.camera_trajectory.metrics",
            "aggregation": "sequence-equal means; corpus frames/time FPS",
        },
        "pred_root": str(pred_root.resolve()),
        "data_root": str(data_root.resolve()),
        "datasets": {},
    }
    for dataset in datasets:
        expected = _expected_sequences(data_root, dataset)
        spec = DATASET_SPECS[dataset]
        actual_frames = sum(record["frames"] for record in expected.values())
        if len(expected) != spec["sequences"] or actual_frames != spec["frames"]:
            raise ValueError(
                f"{dataset} corpus does not match frozen split: "
                f"{len(expected)}/{spec['sequences']} sequences, "
                f"{actual_frames}/{spec['frames']} frames"
            )
        snapshot["datasets"][dataset] = {
            "panel_dataset": spec["panel"],
            "expected_sequences": spec["sequences"],
            "expected_frames": spec["frames"],
            "methods": {
                method: evaluate_method_dataset(pred_root, data_root, method, dataset)
                for method in methods
            },
        }
    return snapshot


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def _panel_payload(snapshot: dict) -> dict:
    dataset_results = snapshot["datasets"]
    hot3d = dataset_results.get("hot3d_val", {}).get("methods", {})
    arctic = dataset_results.get("arctic_val", {}).get("methods", {})

    def mean(result: dict) -> dict:
        return ((result.get("aggregate") or {}).get("mean") or {})

    def full_row(result: dict) -> dict:
        values = mean(result)
        scale_error = values.get("scale_error_pct") if result.get("metric_scale") else None
        return {
            "method": result["label"],
            "values": [
                values.get("ATE_mm"), values.get("RPE_T_mm"), values.get("RPE_R_deg"),
                values.get("ATE_S_mm"), values.get("ATE_pct"), values.get("scale"),
                scale_error, values.get("path_scale"), values.get("FPS"),
            ],
        }

    methods = sorted(set(hot3d) | set(arctic))
    two_dataset_rows = []
    hot3d_rows = []
    arctic_rows = []
    exclusions = []
    for method in methods:
        hot = hot3d.get(method)
        arc = arctic.get(method)
        if hot and hot.get("status") == "complete":
            hot3d_rows.append(full_row(hot))
        elif hot and hot.get("status") == "unsupported_global_trajectory":
            hot3d_rows.append({
                "method": f"{hot['label']} (official parts; no global trajectory)",
                "values": [None] * 9,
            })
        if arc and arc.get("status") == "complete":
            arctic_rows.append(full_row(arc))
        elif arc and arc.get("status") == "unsupported_global_trajectory":
            arctic_rows.append({
                "method": f"{arc['label']} (official parts; no global trajectory)",
                "values": [None] * 9,
            })
        if hot and arc and hot.get("status") == arc.get("status") == "complete":
            two_dataset_rows.append({
                "method": hot["label"],
                "values": [mean(hot).get("ATE_mm"), mean(arc).get("ATE_mm")],
            })
        else:
            statuses = {
                "hot3d": hot.get("status") if hot else "not_requested",
                "arctic": arc.get("status") if arc else "not_requested",
            }
            exclusions.append({"method": METHOD_LABELS.get(method, method), **statuses})
            if hot and arc and all(
                result.get("status") in {"complete", "unsupported_global_trajectory"}
                for result in (hot, arc)
            ):
                two_dataset_rows.append({
                    "method": f"{METHOD_LABELS.get(method, method)} (no global trajectory)",
                    "values": [
                        mean(hot).get("ATE_mm") if hot.get("status") == "complete" else None,
                        mean(arc).get("ATE_mm") if arc.get("status") == "complete" else None,
                    ],
                })
    return {
        "schemaVersion": 1,
        "status": "complete",
        "generatedAt": snapshot["generated_at"],
        "source": "CPFS isolated-conda rerun · whole-sequence Sim(3)",
        "note": (
            "固定行来自 baselines.json；仅显示完整覆盖且支持全局轨迹的方法。"
        ),
        "twoDatasetRows": two_dataset_rows,
        "hot3dRows": hot3d_rows,
        "arcticRows": arctic_rows,
        "exclusions": exclusions,
    }


def _atomic_panel_js(path: Path, snapshot: dict) -> None:
    payload = json.dumps(_panel_payload(snapshot), ensure_ascii=False, indent=2)
    content = (
        "// Generated by camera_trajectory/import_external_results.py.\n"
        "// Do not copy values from legacy metrics files into this snapshot.\n"
        f"const CAMERA_TRAJECTORY_BASELINES = Object.freeze({payload});\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-root", default=str(DEFAULT_PRED_ROOT))
    parser.add_argument("--data-root", default=str(CAMERA_ROOT))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--datasets", default=",".join(DATASET_SPECS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--panel-js", default=None,
                        help="write audited fixed Viewer rows; skipped on incomplete methods")
    parser.add_argument("--log-root", default=None,
                        help="worker logs used to restore cached-stage timing; defaults beside pred")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    try:
        pred_root = Path(args.pred_root)
        log_root = Path(args.log_root) if args.log_root else pred_root.parent / "logs"
        if log_root.is_dir():
            try:
                from .rerun_external import recover_cached_moge_timings
            except ImportError:
                from benchmark.camera_trajectory.rerun_external import (  # type: ignore
                    recover_cached_moge_timings,
                )
            timing_recovery = recover_cached_moge_timings(
                pred_root, log_root, _parse_csv(args.methods),
            )
            print(f"timing recovery: {timing_recovery}")
        snapshot = build_snapshot(
            pred_root, Path(args.data_root),
            _parse_csv(args.methods), _parse_csv(args.datasets),
        )
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _atomic_json(Path(args.output), snapshot)
    problems = []
    for dataset, dataset_result in snapshot["datasets"].items():
        for method, result in dataset_result["methods"].items():
            coverage = result["coverage"]
            print(
                f"{dataset:<11} {method:<26} {result['status']:<30} "
                f"{coverage['sequences']}/{coverage['expected_sequences']} seq "
                f"{coverage['frames']}/{coverage['expected_frames']} frames"
            )
            if result["status"] not in {"complete", "unsupported_global_trajectory"}:
                problems.append((dataset, method, result["status"]))
    if args.panel_js:
        if problems:
            print("panel snapshot not updated because one or more methods are incomplete")
        else:
            _atomic_panel_js(Path(args.panel_js), snapshot)
            print(f"panel snapshot: {args.panel_js}")
    print(f"snapshot: {args.output}")
    return 0 if args.allow_incomplete or not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
