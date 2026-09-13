"""Select the strongest raw benchmark model and create its UKF evaluation variant."""
from __future__ import annotations

import copy
import math
import re
from pathlib import Path


_MAXIMIZE = {"FAcc", "Recall", "F1", "AUC", "delta1", "delta2", "delta3"}
_QUALITY_METRICS = {
    "depth": {"AbsRel", "SqRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3"},
    "extrinsics": {"ATE_RMSE", "ATE_RMSE_metric", "RPE_t", "RPE_rot_deg"},
    "intrinsics": {
        "fx_relerr_pct", "fy_relerr_pct", "fov_x_deg", "fov_y_deg",
        "cx_off_px", "cy_off_px",
    },
    "hands": {"MPJPE", "PA_MPJPE"},
    "hands_world": {
        "PA_MPJPE", "W_MPJPE_poseenc_scaled", "WA_MPJPE_poseenc_scaled",
        "RTE_poseenc_scaled", "Accel_poseenc_scaled",
    },
    "hands_coverage": {
        "FAcc", "Recall", "F1", "MPJPE-p", "PA-MPJPE-p",
        "EPE-p", "GO-p", "CT-p", "Jitter",
    },
}


def auto_ukf_placeholder() -> dict:
    """Return the pending row shown while raw checkpoints are being ranked."""
    return {
        "run": "", "step": "", "ckpt": None, "tag": "auto_ukf_best",
        "label": "UKF融合 · 等待选择平均质量最佳模型",
        "status": "pending", "report": None, "out": None, "error": None,
        "variant": "ukf", "hand_mode": "smooth", "auto_select_ukf": True,
    }


def _metric_rows(report: dict) -> dict[tuple[str, str, str, str], float]:
    rows = {}
    for head, datasets in (report.get("heads") or {}).items():
        allowed = _QUALITY_METRICS.get(head, set())
        if not allowed:
            continue
        for dataset, node in (datasets or {}).items():
            groups = node.get("mean_by_side") or {"overall": node.get("mean") or {}}
            for group, metrics in groups.items():
                for metric in allowed:
                    value = (metrics or {}).get(metric)
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        rows[(head, dataset, group, metric)] = float(value)
    return rows


def select_best_model(models: list[dict]) -> tuple[dict, dict]:
    """Minimize mean normalized loss over comparable quality metrics.

    Each exact head/dataset/group/metric is independently mapped to [0, 1]
    across the raw checkpoints. This makes unlike units comparable. Missing
    values receive the worst loss so incomplete reports cannot win silently.
    """
    candidates = [
        model for model in models
        if model.get("status") == "completed" and model.get("report")
        and model.get("variant") != "ukf" and not model.get("auto_select_ukf")
    ]
    if not candidates:
        raise ValueError("没有成功完成的原始模型可用于 UKF 自动优选")
    if len(candidates) == 1:
        return candidates[0], {"score": 0.0, "metrics": 0, "candidates": 1}

    model_rows = [_metric_rows(model["report"]) for model in candidates]
    metric_keys = sorted(set().union(*(rows.keys() for rows in model_rows)))
    losses = [[] for _ in candidates]
    used = 0
    for key in metric_keys:
        present = [rows.get(key) for rows in model_rows]
        finite = [value for value in present if value is not None]
        if len(finite) < 2:
            continue
        low, high = min(finite), max(finite)
        if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-12):
            continue
        metric = key[-1]
        for index, value in enumerate(present):
            if value is None:
                loss = 1.0
            elif metric in _MAXIMIZE:
                loss = (high - value) / (high - low)
            else:
                loss = (value - low) / (high - low)
            losses[index].append(float(loss))
        used += 1

    # Exact ties or reports without discriminating metrics keep queue order stable.
    scores = [sum(values) / len(values) if values else 0.0 for values in losses]
    best_index = min(range(len(candidates)), key=lambda index: (scores[index], index))
    return candidates[best_index], {
        "score": scores[best_index], "metrics": used, "candidates": len(candidates),
        "scores": [
            {"label": model.get("label") or model.get("ckpt"), "score": scores[index]}
            for index, model in enumerate(candidates)
        ],
    }


def resolve_auto_ukf_model(models: list[dict], *, reuse_cache: bool = True,
                           out_name: str | None = None) -> dict:
    """Create a cache-distinct smooth/UKF variant of the best raw model."""
    winner, ranking = select_best_model(models)
    if not winner.get("benchmark_signature"):
        raise ValueError("最佳模型缺少 benchmark signature，无法建立 UKF 独立缓存")
    signature = copy.deepcopy(winner["benchmark_signature"])
    signature.setdefault("inference", {})["hand_mode"] = "smooth"

    model = {
        key: winner.get(key)
        for key in ("run", "step", "ckpt", "config", "model")
    }
    source_label = str(winner.get("label") or f"{model.get('run')} / {model.get('step')}")
    source_tag = str(winner.get("tag") or Path(str(model.get("ckpt") or "model")).name)
    model.update({
        "tag": re.sub(r"[^A-Za-z0-9._-]+", "-", source_tag).strip("-._")[:70] + "_ukf",
        "label": source_label + " · UKF融合（自动优选）",
        "status": "pending", "report": None, "out": None, "error": None,
        "variant": "ukf", "hand_mode": "smooth", "auto_select_ukf": False,
        "source_model_label": source_label,
        "source_benchmark_signature_key": winner.get("benchmark_signature_key"),
        "quality_ranking": ranking,
        "benchmark_signature": signature,
    })
    if out_name:
        model["out_name"] = out_name

    from benchmark.cache import find_cached_report, signature_key

    model["benchmark_signature_key"] = signature_key(signature)
    hit = find_cached_report(signature) if reuse_cache else None
    if hit:
        model.update(
            status="completed", report=hit["report"], out=hit["out"],
            report_path=hit["report_path"], cache_hit=True,
            cache_manifest=hit["manifest_path"], error=None,
        )
    return model
