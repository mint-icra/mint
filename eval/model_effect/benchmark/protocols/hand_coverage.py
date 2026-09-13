"""Corpus-level ViDiHand hand metrics and public Table 1 references."""
from __future__ import annotations

import math


PUBLIC_HOT3D = {
    "FAcc": 0.948,
    "Recall": 0.974,
    "F1": 0.983,
    "MPJPE-p": 21.514,
    "PA-MPJPE-p": 11.383,
    "EPE-p": 14.953,
    "GO-p": 15.829,
    "CT-p": 0.040,
    "Jitter": 3.741,
}

PUBLIC_ARCTIC = {
    "FAcc": 0.997,
    "Recall": 0.999,
    "F1": 0.999,
    "MPJPE-p": 21.668,
    "PA-MPJPE-p": 9.821,
    "EPE-p": 12.407,
    "GO-p": 14.642,
    "CT-p": 0.047,
    "Jitter": 3.183,
}

PUBLIC_BY_DATASET = {
    "hot3d_hand_coverage": {
        "metrics": PUBLIC_HOT3D,
        "dataset": "HOT3D public Table 1",
        "split": "ViDiHand official test split",
    },
    "arctic_hand_coverage": {
        "metrics": PUBLIC_ARCTIC,
        "dataset": "ARCTIC public Table 1",
        "split": "ARCTIC official test split",
    },
}

METRIC_ORDER = tuple(PUBLIC_HOT3D)


def _sum(seqs: dict, key: str) -> float:
    return float(sum(float(metrics.get(key, 0.0) or 0.0) for metrics in seqs.values()))


def _ratio(numerator: float, denominator: float):
    return float(numerator / denominator) if denominator > 0 else None


def metrics_from_stats(stats: dict) -> dict:
    """Convert additive corpus statistics into the nine public metrics."""
    tp, fn, fp = stats["tp"], stats["fn"], stats["fp"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    pose_count = stats["pose_count"]
    return {
        "FAcc": _ratio(stats["facc_correct"], stats["facc_frames"]),
        "Recall": recall,
        "F1": f1,
        "MPJPE-p": _ratio(stats["mpjpe_sum_mm"], pose_count),
        "PA-MPJPE-p": _ratio(stats["pa_mpjpe_sum_mm"], pose_count),
        "EPE-p": _ratio(stats["epe_sum_px"], stats["epe_joint_count"]),
        "GO-p": _ratio(stats["go_sum_deg"], pose_count),
        "CT-p": _ratio(stats["ct_sum_m"], pose_count),
        "Jitter": _ratio(stats["jitter_sum_mm"], stats["jitter_count"]),
    }


def aggregate_sequence_metrics(
    seqs: dict, dataset_name: str = "hot3d_hand_coverage",
) -> dict:
    """Aggregate hidden additive fields without averaging per-segment means."""
    stats = {
        "facc_correct": _sum(seqs, "_facc_correct"),
        "facc_frames": _sum(seqs, "_facc_frames"),
        "tp": _sum(seqs, "_tp"),
        "fn": _sum(seqs, "_fn"),
        "fp": _sum(seqs, "_fp"),
        "pose_count": _sum(seqs, "_pose_count"),
        "mpjpe_sum_mm": _sum(seqs, "_mpjpe_sum_mm"),
        "pa_mpjpe_sum_mm": _sum(seqs, "_pa_mpjpe_sum_mm"),
        "epe_sum_px": _sum(seqs, "_epe_sum_px"),
        "epe_joint_count": _sum(seqs, "_epe_joint_count"),
        "go_sum_deg": _sum(seqs, "_go_sum_deg"),
        "ct_sum_m": _sum(seqs, "_ct_sum_m"),
        "jitter_sum_mm": _sum(seqs, "_jitter_sum_mm"),
        "jitter_count": _sum(seqs, "_jitter_count"),
    }
    mean = metrics_from_stats(stats)
    reference_spec = PUBLIC_BY_DATASET.get(dataset_name)
    reference_metrics = (reference_spec or {}).get("metrics") or {}
    delta = {
        key: (None if mean[key] is None else float(mean[key] - reference))
        for key, reference in reference_metrics.items()
    }
    single_forward_segments = int(round(_sum(seqs, "_single_forward")))
    first_metrics = next(iter(seqs.values()))
    counts = {
        "segments": len(seqs),
        "frames": int(round(stats["facc_frames"])),
        "TP": int(round(stats["tp"])),
        "FN": int(round(stats["fn"])),
        "FP": int(round(stats["fp"])),
        "pose_hands": int(round(stats["pose_count"])),
        "EPE_joints": int(round(stats["epe_joint_count"])),
        "jitter_samples": int(round(stats["jitter_count"])),
        "presence_error_frames_exported": int(round(_sum(seqs, "_presence_error_frames_exported"))),
    }
    is_arctic = dataset_name == "arctic_hand_coverage"
    official_split = bool(round(float(first_metrics.get("_official_split", 0.0))))
    reference_same_split = bool(round(float(first_metrics.get("_reference_same_split", 0.0))))
    protocol = {
        "name": "ViDiHand相机系手部检测与FN惩罚姿态指标",
        "official_split": official_split,
        "evaluation_split": "ARCTIC protocol_p1 validation" if is_arctic else "deterministic local holdout",
        "segment_frames": 81,
        "fps": 30,
        "image_hw": [480, 672] if is_arctic else [480, 480],
        "intrinsic": (
            "per-sequence calibrated pinhole, scaled from 2800x2000"
            if is_arctic else
            {"fx": 207.8, "fy": 207.8, "cx": 239.8, "cy": 239.8}
        ),
        "presence_output": "hand_presence_logits",
        "presence_decision": "logit >= 0",
        "presence_ground_truth": "valid GT with any of 21 joints on screen",
        "presence_matching": "per-side fixed-slot binary matching; off-screen GT excluded",
        "pose_metric_subset": "all on-screen GT hands (TP + canonical-MANO FN penalty)",
        "false_negative_pose_penalty": True,
        "false_positive_pose_penalty": False,
        "corpus_level": True,
        "clip_single_forward": single_forward_segments == len(seqs),
        "single_forward_segments": single_forward_segments,
        "requested_segments": int(first_metrics.get("_requested_segments", len(seqs))),
        "available_local_segments": int(first_metrics.get("_available_segments", len(seqs))),
        "reference_same_split": reference_same_split,
    }
    if not is_arctic:
        protocol["split_seed"] = int(first_metrics.get("_split_seed", 42))

    result = {
        "mean": mean,
        "counts": counts,
        "protocol": protocol,
    }
    if reference_spec:
        result["reference"] = {
            "source": "ViDiHand, arXiv 2606.30308v2 Table 1",
            "dataset": reference_spec["dataset"],
            "split": reference_spec["split"],
            "same_split_as_local": reference_same_split,
            "metrics": dict(reference_metrics),
        }
        result["delta_vs_reference"] = delta
    return result


def is_finite_metric(value) -> bool:
    return value is not None and math.isfinite(float(value))
