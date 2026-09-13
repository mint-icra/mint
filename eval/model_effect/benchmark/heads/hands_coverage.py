"""ViDiHand-style hand presence and penalized camera-space pose metrics."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from ..core.registry import HEADS
from ..core.schema import GTSequence, HAND_COVERAGE, Prediction
from ..protocols.hand_coverage import metrics_from_stats
from .align import umeyama_sim3
from .base import HeadEvaluator
from .hands import camera_hand_geometry


SIDES = ("left", "right")
PER_HAND = 109
PROTOCOL_HW = (480, 480)
PROTOCOL_K = np.array(
    [[207.8, 0.0, 239.8], [0.0, 207.8, 239.8], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)
DEPTH_FLOOR_M = 0.01
ERROR_BANNER_HEIGHT = 132
_CANONICAL_GEOMETRY = None


def _project(points: np.ndarray, K: np.ndarray, *, floor_depth: bool) -> np.ndarray:
    points = np.asarray(points, np.float64)
    z = points[..., 2]
    denom = np.maximum(z, DEPTH_FLOOR_M) if floor_depth else z
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * points[..., 0] / denom + K[0, 2]
        v = K[1, 1] * points[..., 1] / denom + K[1, 2]
    return np.stack((u, v), axis=-1)


def _on_screen_joints(joints: np.ndarray, K: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    height, width = hw
    uv = _project(joints, K, floor_depth=False)
    z = np.asarray(joints)[..., 2]
    return (
        np.isfinite(uv).all(axis=-1)
        & (z > DEPTH_FLOOR_M)
        & (uv[..., 0] >= 0.0)
        & (uv[..., 0] < width)
        & (uv[..., 1] >= 0.0)
        & (uv[..., 1] < height)
    )


def _geodesic_degrees(pred: np.ndarray, gt: np.ndarray) -> float:
    relative = np.asarray(pred, np.float64).T @ np.asarray(gt, np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    if cosine > 1.0 - 1e-7:
        return 0.0
    return float(np.degrees(np.arccos(cosine)))


def _root_relative_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_rr = pred - pred[0:1]
    gt_rr = gt - gt[0:1]
    return float(np.linalg.norm(pred_rr - gt_rr, axis=-1).mean())


def _pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_rr = pred - pred[0:1]
    gt_rr = gt - gt[0:1]
    scale, rotation, translation = umeyama_sim3(pred_rr, gt_rr, with_scale=True)
    aligned = (scale * (rotation @ pred_rr.T)).T + translation
    return float(np.linalg.norm(aligned - gt_rr, axis=-1).mean())


def _consecutive_runs(mask: np.ndarray):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if run.size >= 3:
            yield run


def _export_presence_errors(
    gt: GTSequence,
    pred_positive: np.ndarray,
    gt_presence: np.ndarray,
    decision_values: np.ndarray,
    decision_name: str,
) -> int:
    root_value = os.environ.get("HAND_PRESENCE_ERROR_DIR")
    if not root_value:
        return 0

    import cv2

    frames = min(
        len(gt.image_paths), len(pred_positive), len(gt_presence), len(decision_values),
    )
    predicted = np.asarray(pred_positive, bool)[:frames]
    target = np.asarray(gt_presence, bool)[:frames]
    values = np.asarray(decision_values, np.float32)[:frames]
    if predicted.shape != (frames, 2) or target.shape != (frames, 2):
        raise ValueError("presence error export 需要 [T,2] 的预测和 GT")

    root = Path(root_value)
    safe_seq = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(gt.seq_id))
    segment_index = int((gt.meta or {}).get("segment_index", 0))
    source_seq = str((gt.meta or {}).get("seq", gt.seq_id))
    records = []
    error_frames = np.flatnonzero(np.any(predicted != target, axis=1))
    for frame in error_frames:
        errors = []
        for side_index in np.flatnonzero(predicted[frame] != target[frame]):
            side = SIDES[int(side_index)]
            error_type = "FN" if target[frame, side_index] else "FP"
            errors.append({"error_type": error_type, "side": side})
        error_labels = [
            f"{item['error_type']}_{item['side'].upper()}" for item in errors
        ]
        source_frame = segment_index * 81 + int(frame)
        category = error_labels[0] if len(error_labels) == 1 else "MULTIPLE"
        output_dir = root / category
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"{safe_seq}__local{int(frame):03d}__source{source_frame:06d}"
            f"__{'+'.join(error_labels)}.jpg"
        )
        output_path = output_dir / filename

        from ..frame_io import read_bgr_frame

        image = read_bgr_frame(gt.image_paths[int(frame)])
        banner = np.full((ERROR_BANNER_HEIGHT, image.shape[1], 3), 24, np.uint8)
        line1 = "ERROR: " + " + ".join(label.replace("_", " ") for label in error_labels)
        line2 = (
            f"GT hand_kept: LEFT={'YES' if target[frame, 0] else 'NO'}"
            f"  RIGHT={'YES' if target[frame, 1] else 'NO'}"
        )
        line3 = (
            f"PRED: LEFT={'YES' if predicted[frame, 0] else 'NO'}"
            f"  RIGHT={'YES' if predicted[frame, 1] else 'NO'}"
            f" | {decision_name} L={float(values[frame, 0]):+.3f} R={float(values[frame, 1]):+.3f}"
        )
        line4 = f"{source_seq} | source frame={source_frame} | segment frame={int(frame)}"
        cv2.putText(banner, line1, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(banner, line2, (8, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 220, 255), 1, cv2.LINE_AA)
        cv2.putText(banner, line3, (8, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (100, 255, 100), 1, cv2.LINE_AA)
        cv2.putText(banner, line4, (8, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, cv2.LINE_AA)
        canvas = np.concatenate((banner, image), axis=0)
        if not cv2.imwrite(str(output_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"无法写入 presence 错误帧: {output_path}")

        records.append({
            "image": str(output_path),
            "source_image": str(gt.image_paths[int(frame)]),
            "seq_id": str(gt.seq_id),
            "source_seq": source_seq,
            "segment_index": segment_index,
            "segment_frame": int(frame),
            "source_frame": source_frame,
            "errors": errors,
            "gt_hand_kept": {
                "left": bool(target[frame, 0]), "right": bool(target[frame, 1]),
            },
            "predicted_presence": {
                "left": bool(predicted[frame, 0]), "right": bool(predicted[frame, 1]),
            },
            "decision_name": decision_name,
            "decision_values": {
                "left": float(values[frame, 0]), "right": float(values[frame, 1]),
            },
        })

    if records:
        manifest_dir = root / "_manifests"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"errors-{os.getpid()}.jsonl"
        with manifest_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    return len(records)


def _decode_mano_array(hand: np.ndarray) -> dict[str, np.ndarray]:
    """Decode ``[T,2,109]`` camera MANO into protocol geometry."""
    from visualization.reproj_core import geometry, mano

    values = np.asarray(hand, np.float32)
    if values.ndim != 3 or values.shape[1:] != (2, PER_HAND):
        raise ValueError(f"hand_mano_6d 应为 [T,2,109]，得到 {values.shape}")
    frames = values.shape[0]
    out = {
        "verts": np.empty((frames, 2, 778, 3), np.float64),
        "joints": np.empty((frames, 2, 21, 3), np.float64),
        "translation": values[..., :3].astype(np.float64),
        "orientation": np.empty((frames, 2, 3, 3), np.float64),
    }
    for side_index, side in enumerate(SIDES):
        segment = values[:, side_index]
        decoded = mano.decode_hand_6d(
            segment[:, :3], segment[:, 3:9], segment[:, 9:99], segment[:, 99:109],
            is_right=(side == "right"),
        )
        verts, joints = mano.run_mano(
            decoded["trans"], decoded["rot"], decoded["hand_pose"], decoded["betas"],
            is_right=(side == "right"),
        )
        out["verts"][:, side_index] = verts
        out["joints"][:, side_index] = joints[:, :21]
        out["orientation"][:, side_index] = geometry.rot6d_to_mat(segment[:, 3:9])
    return out


def _canonical_hand_geometry() -> dict[str, np.ndarray]:
    """Return the identity-rotation, zero-pose, mean-shape MANO placeholder."""
    global _CANONICAL_GEOMETRY
    if _CANONICAL_GEOMETRY is None:
        zeros = np.zeros((1, 2, PER_HAND), np.float32)
        identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32)
        zeros[..., 3:9] = identity_6d
        zeros[..., 9:99] = np.tile(identity_6d, 15)
        decoded = _decode_mano_array(zeros)
        _CANONICAL_GEOMETRY = {
            "joints": decoded["joints"][0],
            "orientation": decoded["orientation"][0],
            "translation": decoded["translation"][0],
        }
    return _CANONICAL_GEOMETRY


def _flatten_prediction_geometry(pred: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([pred[side][key] for side in SIDES], axis=1)
        for key in ("verts", "joints", "translation", "orientation")
    }


def _metric_stats(
    pred: dict[str, np.ndarray],
    gt: dict[str, np.ndarray],
    pred_positive: np.ndarray,
    gt_valid: np.ndarray,
    *,
    K: np.ndarray = PROTOCOL_K,
    hw: tuple[int, int] = PROTOCOL_HW,
    canonical: dict[str, np.ndarray] | None = None,
) -> dict:
    """Compute corpus-additive statistics under the ViDiHand penalty protocol."""
    frames = min(pred["joints"].shape[0], gt["joints"].shape[0], len(pred_positive), len(gt_valid))
    if frames <= 0:
        raise NotImplementedError("相机系手部覆盖率片段没有帧")
    pred = {key: np.asarray(value)[:frames] for key, value in pred.items()}
    gt = {key: np.asarray(value)[:frames] for key, value in gt.items()}
    pred_positive = np.asarray(pred_positive, bool)[:frames]
    gt_valid = np.asarray(gt_valid, bool)[:frames]
    if pred_positive.shape != (frames, 2) or gt_valid.shape != (frames, 2):
        raise ValueError("presence/GT valid 必须为 [T,2]")

    gt_joint_on_screen = _on_screen_joints(gt["joints"], K, hw)
    gt_presence = gt_valid & gt_joint_on_screen.any(axis=-1)
    excluded_offscreen = gt_valid & ~gt_presence
    evaluated_frames = gt_presence.any(axis=1)
    tp_mask = pred_positive & gt_presence
    fn_mask = ~pred_positive & gt_presence
    fp_mask = pred_positive & ~gt_presence & ~excluded_offscreen
    fn_per_frame = fn_mask.sum(axis=1)
    fp_per_frame = fp_mask.sum(axis=1)

    stats = {
        "facc_correct": float(np.sum(
            evaluated_frames & (fn_per_frame == 0) & (fp_per_frame == 0)
        )),
        "facc_frames": float(evaluated_frames.sum()),
        "tp": float(tp_mask.sum()),
        "fn": float(fn_mask.sum()),
        "fp": float(fp_mask.sum()),
        "pose_count": float(tp_mask.sum() + fn_mask.sum()),
        "mpjpe_sum_mm": 0.0,
        "pa_mpjpe_sum_mm": 0.0,
        "epe_sum_px": 0.0,
        "epe_joint_count": 0.0,
        "go_sum_deg": 0.0,
        "ct_sum_m": 0.0,
        "jitter_sum_mm": 0.0,
        "jitter_count": 0.0,
    }
    diagonal = float(np.hypot(hw[1], hw[0]))

    for frame, side in np.argwhere(tp_mask):
        pred_joints = pred["joints"][frame, side]
        gt_joints = gt["joints"][frame, side]
        stats["mpjpe_sum_mm"] += _root_relative_mpjpe(pred_joints, gt_joints) * 1000.0
        stats["pa_mpjpe_sum_mm"] += _pa_mpjpe(pred_joints, gt_joints) * 1000.0
        stats["go_sum_deg"] += _geodesic_degrees(
            pred["orientation"][frame, side], gt["orientation"][frame, side],
        )
        stats["ct_sum_m"] += float(np.linalg.norm(
            pred["translation"][frame, side] - gt["translation"][frame, side]
        ))
        joint_mask = gt_joint_on_screen[frame, side]
        pred_uv = _project(pred_joints, K, floor_depth=True)
        gt_uv = _project(gt_joints, K, floor_depth=False)
        distances = np.linalg.norm(pred_uv[joint_mask] - gt_uv[joint_mask], axis=-1)
        stats["epe_sum_px"] += float(np.minimum(distances, diagonal).sum())
        stats["epe_joint_count"] += float(joint_mask.sum())

    if fn_mask.any():
        canonical = canonical or _canonical_hand_geometry()
        for frame, side in np.argwhere(fn_mask):
            gt_joints = gt["joints"][frame, side]
            canonical_joints = np.asarray(canonical["joints"])[side]
            # The paper intentionally uses raw canonical MPJPE for both MPJPE-p
            # and PA-MPJPE-p false-negative penalties.
            penalty = _root_relative_mpjpe(canonical_joints, gt_joints) * 1000.0
            stats["mpjpe_sum_mm"] += penalty
            stats["pa_mpjpe_sum_mm"] += penalty
            stats["go_sum_deg"] += _geodesic_degrees(
                np.asarray(canonical["orientation"])[side],
                gt["orientation"][frame, side],
            )
            stats["ct_sum_m"] += float(np.linalg.norm(gt["translation"][frame, side]))
            joint_count = float(gt_joint_on_screen[frame, side].sum())
            stats["epe_sum_px"] += diagonal * joint_count
            stats["epe_joint_count"] += joint_count

    for side in range(2):
        for run in _consecutive_runs(tp_mask[:, side]):
            joints = pred["joints"][run, side]
            acceleration = joints[2:] - 2.0 * joints[1:-1] + joints[:-2]
            per_frame = np.linalg.norm(acceleration, axis=-1).mean(axis=-1) * 1000.0
            stats["jitter_sum_mm"] += float(per_frame.sum())
            stats["jitter_count"] += float(len(per_frame))
    return stats


def _stats_as_metrics(stats: dict, *, single_forward: bool) -> dict:
    metrics = metrics_from_stats(stats)
    hidden = {
        "_facc_correct": stats["facc_correct"],
        "_facc_frames": stats["facc_frames"],
        "_tp": stats["tp"],
        "_fn": stats["fn"],
        "_fp": stats["fp"],
        "_pose_count": stats["pose_count"],
        "_mpjpe_sum_mm": stats["mpjpe_sum_mm"],
        "_pa_mpjpe_sum_mm": stats["pa_mpjpe_sum_mm"],
        "_epe_sum_px": stats["epe_sum_px"],
        "_epe_joint_count": stats["epe_joint_count"],
        "_go_sum_deg": stats["go_sum_deg"],
        "_ct_sum_m": stats["ct_sum_m"],
        "_jitter_sum_mm": stats["jitter_sum_mm"],
        "_jitter_count": stats["jitter_count"],
        "_single_forward": float(bool(single_forward)),
    }
    return {**metrics, **hidden}


@HEADS.register("hands_coverage")
class HandCoverageHead(HeadEvaluator):
    name = "hands_coverage"
    required_gt = {HAND_COVERAGE}

    def extract(self, pred: Prediction):
        geometry = _flatten_prediction_geometry(camera_hand_geometry(pred))
        if pred.hand_presence_logits is not None:
            values = np.asarray(pred.hand_presence_logits, np.float32)
            positive = values >= 0.0
            decision_name = "logit"
        elif pred.hand_confidence is not None:
            values = np.asarray(pred.hand_confidence, np.float32)
            positive = values >= 0.5
            decision_name = "confidence"
        else:
            raise NotImplementedError("模型没有 hand presence/confidence 输出")
        if positive.ndim != 2 or positive.shape[1] != 2:
            raise ValueError(f"hand presence 应为 [T,2]，得到 {positive.shape}")
        return {
            "geometry": geometry,
            "positive": positive,
            "decision_values": values,
            "decision_name": decision_name,
            "single_forward": bool((pred.meta or {}).get("single_forward", False)),
        }

    def align(self, item, gt: GTSequence):
        if gt.hand_mano_6d is None or gt.hand_valid_lr is None:
            raise NotImplementedError("数据集未提供覆盖率指标所需的双手 MANO/valid GT")
        intrinsic = np.asarray(gt.intrinsic, np.float64)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("相机系手部覆盖率指标需要有效 3x3 像素内参")
        return {
            "pred": item["geometry"],
            "positive": item["positive"],
            "decision_values": item["decision_values"],
            "decision_name": item["decision_name"],
            "gt": _decode_mano_array(gt.hand_mano_6d),
            "gt_valid": np.asarray(gt.hand_valid_lr, bool),
            "canonical": _canonical_hand_geometry(),
            "single_forward": item["single_forward"],
        }

    def metrics(self, aligned, gt: GTSequence):
        gt_on_screen = _on_screen_joints(
            aligned["gt"]["joints"], np.asarray(gt.intrinsic, np.float64), tuple(gt.hw),
        ).any(axis=-1)
        gt_presence = aligned["gt_valid"] & gt_on_screen
        exported = _export_presence_errors(
            gt,
            aligned["positive"],
            gt_presence,
            aligned["decision_values"],
            aligned["decision_name"],
        )
        stats = _metric_stats(
            aligned["pred"], aligned["gt"], aligned["positive"], aligned["gt_valid"],
            K=np.asarray(gt.intrinsic, np.float64), hw=tuple(gt.hw),
            canonical=aligned["canonical"],
        )
        metrics = _stats_as_metrics(
            stats,
            single_forward=aligned["single_forward"],
        )
        meta = gt.meta or {}
        metrics["_split_seed"] = float(meta.get("split_seed", 42))
        metrics["_requested_segments"] = float(meta.get("requested_segments", 437))
        metrics["_available_segments"] = float(meta.get("available_segments", 0))
        metrics["_official_split"] = float(bool(meta.get("official_split", False)))
        metrics["_reference_same_split"] = float(bool(meta.get("reference_same_split", False)))
        metrics["_presence_error_frames_exported"] = float(exported)
        return metrics
