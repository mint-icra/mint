# -*- coding: utf-8 -*-
"""NumPy camera-trajectory metrics matching the ICRA full-sequence protocol."""
from __future__ import annotations

import math

import numpy as np

PUBLIC_METRICS = (
    "ATE_mm", "RPE_T_mm", "RPE_R_deg", "ATE_S_mm", "RPE_T_S_mm",
    "ATE_pct", "ATE_S_pct", "scale", "scale_error_pct", "path_scale",
    "FPS", "n_frames",
)


def _umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    count, dimensions = src.shape
    src_mean, dst_mean = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - src_mean, dst - dst_mean
    covariance = (dst_centered.T @ src_centered) / count
    u, singular, vh = np.linalg.svd(covariance)
    sign = np.eye(dimensions)
    if np.linalg.det(u) * np.linalg.det(vh) < 0:
        sign[-1, -1] = -1.0
    rotation = u @ sign @ vh
    if with_scale:
        variance = float((src_centered ** 2).sum() / count)
        scale = float((singular * np.diag(sign)).sum() / max(variance, 1e-12))
    else:
        scale = 1.0
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation, translation


def _orthogonalize(poses: np.ndarray) -> np.ndarray:
    result = np.asarray(poses, np.float64).copy()
    for index, rotation in enumerate(result[:, :3, :3]):
        u, _singular, vh = np.linalg.svd(rotation)
        fixed = u @ vh
        if np.linalg.det(fixed) < 0:
            u[:, -1] *= -1
            fixed = u @ vh
        result[index, :3, :3] = fixed
        result[index, 3] = (0.0, 0.0, 0.0, 1.0)
    return result


def _aligned_poses(pred: np.ndarray, gt: np.ndarray, with_scale: bool):
    pred_centers = pred[:, :3, 3]
    gt_centers = gt[:, :3, 3]
    scale, rotation, translation = _umeyama_sim3(
        pred_centers, gt_centers, with_scale=with_scale,
    )
    aligned = pred.copy()
    aligned[:, :3, :3] = np.einsum("ij,tjk->tik", rotation, pred[:, :3, :3])
    aligned[:, :3, 3] = (
        scale * np.einsum("ij,tj->ti", rotation, pred_centers)
        + translation[None]
    )
    return aligned, float(scale)


def _inverse_se3(poses: np.ndarray) -> np.ndarray:
    result = np.tile(np.eye(4, dtype=np.float64), (len(poses), 1, 1))
    rotation_t = np.transpose(poses[:, :3, :3], (0, 2, 1))
    result[:, :3, :3] = rotation_t
    result[:, :3, 3] = -np.einsum("tij,tj->ti", rotation_t, poses[:, :3, 3])
    return result


def _relative(poses: np.ndarray) -> np.ndarray:
    return np.matmul(_inverse_se3(poses[:-1]), poses[1:])


def _rpe(gt: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    gt_relative = _relative(gt)
    pred_relative = _relative(pred)
    error = np.matmul(_inverse_se3(gt_relative), pred_relative)
    translation = np.linalg.norm(error[:, :3, 3], axis=1)
    trace = np.trace(error[:, :3, :3], axis1=1, axis2=2)
    angle = np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0)))
    return (
        float(np.sqrt(np.mean(translation ** 2))),
        float(np.sqrt(np.mean(angle ** 2))),
    )


def trajectory_metrics(
    pred_c2w: np.ndarray,
    gt_c2w: np.ndarray,
    *,
    forward_seconds: float | None = None,
) -> dict[str, float]:
    """Evaluate dense frame-aligned c2w trajectories as one global sequence."""
    pred = np.asarray(pred_c2w, np.float64)
    gt = np.asarray(gt_c2w, np.float64)
    frames = min(len(pred), len(gt))
    if frames < 2:
        raise ValueError("相机轨迹评测至少需要 2 个有效帧对")
    pred, gt = pred[:frames], gt[:frames]
    valid = np.isfinite(pred.reshape(frames, -1)).all(1) & np.isfinite(gt.reshape(frames, -1)).all(1)
    if not valid.all():
        pred, gt = pred[valid], gt[valid]
        frames = len(pred)
    if frames < 2:
        raise ValueError("过滤 NaN/Inf 后不足 2 个相机位姿")
    pred, gt = _orthogonalize(pred), _orthogonalize(gt)

    gt_path = float(np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).sum())
    pred_path = float(np.linalg.norm(np.diff(pred[:, :3, 3], axis=0), axis=1).sum())
    seconds = float(forward_seconds or 0.0)
    if gt_path <= 1e-9 or pred_path <= 1e-9:
        # Very short prefixes can be completely static.  A fitted Sim(3) then
        # collapses to scale=0 and a fake zero ATE, so expose only throughput
        # and provenance instead of ranking the trajectory.
        return {
            "FPS": frames / seconds if seconds > 0 else float("nan"),
            "n_frames": float(frames),
            "degenerate": 1.0,
            "_forward_s": seconds,
            "_gt_path_m": gt_path,
            "_pred_path_m": pred_path,
        }

    pred_sim3, scale = _aligned_poses(pred, gt, with_scale=True)
    pred_se3, _ = _aligned_poses(pred, gt, with_scale=False)
    error = np.linalg.norm(pred_sim3[:, :3, 3] - gt[:, :3, 3], axis=1)
    error_se3 = np.linalg.norm(pred_se3[:, :3, 3] - gt[:, :3, 3], axis=1)
    ate = float(np.sqrt(np.mean(error ** 2)))
    ate_se3 = float(np.sqrt(np.mean(error_se3 ** 2)))
    rpe_t, rpe_r = _rpe(gt, pred_sim3)
    # The ICRA comparison table deliberately keeps metric scale fixed.  Keep
    # this RPE beside the default Sim(3)-aligned value so both protocols can
    # be rendered from the same per-sequence report.
    rpe_t_se3, _rpe_r_se3 = _rpe(gt, pred_se3)

    return {
        "ATE_mm": ate * 1000.0,
        "RPE_T_mm": rpe_t * 1000.0,
        "RPE_R_deg": rpe_r,
        "ATE_S_mm": ate_se3 * 1000.0,
        "RPE_T_S_mm": rpe_t_se3 * 1000.0,
        "ATE_pct": ate / gt_path * 100.0 if gt_path > 1e-9 else float("nan"),
        "ATE_S_pct": ate_se3 / gt_path * 100.0 if gt_path > 1e-9 else float("nan"),
        "scale": scale,
        "scale_error_pct": abs(scale - 1.0) * 100.0,
        "path_scale": gt_path / pred_path if pred_path > 1e-9 else float("nan"),
        "FPS": frames / seconds if seconds > 0 else float("nan"),
        "n_frames": float(frames),
        "degenerate": 0.0,
        "_forward_s": seconds,
        "_gt_path_m": gt_path,
        "_pred_path_m": pred_path,
    }


def _finite_mean(seqs: dict, key: str) -> float | None:
    values = [
        float(metrics[key]) for metrics in seqs.values()
        if key in metrics and math.isfinite(float(metrics[key]))
    ]
    return float(np.mean(values)) if values else None


def _finite_median(seqs: dict, key: str) -> float | None:
    values = [
        float(metrics[key]) for metrics in seqs.values()
        if key in metrics and math.isfinite(float(metrics[key]))
    ]
    return float(np.median(values)) if values else None


def aggregate_trajectory_metrics(seqs: dict, dataset_name: str) -> dict:
    """Aggregate like the ICRA table: sequence-equal metrics and frame/time FPS."""
    mean = {key: _finite_mean(seqs, key) for key in PUBLIC_METRICS}
    # The paper-facing table reports both the sequence-equal mean and median.
    # Preserve the explicit suffixes so the live Viewer rows use the same
    # values as the fixed reference rows.
    for source, target in (
        ("ATE_mm", "ATE_median_mm"),
        ("ATE_S_mm", "ATE_S_median_mm"),
        ("RPE_T_mm", "RPE_T_median_mm"),
        ("RPE_T_S_mm", "RPE_T_S_median_mm"),
        ("RPE_R_deg", "RPE_R_median_deg"),
        ("path_scale", "path_scale_median"),
        ("ATE_pct", "ATE_pct_median"),
        ("ATE_S_pct", "ATE_S_pct_median"),
    ):
        mean[target] = _finite_median(seqs, source)
    total_frames = sum(float(metrics.get("n_frames", 0.0)) for metrics in seqs.values())
    total_seconds = sum(float(metrics.get("_forward_s", 0.0)) for metrics in seqs.values())
    truncated = sum(int(round(float(metrics.get("_truncated", 0.0)))) for metrics in seqs.values())
    degenerate = sum(int(round(float(metrics.get("degenerate", 0.0)))) for metrics in seqs.values())
    mean["FPS"] = total_frames / total_seconds if total_seconds > 0 else None
    mean["n_frames"] = total_frames
    is_hot3d = dataset_name == "camera_hot3d"
    return {
        "mean": mean,
        "counts": {
            "sequences": len(seqs), "frames": int(round(total_frames)),
            "truncated_sequences": truncated,
            "degenerate_sequences": degenerate,
        },
        "protocol": {
            "name": "ICRA全长相机轨迹",
            "version": "icra_full_sequence_v1",
            "official_split": False,
            "evaluation_split": (
                "HOT3D 27-sequence rectified validation export"
                if is_hot3d else
                "ARCTIC protocol P2 validation (s05), 34 full sequences"
            ),
            "sequence_mode": "full_sequence" if not truncated else "prefix_diagnostic",
            "alignment": "whole-sequence Umeyama Sim(3)",
            "rpe": "delta=1 frame, SE(3) relative error, RMSE",
            "comparison_alignment": "whole-sequence Umeyama SE(3), fixed scale=1",
            "comparison_metrics": "ATE_S_mm, RPE_T_S_mm, RPE_R_deg, path_scale, ATE_S_pct",
            "inference": "training clip window + adjacent-window SE(3) chaining",
            "reference_same_split": False,
        },
    }
