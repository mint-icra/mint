# -*- coding: utf-8 -*-
"""hands_world 头 ✅:HaWoR world 系手部指标，两种相机变体对比。

口径对齐 HaWoR(eval_hawor_hot3d.py):pred 相机系手 → world → 与 GT world 关节对齐后算:
  W-MPJPE  = 每 100 帧用前 2 帧对齐(align.hawor_first_align)
  WA-MPJPE = 每 100 帧整段对齐(align.hawor_global_align)
  PA-MPJPE = 逐帧 Procrustes(align.procrustes_per_frame)
  RTE       = root 轨迹刚性对齐后的误差 / GT 总位移 ×100 (%)
  Accel     = 关节二阶差分误差 ×30² (m/s²)
仅 valid 帧；不可见区间会断开轨迹，避免跨断点计算时序指标。

pred 相机系手 → world 三种诊断(都出,对比):
  poseenc: 用 pred.extrinsic_c2w，并只做 SE(3) 对齐——保留模型自身相机轨迹尺度
  poseenc_scaled: 对相机中心额外拟合一个全局最优尺度——公平观察去除尺度误差后的上限
  slam:    用 gt.meta['cam2world_slam']=head@ego(Aria SLAM 相机)——排除相机误差,看手本身
世界轨迹指标带变体后缀(如 W_MPJPE_poseenc / RTE_slam);PA 与相机无关
(Procrustes 吸收),只出一个。

⚠ 需 enable_hand + pose_enc 的学生 ckpt;GT 需含 HAND_WORLD(hand_joints_3d_world)+ EXTRINSIC。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterator

import numpy as np

from .align import (
    hawor_first_align,
    hawor_global_align,
    procrustes_per_frame,
    umeyama_sim3,
)
from .hands import camera_hand_joints
from ..core.registry import HEADS
from ..core.schema import EXTRINSIC, HAND, GTSequence, Prediction
from .base import HeadEvaluator

_MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(_MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(_MODEL_EFFECT))

_CHUNK_LENGTH = 100
_FPS = 30.0


def _contiguous_runs(valid: np.ndarray) -> Iterator[np.ndarray]:
    """Yield original frame indices for each uninterrupted valid-hand track."""
    indices = np.flatnonzero(np.asarray(valid, dtype=bool))
    if not len(indices):
        return
    for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
        if len(run):
            yield run


def _world_mpjpe(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, int]:
    """Compute frame-weighted W/WA-MPJPE over 100-frame valid track chunks."""
    w_errors, wa_errors = [], []
    n_chunks = 0
    for run in _contiguous_runs(valid):
        for start in range(0, len(run), _CHUNK_LENGTH):
            frame_indices = run[start:start + _CHUNK_LENGTH]
            g, p = gt_joints[frame_indices], pred_joints[frame_indices]
            w = hawor_first_align(g, p)
            wa = hawor_global_align(g, p)
            w_errors.append(np.linalg.norm(g - w, axis=-1).mean(axis=-1))
            wa_errors.append(np.linalg.norm(g - wa, axis=-1).mean(axis=-1))
            n_chunks += 1
    if not w_errors:
        raise NotImplementedError("无 valid 帧(手均不在画面 / NaN)")
    return (
        float(np.concatenate(w_errors).mean() * 1000.0),
        float(np.concatenate(wa_errors).mean() * 1000.0),
        n_chunks,
    )


def _root_translation_error(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    valid: np.ndarray,
) -> float:
    """HaWoR RTE (%) using wrist joint 0 as the available MANO root proxy."""
    errors = []
    for run in _contiguous_runs(valid):
        if len(run) < 2:
            continue
        gt_root = gt_joints[run, 0]
        pred_root = pred_joints[run, 0]
        displacement = float(np.linalg.norm(np.diff(gt_root, axis=0), axis=-1).sum())
        if displacement <= 1e-12:
            continue
        _, rotation, translation = umeyama_sim3(
            pred_root, gt_root, with_scale=False,
        )
        pred_aligned = (rotation @ pred_root.T).T + translation
        errors.append(
            np.linalg.norm(gt_root - pred_aligned, axis=-1) / displacement * 100.0
        )
    return float(np.concatenate(errors).mean()) if errors else float("nan")


def _acceleration_error(
    gt_joints: np.ndarray,
    pred_joints: np.ndarray,
    valid: np.ndarray,
) -> float:
    """HaWoR acceleration error (m/s²), evaluated only within valid tracks."""
    errors = []
    for run in _contiguous_runs(valid):
        if len(run) < 5:
            continue
        g, p = gt_joints[run], pred_joints[run]
        accel_gt = g[:-2] - 2.0 * g[1:-1] + g[2:]
        accel_pred = p[:-2] - 2.0 * p[1:-1] + p[2:]
        per_frame = np.linalg.norm(accel_pred - accel_gt, axis=-1).mean(axis=-1)
        # The official HOT3D script discards the first/last acceleration sample.
        errors.append(per_frame[1:-1] * (_FPS ** 2))
    return float(np.concatenate(errors).mean()) if errors else float("nan")


@HEADS.register("hands_world")
class HandsWorldHead(HeadEvaluator):
    name = "hands_world"
    required_gt = {HAND, EXTRINSIC}                              # 模型须产 hand(相机系)+ extrinsic(pose_enc 相机);world 关节由 GT 字段提供(align 守卫)

    def extract(self, pred: Prediction) -> Any:
        """解码左右手相机系关节 + 带上 pred.extrinsic_c2w(pose_enc 相机)。"""
        if pred.hand is None:
            raise NotImplementedError("模型未产出 hand(需 enable_hand 的学生 ckpt)")
        return {
            "cam": camera_hand_joints(pred),
            "c2w_poseenc": np.asarray(pred.extrinsic_c2w, np.float64),
        }

    def align(self, item: Any, gt: GTSequence):
        if gt.hand_joints_3d_world is None:
            raise NotImplementedError("数据集未提供 world 关节(hand_joints_3d_world)")
        side = gt.meta.get("mano_side", "right")
        Jc = item["cam"][side]                                       # pred 相机系 (Sp,21,3)
        gw = np.asarray(gt.hand_joints_3d_world, np.float64)         # GT world (Sg,21,3)
        T = min(Jc.shape[0], gw.shape[0], item["c2w_poseenc"].shape[0])
        Jc, gw = Jc[:T], gw[:T]
        valid = (np.ones(T, bool) if gt.hand_valid is None else np.asarray(gt.hand_valid, bool)[:T])
        valid = valid & np.isfinite(Jc).all((1, 2)) & np.isfinite(gw).all((1, 2))
        # pred → world。相机尺度只作用于相机中心，不缩放相机系 MANO 手尺寸。
        c2w_pe = item["c2w_poseenc"][:T]
        cam2w_slam = np.asarray(gt.meta.get("cam2world_slam"), np.float64)[:T] if gt.meta.get("cam2world_slam") is not None else None
        Jw_sl = (np.einsum("tij,tnj->tni", cam2w_slam[:, :3, :3], Jc) + cam2w_slam[:, None, :3, 3]
                 if cam2w_slam is not None else None)
        gt_c2w = np.linalg.inv(np.asarray(gt.extrinsic_w2c, np.float64)[:T])
        camera_valid = (
            np.isfinite(c2w_pe[:, :3, 3]).all(axis=-1)
            & np.isfinite(gt_c2w[:, :3, 3]).all(axis=-1)
        )
        if camera_valid.sum() < 2:
            raise NotImplementedError("有效相机轨迹不足 2 帧，无法拟合世界系对齐")
        scale, rotation, translation = umeyama_sim3(
            c2w_pe[camera_valid, :3, 3], gt_c2w[camera_valid, :3, 3], with_scale=True,
        )
        if not np.isfinite(scale) or scale <= 1e-6:
            raise NotImplementedError("预测相机轨迹尺度退化，无法进行 Sim(3) 对齐")
        _, rotation_metric, translation_metric = umeyama_sim3(
            c2w_pe[camera_valid, :3, 3], gt_c2w[camera_valid, :3, 3], with_scale=False,
        )

        def world_hand(camera_scale, world_rotation, world_translation):
            camera_centers = (
                camera_scale * np.einsum("ij,tj->ti", world_rotation, c2w_pe[:, :3, 3])
                + world_translation[None]
            )
            camera_rotations = np.einsum("ij,tjk->tik", world_rotation, c2w_pe[:, :3, :3])
            return (
                np.einsum("tij,tnj->tni", camera_rotations, Jc)
                + camera_centers[:, None]
            )

        Jw_pe = world_hand(1.0, rotation_metric, translation_metric)
        Jw_pe_scaled = world_hand(scale, rotation, translation)
        return {
            "gt": gw, "poseenc": Jw_pe, "poseenc_scaled": Jw_pe_scaled,
            "slam": Jw_sl, "valid": valid, "camera_scale_poseenc": float(scale),
        }

    def metrics(self, a: Dict, gt: GTSequence) -> Dict[str, float]:
        valid = a["valid"]
        gt_joints = a["gt"]
        if not valid.any():
            raise NotImplementedError("无 valid 帧(手均不在画面 / NaN)")

        gt_valid = gt_joints[valid]
        out: Dict[str, float] = {
            "n_frames": float(valid.sum()),
            "camera_scale_poseenc": a["camera_scale_poseenc"],
            "camera_scale_error_pct": abs(a["camera_scale_poseenc"] - 1.0) * 100.0,
        }
        pa_done = False
        for tag, Jw in (
            ("poseenc", a["poseenc"]),
            ("poseenc_scaled", a["poseenc_scaled"]),
            ("slam", a["slam"]),
        ):
            if Jw is None:
                continue
            w_mpjpe, wa_mpjpe, n_chunks = _world_mpjpe(
                gt_joints, Jw, valid,
            )
            out[f"W_MPJPE_{tag}"] = w_mpjpe
            out[f"WA_MPJPE_{tag}"] = wa_mpjpe
            out[f"RTE_{tag}"] = _root_translation_error(gt_joints, Jw, valid)
            out[f"Accel_{tag}"] = _acceleration_error(gt_joints, Jw, valid)
            out["n_chunks"] = float(n_chunks)
            if not pa_done:                                          # PA 与相机无关,只算一次
                pred_valid = Jw[valid]
                pred_pa = procrustes_per_frame(gt_valid, pred_valid)
                out["PA_MPJPE"] = float(
                    np.linalg.norm(gt_valid - pred_pa, axis=-1).mean() * 1000.0
                )
                pa_done = True
        return out
