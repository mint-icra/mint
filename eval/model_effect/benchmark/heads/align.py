# -*- coding: utf-8 -*-
"""单目尺度对齐工具(各头共用,纯 numpy)。

单目 3D 预测普遍存在自由度,直接和 GT 比会"全错"——必须先对齐再算指标:
  - 轨迹/点云: 7-DoF 相似变换(旋转+平移+全局 scale),Umeyama 闭式解。
  - 深度: 1-DoF 中值缩放,或 2-DoF 仿射(scale+shift,affine-invariant)。
对齐后应把 scale(及 shift)作为诊断量报出,它本身反映"尺度估得准不准"。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def umeyama_sim3(src: np.ndarray, dst: np.ndarray,
                 with_scale: bool = True) -> Tuple[float, np.ndarray, np.ndarray]:
    """求把 src 对齐到 dst 的相似变换 (s,R,t),使 s*R@src + t ≈ dst。Umeyama(1991) SVD 闭式解。

    Args:
        src: [N,3] 待对齐点(如预测相机中心)。
        dst: [N,3] 目标点(如 GT 相机中心)。
        with_scale: False 时锁 s=1(已知度量尺度)。
    Returns:
        (s, R[3,3], t[3]);对 src 施加 s*R@x+t 即对齐到 dst 系。
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n, dim = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # 保右手系,避免镜像反射
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    if with_scale:
        var_s = (sc ** 2).sum() / n
        s = float((D * np.diag(S)).sum() / max(var_s, 1e-12))
    else:
        s = 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


# ── HaWoR world 系手部对齐(基于 umeyama_sim3;供 hands_world 头出 W/WA/PA-MPJPE)──
def _apply_sim3(joints: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """对 (T,J,3) 施加相似变换 s*R@x+t。"""
    return (s * np.einsum("ij,tnj->tni", R, joints)) + t[None, None]


def hawor_first_align(gt_joints: np.ndarray, pred_joints: np.ndarray) -> np.ndarray:
    """首帧对齐(W-MPJPE 用):用**前 2 帧**关节拟合 sim3(pred→gt),施加到全序列。(T,J,3)。"""
    k = min(2, gt_joints.shape[0])
    s, R, t = umeyama_sim3(pred_joints[:k].reshape(-1, 3), gt_joints[:k].reshape(-1, 3))
    return _apply_sim3(pred_joints, s, R, t)


def hawor_global_align(gt_joints: np.ndarray, pred_joints: np.ndarray) -> np.ndarray:
    """整段对齐(WA-MPJPE 用):用**全序列**关节拟合 sim3(pred→gt),施加到全序列。(T,J,3)。"""
    s, R, t = umeyama_sim3(pred_joints.reshape(-1, 3), gt_joints.reshape(-1, 3))
    return _apply_sim3(pred_joints, s, R, t)


def procrustes_per_frame(gt_joints: np.ndarray, pred_joints: np.ndarray) -> np.ndarray:
    """逐帧 Procrustes(PA-MPJPE 用):每帧独立 sim3 对齐 pred→gt。(T,J,3)。"""
    out = np.empty_like(pred_joints, dtype=np.float64)
    for i in range(pred_joints.shape[0]):
        s, R, t = umeyama_sim3(pred_joints[i], gt_joints[i])
        out[i] = (s * (R @ pred_joints[i].T).T) + t
    return out


def median_scale(pred: np.ndarray, gt: np.ndarray,
                 mask: np.ndarray) -> Tuple[float, np.ndarray]:
    """中值缩放对齐深度:s = median(gt[mask]) / median(pred[mask]),返回 (s, s*pred)。单目 1-DoF。"""
    m = np.asarray(mask, dtype=bool)
    pv, gv = np.asarray(pred, np.float64)[m], np.asarray(gt, np.float64)[m]
    ok = np.isfinite(pv) & np.isfinite(gv) & (pv > 0) & (gv > 0)
    if ok.sum() == 0:
        return 1.0, np.asarray(pred, np.float32).copy()
    s = float(np.median(gv[ok]) / max(float(np.median(pv[ok])), 1e-8))
    return s, (np.asarray(pred, np.float32) * s)


def least_squares_scale_shift(pred: np.ndarray, gt: np.ndarray,
                              mask: np.ndarray) -> Tuple[float, float, np.ndarray]:
    """仿射(scale+shift)对齐深度:最小二乘 a,b 使 a*pred+b ≈ gt,返回 (a,b,a*pred+b)。2-DoF,MiDaS/MoGe 系。"""
    m = np.asarray(mask, dtype=bool)
    pv, gv = np.asarray(pred, np.float64)[m], np.asarray(gt, np.float64)[m]
    ok = np.isfinite(pv) & np.isfinite(gv)
    pv, gv = pv[ok], gv[ok]
    if pv.size < 2:
        return 1.0, 0.0, np.asarray(pred, np.float32).copy()
    A = np.stack([pv, np.ones_like(pv)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, gv, rcond=None)
    return float(a), float(b), (a * np.asarray(pred, np.float32) + b)
