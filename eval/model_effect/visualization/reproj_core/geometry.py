#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval/lingbotmap 自包含几何/解码底座：6D↔旋转矩阵、左手翻转、世界点投影、相机 pose_enc 反解。

本模块刻意不依赖 tools/viewer：MANO 渲染所需的纯几何换算按行为对照 tools/viewer/mano_core.py
复刻于此，使 eval 验证脚本自成一体。GT(lerobot 预算列) 与 模型预测 同为 6D 表示，共用这里的
解码，保证两路在同一约定下可叠加比较。

依赖：numpy + scipy（旋转），相机反解借用仓库内 model_train/_vendor 的 pose_enc 反函数。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_DIR = Path(__file__).resolve().parents[4]   # reproj_core/ -> lingbotmap -> model_effect -> eval -> <repo>

# MANO 21 关节骨架连接（0=腕，1-4 拇…17-20 小），供骨架绘制使用。
MANO_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),
    (0, 5),  (5, 6),  (6, 7),  (7, 8),
    (0, 9),  (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def rot6d_to_mat(r6: np.ndarray) -> np.ndarray:
    """6D（前两行展平，与训练端 _mat_to_6d 取行约定一致）→ 旋转矩阵 (...,3,3)，Gram-Schmidt。"""
    a0 = r6[..., 0:3]
    a1 = r6[..., 3:6]
    b0 = a0 / np.linalg.norm(a0, axis=-1, keepdims=True)
    a1p = a1 - np.sum(b0 * a1, axis=-1, keepdims=True) * b0
    b1 = a1p / np.linalg.norm(a1p, axis=-1, keepdims=True)
    b2 = np.cross(b0, b1)
    return np.stack([b0, b1, b2], axis=-2)


def mat_to_6d(m: np.ndarray) -> np.ndarray:
    """旋转矩阵 (...,3,3) → 6D（取前两行展平，与 rot6d_to_mat / 训练端 _mat_to_6d 约定一致）。"""
    return m[..., :2, :].reshape(*m.shape[:-2], 6)


def rotmat_to_aa(R: np.ndarray) -> np.ndarray:
    """旋转矩阵批量转 axis-angle。R: (...,3,3) → (...,3)。"""
    from scipy.spatial.transform import Rotation as _Rot
    out_shape = R.shape[:-2] + (3,)
    flat = R.reshape(-1, 3, 3).astype(np.float64)
    aa = _Rot.from_matrix(flat).as_rotvec().astype(np.float32)
    return aa.reshape(out_shape)


def apply_left_hand_pose_flip_111(hand_pose_aa: np.ndarray) -> np.ndarray:
    """左手 hand_pose 逐关节乘 [1,-1,-1]（轴角 45 维），与训练端镜像约定还原一致。"""
    out = np.array(hand_pose_aa, copy=True)
    out[..., 1::3] *= -1.0
    out[..., 2::3] *= -1.0
    return out


def project(points_world: np.ndarray, cam_c2w: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """世界系点投影到图像，返回 (uv (N,2), depth (N,))。cam_c2w 为 cam→world，内部求逆得 w2c。

    健壮性：相机后方/过近的点(z<=eps)或坏相机(pred_c2w 退化→NaN/inf)会让 u,v 溢出成 inf/nan，
    下游 int(uv) 会 OverflowError。这里只对 z>eps 的点算投影，其余 uv 置 NaN、depth 置 -inf，
    调用方按 depth 阈值(>0.01)跳过即可；再兜一层把 inf/nan 的 uv 统一成 NaN(depth 相应作废)。"""
    cam_w2c = np.linalg.inv(cam_c2w)
    R = cam_w2c[:3, :3]
    t = cam_w2c[:3, 3]
    p_cam = (R @ points_world.T + t[:, None]).T   # (N, 3)
    z = p_cam[:, 2].astype(np.float64)
    eps = 1e-3
    valid = z > eps                                # 仅相机前方一定距离外的点可投影
    zc = np.where(valid, z, 1.0)                   # 无效点用占位 z 防除零(其 uv 随后置 NaN)
    u = K[0, 0] * p_cam[:, 0] / zc + K[0, 2]
    v = K[1, 1] * p_cam[:, 1] / zc + K[1, 2]
    uv = np.stack([u, v], axis=1)
    bad = ~valid | ~np.isfinite(uv).all(axis=1)    # 后方/过近点 或 溢出成 inf/nan 的点
    uv[bad] = np.nan
    depth = np.where(bad, -np.inf, z)              # 作废点 depth=-inf,必被 depth>0.01 过滤
    return uv, depth


def hand6d_cam_to_world(hands: dict, cam_c2w: np.ndarray) -> dict:
    """相机系双手 6D(orient/transl) --逐帧 cam→world 外参--> 世界系,pose/betas 原样。

    与 build_train_lerobot 的 world→cam 互逆:orient_world = R_c2w · orient_cam,
    transl_world = R_c2w · transl_cam + t_c2w。供 hand_frame='camera' 的数据在可视化前
    转回 world,从而复用现有全部 world 系渲染/3D 逻辑(下游零改)。

    hands: {'left'|'right': {transl_cam(T,3), orient6d(T,6), pose6d(T,90), betas(T,10)}}
           (输入值为相机系;键名统一 transl_cam,输出仍用同键但值已转 world)。cam_c2w: (T,4,4) camera→world。
    """
    c2w = np.asarray(cam_c2w, dtype=np.float32)
    Rcw, tcw = c2w[:, :3, :3], c2w[:, :3, 3]                # (T,3,3),(T,3)
    out = {}
    for side in ("left", "right"):
        h = hands[side]
        o_cam = rot6d_to_mat(np.asarray(h["orient6d"], np.float32))          # (T,3,3)
        o_world = np.einsum("tij,tjk->tik", Rcw, o_cam)                      # 腕朝向 -> world
        t_world = np.einsum("tij,tj->ti", Rcw, np.asarray(h["transl_cam"], np.float32)) + tcw
        out[side] = {
            "transl_cam": t_world.astype(np.float32),
            "orient6d": mat_to_6d(o_world).astype(np.float32),
            "pose6d": np.asarray(h["pose6d"], np.float32),
            "betas": np.asarray(h["betas"], np.float32),
        }
    return out


def K_from_fov(fov_rad: tuple, width: int, height: int) -> np.ndarray:
    """fov(弧度, [fov_h, fov_w]) + 像素分辨率 → 像素内参 K（主点取图像中心）。"""
    fov_x, fov_y = float(fov_rad[0]), float(fov_rad[1])
    fx = (width / 2.0) / np.tan(fov_x / 2.0)
    fy = (height / 2.0) / np.tan(fov_y / 2.0)
    return np.array([[fx, 0.0, width / 2.0],
                     [0.0, fy, height / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def decode_camera_pose_enc(pose_enc: np.ndarray, height: int, width: int,
                           fov_mean: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """相机 pose_enc[T,9]=absT3+quaR4+FoV2 → (cam_c2w (T,4,4), K (3,3))。

    借仓库内 vendor 的权威反函数 pose_encoding_to_extri_intri（OpenCV world→cam），
    再求逆得 c2w；GT 与预测共用本函数，分辨率取显示帧 (H,W)。
    fov_mean=False（默认）→ K 取首帧（intr[0]，现状）；True → K 取整段平均（intr.mean(0)，
    把逐帧抖动的 FoV 拉平成整段一个内参）。
    """
    import torch
    ling = REPO_DIR / 'model_train' / '_vendor'
    if str(ling) not in sys.path:
        sys.path.insert(0, str(ling))
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri
    pe = np.asarray(pose_enc, dtype=np.float32).reshape(-1, 9)
    T = pe.shape[0]
    extr, intr = pose_encoding_to_extri_intri(torch.from_numpy(pe)[None], image_size_hw=(height, width))
    extr = extr[0].numpy().astype(np.float64)   # (T,3,4) world->cam (OpenCV)
    intr = intr[0].numpy().astype(np.float64)   # (T,3,3) 像素内参
    extr44 = np.tile(np.eye(4), (T, 1, 1))
    extr44[:, :3, :4] = extr
    cam_c2w = np.linalg.inv(extr44)             # (T,4,4) cam->world
    K = intr.mean(0) if fov_mean else intr[0]   # 平均内参 vs 首帧内参
    return cam_c2w, K
