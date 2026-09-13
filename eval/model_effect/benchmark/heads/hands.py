# -*- coding: utf-8 -*-
"""hands 头 ✅:手部 3D 关节精度(root-relative MPJPE / Procrustes PA-MPJPE,mm)。

口径对齐 HaWoR(lib/eval_utils/eval_utils.py::compute_errors):
  MPJPE    = pred/gt 各自减腕关节(joint 0)root-relative 后,逐关节 L2 → 帧均。
  PA-MPJPE = 7-DoF 相似变换(Umeyama,复用 align.umeyama_sim3)对齐 pred→gt 后 L2 → 帧均。
均 ×1000 → mm;仅 valid 帧(手在画面、joint 非 NaN)。

extract:pred.hand[S,218] 取右手 [109:218](DexYCB 单右手),经 visualization.reproj_core MANO 前向
**相机系直接解算(不转 world)** → OpenPose 21 关节 [S,21,3],与 DexYCB joint_3d 同序(0=wrist)同系。
⚠ 需 enable_hand 的学生 ckpt(forward 出 'hand');模型手坐标系须为 camera(见 predictor meta.hand_frame)。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .align import umeyama_sim3
from ..core.registry import HEADS
from ..core.schema import HAND, GTSequence, Prediction
from .base import HeadEvaluator

# visualization/ 与 benchmark/ 同在 model_effect/ 下,加进 path 以复用其 MANO 解码(与 predictor.py 一致)。
_MODEL_EFFECT = Path(__file__).resolve().parents[2]        # heads/ -> benchmark/ -> model_effect/
if str(_MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(_MODEL_EFFECT))

_PER_HAND = 109
_SIDE = {"left": slice(0, _PER_HAND), "right": slice(_PER_HAND, 2 * _PER_HAND)}   # 左[0:109]/右[109:218]
# 每手 109 切片:transl3 + orient6d6 + pose6d90 + betas10(与 render/compare.py::_HAND_SLICES 一致)
_SL = {"transl": (0, 3), "orient6d": (3, 9), "pose6d": (9, 99), "betas": (99, 109)}
_CACHE_KEY = "_benchmark_hand_joints_camera"
_GEOMETRY_CACHE_KEY = "_benchmark_hand_geometry_camera"


def camera_hand_geometry(pred: Prediction) -> dict[str, dict[str, np.ndarray]]:
    """Decode camera-space MANO mesh/joints and retain raw wrist pose once."""
    cached = pred.meta.get(_GEOMETRY_CACHE_KEY)
    if cached is not None:
        return cached
    if pred.hand is None:
        raise NotImplementedError("模型未产出 hand(需 enable_hand 的学生 ckpt)")
    from visualization.reproj_core import geometry, mano
    out = {}
    for side, sl in _SIDE.items():
        seg = np.asarray(pred.hand, np.float32)[:, sl]
        params = {key: seg[:, lo:hi] for key, (lo, hi) in _SL.items()}
        decoded = mano.decode_hand_6d(
            params["transl"], params["orient6d"], params["pose6d"], params["betas"],
            is_right=(side == "right"),
        )
        verts, joints = mano.run_mano(
            decoded["trans"], decoded["rot"], decoded["hand_pose"], decoded["betas"],
            is_right=(side == "right"),
        )
        out[side] = {
            "verts": np.asarray(verts, np.float64),
            "joints": np.asarray(joints[:, :21], np.float64),
            "translation": np.asarray(params["transl"], np.float64),
            "orientation": np.asarray(
                geometry.rot6d_to_mat(params["orient6d"].reshape(-1, 6)),
                np.float64,
            ),
        }
    pred.meta[_GEOMETRY_CACHE_KEY] = out
    return out


def camera_hand_joints(pred: Prediction) -> dict[str, np.ndarray]:
    """Decode both camera-space hands once and reuse them across heads/GT sides."""
    cached = pred.meta.get(_CACHE_KEY)
    if cached is not None:
        return cached
    geometry = camera_hand_geometry(pred)
    out = {side: values["joints"] for side, values in geometry.items()}
    pred.meta[_CACHE_KEY] = out
    return out


@HEADS.register("hands")
class HandsHead(HeadEvaluator):
    name = "hands"
    required_gt = {HAND}

    def extract(self, pred: Prediction) -> Any:
        """解码左右手相机系 21 关节;align 按 gt.meta.mano_side 取对应手(DexYCB 无该字段=右手)。"""
        return camera_hand_joints(pred)

    def align(self, item: Any, gt: GTSequence):
        side = gt.meta.get("mano_side", "right")                     # HOT3D 逐手;DexYCB 默认右手
        pj = np.asarray(item[side], np.float64)                      # [Sp,21,3]
        gj = np.asarray(gt.hand_joints_3d, np.float64)               # [Sg,21,3]
        T = min(pj.shape[0], gj.shape[0])
        pj, gj = pj[:T], gj[:T]
        valid = (np.ones(T, bool) if gt.hand_valid is None
                 else np.asarray(gt.hand_valid, bool)[:T])
        valid = valid & np.isfinite(pj).all((1, 2)) & np.isfinite(gj).all((1, 2))
        return {"pred": pj, "gt": gj, "valid": valid}

    def metrics(self, a: Dict, gt: GTSequence) -> Dict[str, float]:
        mpjpe, pa = [], []
        for p, g, v in zip(a["pred"], a["gt"], a["valid"]):
            if not v:
                continue
            pr, gr = p - p[0:1], g - g[0:1]                          # root-relative(减腕 joint0)
            mpjpe.append(float(np.linalg.norm(pr - gr, axis=1).mean()))
            s, R, t = umeyama_sim3(p, g, with_scale=True)            # Procrustes 对齐 pred→gt
            p_pa = (s * (R @ p.T)).T + t
            pa.append(float(np.linalg.norm(p_pa - g, axis=1).mean()))
        if not mpjpe:
            raise NotImplementedError("无 valid 帧(手均不在画面 / NaN)")
        return {"MPJPE": float(np.mean(mpjpe) * 1000.0),             # mm
                "PA_MPJPE": float(np.mean(pa) * 1000.0),
                "n_frames": float(len(mpjpe))}
