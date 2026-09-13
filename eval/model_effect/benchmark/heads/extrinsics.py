# -*- coding: utf-8 -*-
"""extrinsics 头 ✅:相机轨迹精度。并行报告 Sim(3) 最佳尺度与 SE(3) 原始尺度 ATE。

对齐:单目轨迹尺度/位姿不可观测,先用相机中心做 7-DoF 相似变换对齐(见 align.umeyama_sim3),
再算尺度拟合 ATE；同时锁定 s=1 做刚性对齐，保留模型自身度量尺度误差。
对齐系数 s 一并报出(反映尺度估得准不准)。同行口径:DUSt3R/VGGT/MonST3R。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .align import umeyama_sim3
from ..core.registry import HEADS
from ..core.schema import EXTRINSIC, GTSequence, Prediction
from .base import HeadEvaluator


def _w2c_to_c2w(w2c: np.ndarray) -> np.ndarray:
    """[S,4,4] world->cam → camera->world。"""
    return np.linalg.inv(w2c)


def _centers(c2w: np.ndarray) -> np.ndarray:
    """相机中心序列 [S,3] = c2w 平移列。"""
    return np.asarray(c2w, np.float64)[:, :3, 3]


def _rot(c2w: np.ndarray) -> np.ndarray:
    return np.asarray(c2w, np.float64)[:, :3, :3]


@HEADS.register("extrinsics")
class ExtrinsicsHead(HeadEvaluator):
    name = "extrinsics"
    required_gt = {EXTRINSIC}

    def extract(self, pred: Prediction) -> Any:
        return np.asarray(pred.extrinsic_c2w, np.float64)     # [S,4,4] c2w

    def align(self, item: np.ndarray, gt: GTSequence):
        gt_c2w = _w2c_to_c2w(np.asarray(gt.extrinsic_w2c, np.float64))
        T = min(len(item), len(gt_c2w))
        pc, gc = _centers(item[:T]), _centers(gt_c2w[:T])
        s, R, t = umeyama_sim3(pc, gc, with_scale=True)
        pc_al = (s * (R @ pc.T)).T + t                        # 对齐后预测中心
        _, R_metric, t_metric = umeyama_sim3(pc, gc, with_scale=False)
        pc_metric = (R_metric @ pc.T).T + t_metric             # 只消除世界原点/朝向，保留预测尺度
        # 尺度退化:预测轨迹塌缩成一点(近静止)→ s≈0/非有限,Sim3 对齐无意义。
        # 此时不报假 ATE=0,标 degenerate,ATE/RPE 记 nan。
        span = float(np.linalg.norm(gc.max(0) - gc.min(0)))   # GT 轨迹范围(米)
        degenerate = (not np.isfinite(s)) or s <= 1e-6 or span < 1e-6
        return {"pc": pc_al, "pc_metric": pc_metric, "gc": gc,
                "pR": R @ _rot(item[:T]), "gR": _rot(gt_c2w[:T]),
                "s": s, "T": T, "degenerate": degenerate}

    def metrics(self, a: Dict, gt: GTSequence) -> Dict[str, float]:
        if a["degenerate"]:
            return {"ATE_RMSE": float("nan"), "ATE_RMSE_metric": float("nan"),
                    "RPE_t": float("nan"),
                    "RPE_rot_deg": float("nan"), "scale": a["s"],
                    "scale_error_pct": float("nan"), "n": a["T"], "degenerate": 1}
        pc, pc_metric, gc, pR, gR = a["pc"], a["pc_metric"], a["gc"], a["pR"], a["gR"]
        # ATE:Sim3 对齐后各帧相机中心欧氏误差 RMSE
        ate = float(np.sqrt(np.mean(np.sum((pc - gc) ** 2, axis=1))))
        ate_metric = float(np.sqrt(np.mean(np.sum((pc_metric - gc) ** 2, axis=1))))
        # RPE:相邻帧相对位移/相对旋转误差(约定无关,是可信的旋转指标)
        dp = np.diff(pc, axis=0) - np.diff(gc, axis=0)
        rpe_t = float(np.sqrt(np.mean(np.sum(dp ** 2, axis=1)))) if len(pc) > 1 else float("nan")
        rel_p = np.matmul(pR[1:], np.transpose(pR[:-1], (0, 2, 1)))
        rel_g = np.matmul(gR[1:], np.transpose(gR[:-1], (0, 2, 1)))
        rpe_rot = float(np.mean(_geodesic_deg(rel_p, rel_g))) if len(pR) > 1 else float("nan")
        # 注:绝对逐帧旋转误差(rot_deg)依赖对齐旋转,而中心点云估的 R 不代表朝向偏移,
        # 实测虚高且不可靠(见 README §9),故只报约定无关的相对旋转 RPE_rot_deg。
        return {"ATE_RMSE": ate, "ATE_RMSE_metric": ate_metric,
                "RPE_t": rpe_t, "RPE_rot_deg": rpe_rot,
                "scale": a["s"], "scale_error_pct": abs(float(a["s"]) - 1.0) * 100.0,
                "n": a["T"]}


def _geodesic_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """逐帧旋转测地线角(度): acos((tr(Ra Rb^T)-1)/2)。"""
    rel = np.matmul(Ra, np.transpose(Rb, (0, 2, 1)))
    tr = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    cos = np.clip((tr - 1.0) / 2.0, -1.0 + 1e-7, 1.0 - 1e-7)
    return np.degrees(np.arccos(cos))
