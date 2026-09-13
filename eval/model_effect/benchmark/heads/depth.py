# -*- coding: utf-8 -*-
"""depth 头 🔒:深度精度。中值/仿射对齐后 AbsRel/SqRel/RMSE/RMSElog/δ<1.25^{1,2,3}。

指标与对齐**已就绪**;学生当前不产 depth(capability 无 depth)→ 本头被 run 自动 skip。
学生 student.py 开 enable_depth、predictor 多输出 'depth' 后,capability 含 depth,本头自动翻绿,
框架一行不改。extract 里对 pred.depth 缺失显式抛 NotImplementedError,作双保险。
"""
from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

from .align import least_squares_scale_shift, median_scale
from ..core.registry import HEADS
from ..core.schema import DEPTH, GTSequence, Prediction
from .base import HeadEvaluator


@HEADS.register("depth")
class DepthHead(HeadEvaluator):
    name = "depth"
    required_gt = {DEPTH}

    def extract(self, pred: Prediction) -> Any:
        if pred.depth is None:
            raise NotImplementedError("学生暂不产 depth(未开 enable_depth);capability 无 depth 应已 skip")
        return np.asarray(pred.depth, np.float64)             # [S,H,W]

    def align(self, item: np.ndarray, gt: GTSequence):
        gd = np.asarray(gt.depth, np.float64)
        mask = (np.asarray(gt.depth_mask, bool) if gt.depth_mask is not None
                else np.isfinite(gd) & (gd > 0))
        T = min(len(item), len(gd))
        mode = os.environ.get("BENCH_DEPTH_ALIGN", "median")
        if mode == "affine":
            _, _, pal = least_squares_scale_shift(item[:T], gd[:T], mask[:T])
            s = float("nan")
        else:
            s, pal = median_scale(item[:T], gd[:T], mask[:T])
        return {"pred": pal, "gt": gd[:T], "mask": mask[:T], "s": s}

    def metrics(self, a: Dict, gt: GTSequence) -> Dict[str, float]:
        m = depth_errors(a["pred"], a["gt"], a["mask"])
        m["scale"] = a["s"]
        return m


def depth_errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """AbsRel/SqRel/RMSE/RMSElog/δ<1.25^{1,2,3}(KITTI/NYU 标准,只在有效 mask 内、pred/gt>0)。"""
    keys = ["AbsRel", "SqRel", "RMSE", "RMSElog", "delta1", "delta2", "delta3", "num_px"]
    m = (np.asarray(mask, bool) & np.isfinite(pred) & np.isfinite(gt) & (gt > 0) & (pred > 0))
    if m.sum() == 0:
        return {k: float("nan") for k in keys}
    eps = 1e-6
    p = np.clip(np.asarray(pred, np.float64)[m], eps, None)
    g = np.clip(np.asarray(gt, np.float64)[m], eps, None)
    diff = p - g
    ratio = np.maximum(p / g, g / p)
    return {
        "AbsRel": float(np.mean(np.abs(diff) / g)),
        "SqRel": float(np.mean(diff ** 2 / g)),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "RMSElog": float(np.sqrt(np.mean((np.log(p) - np.log(g)) ** 2))),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25 ** 2)),
        "delta3": float(np.mean(ratio < 1.25 ** 3)),
        "num_px": int(m.sum()),
    }
