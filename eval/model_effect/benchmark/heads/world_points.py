# -*- coding: utf-8 -*-
"""world_points 头 🔒:世界点云精度。Umeyama-Sim3 对齐后 Chamfer-L1 / acc·comp@τ / F1@τ。

依赖 depth(+extrinsic+intrinsic)反投影成世界点云——学生开 enable_depth 前不可评,
capability 缺 depth → 本头被 run 自动 skip。指标骨架留好,翻绿只需模型侧出 depth + 补 metrics 数学。
同行口径:DUSt3R / VGGT(ETH3D F1 金标准)。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.registry import HEADS
from ..core.schema import DEPTH, EXTRINSIC, INTRINSIC, GTSequence, Prediction
from .base import HeadEvaluator


@HEADS.register("world_points")
class WorldPointsHead(HeadEvaluator):
    name = "world_points"
    required_gt = {DEPTH, EXTRINSIC, INTRINSIC}
    implemented = False                 # 反投影 + Chamfer/F1 指标待实现

    def extract(self, pred: Prediction) -> Any:
        if pred.depth is None:
            raise NotImplementedError("需 depth 反投影成点云;学生暂不产 depth")
        # TODO(翻绿): depth+K+c2w 逐像素反投影 → 世界点云 [M,3]
        raise NotImplementedError("world_points 反投影与 Chamfer/F1 指标待实现(模型出 depth 后补)")

    def align(self, item: Any, gt: GTSequence):
        raise NotImplementedError

    def metrics(self, aligned: Any, gt: GTSequence) -> Dict[str, float]:
        raise NotImplementedError
