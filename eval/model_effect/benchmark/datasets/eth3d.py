# -*- coding: utf-8 -*-
"""ETH3D 适配器 🔒:激光金标准点云(world_points 头,F1 金标准)。骨架。

落盘: data/benchmark/world_points/eth3d/(hf PeterDAI/eth3d 为已解压目录式,含
dslr_scan_eval/ground_truth_depth)。capability={depth, extrinsic, intrinsic}。
学生开 depth 后接实读 + 补 world_points 头指标。
"""
from __future__ import annotations

from ..core.registry import DATASETS
from ..core.schema import DEPTH, EXTRINSIC, INTRINSIC
from .base import DatasetAdapter


@DATASETS.register("eth3d")
class ETH3DAdapter(DatasetAdapter):
    name = "eth3d"
    root_rel = "world_points/eth3d"
    capability = {DEPTH, EXTRINSIC, INTRINSIC}
    implemented = False                 # iter_sequences 骨架未实现

    def iter_sequences(self, max_seqs=None, max_frames=None):
        raise NotImplementedError("ETH3D(COLMAP + GT 深度/点云)解析待实现")
        yield  # pragma: no cover
