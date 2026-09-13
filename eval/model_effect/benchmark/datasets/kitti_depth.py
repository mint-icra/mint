# -*- coding: utf-8 -*-
"""KITTI(depth) 适配器 🔒:室外 LiDAR 稀疏深度(depth 头)。骨架。

落盘: data/benchmark/depth/kitti_depth/。深度稀疏,评测须走 depth_mask(仅有效像素)。
capability={depth}。学生开 depth 后接实读。
"""
from __future__ import annotations

from ..core.registry import DATASETS
from ..core.schema import DEPTH
from .base import DatasetAdapter


@DATASETS.register("kitti_depth")
class KittiDepthAdapter(DatasetAdapter):
    name = "kitti_depth"
    root_rel = "depth/kitti_depth"
    capability = {DEPTH}
    implemented = False                 # iter_sequences 骨架未实现

    def iter_sequences(self, max_seqs=None, max_frames=None):
        raise NotImplementedError("KITTI depth(16bit/256 + 稀疏 mask)解析待实现")
        yield  # pragma: no cover
