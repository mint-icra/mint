# -*- coding: utf-8 -*-
"""NYUv2 适配器 🔒:室内稠密深度(depth 头)。骨架——学生开 depth 后接实读。

落盘: data/benchmark/depth/nyuv2/(parquet 文件式)。capability={depth}。
当前 iter_sequences 抛 NotImplementedError,run 记 "not_implemented";因学生也无 depth,
即便实现该头也会先被 capability skip,故留骨架、翻绿时再接盘。
"""
from __future__ import annotations

from ..core.registry import DATASETS
from ..core.schema import DEPTH
from .base import DatasetAdapter


@DATASETS.register("nyuv2")
class NYUv2Adapter(DatasetAdapter):
    name = "nyuv2"
    root_rel = "depth/nyuv2"
    capability = {DEPTH}
    implemented = False                 # iter_sequences 骨架未实现

    def iter_sequences(self, max_seqs=None, max_frames=None):
        raise NotImplementedError("NYUv2 parquet 解析待实现(depth 头翻绿时接盘)")
        yield  # pragma: no cover  (使其为生成器)
