# -*- coding: utf-8 -*-
"""极简注册表:输出头 / 数据集按名注册与取用。

两个扩展轴(模型只有学生一个,直接类,不进注册表):
  - HEADS:    一个输出头如何评测(extract/align/metrics)   新头 = heads/<name>.py + @HEADS.register
  - DATASETS: 一个数据集如何产出 GTSequence                新集 = datasets/<name>.py + @DATASETS.register

注册靠 import 触发:heads/__init__.py、datasets/__init__.py import 各实现模块,
run.py import 这两个子包即可让注册表填满,引擎按名取用、零改动。
"""
from __future__ import annotations

from typing import Any, Dict


class Registry:
    def __init__(self, name: str):
        self.name = name
        self._table: Dict[str, Any] = {}

    def register(self, key: str):
        def deco(cls):
            self._table.setdefault(key, cls)   # 幂等:重复 import 保留首个,不报错
            return cls
        return deco

    def get(self, key: str):
        if key not in self._table:
            raise KeyError(f"[{self.name}] 未注册: {key!r};已注册: {self.keys()}")
        return self._table[key]

    def keys(self):
        return sorted(self._table)


HEADS = Registry("heads")
DATASETS = Registry("datasets")
