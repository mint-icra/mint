# -*- coding: utf-8 -*-
"""eval/model_effect/benchmark —— 学生模型定量 benchmark 评测(自包含)。

不引用 unit_test/benchmark 的任何接口;推理/几何解码复用本目录 predictor(HandReprojPredictor
与 reproj_core.geometry,即评蒸馏学生,非原始 lingbot-map 教师)。

公共门面(外部按 `from benchmark import ...` 取用,不必深入子模块):
  run_benchmark   逐 dataset×head 评测编排(CLI run.py 与 viewer store.py 共用)
  capabilities    能力清单(heads/datasets 需求与提供、是否实现 + 模型产出),供选择面板三态联动
  dataset_sizes   每数据集序列条数/总帧数(不加载模型),供面板跑前显示规模
  StudentPredictor 学生推理封装(frames → Prediction)
  HEADS / DATASETS 注册表
  Report / SeqResult 报告聚合
"""
from .core.engine import capabilities, dataset_sizes, run_benchmark
from .predictor import StudentPredictor
from .core.registry import DATASETS, HEADS
from .report import Report, SeqResult

__all__ = ["run_benchmark", "capabilities", "dataset_sizes", "StudentPredictor",
           "HEADS", "DATASETS", "Report", "SeqResult"]
