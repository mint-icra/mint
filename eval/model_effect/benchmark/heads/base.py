# -*- coding: utf-8 -*-
"""HeadEvaluator 基类:一个输出头的评测流程(模板方法),与模型/数据集解耦。

子类声明 name/required_gt,实现三步:
  extract(pred)        从统一 Prediction 抽本头预测
  align(item, gt)      单目尺度对齐(用 align.py),返回对齐后量(+scale 诊断)
  metrics(aligned,gt)  纯指标 -> {名: 标量}
evaluate() 是模板:串起三步,把 NotImplementedError 收敛成 not_implemented 状态,
使骨架阶段能跑通全链路、在报告里看到"哪步还没实现"。头之间互不 import。
"""
from __future__ import annotations

from typing import Any, Dict, Set

from ..core.schema import GTSequence, Prediction


class HeadEvaluator:
    name: str = "base"
    required_gt: Set[str] = set()
    implemented: bool = True            # 骨架头(extract/metrics 未实现)置 False,供能力清单标「待实现」

    def extract(self, pred: Prediction) -> Any:
        raise NotImplementedError

    def align(self, item: Any, gt: GTSequence) -> Any:
        raise NotImplementedError

    def metrics(self, aligned: Any, gt: GTSequence) -> Dict[str, float]:
        raise NotImplementedError

    def evaluate(self, pred: Prediction, gt: GTSequence) -> tuple:
        """返回 (status, metrics, note)。status: evaluated / not_implemented / error。"""
        try:
            item = self.extract(pred)
            aligned = self.align(item, gt)
            return "evaluated", self.metrics(aligned, gt), ""
        except NotImplementedError as e:
            return "not_implemented", {}, str(e) or "TODO"
        except Exception as e:                       # 单序列失败不拖垮整轮
            return "error", {}, f"{type(e).__name__}: {e}"
