#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共用推理的最薄契约层：异常 + 输出 schema 说明。

推理引擎（inference.engine.StudentEngine）与各模型适配器（predictors/<model>）都依赖这里，
可视化（visualization）也从这里取 InferenceCancelled——避免可视化再反向 import 具体引擎。

统一预测契约（引擎 predict 的返回、GT 侧同 schema）：
  {'pose_enc': [N,9]  absT3 + quaR4(xyzw) + FoV2,
   'hand'    : [N,218]=[N,2×109] 双手 MANO 6D（左[0:109]/右[109:218]；每手 transl3+orient6d6+pose6d90+betas10），
   'hand_presence_logits': [N,2] 可选逐帧左右手原始 logit,
   'hand_confidence'     : [N,2] 可选逐帧左右手 sigmoid 概率}
手部参数与存在性输出分别由 enable_hand / enable_hand_presence 控制。
"""
from __future__ import annotations


class InferenceCancelled(Exception):
    """网页端停止活动加载/推理请求时，在数据块、推理窗口或后处理边界抛出。"""


class FullSequenceTooLong(ValueError):
    """exact full 输入超过当前设备的安全帧数上限。"""

    def __init__(self, num_frames: int, max_frames: int):
        self.num_frames = int(num_frames)
        self.max_frames = int(max_frames)
        super().__init__(
            f"视频共 {self.num_frames} 帧，超过 exact full 安全上限 {self.max_frames} 帧；"
            f"请改用最大窗分窗（{self.max_frames} 帧/窗）"
        )
