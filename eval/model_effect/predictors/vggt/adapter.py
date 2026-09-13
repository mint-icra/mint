#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""vggt 推理适配器（占位）：向 inference.registry 注册 "vggt"。

model_train 已注册 vggt 学生，引擎（StudentEngine）本身可跑；但 model_train/configs/ 暂无 vggt 的
distill config，故无默认 config —— 必须显式传 --config，否则抛 NotImplementedError。
待补上 vggt distill config 后，把 DEFAULT_CONFIG 指过去即可（与 lingbotmap 适配器同形）。
"""
from __future__ import annotations

from inference.engine import StudentEngine
from inference.registry import register_predictor


@register_predictor("vggt", default_config=None)
def build(config=None, ckpt=None, window=None, device=None, devices=None,
          full_max_frames=None, compile_mode=None, fp8_mode=None):
    if config is None:
        raise NotImplementedError("vggt 适配器为占位：暂无默认 distill config，请显式 --config 指定。")
    return StudentEngine(
        config, ckpt=ckpt, device=device, window=window, devices=devices,
        full_max_frames=full_max_frames,
        compile_mode=compile_mode, fp8_mode=fp8_mode,
    )
