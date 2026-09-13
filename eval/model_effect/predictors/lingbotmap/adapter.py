#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lingbotmap 推理适配器：注册默认使用 frozen-backbone 配置的 "lingbotmap"。

引擎本体（StudentEngine）模型无关，本适配器只提供「默认 config + 注册名」这层薄封装。
"""
from __future__ import annotations

from pathlib import Path

from inference.engine import StudentEngine
from inference.registry import register_predictor

# lingbotmap -> predictors -> model_effect -> eval -> <repo>
REPO_DIR = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_DIR / "configs" / "training" / "mint_step2.yaml"
)


@register_predictor("lingbotmap", default_config=DEFAULT_CONFIG)
def build(config=None, ckpt=None, window=None, device=None, devices=None,
          full_max_frames=None, compile_mode=None, fp8_mode=None):
    return StudentEngine(
        config or str(DEFAULT_CONFIG), ckpt=ckpt, device=device, window=window,
        devices=devices, full_max_frames=full_max_frames,
        compile_mode=compile_mode, fp8_mode=fp8_mode,
    )
