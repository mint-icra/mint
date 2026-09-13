#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型 → 推理引擎 的注册表。

各模型适配器（predictors/<model>/adapter.py）用 @register_predictor 注册一个工厂 + 默认 config；
入口/网页经 get_predictor(model, ...) 拿到构造好的引擎实例，default_config(model) 拿其默认 config。

改名 predictors（原 models）后与 model_train/models 不再撞名，ensure_adapters 直接 import 即可，
无需任何 sys.modules 清理。
"""
from __future__ import annotations

import importlib

# name -> factory(config, ckpt, window, device, devices) -> 引擎实例
PREDICTORS: dict = {}
# name -> 默认 config 路径（str/Path），无默认则为 None
DEFAULT_CONFIGS: dict = {}

# 各适配器模块路径；import 即触发其 @register_predictor
_ADAPTERS = ("predictors.lingbotmap.adapter", "predictors.vggt.adapter")


def register_predictor(name: str, default_config=None):
    """装饰器：把模型工厂登记进 PREDICTORS，并记录其默认 config。"""
    def _wrap(factory):
        PREDICTORS[name] = factory
        DEFAULT_CONFIGS[name] = None if default_config is None else str(default_config)
        return factory
    return _wrap


def ensure_adapters() -> None:
    """惰性 import 全部适配器模块，触发注册（幂等：已注册则 import 命中缓存）。"""
    for m in _ADAPTERS:
        importlib.import_module(m)


def _known() -> str:
    return ", ".join(sorted(PREDICTORS)) or "(空)"


def default_config(model: str):
    """返回某模型的默认 config 路径（可能为 None，如 vggt 占位）。"""
    ensure_adapters()
    if model not in DEFAULT_CONFIGS:
        raise KeyError(f"未知模型 '{model}'，已注册：{_known()}")
    return DEFAULT_CONFIGS[model]


def get_predictor(model: str, config=None, ckpt=None, window=None, device=None, devices=None,
                  full_max_frames=None, compile_mode=None, fp8_mode=None):
    """按模型名取工厂并构造引擎实例。config=None 时由适配器补其默认 config。"""
    ensure_adapters()
    if model not in PREDICTORS:
        raise KeyError(f"未知模型 '{model}'，已注册：{_known()}")
    return PREDICTORS[model](
        config=config, ckpt=ckpt, window=window, device=device, devices=devices,
        full_max_frames=full_max_frames,
        compile_mode=compile_mode, fp8_mode=fp8_mode,
    )
