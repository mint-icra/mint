#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多模型推理适配器包：每个 predictors/<model>/adapter.py 向 inference.registry 注册一个模型。

import 本包即触发各适配器注册（供 inference.registry.ensure_adapters 使用）。
改名自 models，避免与训练框架顶级包 model_train/models 在同进程撞名。
"""
from . import lingbotmap  # noqa: F401  触发 lingbotmap 适配器注册
from . import vggt        # noqa: F401  触发 vggt 适配器注册
