#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""共用推理包：模型无关的学生推理引擎 + 契约 + 注册表。

- base       统一异常 InferenceCancelled + 预测契约说明（可视化/benchmark 只认此契约）
- engine     StudentEngine —— 走 model_train.build_model 的通用学生引擎
- registry   模型名 → 引擎工厂 的注册表（get_predictor / default_config / ensure_adapters）
"""
