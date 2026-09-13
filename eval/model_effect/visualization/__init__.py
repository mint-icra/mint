#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模型无关的共用可视化包：GT/几何底座（reproj_core）+ 绘制/逐帧 loss（render）+ 网页上位机（viewer）+ 两个入口。

任何模型只要经 inference 引擎产出统一契约（见 inference.base），即可复用本包的离线 mp4 与网页可视化。
"""
