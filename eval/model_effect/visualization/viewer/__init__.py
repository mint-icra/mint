#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""网页端 lingbotmap 效果验证的后端子包（从 viewer_web.py 拆出）。

- const   : 路径 + 枚举常量（MODES/LAYOUTS/CONTENTS/VIDEO_EXTS、REPO_DIR、MODEL_TRAIN_ROOT）。
- ckpts   : model_train 下 ckpt 发现/浏览/tag/config 快照解析。
- netutil : 端口自愈 + 自动开浏览器。
- store   : Store（缓存/推理/GT/mp4/取消/进度/模型&视频懒加载）。
- routes  : create_app(store) —— Flask app 工厂 + 全部 /api 路由 + /video + 静态资源(web/)。

入口仍是 lingbotmap/viewer_web.py（瘦入口：argparse → 建 Store → create_app → run）。
"""
