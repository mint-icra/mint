# -*- coding: utf-8 -*-
"""import 各数据集以触发注册。加新集:建 datasets/<name>.py + @DATASETS.register + 在此 import。"""
from . import arctic, dexycb, eth3d, hot3d, kitti_depth, nyuv2, sintel, tum  # noqa: F401
from ..camera_trajectory import datasets as camera_trajectory  # noqa: F401
