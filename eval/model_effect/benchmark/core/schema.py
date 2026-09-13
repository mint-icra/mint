# -*- coding: utf-8 -*-
"""统一数据契约:把"学生的预测"与"任意数据集的真值"收敛成两个固定容器,
使 head 评测器与模型、数据集完全解耦。

约定(全模块统一):
  - 序列长 S,图像高宽 H/W。
  - 外参两种约定务必区分:
        Prediction.extrinsic_c2w —— camera-to-world(相机在世界系),pose_enc 解码即此。
        GTSequence.extrinsic_w2c —— world-to-camera(投影用),多数数据集 GT 形式。
    互转走 numpy 求逆(见 heads),别双重求逆。
  - 内参 intrinsic 为像素单位 3x3,与所在分辨率绑定;跨分辨率比较前必须归一(intrinsics 头)。
  - 深度 depth 公制或相对,单目预测含未知 scale(+shift),评测前对齐(见 align)。

Modality 常量用于 capability 集合 + head 的 required_gt,做"数据集/模型能力 vs 头需求"匹配。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

# ---- 模态常量(capability / required_gt 的元素) ----
INTRINSIC = "intrinsic"
EXTRINSIC = "extrinsic"
DEPTH = "depth"
POINTS = "points"
HAND = "hand"
HAND_WORLD = "hand_world"                            # world 系手部真值(GT 提供 world 关节 + w2c),供 hands_world 头
HAND_COVERAGE = "hand_coverage"                      # coverage-aware 相机系双手指标
ALL_MODALITIES = (INTRINSIC, EXTRINSIC, DEPTH, POINTS, HAND, HAND_WORLD, HAND_COVERAGE)


@dataclass(frozen=True)
class VideoFrameRef:
    """A frame inside a video, decoded on demand without a JPEG staging cache."""

    video_path: str
    frame_index: int

    def __str__(self) -> str:
        return f"{self.video_path}#frame={int(self.frame_index)}"


@dataclass
class Prediction:
    """学生在一段序列上的统一预测。数组均 numpy(float);None=该模型不产出。"""

    pose_enc: np.ndarray                             # [S,9] absT3+quaR4+FoV2(学生原生)
    extrinsic_c2w: np.ndarray                        # [S,4,4] camera-to-world(pose_enc 解码)
    intrinsic: np.ndarray                            # [S,3,3] 像素单位(pose_enc 解码;主点=中心)
    hw: tuple                                        # (H,W) 内参所在分辨率(解码时的显示帧尺寸)
    depth: Optional[np.ndarray] = None               # [S,H,W]  🔒 学生暂不产
    world_points: Optional[np.ndarray] = None        # [S,H,W,3] 🔒
    hand: Optional[np.ndarray] = None                # [S,218]=[S,2×109] 双手 MANO 6D(左[0:109]/右[109:218];每手 transl3+orient6d6+pose6d90+betas10)
    hand_presence_logits: Optional[np.ndarray] = None # [S,2] 左右手逐帧存在性原始 logit
    hand_confidence: Optional[np.ndarray] = None     # [S,2] sigmoid(presence logit)，可选缓存
    capability: Set[str] = field(default_factory=set)
    meta: Dict = field(default_factory=dict)         # ckpt/窗口等溯源

    @property
    def num_frames(self) -> int:
        return int(self.pose_enc.shape[0])


@dataclass
class GTSequence:
    """某数据集在一段序列上的统一真值。各模态可为 None;capability 标明实际提供哪些。"""

    seq_id: str                                      # 场景/序列名,进报告
    image_paths: List[str | VideoFrameRef]
    hw: tuple                                        # (H,W) GT 图像原分辨率
    intrinsic: Optional[np.ndarray] = None           # [S,3,3] 或 [3,3](恒定)
    extrinsic_w2c: Optional[np.ndarray] = None       # [S,4,4] world-to-camera
    depth: Optional[np.ndarray] = None               # [S,H,W] 公制
    depth_mask: Optional[np.ndarray] = None          # [S,H,W] bool,True=有效像素
    hand_joints_3d: Optional[np.ndarray] = None      # [S,21,3] 相机系公制米(OpenPose 21 序,0=wrist)
    hand_joints_3d_world: Optional[np.ndarray] = None # [S,21,3] world 系公制米(HAND_WORLD 用;hands_world 头把 pred 转 world 后与此对齐)
    hand_verts: Optional[np.ndarray] = None          # [S,778,3] 相机系 MANO 顶点(可选,供 PA-MPVPE;无则跳过)
    hand_valid: Optional[np.ndarray] = None          # [S] bool,该帧手在画面(joint 非 NaN)
    hand_mano_6d: Optional[np.ndarray] = None        # [S,2,109] 相机系左右手 MANO，覆盖率指标用
    hand_valid_lr: Optional[np.ndarray] = None       # [S,2] 左右手 GT 标注有效性；屏内性由协议另行投影判断
    capability: Set[str] = field(default_factory=set)
    meta: Dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return len(self.image_paths)

    def has(self, modalities: Set[str]) -> bool:
        """该数据集是否提供某头评测所需的全部模态。"""
        return set(modalities).issubset(self.capability)
