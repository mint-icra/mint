# -*- coding: utf-8 -*-
"""StudentPredictor:把训练好的学生 ckpt 在一段图像序列上的输出统一成 Prediction。

复用**同在 model_effect/ 下**(非 unit_test)的共用件:
  - inference/engine.py :: StudentEngine(别名 HandReprojPredictor) —— ckpt→build_model→分窗 forward,出 pose_enc[S,9]
  - visualization/reproj_core/geometry.py :: decode_camera_pose_enc —— pose_enc[S,9]→(cam_c2w[S,4,4], K[3,3])

即评的是蒸馏**学生**(model_train/student.py,只出 {pose_enc,hand}),而非原始 lingbot-map 教师。
故 capability 恒含 {extrinsic, intrinsic};depth/points 待学生开 enable_depth 后由 forward 多出 'depth'
时再并入(见 _capability_from_outputs)。
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .core.schema import DEPTH, EXTRINSIC, HAND, HAND_COVERAGE, INTRINSIC, Prediction

# inference/ 与 visualization/ 与本包同在 eval/model_effect/ 下;加进 path 以复用共用引擎与 reproj_core。
_MODEL_EFFECT = Path(__file__).resolve().parents[1]        # benchmark/ -> model_effect/
if str(_MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(_MODEL_EFFECT))


class StudentPredictor:
    """学生推理封装:frames → pose_enc → (cam_c2w, K) → Prediction。"""

    def __init__(self, config_path: str, ckpt: str | None = None,
                 device: str | None = None, window: int | None = None,
                 single_forward: bool = True, window_batch_size: int = 1,
                 image_workers: int = 1, hand_mode: str = "hard",
                 compile_mode: str | None = None, fp8_mode: str | None = None):
        from inference.engine import HandReprojPredictor
        self._engine = HandReprojPredictor(
            config_path, ckpt=ckpt, device=device, window=window,
            compile_mode=compile_mode, fp8_mode=fp8_mode,
        )
        self.ckpt = ckpt
        # single_forward=True:整段一次前向(window=N),全帧共用同一坐标原点 → 全局轨迹连续,
        #   是轨迹评测的正确对齐方式(模型 forward 支持任意 S,非只 8 帧)。
        # False:退回按训练 clip_len 分窗独立前向(会在窗间断裂,仅用于对照/超长序列省显存)。
        self.single_forward = single_forward
        self.window = self._engine.window
        self.window_batch_size = max(1, int(window_batch_size))
        self.image_workers = max(1, int(image_workers))
        self.hand_mode = str(hand_mode)
        self._dataset_name = ""

    @classmethod
    def from_engine(cls, engine, *, ckpt: str | None = None,
                    single_forward: bool = True, window_batch_size: int = 1,
                    image_workers: int = 1,
                    hand_mode: str = "hard") -> "StudentPredictor":
        """复用一个**已加载**的 StudentEngine(如 viewer 常驻引擎),不再重复加载 4.6G 权重。
        window 由 predict 按 single_forward 临时设置并复原(见 predict),不污染引擎供交互推理的原 window。"""
        self = cls.__new__(cls)
        self._engine = engine
        self.ckpt = ckpt
        self.single_forward = single_forward
        self.window = engine.window
        self.window_batch_size = max(1, int(window_batch_size))
        self.image_workers = max(1, int(image_workers))
        self.hand_mode = str(hand_mode)
        self._dataset_name = ""
        return self

    def set_benchmark_dataset(self, dataset_name: str) -> None:
        """Select per-protocol inference mode without rebuilding the model."""
        self._dataset_name = str(dataset_name or "")

    @property
    def effective_single_forward(self) -> bool:
        """Select the protocol-safe long/short sequence inference behavior."""
        if self._dataset_name in {"camera_hot3d", "camera_arctic"}:
            # The ICRA export contains 624-4439 frame sequences.  Its canonical
            # model protocol is training-window chunking + SE(3) chaining; a
            # single forward is both a different protocol and normally OOM.
            return False
        return bool(self.single_forward or self._dataset_name.endswith("_hand_coverage"))

    def predict(self, image_paths, hw: tuple | None = None, on_step=None) -> Prediction:
        """image_paths: 帧路径列表。hw:(H,W) 用于内参解码所在分辨率;默认按首帧原分辨率。

        内参绑分辨率:解码用 GT 原分辨率 hw,使 pred K 与 GT K 同像素单位、便于比较(intrinsics 头
        还会再做分辨率归一,故此处 hw 传 GT 分辨率最稳)。
        on_step(done, total):可选,透传给引擎分窗前向,逐窗回调(done=当前窗号, total=总窗数),
        供上层展示单条序列内的窗口级推理进度。
        """
        from visualization.reproj_core.geometry import decode_camera_pose_enc

        load_started = time.perf_counter()
        frames, input_hw = _load_preprocessed_rgb(
            image_paths, self._engine.size_hw, workers=self.image_workers,
        )
        image_load_s = time.perf_counter() - load_started
        H0, W0 = input_hw
        H, W = hw if hw is not None else (H0, W0)

        use_single_forward = self.effective_single_forward
        prev_window = self._engine.window
        if use_single_forward:                            # 整段一次前向:全帧同一坐标原点(轨迹连续)
            self._engine.window = int(frames.shape[0])
        try:
            out = self._engine.predict(
                frames,
                on_step=on_step,
                window_batch_size=self.window_batch_size,
                hand_mode=self.hand_mode,
                preprocessed=True,
            )
        finally:
            self._engine.window = prev_window              # 复原:复用外部引擎时不污染其 window(交互推理仍用原值)
        pose_enc = np.asarray(out["pose_enc"], np.float32)  # [N,9]
        cam_c2w, K = decode_camera_pose_enc(pose_enc, H, W)  # [N,4,4], [3,3]
        intr = np.tile(np.asarray(K, np.float64), (pose_enc.shape[0], 1, 1))  # [N,3,3]

        cap = {EXTRINSIC, INTRINSIC}
        depth = None
        if "depth" in out:                                 # 学生开 enable_depth 后自动并入,框架不改
            depth = np.asarray(out["depth"], np.float32)
            cap.add(DEPTH)
        hand = None
        if "hand" in out:                                  # 学生开 enable_hand 后 forward 出 [N,218] 双手 MANO 6D
            hand = np.asarray(out["hand"], np.float32)
            cap.add(HAND)
        presence_logits = None
        hand_confidence = None
        if "hand_presence_logits" in out:
            presence_logits = np.asarray(out["hand_presence_logits"], np.float32)
        if "hand_confidence" in out:
            hand_confidence = np.asarray(out["hand_confidence"], np.float32)
        if hand is not None and (presence_logits is not None or hand_confidence is not None):
            cap.add(HAND_COVERAGE)

        engine_timings = dict(out.get("_timings") or {})
        return Prediction(
            pose_enc=pose_enc, extrinsic_c2w=np.asarray(cam_c2w, np.float64),
            intrinsic=intr, hw=(H, W), depth=depth, hand=hand,
            hand_presence_logits=presence_logits, hand_confidence=hand_confidence,
            capability=cap,
            meta={
                "ckpt": self.ckpt,
                "window": self.window,
                "window_batch_size": self.window_batch_size,
                "image_workers": self.image_workers,
                "hand_mode": self.hand_mode,
                "n_input_hw": (H0, W0),
                "hand_frame": "camera",
                "single_forward": use_single_forward,
                "timings": {"image_load_s": image_load_s, **engine_timings},
            },
        )


def _load_rgb_uint8(image_paths, workers: int = 1) -> np.ndarray:
    """按原分辨率加载 RGB → [N,H,W,3] uint8(统一到首帧尺寸,防混分辨率)。"""
    import cv2

    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("image_paths must contain at least one frame")

    def _read(path):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"读图失败: {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if workers <= 1:
        imgs = [_read(path) for path in image_paths]
    else:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(image_paths))) as executor:
            imgs = list(executor.map(_read, image_paths))

    H0, W0 = imgs[0].shape[:2]
    for index, rgb in enumerate(imgs):
        if rgb.shape[:2] != (H0, W0):
            imgs[index] = cv2.resize(rgb, (W0, H0), interpolation=cv2.INTER_AREA)
    return np.stack(imgs, axis=0).astype(np.uint8)


def _load_preprocessed_rgb(image_paths, size_hw, workers: int = 1):
    """Decode and resize bounded chunks, avoiding a full-resolution float tensor."""
    import cv2
    import torch
    import torch.nn.functional as F

    image_paths = list(image_paths)
    if not image_paths:
        raise ValueError("image_paths must contain at least one frame")

    from .frame_io import read_rgb_frames

    H0 = W0 = None

    def _normalize(rgb):
        if rgb.shape[:2] != (H0, W0):
            return cv2.resize(rgb, (W0, H0), interpolation=cv2.INTER_AREA)
        return rgb

    # F.interpolate is batch-independent, so bounded chunks are numerically
    # equivalent to preprocessing the entire sequence in one allocation.
    chunk_size = max(1, min(32, int(workers) * 2))
    resized = []
    executor = (ThreadPoolExecutor(max_workers=min(int(workers), len(image_paths)))
                if workers > 1 else None)
    try:
        for start in range(0, len(image_paths), chunk_size):
            stop = min(len(image_paths), start + chunk_size)
            images = read_rgb_frames(image_paths[start:stop], executor=executor)
            if H0 is None:
                H0, W0 = images[0].shape[:2]
            batch = np.stack([_normalize(image) for image in images], axis=0)
            tensor = torch.from_numpy(batch).permute(0, 3, 1, 2).float().div_(255.0)
            resized.append(F.interpolate(
                tensor, size=tuple(size_hw), mode="bilinear", align_corners=False,
            ))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return torch.cat(resized, dim=0).contiguous(), (H0, W0)
