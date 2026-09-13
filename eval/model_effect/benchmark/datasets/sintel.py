# -*- coding: utf-8 -*-
"""Sintel 适配器 ✅:内参 + 外参(camdata_left/*.cam 同时含 M(3x3 内参) 与 N(3x4 world->cam))。

落盘: data/benchmark/intrinsics/sintel/raw/Sintel/Sintel/training/{camdata_left,clean}/<场景>/frame_*.cam|png
故 capability = {intrinsic, extrinsic},P0 阶段可同时供 intrinsics 与 extrinsics 两头。
.cam 二进制格式(Sintel SDK cam_read):float32 check(202021.25) + float64 3x3 M + float64 3x4 N。
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.registry import DATASETS
from ..core.schema import EXTRINSIC, INTRINSIC, GTSequence
from .base import DatasetAdapter

_TAG_FLOAT = 202021.25


def _cam_read(path: str):
    """返回 (M[3,3] 内参, N[3,4] world->cam)。"""
    with open(path, "rb") as f:
        check = np.fromfile(f, dtype=np.float32, count=1)[0]
        if abs(float(check) - _TAG_FLOAT) > 1e-2:
            raise ValueError(f"{path} 非法 Sintel .cam(check={check})")
        M = np.fromfile(f, dtype=np.float64, count=9).reshape(3, 3)
        N = np.fromfile(f, dtype=np.float64, count=12).reshape(3, 4)
    return M, N


def _find_training(root: str) -> str | None:
    """定位 .../training 目录(解压层级可能嵌套),要求其下有 camdata_left。"""
    hits = glob.glob(os.path.join(root, "**", "training", "camdata_left"), recursive=True)
    return os.path.dirname(hits[0]) if hits else None


@DATASETS.register("sintel")
class SintelAdapter(DatasetAdapter):
    name = "sintel"
    root_rel = "intrinsics/sintel"
    capability = {INTRINSIC, EXTRINSIC}

    def _scene_inputs(self, scene: str, max_frames=None) -> list[str]:
        training = _find_training(self.root)
        if training is None:
            raise FileNotFoundError(
                f"未找到 Sintel training/camdata_left(在 {self.root} 下);先下载解压")
        cam_root = os.path.join(training, "camdata_left", scene)
        img_root = os.path.join(training, "clean", scene)
        cams = sorted(glob.glob(os.path.join(cam_root, "frame_*.cam")))
        if max_frames:
            cams = cams[:max_frames]
        return [
            os.path.join(img_root, os.path.splitext(os.path.basename(cam))[0] + ".png")
            for cam in cams
            if os.path.exists(os.path.join(
                img_root, os.path.splitext(os.path.basename(cam))[0] + ".png"))
        ]

    def list_visual_sequences(self) -> list[dict]:
        training = _find_training(self.root)
        if training is None:
            raise FileNotFoundError(
                f"未找到 Sintel training/camdata_left(在 {self.root} 下);先下载解压")
        cam_root = os.path.join(training, "camdata_left")
        scenes = sorted(d for d in os.listdir(cam_root)
                        if os.path.isdir(os.path.join(cam_root, d)))
        return [{"seq_id": scene, "label": scene,
                 "frame_count": len(self._scene_inputs(scene))}
                for scene in scenes]

    def load_visual_sequence(self, seq_id: str, max_frames=None) -> dict:
        entries = {item["seq_id"] for item in self.list_visual_sequences()}
        if seq_id not in entries:
            raise KeyError(f"Sintel 中不存在场景 {seq_id!r}")
        paths = self._scene_inputs(seq_id, max_frames=max_frames)
        if not paths:
            raise FileNotFoundError(f"Sintel 场景 {seq_id!r} 没有可读输入帧")
        import cv2
        image = cv2.imread(paths[0])
        if image is None:
            raise FileNotFoundError(f"读图失败: {paths[0]}")
        return {"seq_id": seq_id, "label": seq_id, "image_paths": paths,
                "hw": tuple(image.shape[:2]), "frame_count": len(paths),
                "fps": 24.0, "source_path": str(Path(paths[0]).parent)}

    def iter_sequences(self, max_seqs=None, max_frames=None) -> Iterator[GTSequence]:
        training = _find_training(self.root)
        if training is None:
            raise FileNotFoundError(
                f"未找到 Sintel training/camdata_left(在 {self.root} 下);先下载解压")
        cam_root = os.path.join(training, "camdata_left")
        img_root = os.path.join(training, "clean")
        scenes = sorted(d for d in os.listdir(cam_root)
                        if os.path.isdir(os.path.join(cam_root, d)))
        if max_seqs:
            scenes = scenes[:max_seqs]

        for scene in scenes:
            cams = sorted(glob.glob(os.path.join(cam_root, scene, "frame_*.cam")))
            if max_frames:
                cams = cams[:max_frames]
            img_paths, intr, extr = [], [], []
            for c in cams:
                stem = os.path.splitext(os.path.basename(c))[0]
                img = os.path.join(img_root, scene, stem + ".png")
                if not os.path.exists(img):
                    continue
                M, N = _cam_read(c)
                extr44 = np.eye(4); extr44[:3, :4] = N       # world->cam
                img_paths.append(img); intr.append(M); extr.append(extr44)
            if not img_paths:
                continue
            import cv2
            # 取首张**可读**图定 hw:cv2.imread 可能返回 None(损坏/LFS 指针/挂载抖动),不能 .shape 直接崩。
            hw = None
            for p in img_paths:
                im = cv2.imread(p)
                if im is not None:
                    hw = im.shape[:2]
                    break
            if hw is None:
                print(f"[sintel] 场景 {scene} 图像均不可读(损坏/未下全/挂载异常),跳过", flush=True)
                continue
            h, w = hw
            yield GTSequence(
                seq_id=scene, image_paths=img_paths, hw=(h, w),
                intrinsic=np.stack(intr), extrinsic_w2c=np.stack(extr),
                capability=self.capability, meta={"dataset": "sintel"},
            )
