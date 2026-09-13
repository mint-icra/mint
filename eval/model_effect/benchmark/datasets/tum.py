# -*- coding: utf-8 -*-
"""TUM RGB-D 适配器 ✅(数据未下载时优雅降级):外参 GT(ATE 金标准)。

落盘: data/benchmark/extrinsics/tum/<seq>/{rgb.txt, groundtruth.txt, rgb/*.png}
groundtruth.txt: `timestamp tx ty tz qx qy qz qw`(world->cam? 实为 cam->world 位姿,见 TUM 约定)。
rgb.txt: `timestamp filename`。按时间戳最近邻把图与位姿配对。
未下载(目录空)→ iter_sequences 抛 FileNotFoundError,run 记 "skipped(缺数据)"。
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np

from ..core.registry import DATASETS
from ..core.schema import EXTRINSIC, GTSequence
from .base import DatasetAdapter


def _quat_to_R(q):
    """q=[qx,qy,qz,qw] → R[3,3]。"""
    x, y, z, w = q
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ])


def _read_tum_txt(path):
    """读 `ts v1 v2 ...` 表,跳过 # 注释。返回 [(ts, [values...])]。"""
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            rows.append((float(parts[0]), parts[1:]))
    return rows


@DATASETS.register("tum")
class TUMAdapter(DatasetAdapter):
    name = "tum"
    root_rel = "extrinsics/tum"
    capability = {EXTRINSIC}

    def _sequence_dirs(self) -> list[str]:
        if not os.path.isdir(self.root) or not os.listdir(self.root):
            raise FileNotFoundError(f"TUM 未下载(空目录 {self.root});见 readme/datasets_download/extrinsics")
        seqs = sorted(d for d in glob.glob(os.path.join(self.root, "*"))
                      if os.path.isfile(os.path.join(d, "groundtruth.txt")))
        if not seqs:
            raise FileNotFoundError(f"{self.root} 下无含 groundtruth.txt 的序列")
        return seqs

    def list_visual_sequences(self) -> list[dict]:
        out = []
        for seq_dir in self._sequence_dirs():
            rgb = _read_tum_txt(os.path.join(seq_dir, "rgb.txt"))
            out.append({"seq_id": os.path.basename(seq_dir),
                        "label": os.path.basename(seq_dir),
                        "frame_count": len(rgb)})
        return out

    def load_visual_sequence(self, seq_id: str, max_frames=None) -> dict:
        matches = {os.path.basename(path): path for path in self._sequence_dirs()}
        seq_dir = matches.get(seq_id)
        if seq_dir is None:
            raise KeyError(f"TUM 中不存在序列 {seq_id!r}")
        rgb = _read_tum_txt(os.path.join(seq_dir, "rgb.txt"))
        if max_frames:
            rgb = rgb[:max_frames]
        paths = [os.path.join(seq_dir, values[0]) for _ts, values in rgb]
        if not paths:
            raise FileNotFoundError(f"TUM 序列 {seq_id!r} 没有 RGB 帧")
        import cv2
        image = cv2.imread(paths[0])
        if image is None:
            raise FileNotFoundError(f"读图失败: {paths[0]}")
        timestamps = np.asarray([timestamp for timestamp, _values in rgb], dtype=float)
        intervals = np.diff(timestamps)
        fps = float(1.0 / np.median(intervals[intervals > 0])) \
            if np.any(intervals > 0) else 30.0
        return {"seq_id": seq_id, "label": seq_id, "image_paths": paths,
                "hw": tuple(image.shape[:2]), "frame_count": len(paths),
                "fps": fps, "source_path": seq_dir}

    def iter_sequences(self, max_seqs=None, max_frames=None) -> Iterator[GTSequence]:
        seqs = self._sequence_dirs()
        if max_seqs:
            seqs = seqs[:max_seqs]

        for sd in seqs:
            gt = _read_tum_txt(os.path.join(sd, "groundtruth.txt"))
            rgb = _read_tum_txt(os.path.join(sd, "rgb.txt"))
            gt_ts = np.array([t for t, _ in gt])
            img_paths, extr = [], []
            for ts, (fn,) in ([(t, v) for t, v in rgb][:max_frames] if max_frames else rgb):
                j = int(np.argmin(np.abs(gt_ts - ts)))       # 最近邻位姿
                vals = np.array([float(x) for x in gt[j][1]])
                R = _quat_to_R(vals[3:7]); tt = vals[0:3]
                c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = tt   # TUM gt 为 cam->world
                extr.append(np.linalg.inv(c2w))               # 存 world->cam(GT 约定)
                img_paths.append(os.path.join(sd, fn))
            if not img_paths:
                continue
            import cv2
            h, w = cv2.imread(img_paths[0]).shape[:2]
            yield GTSequence(
                seq_id=os.path.basename(sd), image_paths=img_paths, hw=(h, w),
                extrinsic_w2c=np.stack(extr), capability=self.capability,
                meta={"dataset": "tum"},
            )
