# -*- coding: utf-8 -*-
"""Adapters for the 27 HOT3D + 34 ARCTIC full-sequence trajectory export.

Expected layout (each dataset directory contains sequence directories)::

    camera_trajectory/
      hot3d_val/<sequence>/{images/*.jpg,gt.npz,meta.json}
      arctic_val/<sequence>/{images/*.jpg,gt.npz,meta.json}

Resolution order:
  1. ``CAMERA_TRAJECTORY_ROOT`` (the directory containing hot3d_val/arctic_val)
  2. ``<data_root>/camera_trajectory``
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.registry import DATASETS
from ..core.schema import EXTRINSIC, GTSequence
from ..datasets.base import DatasetAdapter, deterministic_diverse_sample


ENV_ROOT = "CAMERA_TRAJECTORY_ROOT"


def _contains_dataset(root: Path, subdir: str) -> bool:
    directory = root / subdir
    return directory.is_dir() and any(
        child.is_dir() and (child / "gt.npz").is_file()
        for child in directory.iterdir()
    )


class CameraTrajectoryDataset(DatasetAdapter):
    """Shared implementation for a dense, full-length camera trajectory set."""

    capability = {EXTRINSIC}
    default_enabled = False
    dataset_subdir = ""
    source_dataset = ""

    def __init__(self, data_root: str):
        super().__init__(data_root)
        configured = os.environ.get(ENV_ROOT, "").strip()
        candidates = []
        if configured:
            candidates.append(Path(configured).expanduser())
        candidates.append(Path(data_root) / "camera_trajectory")

        selected = next(
            (candidate.resolve() for candidate in candidates
             if _contains_dataset(candidate, self.dataset_subdir)),
            None,
        )
        # Preserve the preferred path in error messages even when data is absent.
        preferred = candidates[0].expanduser().resolve()
        self.data_root = selected or preferred
        self.root = str(self.data_root / self.dataset_subdir)
        canonical = (Path(data_root) / "camera_trajectory").expanduser().resolve()
        self.data_source_note = (
            "仓库内标准路径" if self.data_root == canonical
            else f"兼容数据源: {self.data_root}"
        )

    @property
    def directory(self) -> Path:
        return Path(self.root)

    def _missing_message(self) -> str:
        return (
            f"未找到 {self.name} 相机轨迹数据: {self.directory}。期望 "
            f"{self.dataset_subdir}/<seq>/images + gt.npz + meta.json；可设置 "
            f"{ENV_ROOT}=/path/containing/hot3d_val_and_arctic_val"
        )

    def _sequence_names(self) -> list[str]:
        if not self.directory.is_dir():
            raise FileNotFoundError(self._missing_message())
        names = [
            child.name for child in sorted(self.directory.iterdir())
            if child.is_dir() and (child / "gt.npz").is_file()
            and (child / "meta.json").is_file() and (child / "images").is_dir()
        ]
        if not names:
            raise FileNotFoundError(self._missing_message())
        return names

    def _meta(self, name: str) -> dict:
        path = self.directory / name / "meta.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"相机轨迹 meta 无效: {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"相机轨迹 meta 必须是对象: {path}")
        return value

    def _frame_count(self, name: str) -> int:
        meta = self._meta(name)
        frames = int(meta.get("frames") or 0)
        if frames > 0:
            return frames
        return len(list((self.directory / name / "images").glob("*.jpg")))

    def _benchmark_names(self) -> list[str]:
        names = self._sequence_names()
        options = self.benchmark_selection
        count = options.get("sample_count") if options.get("sampling") == "diverse" else None
        return deterministic_diverse_sample(
            names, count, group_key=self._sampling_group,
            seed=options.get("seed", 42),
        )

    def _sampling_group(self, name: str) -> str:
        # HOT3D groups by participant; ARCTIC export is s05-only, so grouping by
        # manipulated object keeps small smoke samples diverse.
        if self.source_dataset == "hot3d":
            return name.split("_", 1)[0]
        value = name.removeprefix("s05_")
        for action in ("_grab_", "_use_"):
            if action in value:
                return value.split(action, 1)[0]
        return value

    def count_sequences(self):
        names = self._sequence_names()
        return len(names), sum(self._frame_count(name) for name in names)

    def iter_sequences(self, max_seqs=None, max_frames=None) -> Iterator[GTSequence]:
        names = self._benchmark_names()
        if max_seqs is not None:
            names = names[:int(max_seqs)]
        yield from self._iter_names(names, max_frames=max_frames)

    def _names_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> list[str]:
        start = int(seq_start or 0)
        stop = seq_end
        if max_seqs is not None:
            max_stop = start + int(max_seqs)
            stop = max_stop if stop is None else min(int(stop), max_stop)
        names = self._benchmark_names()[start:stop]
        costs = [
            (min(self._frame_count(name), int(max_frames)) if max_frames else self._frame_count(name),
             order, name)
            for order, name in enumerate(names)
        ]
        loads = [0] * int(shard_count)
        selected = set()
        for frames, _order, name in sorted(costs, key=lambda item: (-item[0], item[1])):
            target = min(range(int(shard_count)), key=lambda index: (loads[index], index))
            loads[target] += frames
            if target == int(shard_index):
                selected.add(name)
        return [name for name in names if name in selected]

    def iter_sequences_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> Iterator[GTSequence]:
        names = self._names_for_shard(
            shard_index, shard_count, max_seqs=max_seqs, max_frames=max_frames,
            seq_start=seq_start, seq_end=seq_end,
        )
        yield from self._iter_names(names, max_frames=max_frames)

    def count_sequences_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> int:
        return len(self._names_for_shard(
            shard_index, shard_count, max_seqs=max_seqs, max_frames=max_frames,
            seq_start=seq_start, seq_end=seq_end,
        ))

    def _iter_names(self, names: list[str], max_frames=None) -> Iterator[GTSequence]:
        for name in names:
            directory = self.directory / name
            images = sorted((directory / "images").glob("*.jpg"))
            if not images:
                raise FileNotFoundError(f"相机轨迹序列没有 JPEG: {directory / 'images'}")
            gt_path = directory / "gt.npz"
            with np.load(gt_path, allow_pickle=False) as archive:
                if "c2w" not in archive:
                    raise ValueError(f"相机轨迹 GT 缺 c2w: {gt_path}")
                c2w = np.asarray(archive["c2w"], np.float64)
                intrinsic = np.asarray(archive["K"], np.float64) if "K" in archive else None
                hw = tuple(int(value) for value in archive["hw"]) if "hw" in archive else None
            if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
                raise ValueError(f"相机轨迹 c2w 形状应为 [T,4,4]: {gt_path} -> {c2w.shape}")
            meta = self._meta(name)
            if hw is None:
                hw = tuple(int(value) for value in meta.get("hw") or ())
            if len(hw) != 2:
                raise ValueError(f"相机轨迹缺有效 hw: {gt_path}")
            if intrinsic is None and meta.get("K") is not None:
                intrinsic = np.asarray(meta["K"], np.float64)

            total = min(len(images), len(c2w))
            if total < 2:
                raise ValueError(f"相机轨迹至少需要 2 帧: {directory}")
            if len(images) != len(c2w):
                raise ValueError(
                    f"图像与 GT 帧数不一致: {directory} images={len(images)} c2w={len(c2w)}"
                )
            limit = min(total, int(max_frames)) if max_frames else total
            c2w = c2w[:limit]
            yield GTSequence(
                seq_id=name,
                image_paths=[str(path) for path in images[:limit]],
                hw=hw,
                intrinsic=intrinsic,
                extrinsic_w2c=np.linalg.inv(c2w),
                capability=self.capability,
                meta={
                    "dataset": self.name,
                    "source_dataset": self.source_dataset,
                    "source_path": str(directory),
                    "gt_source": meta.get("gt_source", ""),
                    "fps": float(meta.get("fps", 30.0)),
                    "full_sequence_frames": total,
                    "truncated_for_benchmark": limit < total,
                    "trajectory_protocol": "icra_full_sequence_v1",
                },
            )

    def list_visual_sequences(self) -> list[dict]:
        return [
            {
                "seq_id": name,
                "label": name,
                "frame_count": self._frame_count(name),
                "source_path": str(self.directory / name),
            }
            for name in self._sequence_names()
        ]

    def load_visual_sequence(self, seq_id: str, max_frames: int | None = None) -> dict:
        if seq_id not in set(self._sequence_names()):
            raise KeyError(f"数据集 {self.name} 中不存在序列 {seq_id!r}")
        directory = self.directory / seq_id
        images = sorted((directory / "images").glob("*.jpg"))
        if max_frames:
            images = images[:int(max_frames)]
        meta = self._meta(seq_id)
        return {
            "seq_id": seq_id,
            "label": seq_id,
            "image_paths": [str(path) for path in images],
            "hw": tuple(int(value) for value in meta["hw"]),
            "frame_count": len(images),
            "fps": float(meta.get("fps", 30.0)),
            "source_path": str(directory),
        }


@DATASETS.register("camera_hot3d")
class HOT3DCameraTrajectoryDataset(CameraTrajectoryDataset):
    name = "camera_hot3d"
    dataset_subdir = "hot3d_val"
    source_dataset = "hot3d"


@DATASETS.register("camera_arctic")
class ARCTICCameraTrajectoryDataset(CameraTrajectoryDataset):
    name = "camera_arctic"
    dataset_subdir = "arctic_val"
    source_dataset = "arctic"
