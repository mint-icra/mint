"""ARCTIC egocentric hand-coverage adapter using the official validation split.

The public ARCTIC test subject (s03) does not include MANO labels in the
downloaded release.  This adapter therefore defaults to the official ``val``
split (s05), divides each sequence into non-overlapping 81-frame clips, and
evaluates camera-frame two-hand geometry only.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.registry import DATASETS
from ..core.schema import HAND_COVERAGE, GTSequence
from .base import DatasetAdapter, deterministic_diverse_sample, fixed_split_items


TARGET_HW = (480, 672)
SEGMENT_FRAMES = 81


def _load_dict(path: Path) -> dict:
    return np.load(path, allow_pickle=True).item()


def _camera_mano_6d(mano_data: dict, egocam: dict) -> tuple[np.ndarray, np.ndarray]:
    """Convert ARCTIC world-frame MANO parameters to this repo's camera 6D format."""
    from scipy.spatial.transform import Rotation
    from visualization.reproj_core import geometry, mano

    frames = len(mano_data["right"]["rot"])
    world_to_camera = np.asarray(egocam["R_k_cam_np"], np.float32)[:frames]
    camera_translation = np.asarray(egocam["T_k_cam_np"], np.float32)[:frames].reshape(frames, 3)
    encoded = []
    valid = []
    for side in ("left", "right"):
        values = mano_data[side]
        orient_world = np.asarray(values["rot"], np.float32)[:frames]
        pose = np.asarray(values["pose"], np.float32)[:frames]
        translation_world = np.asarray(values["trans"], np.float32)[:frames]
        betas = np.broadcast_to(
            np.asarray(values["shape"], np.float32).reshape(1, 10), (frames, 10),
        ).copy()

        _, joints_world = mano.run_mano(
            translation_world, orient_world, pose, betas, is_right=(side == "right"),
        )
        wrist_camera = (
            np.einsum("tij,tj->ti", world_to_camera, joints_world[:, 0])
            + camera_translation
        )
        orient_camera = np.einsum(
            "tij,tjk->tik", world_to_camera,
            Rotation.from_rotvec(orient_world).as_matrix(),
        )

        # decode_hand_6d applies this involutive mirror correction for the left hand.
        pose_for_encoding = pose.copy()
        if side == "left":
            pose_for_encoding[:, 1::3] *= -1.0
            pose_for_encoding[:, 2::3] *= -1.0
        pose_matrices = Rotation.from_rotvec(
            pose_for_encoding.reshape(-1, 3)
        ).as_matrix().reshape(frames, 15, 3, 3)

        hand = np.concatenate([
            wrist_camera,
            geometry.mat_to_6d(orient_camera),
            geometry.mat_to_6d(pose_matrices).reshape(frames, 90),
            betas,
        ], axis=-1).astype(np.float32)
        encoded.append(hand)
        valid.append(np.isfinite(hand).all(axis=-1))
    return np.stack(encoded, axis=1), np.stack(valid, axis=1)


@DATASETS.register("arctic_hand_coverage")
class ARCTICHandCoverageAdapter(DatasetAdapter):
    name = "arctic_hand_coverage"
    root_rel = "hand_pose/arctic"
    capability = {HAND_COVERAGE}
    default_enabled = False
    segment_frames = SEGMENT_FRAMES

    def __init__(self, data_root: str):
        super().__init__(data_root)
        self.root = Path(self.root)
        self.data = self.root / "data"
        self.split = os.environ.get("ARCTIC_SPLIT", "val").strip().lower()
        if self.split not in {"train", "val", "test"}:
            raise ValueError("ARCTIC_SPLIT must be train, val, or test")
        cache_value = os.environ.get("ARCTIC_CACHE_DIR")
        self.cache_root = (
            Path(cache_value).expanduser().resolve() if cache_value
            else Path(tempfile.gettempdir()) / "mint-arctic-benchmark-480p"
        )
        self.image_workers = max(1, int(os.environ.get("ARCTIC_IMAGE_WORKERS", "8")))
        self._entries_cache = None

    def _split_sequences(self) -> list[str]:
        split_path = self.data / "splits_json" / "protocol_p1.json"
        if not split_path.is_file():
            raise FileNotFoundError(f"ARCTIC split manifest is missing: {split_path}")
        values = json.loads(split_path.read_text(encoding="utf-8")).get(self.split) or []
        if not values:
            raise FileNotFoundError(f"ARCTIC protocol_p1 has no {self.split!r} sequences")
        return sorted(str(value) for value in values)

    def _label_path(self, sequence: str, suffix: str) -> Path:
        subject, name = sequence.split("/", 1)
        return self.data / "raw_seqs" / subject / f"{name}.{suffix}.npy"

    def _frame_count(self, sequence: str) -> int:
        path = self._label_path(sequence, "mano")
        if not path.is_file():
            if self.split == "test":
                raise FileNotFoundError(
                    "ARCTIC official test images are present, but the public release omits "
                    f"MANO ground truth ({path}); use ARCTIC_SPLIT=val for local metrics"
                )
            raise FileNotFoundError(f"ARCTIC MANO labels are missing: {path}")
        return int(len(_load_dict(path)["right"]["rot"]))

    def _selected_segments(self) -> list[tuple[str, int]]:
        if self._entries_cache is None:
            self._entries_cache = [
                (sequence, segment_index)
                for sequence in self._split_sequences()
                for segment_index in range(self._frame_count(sequence) // self.segment_frames)
            ]
        return self._entries_cache

    def _benchmark_segments(self) -> list[tuple[str, int]]:
        entries = self._selected_segments()
        options = self.benchmark_selection
        if options.get("fixed_tier"):
            return fixed_split_items(
                entries, self.name, options["fixed_tier"],
                item_id=lambda entry: f"{entry[0]}#seg{entry[1]:04d}",
                version=options["split_version"],
            )
        count = options.get("sample_count") if options.get("sampling") == "diverse" else None
        return deterministic_diverse_sample(
            entries, count, group_key=lambda entry: entry[0], seed=options.get("seed", 42),
        )

    @staticmethod
    def _validate_max_frames(max_frames):
        if max_frames is not None and int(max_frames) < SEGMENT_FRAMES:
            raise ValueError("ARCTIC hand coverage requires complete 81-frame segments")

    def count_sequences(self):
        count = len(self._selected_segments())
        return count, count * self.segment_frames

    def iter_sequences(self, max_seqs=None, max_frames=None) -> Iterator[GTSequence]:
        self._validate_max_frames(max_frames)
        entries = self._benchmark_segments()
        if max_seqs is not None:
            entries = entries[:int(max_seqs)]
        yield from self._iter_protocol_entries(entries)

    def iter_sequences_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> Iterator[GTSequence]:
        entries = self._entries_for_shard(
            shard_index, shard_count, max_seqs=max_seqs, max_frames=max_frames,
            seq_start=seq_start, seq_end=seq_end,
        )
        yield from self._iter_protocol_entries(entries)

    def _entries_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> list[tuple[str, int]]:
        self._validate_max_frames(max_frames)
        start = int(seq_start or 0)
        end = int(seq_end) if seq_end is not None else None
        if max_seqs is not None:
            max_end = start + int(max_seqs)
            end = max_end if end is None else min(end, max_end)
        entries = self._benchmark_segments()[start:end]

        # Keep all clips from a source video on one GPU to avoid rebuilding MANO GT.
        by_sequence = defaultdict(list)
        for entry in entries:
            by_sequence[entry[0]].append(entry)
        loads = [0] * int(shard_count)
        assigned = {}
        for sequence, sequence_entries in sorted(
            by_sequence.items(), key=lambda item: (-len(item[1]), item[0]),
        ):
            target = min(range(int(shard_count)), key=lambda index: (loads[index], index))
            assigned[sequence] = target
            loads[target] += len(sequence_entries)
        return [entry for entry in entries if assigned[entry[0]] == int(shard_index)]

    def count_sequences_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> int:
        return len(self._entries_for_shard(
            shard_index, shard_count, max_seqs=max_seqs, max_frames=max_frames,
            seq_start=seq_start, seq_end=seq_end,
        ))

    def _source_images(self, sequence: str, frames: int) -> list[Path]:
        subject, name = sequence.split("/", 1)
        directory = self.data / "cropped_images" / subject / name / "0"
        paths = sorted(directory.glob("*.jpg"))
        if len(paths) < frames:
            raise FileNotFoundError(
                f"ARCTIC {sequence} has {len(paths)} ego images but {frames} label frames"
            )
        return paths[:frames]

    def _resize_paths(self, sequence: str, sources: list[Path]) -> list[str]:
        import cv2

        subject, name = sequence.split("/", 1)
        target_dir = self.cache_root / subject / name
        target_dir.mkdir(parents=True, exist_ok=True)

        def resize(source: Path) -> str:
            target = target_dir / source.name
            if target.is_file() and target.stat().st_size > 0:
                return str(target)
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(f"Cannot read ARCTIC image: {source}")
            resized = cv2.resize(
                image, (TARGET_HW[1], TARGET_HW[0]), interpolation=cv2.INTER_AREA,
            )
            temporary = target.with_name(f".{target.stem}.{os.getpid()}.tmp.jpg")
            if not cv2.imwrite(str(temporary), resized, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise OSError(f"Cannot write ARCTIC cache image: {temporary}")
            os.replace(temporary, target)
            return str(target)

        workers = min(self.image_workers, len(sources))
        if workers <= 1:
            return [resize(source) for source in sources]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(resize, sources))

    def _sequence_payload(self, sequence: str):
        mano_path = self._label_path(sequence, "mano")
        ego_path = self._label_path(sequence, "egocam.dist")
        if not ego_path.is_file():
            raise FileNotFoundError(f"ARCTIC ego-camera labels are missing: {ego_path}")
        mano_data = _load_dict(mano_path)
        egocam = _load_dict(ego_path)
        mano_lr, valid_lr = _camera_mano_6d(mano_data, egocam)
        frames = len(mano_lr)
        sources = self._source_images(sequence, frames)

        intrinsic = np.asarray(egocam["intrinsics"], np.float64).copy()
        intrinsic[0] *= TARGET_HW[1] / 2800.0
        intrinsic[1] *= TARGET_HW[0] / 2000.0
        return mano_lr, valid_lr, sources, intrinsic

    def _iter_protocol_entries(self, entries) -> Iterator[GTSequence]:
        by_sequence = defaultdict(list)
        for sequence, segment_index in entries:
            by_sequence[sequence].append(int(segment_index))
        available = len(self._selected_segments())
        for sequence in sorted(by_sequence):
            mano_lr, valid_lr, sources, intrinsic = self._sequence_payload(sequence)
            for segment_index in sorted(by_sequence[sequence]):
                start = segment_index * self.segment_frames
                end = start + self.segment_frames
                image_paths = self._resize_paths(sequence, sources[start:end])
                yield GTSequence(
                    seq_id=f"{sequence}#seg{segment_index:04d}",
                    image_paths=image_paths,
                    hw=TARGET_HW,
                    intrinsic=intrinsic,
                    hand_mano_6d=mano_lr[start:end],
                    hand_valid_lr=valid_lr[start:end],
                    capability=self.capability,
                    meta={
                        "dataset": self.name,
                        "source_dataset": "arctic",
                        "seq": sequence,
                        "segment_index": segment_index,
                        "prediction_group": f"{self.name}:{sequence}:{segment_index}",
                        "official_split": True,
                        "split_name": self.split,
                        "reference_same_split": False,
                        "requested_segments": available,
                        "available_segments": available,
                    },
                )

    def list_visual_sequences(self) -> list[dict]:
        return [
            {
                "seq_id": f"{sequence}#seg{segment_index:04d}",
                "label": f"{sequence} / segment {segment_index}",
                "frame_count": self.segment_frames,
            }
            for sequence, segment_index in self._selected_segments()
        ]

    def load_visual_sequence(self, seq_id: str, max_frames=None) -> dict:
        self._validate_max_frames(max_frames)
        match = re.fullmatch(r"(.+)#seg(\d+)", str(seq_id))
        if match is None:
            raise KeyError(f"Invalid ARCTIC hand-coverage segment: {seq_id!r}")
        entry = (match.group(1), int(match.group(2)))
        if entry not in set(self._selected_segments()):
            raise KeyError(f"Segment is not in ARCTIC {self.split}: {seq_id!r}")
        seq = next(self._iter_protocol_entries([entry]))
        return {
            "seq_id": seq.seq_id,
            "label": seq.seq_id,
            "image_paths": list(seq.image_paths),
            "hw": tuple(seq.hw),
            "frame_count": seq.num_frames,
            "fps": 30.0,
            "source_path": str(Path(seq.image_paths[0]).parent),
        }
