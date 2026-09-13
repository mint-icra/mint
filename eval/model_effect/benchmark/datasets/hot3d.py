# -*- coding: utf-8 -*-
"""HOT3D 适配器 ✅:第一视角(ego)双手抓取,Aria FISHEYE624 鱼眼。供 hands(相机系)+ hands_world(world 系)头。

量化评测优先读取训练就绪的 `output/.../build_train_lerobot/hot3d/lerobot_v3`：其中视频已去畸变，
相机 pose_enc 与双手 camera MANO 真值可直接复用。`HOT3D_LEROBOT_ROOT` 可覆盖该路径。

兼容旧数据 `data/benchmark/hand_pose/hot3d/<seq>/`(自包含):
  extracted_images/*.jpg(1408² 鱼眼)、anno.pth(双手 world MANO 轴角 rot/trans/pose/betas,_l/_r)、
  ego_extrinsics.pkl(T_Device_Camera 恒定)、head_pose.pkl(T_world_device=Aria SLAM 轨迹)、
  camera_models.json(FISHEYE624 15 参)、masks/mask_hand_visible.csv(逐帧手可见)、mps/(SLAM 原始,未用)。

**默认去畸变成针孔(HOT3D_RECTIFY=1;鱼眼禁用于测评)**:模型在针孔图 in-distribution(相机系 85→28mm)。
每序列 × 每只手 yield 一条 GTSequence(seq_id=`<seq>#left|#right`,meta.mano_side 标手)。

坐标(实证):w2c=inv(ego)@inv(head);针孔 upright=R90@sensor(R90=[[0,-1,0],[1,0,0],[0,0,1]]);
GT 相机系关节=(针孔:R90@w2c@world;鱼眼:w2c@world);world 关节=anno run_mano 原样。
hands_world 头用 pred.extrinsic_c2w(pose_enc) / meta 里 head@ego(SLAM)两变体把 pred 转 world 比。
"""
from __future__ import annotations

import glob
import csv
import hashlib
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np

from ..core.registry import DATASETS
from ..core.schema import EXTRINSIC, HAND, HAND_COVERAGE, GTSequence, VideoFrameRef
from .base import DatasetAdapter, deterministic_diverse_sample, fixed_split_items
from ..preprocess import fisheye as FE

_ME = Path(__file__).resolve().parents[2]                  # datasets/ -> benchmark/ -> model_effect/
if str(_ME) not in sys.path:
    sys.path.insert(0, str(_ME))

_RECTIFY = os.environ.get("HOT3D_RECTIFY", "1") != "0"     # 默认去畸变成针孔;=0 才用鱼眼(不推荐)
_WIDTH = int(os.environ.get("HOT3D_WIDTH", "720"))         # 针孔输出边长
_HANDS = [h for h in os.environ.get("HOT3D_HANDS", "left,right").split(",") if h]  # 评哪些手
_SEQS = [s for s in os.environ.get("HOT3D_SEQS", "").split(",") if s]  # 只评这些序列(逗号分隔;空=全部)。多卡分片用


@DATASETS.register("hot3d")
class HOT3DAdapter(DatasetAdapter):
    name = "hot3d"
    root_rel = "hand_pose/hot3d"
    capability = {HAND, EXTRINSIC}                             # HAND=相机系(hands头);world 关节存 hand_joints_3d_world 供 hands_world 头

    def __init__(self, data_root: str):
        super().__init__(data_root)
        default_data = (_ME.parent.parent / "data" / "benchmark").resolve()
        converted = os.environ.get("HOT3D_LEROBOT_ROOT")
        if converted:
            self._lerobot_root = Path(converted).expanduser().resolve()
        elif Path(data_root).resolve() == default_data:
            self._lerobot_root = (
                _ME.parent.parent / "output" / "scripts" / "data_processed" /
                "pipeline_lerobot" / "build_train_lerobot" / "hot3d" / "lerobot_v3"
            ).resolve()
        else:
            self._lerobot_root = None
        self._converted_cache = None

    def _converted_episodes(self) -> dict[str, dict]:
        """Map official sequence names to the already rectified LeRobot evaluation export."""
        if self._converted_cache is not None:
            return self._converted_cache
        root = self._lerobot_root
        if root is None or not (root / "meta" / "info.json").is_file():
            self._converted_cache = {}
            return self._converted_cache
        from visualization.reproj_core import lerobot_io

        mapped = {}
        for episode in lerobot_io.discover_episodes(root):
            resolved = Path(episode["video"]).resolve()
            name = next((part for part in reversed(resolved.parts)
                         if re.fullmatch(r"P\d{4}_[0-9A-Za-z]+", part)), None)
            if name:
                mapped[name] = episode
        self._converted_cache = mapped
        return mapped

    @staticmethod
    def _extracted_images(directory: str) -> list[str]:
        return sorted(glob.glob(os.path.join(directory, "extracted_images", "*.jpg")))

    @staticmethod
    def _raw_rgb_frame_count(directory: str) -> int:
        """Count completed official-format RGB frames without opening the multi-GB VRS."""
        if not os.path.isfile(os.path.join(directory, "recording.vrs")):
            return 0
        mask = os.path.join(directory, "masks", "mask_hand_visible.csv")
        if not os.path.isfile(mask):
            return 0
        with open(mask, newline="") as handle:
            return sum(1 for row in csv.DictReader(handle)
                       if str(row.get("stream_id", "")) == "214-1")

    def _frame_count(self, name: str) -> int:
        converted = self._converted_episodes().get(name)
        if converted is not None:
            return int(converted["length"])
        directory = os.path.join(self.root, name)
        images = self._extracted_images(directory)
        return len(images) if images else self._raw_rgb_frame_count(directory)

    def _extract_raw_rgb(self, name: str, max_frames=None) -> list[str]:
        """Lazily materialize official HOT3D RGB frames for Viewer preview."""
        import cv2
        from projectaria_tools.core import data_provider

        directory = os.path.join(self.root, name)
        vrs = os.path.join(directory, "recording.vrs")
        if not os.path.isfile(vrs):
            raise FileNotFoundError(f"HOT3D 序列 {name!r} 缺 recording.vrs")
        provider = data_provider.create_vrs_data_provider(vrs)
        if provider is None:
            raise RuntimeError(f"无法打开 HOT3D VRS: {vrs}")
        stream = provider.get_stream_id_from_label("camera-rgb")
        total = int(provider.get_num_data(stream))
        limit = min(total, int(max_frames)) if max_frames else total
        cache = (_ME.parent.parent / "output" / "eval" / "benchmark" /
                 "_hot3d_vrs_cache" / name)
        cache.mkdir(parents=True, exist_ok=True)
        paths = []
        for index in range(limit):
            dst = cache / f"{index:06d}.jpg"
            if not dst.exists():
                image = np.asarray(provider.get_image_data_by_index(stream, index)[0].to_numpy_array())
                if image.ndim != 3 or image.shape[2] != 3:
                    raise ValueError(f"HOT3D RGB 帧形状异常: {image.shape}")
                temporary = dst.with_name(f".{dst.stem}.{os.getpid()}.tmp.jpg")
                if not cv2.imwrite(str(temporary), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"写入 HOT3D 帧失败: {temporary}")
                os.replace(temporary, dst)
            paths.append(str(dst))
        return paths

    @staticmethod
    def _write_rgb_cache_indices(name: str, episode: dict, indices) -> list[str]:
        """Materialize explicitly requested frames for Viewer-only filesystem inputs."""
        import cv2
        from decord import VideoReader

        cache = (_ME.parent.parent / "output" / "eval" / "benchmark" /
                 "_hot3d_lerobot_cache" / name)
        cache.mkdir(parents=True, exist_ok=True)
        indices = [int(index) for index in indices]
        paths = [cache / f"{index:06d}.jpg" for index in indices]
        path_by_index = dict(zip(indices, paths))
        missing = [index for index, path in zip(indices, paths)
                   if not path.is_file() or path.stat().st_size == 0]
        if missing:
            reader = VideoReader(str(episode["video"]), num_threads=min(8, os.cpu_count() or 4))
            start = int(episode.get("video_start", 0))
            for offset in range(0, len(missing), 64):
                local_indices = missing[offset:offset + 64]
                absolute = [start + index for index in local_indices]
                if absolute[-1] >= len(reader):
                    raise RuntimeError(
                        f"HOT3D {name} 视频仅 {len(reader)} 帧，无法读取第 {absolute[-1]} 帧"
                    )
                images = reader.get_batch(absolute).asnumpy()
                for index, image in zip(local_indices, images):
                    dst = path_by_index[index]
                    temporary = dst.with_name(f".{dst.stem}.{os.getpid()}.tmp.jpg")
                    ok = cv2.imwrite(str(temporary), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    if not ok:
                        raise OSError(f"写入 HOT3D 缓存帧失败: {temporary}")
                    os.replace(temporary, dst)
        return [str(path) for path in paths]

    @staticmethod
    def _write_rgb_cache(name: str, episode: dict, count: int) -> list[str]:
        return HOT3DAdapter._write_rgb_cache_indices(name, episode, range(int(count)))

    @staticmethod
    def _video_frames(episode: dict, count: int) -> list[VideoFrameRef]:
        """Reference MP4 frames directly so benchmark setup does not create huge JPEG caches."""
        start = int(episode.get("video_start", 0))
        path = str(episode["video"])
        return [VideoFrameRef(path, start + index) for index in range(int(count))]

    def list_visual_sequences(self) -> list[dict]:
        out = []
        for name in self._sequence_names():
            count = self._frame_count(name)
            if count:
                out.append({"seq_id": name, "label": name, "frame_count": count})
        return out

    def load_visual_sequence(self, seq_id: str, max_frames=None) -> dict:
        # Reports use "<video>#left|right"; both rows share the same visual input.
        name = str(seq_id).split("#", 1)[0]
        if name not in set(self._sequence_names()):
            raise KeyError(f"HOT3D 中不存在序列 {seq_id!r}")
        converted = self._converted_episodes().get(name)
        if converted is not None:
            count = int(converted["length"])
            count = min(count, int(max_frames)) if max_frames else count
            paths = self._write_rgb_cache(name, converted, count)
            from visualization.reproj_core import lerobot_io
            hw = lerobot_io.video_hw(self._lerobot_root)
            return {"seq_id": name, "label": name, "image_paths": paths,
                    "hw": hw, "frame_count": len(paths), "fps": 30.0,
                    "source_path": str(converted["video"])}
        directory = os.path.join(self.root, name)
        paths = self._extracted_images(directory)
        if paths:
            paths = paths[:max_frames] if max_frames else paths
        else:
            paths = self._extract_raw_rgb(name, max_frames=max_frames)
        if not paths:
            raise FileNotFoundError(f"HOT3D 序列 {name!r} 没有输入帧")
        if _RECTIFY:
            cam = FE.load_fisheye624(os.path.join(directory, "camera_models.json"))
            cache = _ME.parent.parent / "output" / "eval" / "benchmark" / "_rectify_cache" / name
            paths = FE.rectify_to_cache(paths, cam, _WIDTH, cache)
            hw = (_WIDTH, _WIDTH)
        else:
            import cv2
            image = cv2.imread(paths[0])
            if image is None:
                raise FileNotFoundError(f"读图失败: {paths[0]}")
            hw = tuple(image.shape[:2])
        return {"seq_id": name, "label": name, "image_paths": list(paths),
                "hw": hw, "frame_count": len(paths), "fps": 30.0,
                "source_path": directory}

    def _sequence_names(self, max_seqs=None) -> list[str]:
        converted = self._converted_episodes()
        if converted:
            seqs = sorted(converted)
        elif not os.path.isdir(self.root):
            raise FileNotFoundError(f"未找到 HOT3D(缺 {self.root});数据应在 hand_pose/hot3d")
        else:
            seqs = sorted(d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d)) and d.startswith("P0")
                          and self._frame_count(d) > 0)
        if not seqs:
            raise FileNotFoundError(f"{self.root} 下无已下载完成的 P0* 序列")
        if _SEQS:
            seqs = [name for name in seqs if name in _SEQS]
        return seqs[:max_seqs] if max_seqs else seqs

    def count_sequences(self):
        """Cheap panel count from LeRobot episode lengths or legacy frame manifests."""
        seqs = self._sequence_names()
        n_seqs = n_frames = 0
        for sq in seqs:
            frames = self._frame_count(sq)
            if not frames:
                continue
            n_seqs += 1
            n_frames += frames
        return n_seqs, n_frames

    def _benchmark_names(self) -> list[str]:
        names = self._sequence_names()
        options = self.benchmark_selection
        if options.get("fixed_tier"):
            return fixed_split_items(
                names, self.name, options["fixed_tier"], item_id=str,
                version=options["split_version"],
            )
        count = options.get("sample_count") if options.get("sampling") == "diverse" else None
        return deterministic_diverse_sample(
            names,
            count,
            group_key=lambda name: name.split("_", 1)[0],
            seed=options.get("seed", 42),
        )

    def iter_sequences(self, max_seqs=None, max_frames=None) -> Iterator[GTSequence]:
        names = self._benchmark_names()
        if max_seqs is not None:
            names = names[:int(max_seqs)]
        yield from self._iter_named_sequences(
            names, max_frames=max_frames
        )

    def iter_sequences_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> Iterator[GTSequence]:
        """Balance videos before loading annotations/MANO so each GPU builds only its GT."""
        names = self._names_for_shard(
            shard_index, shard_count, max_seqs=max_seqs, max_frames=max_frames,
            seq_start=seq_start, seq_end=seq_end,
        )
        yield from self._iter_named_sequences(names, max_frames=max_frames)

    def _names_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> list[str]:
        limit = seq_end
        if max_seqs is not None:
            max_end = int(seq_start or 0) + int(max_seqs)
            limit = max_end if limit is None else min(int(limit), max_end)
        seqs = self._benchmark_names()[int(seq_start or 0):limit]
        costs = []
        for order, name in enumerate(seqs):
            frames = self._frame_count(name)
            costs.append((min(frames, max_frames) if max_frames else frames, order, name))
        loads = [0] * int(shard_count)
        selected = set()
        for frames, _order, name in sorted(costs, key=lambda item: (-item[0], item[1])):
            target = min(range(int(shard_count)), key=lambda idx: (loads[idx], idx))
            loads[target] += frames
            if target == int(shard_index):
                selected.add(name)
        return [name for name in seqs if name in selected]

    def count_sequences_for_shard(
        self, shard_index, shard_count, max_seqs=None, max_frames=None,
        seq_start=0, seq_end=None,
    ) -> int:
        names = self._names_for_shard(
            shard_index, shard_count, max_seqs=max_seqs, max_frames=max_frames,
            seq_start=seq_start, seq_end=seq_end,
        )
        return len(names) * len(_HANDS)

    def _iter_named_sequences(self, seqs, max_frames=None) -> Iterator[GTSequence]:
        import cv2
        import torch
        from visualization.reproj_core import mano as M

        def tonp(x):
            return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

        def load_valid(d, n):
            f = os.path.join(d, "masks", "mask_hand_visible.csv")
            if not os.path.exists(f):
                return np.ones(n, bool)
            import csv
            ok = np.zeros(n, bool); i = 0
            with open(f) as fh:
                for row in csv.DictReader(fh):
                    if i >= n:
                        break
                    if str(row.get("mask", "")).strip().lower() == "true":
                        ok[i] = True
                    i += 1
            return ok if ok.any() else np.ones(n, bool)

        for sq in seqs:
            converted = self._converted_episodes().get(sq)
            if converted is not None:
                yield from self._iter_converted_sequence(sq, converted, max_frames, M)
                continue
            d = os.path.join(self.root, sq)
            imgs = self._extracted_images(d)
            if not imgs:
                raise FileNotFoundError(
                    f"HOT3D 序列 {sq} 是官方 VRS 原始格式，可直接在 Viewer 预览；"
                    "量化评测还需要 extracted_images/anno.pth/ego_extrinsics.pkl/head_pose.pkl"
                )
            if max_frames:
                imgs = imgs[:max_frames]
            n = len(imgs)
            a = self._load_serialized(os.path.join(d, "anno.pth"))
            ego = np.asarray(self._load_serialized(os.path.join(d, "ego_extrinsics.pkl")), np.float64)[:n]  # T_Device_Camera
            head = np.asarray(self._load_serialized(os.path.join(d, "head_pose.pkl")), np.float64)[:n]      # T_world_device(=SLAM)
            cam = FE.load_fisheye624(os.path.join(d, "camera_models.json"))
            w2c = np.linalg.inv(ego) @ np.linalg.inv(head)                                        # (n,4,4)

            # 去畸变成针孔(默认):image_paths 指向针孔缓存;鱼眼时用原图
            if _RECTIFY:
                cache = _ME.parent.parent / "output" / "eval" / "benchmark" / "_rectify_cache" / sq
                image_paths = FE.rectify_to_cache(imgs, cam, _WIDTH, cache)
                H = W = _WIDTH
                K = np.array([[FE.pinhole_f(cam, W), 0, W / 2.0], [0, FE.pinhole_f(cam, W), W / 2.0], [0, 0, 1]], np.float64)
            else:
                image_paths = imgs
                H, W = cv2.imread(imgs[0]).shape[:2]
                K = np.array([[cam["f"], 0, W / 2.0], [0, cam["f"], H / 2.0], [0, 0, 1]], np.float64)

            valid_frame = load_valid(d, n)

            for side, sfx, rt in (("left", "l", False), ("right", "r", True)):
                if side not in _HANDS:
                    continue
                r = tonp(a[f"rot_{sfx}"])[:n].astype(np.float32); t = tonp(a[f"trans_{sfx}"])[:n].astype(np.float32)
                p = tonp(a[f"pose_{sfx}"])[:n].astype(np.float32); b = tonp(a[f"betas_{sfx}"])[:n].astype(np.float32)
                _, j = M.run_mano(t, r, p, b, is_right=rt)
                Jw = np.asarray(j[:, :21], np.float64)                                            # world 21 关节
                Jc = np.einsum("tij,tnj->tni", w2c[:, :3, :3], Jw) + w2c[:, None, :3, 3]           # sensor 相机系
                if _RECTIFY:
                    Jc = np.einsum("ij,tnj->tni", FE.R90, Jc)                                     # upright 针孔系(与 pred 同)
                mano_ok = np.abs(tonp(a[f"rot_{sfx}"])[:n]).sum(-1) > 1e-6
                valid = valid_frame & mano_ok & np.isfinite(Jc).all((1, 2))
                yield GTSequence(
                    seq_id=f"{sq}#{side}", image_paths=image_paths, hw=(H, W),
                    intrinsic=K, extrinsic_w2c=w2c,
                    hand_joints_3d=Jc, hand_joints_3d_world=Jw, hand_valid=valid,
                    capability=self.capability,
                    meta={"dataset": "hot3d", "seq": sq, "mano_side": side,
                          # 左右手共享完全相同的输入帧；benchmark 据此同卡分组且只推理一次。
                          "prediction_group": f"hot3d:{sq}",
                          "proj": "pinhole" if _RECTIFY else "fisheye",
                          "cam2world_slam": (head @ ego)},               # head(=SLAM)@ego,供 world 头 SLAM 变体
                )

    @staticmethod
    def _load_serialized(path: str):
        """Load legacy pickle artifacts without requiring joblib for plain pickle files."""
        import pickle

        with open(path, "rb") as handle:
            try:
                return pickle.load(handle)
            except Exception as pickle_error:  # noqa: BLE001
                try:
                    import joblib
                except ModuleNotFoundError:
                    raise RuntimeError(
                        f"旧 HOT3D 标注 {path} 不是标准 pickle，且当前环境未安装 joblib"
                    ) from pickle_error
                return joblib.load(path)

    def _iter_converted_sequence(self, name, episode, max_frames, mano_module):
        """Build camera/world MANO GT from the rectified LeRobot export used for training."""
        from visualization.reproj_core import lerobot_io

        labels_only = dict(episode)
        labels_only["video"] = None
        raw = lerobot_io.load_episode_raw(
            labels_only, max_frames=max_frames, allow_missing_video=True,
        )
        count = int(raw["T"])
        image_paths = self._video_frames(episode, count)
        c2w = np.asarray(raw["cam_c2w"], np.float64)[:count]
        w2c = np.linalg.inv(c2w)
        kept = np.asarray(raw["kept"], bool)[:count]
        for side_index, side in enumerate(("left", "right")):
            if side not in _HANDS:
                continue
            hand = raw["hands"].get(side)
            if hand is None:
                continue
            decoded = mano_module.decode_hand_6d(
                hand["transl_cam"], hand["orient6d"], hand["pose6d"], hand["betas"],
                is_right=(side == "right"),
            )
            _, joints = mano_module.run_mano(
                decoded["trans"], decoded["rot"], decoded["hand_pose"], decoded["betas"],
                is_right=(side == "right"),
            )
            joints_camera = np.asarray(joints[:, :21], np.float64)
            joints_world = (
                np.einsum("tij,tnj->tni", c2w[:, :3, :3], joints_camera)
                + c2w[:, None, :3, 3]
            )
            valid = kept[:, side_index] & np.isfinite(joints_camera).all((1, 2))
            yield GTSequence(
                seq_id=f"{name}#{side}", image_paths=image_paths, hw=tuple(raw["hw"]),
                intrinsic=np.asarray(raw["K"], np.float64), extrinsic_w2c=w2c,
                hand_joints_3d=joints_camera, hand_joints_3d_world=joints_world,
                hand_valid=valid, capability=self.capability,
                meta={"dataset": "hot3d", "seq": name, "mano_side": side,
                      "prediction_group": f"hot3d:{name}", "proj": "pinhole",
                      "cam2world_slam": c2w},
            )


@DATASETS.register("hot3d_hand_coverage")
class HOT3DHandCoverageAdapter(HOT3DAdapter):
    """Deterministic non-official HOT3D split for coverage-aware hand metrics."""

    name = "hot3d_hand_coverage"
    capability = {HAND_COVERAGE}
    default_enabled = False
    segment_frames = 81

    def __init__(self, data_root: str):
        super().__init__(data_root)
        self.split_seed = int(os.environ.get("HAND_COVERAGE_SPLIT_SEED", "42"))
        self.test_segments = int(os.environ.get("HAND_COVERAGE_TEST_SEGMENTS", "437"))
        if self.test_segments < 0:
            raise ValueError("HAND_COVERAGE_TEST_SEGMENTS 必须 >= 0；0 表示使用全部片段")
        self._segment_cache = None

    def _selected_segments(self) -> list[tuple[str, int]]:
        if self._segment_cache is not None:
            return self._segment_cache
        if not self._converted_episodes():
            raise FileNotFoundError(
                "hot3d_hand_coverage 需要含 camera-frame 双手 MANO 的 HOT3D LeRobot 转换数据"
            )
        candidates = [
            (name, segment_index)
            for name in self._sequence_names()
            for segment_index in range(self._frame_count(name) // self.segment_frames)
        ]
        available = len(candidates)
        if self.test_segments and available > self.test_segments:
            def score(entry):
                payload = f"{self.split_seed}:{entry[0]}:{entry[1]}".encode("ascii")
                return hashlib.sha256(payload).digest()

            candidates = sorted(candidates, key=score)[:self.test_segments]
        self._segment_cache = sorted(candidates)
        return self._segment_cache

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
        if max_frames is not None and int(max_frames) < HOT3DHandCoverageAdapter.segment_frames:
            raise ValueError("相机系手部覆盖率指标固定使用完整 81 帧片段，--max-frames 不能小于 81")

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
        # Keep clips from the same video together so concurrent shards do not
        # redundantly decode the same episode or contend on its RGB cache.
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

    def _iter_protocol_entries(self, entries) -> Iterator[GTSequence]:
        from visualization.reproj_core import lerobot_io

        by_sequence = defaultdict(list)
        for name, segment_index in entries:
            by_sequence[name].append(int(segment_index))
        episodes = self._converted_episodes()
        total_available = sum(
            self._frame_count(name) // self.segment_frames for name in self._sequence_names()
        )
        for name in sorted(by_sequence):
            episode = episodes[name]
            segment_indices = sorted(by_sequence[name])
            max_end = (segment_indices[-1] + 1) * self.segment_frames
            labels_only = dict(episode)
            labels_only["video"] = None
            raw = lerobot_io.load_episode_raw(
                labels_only, max_frames=max_end, allow_missing_video=True,
            )
            if raw.get("hand_frame") != "camera":
                raise NotImplementedError(
                    f"{name} 的 hand_frame={raw.get('hand_frame')!r}，协议要求 camera"
                )
            if not all(side in raw["hands"] for side in ("left", "right")):
                raise NotImplementedError(f"{name} 缺少左右手完整 MANO GT")
            paths = self._video_frames(episode, max_end)
            mano_lr = np.stack([
                np.concatenate([
                    raw["hands"][side]["transl_cam"],
                    raw["hands"][side]["orient6d"],
                    raw["hands"][side]["pose6d"],
                    raw["hands"][side]["betas"],
                ], axis=-1)
                for side in ("left", "right")
            ], axis=1).astype(np.float32)
            valid_lr = np.asarray(raw["kept"], bool)
            valid_lr &= np.isfinite(mano_lr).all(axis=-1)
            for segment_index in segment_indices:
                start = segment_index * self.segment_frames
                end = start + self.segment_frames
                yield GTSequence(
                    seq_id=f"{name}#seg{segment_index:04d}",
                    image_paths=paths[start:end],
                    hw=(480, 480),
                    intrinsic=np.array(
                        [[207.8, 0.0, 239.8], [0.0, 207.8, 239.8], [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    ),
                    hand_mano_6d=mano_lr[start:end],
                    hand_valid_lr=valid_lr[start:end],
                    capability=self.capability,
                    meta={
                        "dataset": self.name,
                        "source_dataset": "hot3d",
                        "seq": name,
                        "segment_index": segment_index,
                        "prediction_group": f"{self.name}:{name}:{segment_index}",
                        "official_split": False,
                        "split_seed": self.split_seed,
                        "requested_segments": self.test_segments,
                        "available_segments": total_available,
                    },
                )

    def list_visual_sequences(self) -> list[dict]:
        return [
            {
                "seq_id": f"{name}#seg{segment_index:04d}",
                "label": f"{name} / segment {segment_index}",
                "frame_count": self.segment_frames,
            }
            for name, segment_index in self._selected_segments()
        ]

    def load_visual_sequence(self, seq_id: str, max_frames=None) -> dict:
        self._validate_max_frames(max_frames)
        match = re.fullmatch(r"(.+)#seg(\d+)", str(seq_id))
        if match is None:
            raise KeyError(f"无效 hand-coverage segment id: {seq_id!r}")
        entry = (match.group(1), int(match.group(2)))
        if entry not in set(self._selected_segments()):
            raise KeyError(f"非当前 hand-coverage split 片段: {seq_id!r}")
        name, segment_index = entry
        episode = self._converted_episodes()[name]
        start = segment_index * self.segment_frames
        end = start + self.segment_frames
        image_paths = self._write_rgb_cache_indices(name, episode, range(start, end))
        return {
            "seq_id": str(seq_id),
            "label": str(seq_id),
            "image_paths": image_paths,
            "hw": (480, 480),
            "frame_count": len(image_paths),
            "fps": 30.0,
            "source_path": str(episode["video"]),
        }
