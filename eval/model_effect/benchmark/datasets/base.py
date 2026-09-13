# -*- coding: utf-8 -*-
"""DatasetAdapter 基类:把某公开数据集解析成统一 GTSequence(逐序列)。

子类声明 name / root_rel(相对 data-root 的落盘子目录)/ capability,实现 iter_sequences。
加数据集只碰 datasets/;对齐/指标/入口不动。缺数据时 iter_sequences 抛 FileNotFoundError,
由 run 优雅降级成该数据集级 "skipped(缺数据)"。
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterator, Set, TypeVar

from ..core.schema import GTSequence


_T = TypeVar("_T")
FIXED_SOTA_SPLIT_VERSION = "sota-fixed-v1"
_FIXED_SPLIT_DIR = Path(__file__).resolve().parents[1] / "splits"


@lru_cache(maxsize=4)
def fixed_split_manifest(version: str = FIXED_SOTA_SPLIT_VERSION) -> dict:
    if version != FIXED_SOTA_SPLIT_VERSION:
        raise ValueError(f"未知固定 SOTA 数据清单版本: {version}")
    path = _FIXED_SPLIT_DIR / f"{version}.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"固定 SOTA 数据清单不存在: {path}") from exc
    datasets = manifest.get("datasets") or {}
    encoded = json.dumps(datasets, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    actual_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    if manifest.get("content_sha256") != actual_hash:
        raise ValueError(f"固定 SOTA 数据清单校验失败: {path}")
    return manifest


def fixed_split_items(
    items: list[_T], dataset: str, tier: str, item_id: Callable[[_T], str],
    version: str = FIXED_SOTA_SPLIT_VERSION,
) -> list[_T]:
    manifest = fixed_split_manifest(version)
    dataset_node = (manifest.get("datasets") or {}).get(dataset) or {}
    ids = ((dataset_node.get("tiers") or {}).get(tier))
    if not isinstance(ids, list):
        raise ValueError(f"固定清单 {version} 不含 {dataset}/{tier}")
    by_id = {str(item_id(item)): item for item in items}
    missing = [value for value in ids if value not in by_id]
    if missing:
        raise FileNotFoundError(
            f"固定清单 {version} 的 {dataset}/{tier} 缺 {len(missing)} 条本地数据，"
            f"首条为 {missing[0]}"
        )
    return [by_id[value] for value in ids]


def normalize_dataset_selection(selection) -> dict[str, dict]:
    """Validate and canonicalize per-dataset benchmark sampling options."""
    if selection in (None, ""):
        return {}
    if not isinstance(selection, dict):
        raise ValueError("dataset_selection 必须是对象")
    normalized = {}
    for dataset, raw in selection.items():
        if not isinstance(raw, dict):
            raise ValueError(f"dataset_selection[{dataset!r}] 必须是对象")
        options = {}
        sampling = str(raw.get("sampling") or "diverse").strip().lower()
        if sampling not in {"diverse", "all"}:
            raise ValueError(f"dataset_selection[{dataset!r}].sampling 仅支持 diverse/all")
        options["sampling"] = sampling
        for key in ("sample_count", "max_frames"):
            value = raw.get(key)
            if value not in (None, ""):
                value = int(value)
                if value <= 0:
                    raise ValueError(f"dataset_selection[{dataset!r}].{key} 必须 > 0")
                options[key] = value
        options["seed"] = int(raw.get("seed", 42))
        fixed_tier = raw.get("fixed_tier")
        if fixed_tier not in (None, ""):
            fixed_tier = str(fixed_tier).strip().lower()
            if fixed_tier not in {"minimum", "quarter", "half", "full"}:
                raise ValueError(f"dataset_selection[{dataset!r}].fixed_tier 无效")
            split_version = str(raw.get("split_version") or FIXED_SOTA_SPLIT_VERSION)
            manifest = fixed_split_manifest(split_version)
            dataset_node = (manifest.get("datasets") or {}).get(str(dataset)) or {}
            ids = ((dataset_node.get("tiers") or {}).get(fixed_tier))
            if not isinstance(ids, list):
                raise ValueError(f"固定清单 {split_version} 不含 {dataset}/{fixed_tier}")
            if "sample_count" in options and options["sample_count"] != len(ids):
                raise ValueError(
                    f"{dataset}/{fixed_tier} 固定为 {len(ids)} 条，不是 {options['sample_count']} 条"
                )
            options.update(
                fixed_tier=fixed_tier,
                split_version=split_version,
                split_hash=manifest["content_sha256"],
                sample_count=len(ids),
            )
        normalized[str(dataset)] = options
    return {name: normalized[name] for name in sorted(normalized)}


def deterministic_diverse_sample(
    items: list[_T], count: int | None, group_key: Callable[[_T], object], seed: int = 42,
) -> list[_T]:
    """Hash-rank within groups, then round-robin groups; preserve source order in output."""
    values = list(items)
    if count is None or int(count) >= len(values):
        return values
    count = max(0, int(count))

    def digest(kind: str, value: object) -> bytes:
        encoded = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
        return hashlib.sha256(f"{int(seed)}:{kind}:{encoded}".encode("utf-8")).digest()

    groups: dict[str, list[tuple[int, _T]]] = {}
    for index, item in enumerate(values):
        group = json.dumps(group_key(item), ensure_ascii=True, sort_keys=True, default=str)
        groups.setdefault(group, []).append((index, item))
    ordered_groups = sorted(groups, key=lambda group: (digest("group", group), group))
    queues = {
        group: sorted(groups[group], key=lambda pair: (digest("item", pair[1]), pair[0]))
        for group in ordered_groups
    }
    selected = []
    depth = 0
    while len(selected) < count:
        added = False
        for group in ordered_groups:
            queue = queues[group]
            if depth < len(queue):
                selected.append(queue[depth][0])
                added = True
                if len(selected) >= count:
                    break
        if not added:
            break
        depth += 1
    return [values[index] for index in sorted(selected)]


class DatasetAdapter:
    name: str = "base"
    root_rel: str = ""                 # 相对 data-root,如 "intrinsics/sintel"
    capability: Set[str] = set()
    implemented: bool = True           # 骨架数据集(iter_sequences 未实现)置 False,供能力清单标「待实现」
    default_enabled: bool = True       # 专项协议数据集可保留在能力面板，但不随 --datasets all 自动运行

    def __init__(self, data_root: str):
        self.root = os.path.join(data_root, self.root_rel)
        self.benchmark_selection: dict = {}

    def set_benchmark_selection(self, options: dict | None) -> None:
        self.benchmark_selection = dict(options or {})

    def benchmark_max_frames(self, fallback: int | None = None) -> int | None:
        value = self.benchmark_selection.get("max_frames", fallback)
        return None if value is None else int(value)

    def iter_sequences(self, max_seqs: int | None = None,
                       max_frames: int | None = None) -> Iterator[GTSequence]:
        raise NotImplementedError

    def iter_sequences_for_shard(
        self,
        shard_index: int,
        shard_count: int,
        max_seqs: int | None = None,
        max_frames: int | None = None,
        seq_start: int = 0,
        seq_end: int | None = None,
    ) -> Iterator[GTSequence] | None:
        """Optionally shard before expensive GT construction; None uses engine fallback."""
        return None

    def count_sequences_for_shard(
        self,
        shard_index: int,
        shard_count: int,
        max_seqs: int | None = None,
        max_frames: int | None = None,
        seq_start: int = 0,
        seq_end: int | None = None,
    ) -> int | None:
        """Return a cheap exact shard size to let the engine stream expensive inputs."""
        return None

    def list_visual_sequences(self) -> list[dict]:
        """Return inputs that can be opened in the Viewer without running the model.

        Dataset adapters with expensive GT construction should override this with a
        filesystem-only implementation.  The default keeps new lightweight adapters
        previewable without adding another required method.
        """
        return [
            {
                "seq_id": seq.seq_id,
                "label": seq.seq_id,
                "frame_count": seq.num_frames,
            }
            for seq in self.iter_sequences()
        ]

    def load_visual_sequence(self, seq_id: str, max_frames: int | None = None) -> dict:
        """Resolve one evaluation input to the image-sequence contract used by Viewer."""
        for seq in self.iter_sequences(max_frames=max_frames):
            if seq.seq_id == seq_id:
                return {
                    "seq_id": seq.seq_id,
                    "label": seq.seq_id,
                    "image_paths": list(seq.image_paths),
                    "hw": tuple(seq.hw),
                    "frame_count": seq.num_frames,
                    "source_path": str(Path(seq.image_paths[0]).parent),
                }
        raise KeyError(f"数据集 {self.name} 中不存在序列 {seq_id!r}")

    def count_sequences(self):
        """返回 (序列条数, 总帧数|None),供面板跑前显示规模(不加载模型)。
        默认走 iter_sequences 精确计数——会逐序列构造 GT,枚举成本高的数据集(需逐帧读标注/去畸变
        等)应**重写**为「只走目录、不加载 GT」的廉价版,n_frames 可用原图文件数估计或返回 None。
        缺数据时与 iter_sequences 一样抛 FileNotFoundError,由上层降级成「—(缺数据)」。"""
        n_seqs = 0
        n_frames = 0
        for seq in self.iter_sequences():
            n_seqs += 1
            try:
                n_frames += len(seq.image_paths)
            except Exception:                      # noqa: BLE001  个别序列取帧数失败不影响条数
                pass
        return n_seqs, n_frames
