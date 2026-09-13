#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LeRobot task-language and motion diversity analysis for the viewer."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .const import REPO_DIR

SCHEMA_VERSION = 3
COMPONENT_VERSION = 1
SCENE_TAXONOMY_VERSION = 2
DEFAULT_ROOT = (REPO_DIR / "output" / "scripts" / "data_processed" / "pipeline_lerobot"
                / "build_train_lerobot")
DATASETS = {
    "ego4d": {"label": "Ego4D", "color": "#58d5d2", "dirs": ("ego4d",)},
    "egodex": {"label": "EgoDex", "color": "#ff9e64", "dirs": ("egodex",)},
    "epickitchen": {"label": "EPIC-KITCHENS", "color": "#a1e26e",
                    "dirs": ("epickitchen", "epic-kitchen")},
}


class AnalysisCancelled(RuntimeError):
    pass


_SCENE_PATTERNS = [
    ("厨房/烹饪", r"\b(?:kitchen|food|cook\w*|bread|egg|sandwich|fruit|vegetable|onion|"
                    r"potato|tomato|garlic|ginger|rice|noodles?|meat|dough|flour|banana|"
                    r"croissant|candy|snack|seasoning|pepsi|beverage|soda|coffee|tea|milk|"
                    r"juice|knife|tongs?|cutting board|frying pan|pan|pot|bowl|plate|cup|"
                    r"mug|fork|spoon|spatula|ladle|kettle|faucet|sink|fridge|refrigerator|"
                    r"oven|stove|dishwasher|colander)\b"),
    ("清洁/洗涤", r"\b(?:clean\w*|wash\w*|wipe\w*|rins\w*|scrub\w*|sweep\w*|mop\w*|"
                   r"vacuum\w*|dust\w*|trash|garbage|sponge|detergent|soap|rags?|"
                   r"declutter\w*|tidy\w*|housework)\b"),
    ("电子/线缆/设备", r"\b(?:computer|laptop|keyboard|mouse|smartphone|phone|cellphone|"
                      r"tablet|usb|airpods?|earbuds?|headphones?|charger|charging|cables?|"
                      r"wires?|wire harness|adapter|plugs?|socket|power strips?|power cords?|"
                      r"device|printer|remote control|switch(?:es)?|battery|circuit|sensor|"
                      r"connector|dashboard)\b"),
    ("木工/装修/建材", r"\b(?:woodwork\w*|wood(?:en)? (?:planks?|strips?|posts?|boards?|"
                      r"railings?|beams?|pieces?)|planks?|lumber|timber|sandpaper|sanders?|"
                      r"sanding|orbital sander|routers?|miter saw|miter gauge|table saw|"
                      r"baseboards?|molding|wallpaper|ceilings?|walls?|flooring|beams?|"
                      r"stud finder|nail gun|construction|bricks?|mortar|trowel|cement|"
                      r"grout|tiles?|drywall|paint roller|measuring tape)\b"),
    ("车辆/驾驶", r"\b(?:cars?|automobile|vehicle|dashboard|steering wheel|ignition|"
                   r"windshield|seat ?belt|gear shift|hood|trunk|motorcycle|bicycle|bike|"
                   r"brake caliper|brake rotor)\b"),
    ("机械/维修/工具", r"\b(?:wrenches?|screwdrivers?|pliers?|hammers?|drills?|chainsaw|"
                      r"tools?|toolbox|tool chest|tool tray|tool cart|fasteners?|lubricant|"
                      r"mechanism|tires?|wheels?|lawnmower|mower|engine|repair\w*|nuts?|"
                      r"bolts?|clamps?|vise|screws?|unscrew\w*|metal pins?|guide pins?|"
                      r"rubber hoses?|pipes?)\b"),
    ("包装/拆封", r"\b(?:packages?|packaging|packets?|wrappers?|wrapping|plastic-wrapped|"
                   r"zip[ -]?lock|blister packs?|seals?|tape|cartons?|envelopes?|parcels?|"
                   r"adhesive|labels?)\b"),
    ("模型/结构组装", r"\b(?:assembl\w*|assembly pieces?|3d print\w*|interlocking|"
                     r"construction set|model parts?|vertical posts?|pillars?|pegs?|"
                     r"structural frames?|mounting brackets?|blocky pieces?)\b"),
    ("收纳/容器", r"\b(?:containers?|boxes?|cases?|bags?|baskets?|trays?|shelves?|shelf|"
                   r"cabinets?|drawers?|bins?|bottles?|jars?|buckets?|canisters?|lids?|"
                   r"caps?|pouches?|holders?|racks?|compartments?)\b"),
    ("纺织/衣物/洗衣", r"\b(?:fabric|cloth|clothing|laundry|iron(?:ing)?|drying rack|shirts?|"
                      r"pants|trousers|shorts|socks?|shoes?|shoelaces?|jackets?|gloves?|"
                      r"towels?|blankets?|bedsheets?|bed sheets?|garments?|dresses?|"
                      r"placemats?|folded clothes)\b"),
    ("手工/编织/饰品", r"\b(?:crochet\w*|yarn|knit\w*|sew\w*|stitch\w*|beads?|beaded|"
                     r"necklace|bracelet|jewelry|ribbon|rubber bands?|origami|craft\w*|"
                     r"sculpt\w*|clay|putty|glue gun|thread|string|loom)\b"),
    ("绘画/艺术", r"\b(?:canvas|paint\w*|draw\w*|color\w*|watercolor|markers?|crayons?|"
                   r"easel|palette|sketch\w*|stamp\w*)\b"),
    ("办公/文具/阅读", r"\b(?:books?|notebooks?|paper|pens?|pencils?|documents?|desk|"
                     r"paperclips?|stapler|folders?|binders?|rulers?|erasers?|scissors|"
                     r"clipboard)\b"),
    ("益智/桌游", r"\b(?:puzzles?|jigsaw|lego|blocks?|domino(?:es)?|jenga|chess|chessboard|"
                   r"cards?|dice|mancala|reversi|board games?|game board|game pieces?|"
                   r"playing pieces?|marbles?|target pits?)\b"),
    ("玩具/模型", r"\b(?:toys?|dolls?|figurines?|slime|spinners?|rubber ducks?|model parts?|"
                   r"miniature|toy fruit|plastic banana|plastic grapes)\b"),
    ("个人护理/穿戴", r"\b(?:makeup|lipstick|mascara|hair|comb\w*|braid\w*|wig|toothbrush|"
                       r"toothpaste|shav\w*|skin care|face cream|eyeglasses|glasses case|"
                       r"scrunchie)\b"),
    ("家居/家具/床品", r"\b(?:beds?|mattress|bedsheet|bed sheet|pillows?|curtains?|sofas?|"
                     r"couches?|chairs?|furniture|doors?|windows?|closet|lamp|carpet|rug|"
                     r"room)\b"),
    ("园艺/户外", r"\b(?:garden\w*|plant\w*|flower\w*|soil|branch\w*|leaves|leaf|seed\w*|"
                   r"lawn|outdoor|tree|grass|watering can)\b"),
    ("运动/健身", r"\b(?:sport\w*|exercise|workout|fitness|dumbbell|tennis|basketball|"
                   r"football|soccer|baseball|golf|yoga)\b"),
]
_ACTION_PATTERNS = [
    ("拿取/举起", r"\b(?:pick\w* up|take\w* (?:out|up)|lift\w*)\b"),
    ("放置/放下", r"\b(?:plac\w*|put\w*|set\w* down)\b"),
    ("移动/搬运", r"\b(?:mov\w*|transfer\w*|carr\w*|bring\w*|brought)\b"),
    ("伸手/触碰", r"\b(?:reach\w*|touch\w*|extend\w* (?:the )?(?:left |right )?hand)\b"),
    ("抓握/持握", r"\b(?:grasp\w*|grab\w*|hold\w*|held)\b"),
    ("开合", r"\b(?:open\w*|clos\w*|shut\w*)\b"),
    ("插入/移除", r"\b(?:insert\w*|remov\w*|extract\w*|take\w* off)\b"),
    ("切削/剥皮", r"\b(?:cut\w*|slic\w*|chop\w*|peel\w*|trim\w*|carv\w*)\b"),
    ("清洁/擦洗", r"\b(?:clean\w*|wash\w*|wip\w*|rins\w*|scrub\w*|sweep\w*|mop\w*)\b"),
    ("倾倒/装填", r"\b(?:pour\w*|fill\w*|drain\w*|dump\w*)\b"),
    ("搅拌/混合", r"\b(?:stir\w*|mix\w*|whisk\w*|knead\w*|blend\w*)\b"),
    ("按压/推动", r"\b(?:press\w*|push\w*|squeez\w*)\b"),
    ("拉动", r"\b(?:pull\w*|drag\w*)\b"),
    ("旋转/翻转", r"\b(?:rotat\w*|turn\w*|twist\w*|flip\w*)\b"),
    ("调整/整理", r"\b(?:adjust\w*|align\w*|arrang\w*|organiz\w*|tidy\w*|straighten\w*)\b"),
    ("组装/连接", r"\b(?:assembl\w*|build\w*|built|connect\w*|attach\w*)\b"),
    ("拆卸/分离", r"\b(?:disassembl\w*|detach\w*|separat\w*|unscrew\w*)\b"),
    ("折叠/包裹", r"\b(?:fold\w*|unfold\w*|wrap\w*)\b"),
    ("绘画/书写", r"\b(?:paint\w*|draw\w*|drew|writ\w*|wrote|color\w*)\b"),
    ("工具操作", r"\b(?:hammer\w*|screw\w*|drill\w*|saw\w*|brush\w*|comb\w*)\b"),
    ("进食/饮用", r"\b(?:eat\w*|ate|drink\w*|drank|sip\w*|bit\w*)\b"),
]
_SCENE_RE = re.compile("|".join(f"(?P<s{index}>{pattern})"
                                  for index, (_, pattern) in enumerate(_SCENE_PATTERNS)))
_ACTION_RE = re.compile("|".join(f"(?P<a{i}>{pattern})"
                                   for i, (_, pattern) in enumerate(_ACTION_PATTERNS)))
_TOKEN_RE = re.compile(r"[a-z][a-z'-]+")
_NORMALIZE_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = set("""
the a an and or to from with using use used into onto in on at of for by then while after
before it its this that these those is are be both left right hand hands side top bottom upper
lower middle center front back one two another repeatedly continuously corresponding each
place put move pick take hold grasp reach open close remove insert clean wash wipe cut turn
press push pull adjust organize object objects thing piece pieces item items
""".split())


def _cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise AnalysisCancelled("多样性分析已取消")


def _english_task(value) -> str:
    text = str(value)
    if "英文:" in text:
        text = text.split("英文:", 1)[1]
    return " ".join(text.lower().split())


def _scene_category(text: str, dataset: str) -> str:
    if dataset == "epickitchen":
        return "厨房/烹饪"
    matches = [set() for _ in _SCENE_PATTERNS]
    for match in _SCENE_RE.finditer(text):
        matches[int(match.lastgroup[1:])].add(match.group(0))
    best_index = max(range(len(matches)), key=lambda index: (len(matches[index]), -index))
    return (_SCENE_PATTERNS[best_index][0]
            if matches[best_index] else "其他")


def _entropy(values: dict[str, int]) -> dict:
    counts = np.asarray([value for value in values.values() if value > 0], np.float64)
    if not len(counts):
        return {"shannon": 0.0, "normalized": 0.0, "effective_categories": 0.0}
    probs = counts / counts.sum()
    shannon = float(-(probs * np.log(probs)).sum())
    return {
        "shannon": shannon,
        "normalized": float(shannon / math.log(len(probs))) if len(probs) > 1 else 0.0,
        "effective_categories": float(math.exp(shannon)),
    }


def _quantiles(values) -> dict:
    array = np.asarray(values, np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"n": 0, "p10": None, "p50": None, "p90": None, "p99": None}
    quantile = np.quantile(array, [.1, .5, .9, .99])
    return {"n": int(len(array)), "p10": float(quantile[0]), "p50": float(quantile[1]),
            "p90": float(quantile[2]), "p99": float(quantile[3])}


def _distribution(values: dict[str, int], denominator: int | None = None) -> list[dict]:
    total = int(denominator if denominator is not None else sum(values.values()))
    return [{"label": label, "count": int(count),
             "percent": 100.0 * count / max(total, 1)}
            for label, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))]


def _selected_files(files: list[Path], count: int) -> list[Path]:
    if len(files) <= count:
        return files
    indices = np.linspace(0, len(files) - 1, count, dtype=np.int64)
    return [files[int(index)] for index in indices]


def _signature(paths: list[Path], scope: str) -> str:
    digest = hashlib.sha256(f"{COMPONENT_VERSION}:{scope}".encode())
    for path in paths:
        stat = path.stat()
        digest.update(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def _read_component(path: Path, signature: str):
    if not path.is_file():
        return None
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved.get("signature") == signature:
            return saved.get("value")
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _write_component(path: Path, signature: str, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temp.write_text(json.dumps({"signature": signature, "value": value},
                               ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _dataset_name_from_path(path: Path) -> str | None:
    for part in (path, *path.parents):
        dirname = part.name.casefold()
        for name, config in DATASETS.items():
            if dirname in config["dirs"]:
                return name
    return None


def discover_datasets(input_root: Path, dataset_names: list[str]) -> list[dict]:
    root = Path(input_root).expanduser().resolve()
    unknown = [name for name in dataset_names if name not in DATASETS]
    if unknown:
        raise ValueError(f"未知数据集: {', '.join(unknown)}")
    if not dataset_names:
        raise ValueError("请至少选择一个数据集")
    collection_roots = [
        root,
        root / "pipeline_lerobot" / "build_train_lerobot",
        root / "build_train_lerobot",
        root / "output" / "scripts" / "data_processed" / "pipeline_lerobot"
        / "build_train_lerobot",
    ]
    discovered = []
    for name in dataset_names:
        config = DATASETS[name]
        candidates = []
        if root.name in config["dirs"]:
            candidates.extend((root / "lerobot_v3", root))
        elif root.name == "lerobot_v3" and root.parent.name in config["dirs"]:
            candidates.append(root)
        elif len(dataset_names) == 1 and (root / "meta" / "info.json").is_file() \
                and _dataset_name_from_path(root) in {None, name}:
            candidates.append(root)
        for collection_root in collection_roots:
            for dirname in config["dirs"]:
                candidates.extend((collection_root / dirname / "lerobot_v3",
                                   collection_root / dirname))
        candidates = list(dict.fromkeys(candidates))
        dataset_root = next((path for path in candidates
                             if (path / "meta" / "info.json").is_file()), None)
        if dataset_root is None:
            tried = ", ".join(str(path) for path in candidates[:4])
            raise FileNotFoundError(f"{config['label']} 不存在；尝试过: {tried}")
        task_path = dataset_root / "meta" / "tasks.parquet"
        if not task_path.is_file():
            raise FileNotFoundError(f"缺少自然语言任务标签: {task_path}")
        data_files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"没有轨迹 parquet: {dataset_root / 'data'}")
        discovered.append({"name": name, "label": config["label"], "color": config["color"],
                           "root": dataset_root, "task_path": task_path,
                           "data_files": data_files})
    return discovered


def discover_input_paths(input_roots: list[Path]) -> list[dict]:
    """Resolve one or more selected paths and infer their supported dataset names."""
    if not input_roots:
        raise ValueError("请至少添加一个数据集目录")
    discovered_by_name = {}
    for raw_root in input_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"目录不存在或不可访问: {root}")
        direct_info = root / "meta" / "info.json"
        if direct_info.is_file():
            name = _dataset_name_from_path(root)
            if name is None:
                raise ValueError(f"无法从路径识别数据集名称，请选择 ego4d、egodex 或 epickitchen 目录: {root}")
            candidates = discover_datasets(root, [name])
        else:
            candidates = []
            for name in DATASETS:
                try:
                    candidates.extend(discover_datasets(root, [name]))
                except FileNotFoundError as exc:
                    if "不存在；尝试过:" not in str(exc):
                        raise
            if not candidates:
                raise FileNotFoundError(
                    f"所选目录下没有 Ego4D、EgoDex 或 EPIC-KITCHENS LeRobot 数据集: {root}")
        for dataset in candidates:
            previous = discovered_by_name.get(dataset["name"])
            if previous is not None and previous["root"] != dataset["root"]:
                raise ValueError(
                    f"{dataset['label']} 被选择了两次且路径不同: {previous['root']} / {dataset['root']}")
            discovered_by_name[dataset["name"]] = dataset
    return list(discovered_by_name.values())


def analysis_cache_key(datasets: list[dict], sample_files: int) -> str:
    paths = []
    names = []
    for dataset in datasets:
        selected = _selected_files(dataset["data_files"], sample_files)
        paths.extend([dataset["root"] / "meta" / "info.json", dataset["task_path"], *selected])
        names.append(dataset["name"])
    return _signature(paths, f"report:{','.join(names)}:{sample_files}:schema{SCHEMA_VERSION}")


def _task_text_stats(dataset: dict, cancel_event=None, progress_callback=None) -> dict:
    import pyarrow.parquet as pq

    labels = pq.read_table(dataset["task_path"])["__index_level_0__"].to_pylist()
    scenes, actions, tokens = Counter(), Counter(), Counter()
    normalized_hashes = set()
    action_covered = 0
    token_stride = max(1, math.ceil(len(labels) / 200_000))
    token_samples = 0
    for index, raw in enumerate(labels):
        if index % 16384 == 0:
            _cancelled(cancel_event)
            if progress_callback is not None:
                progress_callback({"stage": "diversity_labels", "current": dataset["label"],
                                   "label_done": index, "label_total": len(labels)})
        text = _english_task(raw)
        normalized_hashes.add(hash(_NORMALIZE_RE.sub(" ", text).strip()))
        scenes[_scene_category(text, dataset["name"])] += 1
        found = {int(match.lastgroup[1:]) for match in _ACTION_RE.finditer(text)}
        action_covered += int(bool(found))
        for action_index in found:
            actions[_ACTION_PATTERNS[action_index][0]] += 1
        if index % token_stride == 0:
            token_samples += 1
            for token in _TOKEN_RE.findall(text):
                if len(token) > 2 and token not in _STOPWORDS:
                    tokens[token] += 1
    total = len(labels)
    return {
        "task_ids": total, "normalized_unique": len(normalized_hashes),
        "normalized_unique_ratio": len(normalized_hashes) / max(total, 1),
        "scene_counts": dict(scenes), "scene_entropy": _entropy(dict(scenes)),
        "action_counts": dict(actions), "action_coverage": action_covered / max(total, 1),
        "top_entity_terms": [{"term": term, "count": count}
                             for term, count in tokens.most_common(16)],
        "entity_term_sample_labels": token_samples,
    }


def _fixed_list_numpy(column, width: int, dtype=np.float64) -> np.ndarray:
    array = column.combine_chunks()
    return np.asarray(array.values, dtype=dtype).reshape(-1, width)


def _motion_stats(dataset: dict, sample_files: int, fps: float,
                  cancel_event=None, progress_callback=None) -> dict:
    import pyarrow.parquet as pq

    selected = _selected_files(dataset["data_files"], sample_files)
    spans, net, paths, head_sweep, durations, hand_spans = [], [], [], [], [], []
    hand_usage = Counter()
    sampled_frames = 0
    for file_index, path in enumerate(selected):
        _cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback({"stage": "diversity_motion", "current": str(path),
                               "motion_done": file_index, "motion_total": len(selected)})
        schema = set(pq.read_schema(path).names)
        kept_column = "hand_kept" if "hand_kept" in schema else "state_mask"
        columns = ["episode_index", "cam_trans", "cam_quat", kept_column]
        if "left_mano_transl_cam" in schema and "right_mano_transl_cam" in schema:
            columns.extend(["left_mano_transl_cam", "right_mano_transl_cam"])
        table = pq.read_table(path, columns=columns)
        episode = np.asarray(table["episode_index"], dtype=np.int64)
        cam = _fixed_list_numpy(table["cam_trans"], 3)
        quat = _fixed_list_numpy(table["cam_quat"], 4)
        kept = _fixed_list_numpy(table[kept_column], 2, dtype=bool)
        left = (_fixed_list_numpy(table["left_mano_transl_cam"], 3)
                if "left_mano_transl_cam" in table.column_names else None)
        right = (_fixed_list_numpy(table["right_mano_transl_cam"], 3)
                 if "right_mano_transl_cam" in table.column_names else None)
        sampled_frames += len(episode)
        hand_usage["双手"] += int(np.sum(kept[:, 0] & kept[:, 1]))
        hand_usage["仅左手"] += int(np.sum(kept[:, 0] & ~kept[:, 1]))
        hand_usage["仅右手"] += int(np.sum(~kept[:, 0] & kept[:, 1]))
        hand_usage["无有效手"] += int(np.sum(~kept[:, 0] & ~kept[:, 1]))
        boundaries = np.r_[0, np.flatnonzero(episode[1:] != episode[:-1]) + 1, len(episode)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            xyz = cam[start:end]
            finite = np.isfinite(xyz).all(1)
            xyz = xyz[finite]
            if len(xyz) < 2:
                continue
            spans.append(float(np.linalg.norm(np.ptp(xyz, axis=0))))
            net.append(float(np.linalg.norm(xyz[-1] - xyz[0])))
            one_hz = xyz[::max(1, int(round(fps)))]
            if not np.array_equal(one_hz[-1], xyz[-1]):
                one_hz = np.vstack((one_hz, xyz[-1]))
            paths.append(float(np.linalg.norm(np.diff(one_hz, axis=0), axis=1).sum()))
            durations.append((end - start) / fps)
            q = quat[start:end][finite]
            norm = np.linalg.norm(q, axis=1, keepdims=True)
            valid_q = np.isfinite(q).all(1) & (norm[:, 0] > 1e-8)
            q = q[valid_q] / norm[valid_q]
            if len(q):
                dots = np.clip(np.abs(q @ q[0]), 0, 1)
                head_sweep.append(float(np.degrees(2 * np.arccos(dots)).max()))
            if left is not None:
                per_hand = []
                for points, valid in ((left[start:end], kept[start:end, 0]),
                                      (right[start:end], kept[start:end, 1])):
                    valid = valid & np.isfinite(points).all(1)
                    if valid.sum() >= 2:
                        per_hand.append(float(np.linalg.norm(np.ptp(points[valid], axis=0))))
                if per_hand:
                    hand_spans.append(max(per_hand))
    activity = Counter()
    for value in spans:
        if value < .15:
            activity["局部 <0.15m"] += 1
        elif value < .5:
            activity["近距 0.15-0.5m"] += 1
        elif value < 1.5:
            activity["中等 0.5-1.5m"] += 1
        else:
            activity["大范围 ≥1.5m"] += 1
    return {
        "available_files": len(dataset["data_files"]), "sampled_files": len(selected),
        "sampled_frames": sampled_frames, "sampled_episodes": len(spans),
        "camera_span_m": _quantiles(spans), "net_displacement_m": _quantiles(net),
        "path_length_1hz_m": _quantiles(paths), "head_sweep_deg": _quantiles(head_sweep),
        "hand_workspace_m": _quantiles(hand_spans), "episode_duration_s": _quantiles(durations),
        "activity_counts": dict(activity), "hand_usage_counts": dict(hand_usage),
    }


def _component(dataset: dict, sample_files: int, component_cache_root: Path,
               cancel_event=None, progress_callback=None, refresh: bool = False) -> dict:
    info = json.loads((dataset["root"] / "meta" / "info.json").read_text(encoding="utf-8"))
    text_signature = _signature(
        [dataset["task_path"]], f"{dataset['name']}:text:taxonomy{SCENE_TAXONOMY_VERSION}")
    text_path = component_cache_root / f"{dataset['name']}_text.json"
    text = None if refresh else _read_component(text_path, text_signature)
    if text is None:
        text = _task_text_stats(dataset, cancel_event, progress_callback)
        _write_component(text_path, text_signature, text)
    elif "未归类/场景不明" in text.get("scene_counts", {}):
        counts = text["scene_counts"]
        counts["其他"] = counts.get("其他", 0) + counts.pop("未归类/场景不明")
        text["scene_entropy"] = _entropy(counts)
        _write_component(text_path, text_signature, text)
    selected = _selected_files(dataset["data_files"], sample_files)
    motion_signature = _signature(selected, f"{dataset['name']}:motion:{sample_files}")
    motion_path = component_cache_root / f"{dataset['name']}_motion_{sample_files}.json"
    motion = None if refresh else _read_component(motion_path, motion_signature)
    if motion is None:
        motion = _motion_stats(dataset, sample_files, float(info["fps"]),
                               cancel_event, progress_callback)
        _write_component(motion_path, motion_signature, motion)
    return {
        "dataset": dataset["name"], "label": dataset["label"], "color": dataset["color"],
        "root": str(dataset["root"]), "episodes": int(info["total_episodes"]),
        "frames": int(info["total_frames"]), "videos": int(info["total_videos"]),
        "fps": float(info["fps"]), "hours": float(info["total_frames"] / info["fps"] / 3600),
        "mean_episode_s": float(info["total_frames"] / info["fps"] / info["total_episodes"]),
        "text": text, "motion": motion,
    }


def analyze(input_root: Path, dataset_names: list[str], sample_files: int,
            component_cache_root: Path, cancel_event=None, progress_callback=None,
            datasets: list[dict] | None = None, cache_key: str | None = None,
            refresh: bool = False) -> dict:
    sample_files = max(1, min(int(sample_files), 96))
    datasets = datasets or discover_datasets(input_root, dataset_names)
    cache_key = cache_key or analysis_cache_key(datasets, sample_files)
    results = []
    total_phases = len(datasets) * 2
    for index, dataset in enumerate(datasets):
        _cancelled(cancel_event)
        if progress_callback is not None:
            progress_callback({"stage": "diversity_labels", "current": dataset["label"],
                               "done": index * 2, "total": total_phases})
        result = _component(dataset, sample_files, component_cache_root,
                            cancel_event, progress_callback, refresh)
        results.append(result)
        if progress_callback is not None:
            progress_callback({"stage": "diversity_motion", "current": dataset["label"],
                               "done": index * 2 + 2, "total": total_phases})
    totals = {key: sum(item[key] for item in results)
              for key in ("episodes", "frames", "videos", "hours")}
    totals["task_ids"] = sum(item["text"]["task_ids"] for item in results)
    totals["normalized_unique"] = sum(item["text"]["normalized_unique"] for item in results)
    scene_counts, action_counts, activity_counts, hand_counts = Counter(), Counter(), Counter(), Counter()
    for item in results:
        scene_counts.update(item["text"]["scene_counts"])
        action_counts.update(item["text"]["action_counts"])
        activity_counts.update(item["motion"]["activity_counts"])
        hand_counts.update(item["motion"]["hand_usage_counts"])
    return {
        "schema_version": SCHEMA_VERSION, "analysis_type": "diversity",
        "cache_key": cache_key, "generated_at": time.time(), "input_root": str(input_root),
        "selected_datasets": dataset_names, "sample_files": sample_files,
        "overview": {**totals, "datasets": len(results),
                     "scene_kinds": len(scene_counts), "action_kinds": len(action_counts)},
        "datasets": results,
        "distributions": {
            "dataset_frames": [{"label": item["label"], "count": item["frames"],
                                "percent": 100.0 * item["frames"] / max(totals["frames"], 1)}
                               for item in results],
            "scenes": _distribution(dict(scene_counts), totals["task_ids"]),
            "actions": _distribution(dict(action_counts), totals["task_ids"]),
            "activity": _distribution(dict(activity_counts)),
            "hand_usage": _distribution(dict(hand_counts)),
        },
        "methodology": {
            "scene": "自然语言任务标签按细粒度关键词得分归入任务语义域；未命中项归入其他。EPIC-KITCHENS 固定为厨房，不是视觉场景真值。",
            "action": "一个描述可命中多个动作族，因此动作百分比之和可以超过 100%。",
            "motion": "每套数据均匀抽取 data parquet，按 episode 统计 cam_trans、cam_quat 与手腕工作区。",
            "unique": "自然语言仅归一大小写、标点和空白；跨数据集不做语义去重。",
        },
    }


def write_report(report: dict, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "result.json"
    csv_path = output_dir / "datasets.csv"
    summary_path = output_dir / "summary.txt"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["dataset", "episodes", "frames", "hours", "videos", "task_ids",
              "normalized_unique", "action_coverage", "scene_effective_categories",
              "sampled_episodes", "camera_span_p50_m", "camera_span_p90_m",
              "head_sweep_p50_deg", "hand_workspace_p50_m"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in report.get("datasets") or []:
            text, motion = item["text"], item["motion"]
            writer.writerow({
                "dataset": item["dataset"], "episodes": item["episodes"],
                "frames": item["frames"], "hours": item["hours"], "videos": item["videos"],
                "task_ids": text["task_ids"], "normalized_unique": text["normalized_unique"],
                "action_coverage": text["action_coverage"],
                "scene_effective_categories": text["scene_entropy"]["effective_categories"],
                "sampled_episodes": motion["sampled_episodes"],
                "camera_span_p50_m": motion["camera_span_m"]["p50"],
                "camera_span_p90_m": motion["camera_span_m"]["p90"],
                "head_sweep_p50_deg": motion["head_sweep_deg"]["p50"],
                "hand_workspace_p50_m": motion["hand_workspace_m"]["p50"],
            })
    overview = report.get("overview") or {}
    lines = [
        "LeRobot 数据多样性分析",
        f"数据集: {', '.join(report.get('selected_datasets') or [])}",
        f"总 episode: {overview.get('episodes', 0):,}",
        f"总帧数: {overview.get('frames', 0):,}",
        f"总时长: {overview.get('hours', 0):.3f} h",
        f"任务 ID: {overview.get('task_ids', 0):,}",
        f"语义场景 / 动作族: {overview.get('scene_kinds', 0)} / {overview.get('action_kinds', 0)}",
        "",
    ]
    for item in report.get("datasets") or []:
        span = item["motion"]["camera_span_m"]
        lines.append(f"{item['label']}: {item['hours']:.2f} h, "
                     f"相机跨度 P50/P90={span['p50']:.3f}/{span['p90']:.3f} m")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "summary": summary_path}
