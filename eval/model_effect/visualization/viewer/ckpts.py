#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_train 下 ckpt 的发现 / 逐级浏览 / 短标签 / 所属 run config 快照解析。

run 层级不固定：老结构 <gpu>/<ts>/step_*（2 层），新结构 <gpu>/<ts>/<task>/step_*（3 层）。
不写死层数，递归找到任何直接含 step_* 的目录即认作一个 run，用相对 model_train 的 posix 路径当 run 名。
"""
from __future__ import annotations

import re
from pathlib import Path

from .const import (DEFAULT_CHECKPOINT, DEFAULT_CHECKPOINT_RUN,
                    MODEL_TRAIN_ROOT, REPO_DIR, VIDEO_EXTS)


CHECKPOINT_FILE_EXTS = {".safetensors", ".bin", ".pt", ".pth", ".ckpt"}
CHECKPOINT_PATTERNS = (
    "model.safetensors", "pytorch_model.bin", "*.safetensors",
    "pytorch_model*.bin", "*.pt", "*.pth", "*.ckpt",
)


def checkpoint_weight_file(path: str | Path) -> Path | None:
    """Return the weight file represented by a checkpoint file or directory."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate if candidate.suffix.lower() in CHECKPOINT_FILE_EXTS else None
    if not candidate.is_dir():
        return None
    for pattern in CHECKPOINT_PATTERNS:
        hits = sorted(item for item in candidate.glob(pattern) if item.is_file())
        if hits:
            return hits[0]
    return None


def resolve_checkpoint_path(value: str | Path | None) -> Path | None:
    """Resolve a user-selected checkpoint anywhere on the server filesystem."""
    if value is None or not str(value).strip():
        return None
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_DIR / candidate
    try:
        candidate = candidate.resolve()
    except OSError:
        return None
    return candidate if checkpoint_weight_file(candidate) is not None else None


def run_step_for_path(value: str | Path | None) -> tuple[str | None, str | None]:
    """Map known project checkpoints back to the legacy Run/Step selectors."""
    if value is None:
        return None, None
    path = Path(value).expanduser().resolve()
    if path == DEFAULT_CHECKPOINT.resolve():
        return DEFAULT_CHECKPOINT_RUN, DEFAULT_CHECKPOINT.name
    step_dir = path.parent if path.is_file() else path
    if not step_dir.name.startswith("step_"):
        return None, None
    try:
        run = step_dir.parent.relative_to(MODEL_TRAIN_ROOT.resolve()).as_posix()
    except ValueError:
        return None, None
    return run, step_dir.name


def list_runs() -> list[str]:
    """model_train 下所有「含 step_* 子目录」的目录，返回其相对 model_train 的 posix 路径（新→旧）。"""
    runs = set()
    if MODEL_TRAIN_ROOT.is_dir():
        for step_dir in MODEL_TRAIN_ROOT.rglob("step_*"):
            if step_dir.is_dir():
                runs.add(step_dir.parent.relative_to(MODEL_TRAIN_ROOT).as_posix())
    if DEFAULT_CHECKPOINT.is_file():
        runs.add(DEFAULT_CHECKPOINT_RUN)
    return sorted(runs, reverse=True)


def list_log_runs() -> list[str]:
    """列出可供 log_diff 选择的 run；只要求其 ``logs/`` 下存在 ``node*.log``。"""
    if not MODEL_TRAIN_ROOT.is_dir():
        return []
    runs = {
        log_file.parent.parent.relative_to(MODEL_TRAIN_ROOT).as_posix()
        for log_file in MODEL_TRAIN_ROOT.rglob("node*.log")
        if log_file.is_file() and log_file.parent.name == "logs"
    }
    return sorted(runs, reverse=True)


def list_steps(run: str) -> list[str]:
    """某 run 下的 step_* 目录名（按 step 升序）。"""
    if run == DEFAULT_CHECKPOINT_RUN:
        return [DEFAULT_CHECKPOINT.name] if DEFAULT_CHECKPOINT.is_file() else []
    rd = MODEL_TRAIN_ROOT / run
    if not rd.is_dir():
        return []
    return sorted(d.name for d in rd.iterdir() if d.is_dir() and d.name.startswith("step_"))


def browse(rel: str) -> dict:
    """逐级浏览 model_train:列出 rel(相对 model_train 的 posix 路径,''=根)下的直接子目录。

    返回 {rel, dirs:[子目录名...], is_run:bool, steps:[...]}。is_run=该目录直接含 step_*（到达 run 层,
    前端应停止下钻、改选 step）。防目录穿越:解析后必须仍在 model_train 内。"""
    if rel == DEFAULT_CHECKPOINT_RUN:
        steps = list_steps(rel)
        return {"rel": rel, "dirs": [], "is_run": bool(steps), "steps": steps}

    root = MODEL_TRAIN_ROOT.resolve()
    d = (MODEL_TRAIN_ROOT / rel).resolve() if rel else root
    if d != root and root not in d.parents:      # 穿越/非法路径 → 视作根
        d, rel = root, ""
    if not d.is_dir():
        dirs = ([DEFAULT_CHECKPOINT_RUN]
                if not rel and DEFAULT_CHECKPOINT.is_file() else [])
        return {"rel": rel, "dirs": dirs, "is_run": False}
    subs = sorted((x.name for x in d.iterdir() if x.is_dir()), reverse=True)
    if not rel and DEFAULT_CHECKPOINT.is_file() and DEFAULT_CHECKPOINT_RUN not in subs:
        subs.insert(0, DEFAULT_CHECKPOINT_RUN)
    is_run = any(n.startswith("step_") for n in subs)
    # run 层时不把 step_* 当作可下钻子目录返回（step 作为叶子由 steps 单独给前端展示为可选 ckpt）
    dirs = [] if is_run else subs
    return {"rel": rel, "dirs": dirs, "is_run": is_run, "steps": list_steps(rel) if is_run else []}


def innermost(root: Path) -> str:
    """--no_truth 文件浏览默认起点：从 root 沿「单一子目录且当前层无视频」链下探到「最里层」，
    返回相对 root 的 posix 路径（root 本身直接有视频/有分叉则返回 ''）。避免用户从空壳目录一层层点进去。"""
    root = Path(root).resolve()
    cur = root
    for _ in range(64):                       # 防环/防超深，上限 64 层
        try:
            entries = list(cur.iterdir())
        except OSError:
            break
        subs = [x for x in entries if x.is_dir()]
        vids = [x for x in entries if x.is_file() and x.suffix.lower() in VIDEO_EXTS]
        if vids or len(subs) != 1:            # 当前层已有视频，或不是单链 → 停在此
            break
        cur = subs[0]
    return cur.relative_to(root).as_posix() if cur != root else ""


def resolve_ckpt(run: str, step: str) -> Path | None:
    """(run, step) → 校验后的 ckpt 目录；越界 / 不存在返回 None。"""
    if not run or not step:
        return None
    if run == DEFAULT_CHECKPOINT_RUN:
        if step == DEFAULT_CHECKPOINT.name and DEFAULT_CHECKPOINT.is_file():
            return DEFAULT_CHECKPOINT.resolve()
        return None
    ckpt = (MODEL_TRAIN_ROOT / run / step).resolve()
    root = MODEL_TRAIN_ROOT.resolve()
    if root not in ckpt.parents or not ckpt.is_dir():   # 防目录穿越
        return None
    return ckpt


def ckpt_tag(ckpt: str | None) -> str:
    """ckpt → 缓存/文件名用短标签；无 ckpt（smoke）用 'smoke'。

    用 ckpt 相对 model_train 的完整 run 路径 + step 生成，避免不同 run 撞名。路径分隔符统一转 '_'。"""
    if not ckpt:
        return "smoke"
    p = Path(ckpt).resolve()
    try:
        rel = p.relative_to(MODEL_TRAIN_ROOT.resolve()).as_posix()   # <run...>/step_*
    except ValueError:
        rel = f"{p.parent.name}/{p.name}"                            # 非 model_train 下的 ckpt 回退
    return re.sub(r"[^A-Za-z0-9_.-]", "_", rel)


def auto_pick_ckpt() -> str | None:
    """Pick the latest training step, then fall back to the downloaded public checkpoint."""
    for run in (item for item in list_runs() if item != DEFAULT_CHECKPOINT_RUN):
        steps = list_steps(run)
        if steps:
            return str(MODEL_TRAIN_ROOT / run / steps[-1])
    if DEFAULT_CHECKPOINT.is_file():
        return str(DEFAULT_CHECKPOINT)
    return None


def config_for_ckpt(ckpt: str | None, fallback: str) -> str:
    """优先用该 ckpt 所属 run 自带的 config 快照，保证模型/loss/data 与训练一致。

    新训练脚本把解析后的真实配置写到 <run>/logs/record/config.yaml；老 run 可能仍是
    <run>/config/config.yaml。两者都没有（或没给 ckpt）时才回退到 fallback（命令行/默认模板）。
    """
    if ckpt:
        p = Path(ckpt).resolve()
        if p.is_file():                         # ckpt 可直接传权重文件，先回到 step_* 目录
            p = p.parent
        run = p.parent if p.name.startswith("step_") else p
        for snap in (run / "logs" / "record" / "config.yaml",
                     run / "config" / "config.yaml"):
            if snap.is_file():
                return str(snap)
    return fallback


def load_loss_cfg(cfg_path: str) -> dict:
    """从 config 读逐帧 loss 面板要用的段：{loss, model, data}。

    · loss  —— 按输出归属的模块权重 + terms（面板直接复用训练 Criterion）。
    · model —— 头开关/结构（enable_hand、backbone.enable_camera、hand_head.name…），
                面板据此按「输出头」分组、判定头是否启用。
    · data  —— clip_len / clip_stride，面板按训练同一切窗规则分窗计算（逐字一致）。
    读不到时各段回空 dict（面板降级：无头信息/回退整段）。"""
    try:
        import yaml
        cfg = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001
        cfg = {}
    return {"loss": cfg.get("loss", {}) or {},
            "model": cfg.get("model", {}) or {},
            "data": cfg.get("data", {}) or {}}
