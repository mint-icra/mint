#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch-video discovery, mirrored output planning, and atomic result writers."""
from __future__ import annotations

import json
import os
import shutil
import string
import tempfile
from pathlib import Path

import numpy as np

from .const import VIDEO_EXTS


ALLOWED_TEMPLATE_FIELDS = {"stem", "name", "ext", "parent", "index"}
DEFAULT_NAME_TEMPLATE = "{stem}_pred"


def resolve_batch_roots(input_dir: str, output_dir: str) -> tuple[Path, Path]:
    """Validate server-side roots and prevent generated videos from being rescanned."""
    raw_input = Path(str(input_dir or "")).expanduser()
    raw_output = Path(str(output_dir or "")).expanduser()
    if not raw_input.is_absolute() or not raw_output.is_absolute():
        raise ValueError("输入目录和输出目录必须是绝对路径")
    input_root = raw_input.resolve()
    output_root = raw_output.resolve()
    if not input_root.is_dir():
        raise ValueError(f"输入目录不存在或不可读: {input_root}")
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise ValueError("输出目录不能等于输入目录，也不能位于输入目录内部")
    return input_root, output_root


def validate_name_template(name_template: str) -> str:
    """Allow useful filename fields without attribute access or path traversal."""
    template = str(name_template or "").strip()
    if not template:
        raise ValueError("文件名模板不能为空")
    formatter = string.Formatter()
    try:
        parts = list(formatter.parse(template))
    except ValueError as exc:
        raise ValueError(f"文件名模板无效: {exc}") from exc
    for _literal, field, format_spec, conversion in parts:
        if field is None:
            continue
        if field not in ALLOWED_TEMPLATE_FIELDS:
            raise ValueError(
                f"不支持模板字段 {{{field}}}；可用字段: "
                + ", ".join(sorted(ALLOWED_TEMPLATE_FIELDS))
            )
        if conversion:
            raise ValueError("文件名模板不支持 !r、!s 等转换")
        if format_spec and field != "index":
            raise ValueError("仅 {index} 支持格式，例如 {index:04d}")
        if "{" in format_spec or "}" in format_spec:
            raise ValueError("文件名模板不支持嵌套格式字段")
    return template


def render_output_name(source: Path, index: int, name_template: str) -> str:
    template = validate_name_template(name_template)
    values = {
        "stem": source.stem,
        "name": source.name,
        "ext": source.suffix.lstrip("."),
        "parent": source.parent.name,
        "index": int(index),
    }
    try:
        output_name = template.format(**values).strip()
    except (KeyError, ValueError) as exc:
        raise ValueError(f"文件名模板格式化失败: {exc}") from exc
    if (not output_name or output_name in {".", ".."} or ".." in output_name
            or "/" in output_name or "\\" in output_name
            or Path(output_name).name != output_name):
        raise ValueError(f"模板生成了非法文件名: {output_name!r}")
    return output_name


def discover_videos(input_root: Path) -> list[Path]:
    """Recursively enumerate supported videos in a deterministic order."""
    videos: list[Path] = []
    for root, dirs, files in os.walk(input_root, followlinks=False):
        dirs.sort(key=str.lower)
        for filename in sorted(files, key=str.lower):
            if Path(filename).suffix.lower() in VIDEO_EXTS:
                videos.append(Path(root) / filename)
    return videos


def build_output_plan(input_root: Path, output_root: Path,
                      name_template: str = DEFAULT_NAME_TEMPLATE) -> list[dict]:
    """Build mirror-layout MP4/NPZ targets and reject template collisions."""
    template = validate_name_template(name_template)
    plan = []
    claimed: dict[Path, Path] = {}
    for index, source in enumerate(discover_videos(input_root), start=1):
        relative = source.relative_to(input_root)
        output_dir = output_root / relative.parent
        output_name = render_output_name(source, index, template)
        output_mp4 = output_dir / f"{output_name}.mp4"
        output_npz = output_dir / f"{output_name}.npz"
        for target in (output_mp4, output_npz):
            previous = claimed.get(target)
            if previous is not None:
                raise ValueError(
                    f"文件名模板产生重名输出: {previous} 与 {source} -> {target}"
                )
            claimed[target] = source
        plan.append({
            "index": index,
            "source": source,
            "relative": relative,
            "output_mp4": output_mp4,
            "output_npz": output_npz,
        })
    return plan


def atomic_save_npz(path: Path, **arrays) -> None:
    """Write compressed predictions without exposing a partial final file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_copy_file(source: Path, target: Path) -> None:
    """Sequentially upload a local artifact, then expose it under the final name."""
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.stem}.", suffix=f".part{target.suffix}", dir=target.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copyfile(source, tmp_path)
        os.replace(tmp_path, target)
    finally:
        tmp_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".json", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)
