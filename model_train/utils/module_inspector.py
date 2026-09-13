#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import Dict, List

import torch.nn as nn


def _human(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


def _insert(tree: dict, parts: List[str], numel: int, trainable: bool):
    node = tree
    for p in parts[:-1]:
        node = node.setdefault(p, {"__num__": 0, "__train__": 0, "__child__": {}})
        node["__num__"] += numel
        node["__train__"] += numel if trainable else 0
        node = node["__child__"]
    leaf = node.setdefault(parts[-1], {"__num__": 0, "__train__": 0, "__child__": {}})
    leaf["__num__"] += numel
    leaf["__train__"] += numel if trainable else 0


def _collapse_int_keys(node: dict) -> bool:
    keys = [k for k in node if not k.startswith("__")]
    return len(keys) >= 3 and all(k.isdigit() for k in keys)


def _render(node: dict, lines: List[str], prefix: str = ""):
    children = node["__child__"] if "__child__" in node else node
    keys = [k for k in children if not k.startswith("__")]

    if _collapse_int_keys(children):
        ikeys = sorted(keys, key=int)
        agg_num = sum(children[k]["__num__"] for k in ikeys)
        agg_tr = sum(children[k]["__train__"] for k in ikeys)
        flag = _flag(agg_num, agg_tr)
        lines.append(f"[train]  {prefix}; {ikeys[0]}; {ikeys[-1]}; {len(ikeys)}."
                     f"{_human(agg_num)}  {flag}")
        _render(children[ikeys[0]], lines, prefix + "[train]")
        return
    int_first = sorted(keys, key=lambda k: (0, int(k)) if k.isdigit() else (1, k))
    for i, k in enumerate(int_first):
        sub = children[k]
        last = i == len(int_first) - 1
        branch = "[train]" if last else "[train]"
        flag = _flag(sub["__num__"], sub["__train__"])
        lines.append(f"{prefix}{branch} {k}  {_human(sub['__num__'])}  {flag}")
        if sub["__child__"]:
            _render(sub, lines, prefix + ("    " if last else "[train]"))


def _flag(num: int, train: int) -> str:
    if num == 0:
        return ""
    if train == 0:
        return "[FROZEN]"
    if train == num:
        return "[train]"
    return f"[partial {_human(train)}/{_human(num)} train]"


def summarize(model: nn.Module, title: str = "model") -> str:
    """Internal helper."""
    tree: Dict = {"__num__": 0, "__train__": 0, "__child__": {}}
    total, trainable = 0, 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        trainable += n if p.requires_grad else 0
        _insert(tree["__child__"], name.split("."), n, p.requires_grad)
        tree["__num__"] += n
        tree["__train__"] += n if p.requires_grad else 0
    lines = [f"[train]  {title}.",
             f"[train]  {_human(total)}; {_human(trainable)}."
             f"[train]  {_human(total - trainable)}.", ""]
    _render(tree, lines)
    return "\n".join(lines)


def list_freezable_names(model: nn.Module, depth: int = 2) -> List[str]:
    """Internal helper."""
    names = set()
    for name, _ in model.named_parameters():
        parts = name.split(".")
        names.add(".".join(parts[:depth]))
    return sorted(names)
