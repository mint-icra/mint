#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math

import torch

from utils.logging import rank0_print


_RUNTIME_GROUP_KEYS = ("name", "grad_clip", "grad_value_clip")


def capture_optimizer_group_runtime_config(optimizer) -> list[dict]:
    """Keep YAML-owned group metadata that optimizer checkpoints overwrite."""
    return [
        {key: group[key] for key in _RUNTIME_GROUP_KEYS if key in group}
        for group in optimizer.param_groups
    ]


def restore_optimizer_group_runtime_config(optimizer, runtime_config) -> None:
    if len(optimizer.param_groups) != len(runtime_config):
        raise RuntimeError(
            "[train]"
            f"checkpoint={len(optimizer.param_groups)}, yaml={len(runtime_config)}"
        )
    for group, configured in zip(optimizer.param_groups, runtime_config):
        for key in _RUNTIME_GROUP_KEYS:
            group.pop(key, None)
        group.update(configured)


def build_optimizer(model, cfg: dict):
    cfg = cfg or {}
    lr_default = float(cfg.get("lr", 1e-4))
    wd = float(cfg.get("weight_decay", 0.05))
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]


    labels = []
    pg_cfg = cfg.get("param_groups")
    if pg_cfg:




        specs = [
            (
                str(g["match"]),
                float(g["lr"]),
                g.get("grad_clip"),
                g.get("grad_value_clip"),
                [],
                [],
            )
            for g in pg_cfg
        ]
        misc, misc_names = [], []
        for n, p in trainable:
            for match, _lr, _gc, _gvc, plist, pnames in specs:
                if match in n:
                    plist.append(p)
                    if not pnames:
                        pnames.append(n)
                    break
            else:
                misc.append(p)
                if not misc_names:
                    misc_names.append(n)
        groups = []
        for match, lr, gc, gvc, plist, pnames in specs:
            if plist:

                grp = {"params": plist, "lr": lr, "name": match.split(".")[-1]}
                if gc is not None:
                    grp["grad_clip"] = float(gc)
                if gvc is not None:
                    grp["grad_value_clip"] = float(gvc)
                groups.append(grp)
                labels.append(
                    (match, lr, gc, gvc, len(plist), pnames[0] if pnames else "", False)
                )
        if misc:
            groups.append({"params": misc, "lr": lr_default, "name": "misc"})

        labels.append(("[train]", lr_default, None, None, len(misc),
                       misc_names[0] if misc_names else "", True))
    else:

        lr_backbone = float(cfg.get("lr_backbone", lr_default * 0.1))
        backbone, heads = [], []
        bb_name, hd_name = "", ""
        for n, p in trainable:
            if "backbone." in n:
                backbone.append(p)
                bb_name = bb_name or n
            else:
                heads.append(p)
                hd_name = hd_name or n
        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "name": "backbone"})
            labels.append(("backbone.", lr_backbone, None, None,
                           len(backbone), bb_name, False))
        if heads:
            groups.append({"params": heads, "lr": lr_default, "name": "heads"})
            labels.append(("<heads>", lr_default, None, None,
                           len(heads), hd_name, False))

    if not groups:
        raise RuntimeError("[train]")

    total = sum(cnt for _, _, _, _, cnt, _, _ in labels)
    summaries = []
    for name, lr, gc, gvc, cnt, sample, is_misc in labels:
        clip_s = f"{gc:g}" if gc is not None else "[train]"
        value_clip_s = f"{gvc:g}" if gvc is not None else "[train]"
        summaries.append(
            f"{name}(lr={lr:g},clip={clip_s},value_clip={value_clip_s},n={cnt})"
        )
        if is_misc and cnt:
            rank0_print(
                f"[train]  {cnt}."
                f"[train]  {sample!r}."
            )
    rank0_print(f"[optim] groups(total={total}): " + "; ".join(summaries))
    return torch.optim.AdamW(groups, lr=lr_default, weight_decay=wd, betas=(0.9, 0.95))


def build_scheduler(optimizer, cfg: dict, total_steps: int, step_scale: int = 1):
    cfg = cfg or {}
    step_scale = max(1, int(step_scale))
    total = int(total_steps) * step_scale
    warmup = int(cfg.get("warmup_steps", max(1, int(total_steps) // 20))) * step_scale

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
