#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from .train_logger import TrainLogger


def _default_wandb_run_name(out_dir: str) -> str:
    run_dir = os.path.normpath(out_dir)
    task_name = os.path.basename(run_dir)
    timestamp = os.path.basename(os.path.dirname(run_dir))
    return f"{timestamp}_{task_name}"


def build_logger(cfg: dict, out_dir: str, accelerator, total: int) -> TrainLogger:
    """Internal helper."""
    is_main = accelerator.is_main_process
    sinks = []
    if is_main:
        wcfg = (cfg.get("train", {}) or {}).get("wandb") or {}
        if wcfg.get("enabled"):
            from .wandb_sink import WandbSink
            run_name = wcfg.get("name") or _default_wandb_run_name(out_dir)
            sinks.append(WandbSink(wcfg, full_cfg=cfg, out_dir=out_dir, run_name=run_name))
    return TrainLogger(total=total, sinks=sinks, is_main=is_main)


__all__ = ["TrainLogger", "build_logger"]
