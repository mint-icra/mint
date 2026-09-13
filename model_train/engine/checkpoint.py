#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal helper."""
import os

from utils.logging import rank0_print


def save_checkpoint(accelerator, out_dir: str, step: int):
    ckpt_dir = os.path.join(out_dir, f"step_{step:08d}")
    accelerator.save_state(ckpt_dir)
    rank0_print(f"[train]  {ckpt_dir}.")
    return ckpt_dir


def load_checkpoint(accelerator, ckpt_dir: str):
    accelerator.load_state(ckpt_dir)
    rank0_print(f"[train]  {ckpt_dir}.")
