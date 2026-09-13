#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import fnmatch
from typing import Dict, List, Optional

import torch.nn as nn

from utils.logging import rank0_print


def apply_freeze(model: nn.Module, patterns: Optional[List[str]], verbose: bool = True) -> Dict:
    patterns = patterns or []
    n_frozen_p, n_total_p = 0, 0
    frozen_params, total_params = 0, 0
    hit_patterns = {p: 0 for p in patterns}
    for name, p in model.named_parameters():
        n_total_p += 1
        total_params += p.numel()
        matched = next((pat for pat in patterns if fnmatch.fnmatch(name, pat)), None)
        if matched is not None:
            p.requires_grad_(False)
            n_frozen_p += 1
            frozen_params += p.numel()
            hit_patterns[matched] += 1
    summary = {
        "patterns": patterns,
        "frozen_tensors": n_frozen_p,
        "total_tensors": n_total_p,
        "frozen_params": frozen_params,
        "trainable_params": total_params - frozen_params,
        "pattern_hits": hit_patterns,
    }
    if verbose and patterns:
        rank0_print(f"[train]  {n_frozen_p}; {n_total_p}.")
        for pat, n in hit_patterns.items():
            if n == 0:
                rank0_print(f"[train]  {pat!r}.")
    return summary
