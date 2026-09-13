#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from tqdm.auto import tqdm


class TrainLogger:
    def __init__(self, total: int, sinks: list, is_main: bool, desc: str = "train"):
        self.sinks = sinks
        self.pbar = tqdm(total=total, disable=not is_main,
                         desc=desc, dynamic_ncols=True)

    def update(self, n: int = 1):
        """Internal helper."""
        self.pbar.update(n)

    def log(self, step: int, metrics: dict, lr: float = None):
        # High-cardinality diagnostics stay in metric sinks (for example W&B) instead
        # of making every tqdm refresh and node log line excessively wide.
        post = {
            k: f"{v:.4f}" for k, v in metrics.items()
            if not k.startswith("diag/")
        }
        if lr is not None:
            post["lr"] = f"{lr:.2e}"
        self.pbar.set_postfix(post, refresh=False)

        payload = dict(metrics)
        if lr is not None:
            payload["lr"] = lr
        for s in self.sinks:
            s.log(step, payload)

    def info(self, msg: str):
        """Internal helper."""
        self.pbar.write(msg)

    def close(self):
        self.pbar.close()
        for s in self.sinks:
            s.close()
