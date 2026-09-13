#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path

from .base import MetricSink


_MODEL_TRAIN_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_API_KEY_FILE = _MODEL_TRAIN_ROOT / "configs" / "wandb" / "wandb_key.md"


def _wandb_is_authenticated() -> bool:
    """Check env, netrc and session credentials without triggering a login prompt."""
    from wandb.apis.internal import Api

    return Api().is_authenticated


def _configure_wandb_credentials(wcfg: dict, mode: str) -> None:
    if mode not in {"online", "shared"} or _wandb_is_authenticated():
        return

    configured_path = wcfg.get("api_key_file")
    key_path = Path(configured_path).expanduser() if configured_path else _DEFAULT_API_KEY_FILE
    if not key_path.is_absolute():
        key_path = _MODEL_TRAIN_ROOT / key_path

    try:
        api_key = key_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"[train]  {key_path}."
        ) from None
    except OSError as exc:
        raise RuntimeError(f"[train]  {key_path}; {exc}.") from exc

    if not api_key:
        raise RuntimeError(f"[train]  {key_path}.")
    if any(char.isspace() for char in api_key):
        raise RuntimeError(f"[train]  {key_path}.")

    # Set the key only for this process; do not persist it to netrc or the run config.
    os.environ["WANDB_API_KEY"] = api_key
    print(f"[train]  {key_path}.")


class WandbSink(MetricSink):
    def __init__(self, wcfg: dict, full_cfg: dict, out_dir: str, run_name: str):
        import wandb

        self._wandb = wandb
        mode = str(wcfg.get("mode", "online")).lower()
        _configure_wandb_credentials(wcfg, mode)
        self.run = wandb.init(
            project=wcfg.get("project", "model_train"),
            name=wcfg.get("name") or run_name,
            dir=out_dir,
            mode=mode,                         # online / offline
            config=full_cfg,
        )

    def log(self, step: int, metrics: dict):
        self._wandb.log(metrics, step=step)

    def close(self):
        self.run.finish()
