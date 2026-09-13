#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train or inspect the MINT camera-and-hand student model."""
import argparse
import os
import sys
import time
import warnings


warnings.filterwarnings(
    "ignore",
    message=r"Failed to JIT torch c dlpack extension.*",
    category=UserWarning,
    module=r"tvm_ffi\._optional_torch_c_dlpack",
)



_ROOT = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_ROOT)
for _p in (_ROOT,
           os.path.join(_ROOT, "_vendor")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import yaml  # noqa: E402

from models import build_model            # noqa: E402
from utils.freeze import apply_freeze      # noqa: E402
from utils import module_inspector         # noqa: E402
from utils.logging import rank0_print      # noqa: E402


def _abs(p):
    return p if (p is None or os.path.isabs(p)) else os.path.join(_REPO, p)


def _resolve_init_from(path):
    """Resolve a model-only initialization path to its weight file."""
    path = _abs(path)
    if os.path.isdir(path):
        weight_file = os.path.join(path, "model.safetensors")
        if not os.path.isfile(weight_file):
            raise FileNotFoundError(
                f"No model.safetensors file exists in initialization directory: {path}"
            )
        return weight_file
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {path}")
    return path


def _resolve_data_root(root):
    """Resolve string roots while preserving per-dataset override dictionaries."""
    if not isinstance(root, list):
        return _abs(root)
    resolved = []
    for item in root:
        if isinstance(item, dict):
            if "root" not in item:
                raise KeyError("Each dataset override must define a 'root' field.")
            resolved.append({**item, "root": _abs(item["root"])})
        else:
            resolved.append(_abs(item))
    return resolved


def _snapshot_code(record_dir):
    import shutil
    dirs = ["engine", "losses", "models", "data", "_vendor/lingbot_map"]
    files = ["train.py"]
    code_dir = os.path.join(record_dir, "code")
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    try:
        for d in dirs:
            src = os.path.join(_ROOT, d)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(code_dir, d),
                                ignore=ignore, dirs_exist_ok=True)
        for fn in files:
            src = os.path.join(_ROOT, fn)
            if os.path.isfile(src):
                os.makedirs(code_dir, exist_ok=True)
                shutil.copy2(src, os.path.join(code_dir, fn))
    except Exception as e:
        rank0_print(f"[warning] Could not snapshot source code: {e}")


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _infer_hand_supervision(cfg):
    """Keep old checkpoints working while allowing MANO-only or kp21-only datasets."""
    loss_cfg = cfg.get("loss", {}) or {}
    term_names = {
        term.get("name")
        for group in loss_cfg.values()
        if isinstance(group, dict)
        for term in group.get("terms", [])
        if isinstance(term, dict)
    }
    needs_kpt21 = bool(
        {"hand_kp21_loss", "world_kp21_loss"} & set(loss_cfg)
        or any(name and ("kp21" in name or name == "betas_reg") for name in term_names)
    )
    needs_mano = bool(
        {"mano_param", "hand_loss", "world_loss", "image_hand"} & set(loss_cfg)
        or any(name in {"rootrel_mpjpe", "rootrel_vel_mpjpe", "abs_mpjpe"}
               for name in term_names)
        or any(name and name.startswith(
            ("world_trans", "world_orient", "transl_world", "orient_world")
        )
               for name in term_names)
    )
    cfg["data"].setdefault("require_mano_gt", needs_mano)
    cfg["data"].setdefault("require_kpt21_gt", needs_kpt21)


def _monitoring_options(train_cfg):
    """Return validated opt-in monitoring options, or None when disabled."""
    monitoring_cfg = train_cfg.get("monitoring")
    if monitoring_cfg is None:
        return None
    if not isinstance(monitoring_cfg, dict):
        raise TypeError("train.monitoring must be a mapping.")
    if not bool(monitoring_cfg.get("enabled", False)):
        return None
    interval = float(monitoring_cfg.get("interval", 1.0))
    if interval <= 0:
        raise ValueError("train.monitoring.interval must be positive.")
    return {"interval": interval}


def _reproducibility_options(train_cfg):
    """Return validated opt-in reproducibility settings."""
    reproducibility_cfg = train_cfg.get("reproducibility")
    if reproducibility_cfg is None:
        return None
    if not isinstance(reproducibility_cfg, dict):
        raise TypeError("train.reproducibility must be a mapping.")
    if not bool(reproducibility_cfg.get("enabled", False)):
        return None

    seed = reproducibility_cfg.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("train.reproducibility.seed must be an integer.")
    if not 0 <= seed < 2 ** 32:
        raise ValueError("train.reproducibility.seed must be in [0, 2**32).")

    deterministic = reproducibility_cfg.get("deterministic_algorithms", True)
    if not isinstance(deterministic, bool):
        raise TypeError("deterministic_algorithms must be true or false.")
    return {"seed": seed, "deterministic_algorithms": deterministic}


def _ensure_python_hash_seed(seed):
    """Re-exec once so PYTHONHASHSEED takes effect for the current interpreter."""
    expected = str(seed)
    if os.environ.get("PYTHONHASHSEED") == expected:
        return
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = expected
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def _configure_reproducibility(options):
    """Seed all RNGs before constructing either the model or the DataLoader."""
    import random

    import numpy as np
    import torch

    seed = options["seed"]
    deterministic = options["deterministic_algorithms"]
    if deterministic:
        workspace_config = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace_config not in {":4096:8", ":16:8"}:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def _gpu_tag():
    """Internal helper."""
    import re
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    name = torch.cuda.get_device_name(0)
    return re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_") or "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None,
                    help="Training YAML path; required unless --resume is used")
    ap.add_argument("--inspect", action="store_true",
                    help="Build the model and print its structure without training")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Override train.max_steps from the configuration")
    ap.add_argument("--resume", default=None,
                    help="Accelerate checkpoint directory to resume")
    ap.add_argument("--init-from", default=None,
                    help="Model-only checkpoint used to initialize a new run")
    args = ap.parse_args()

    resume_dir = _abs(args.resume) if args.resume else None
    if resume_dir and not os.path.isdir(resume_dir):
        ap.error(f"resume directory does not exist: {resume_dir}")



    if args.config:
        config_path = _abs(args.config)
    elif resume_dir:
        run_root = os.path.dirname(os.path.normpath(resume_dir))
        config_path = os.path.join(run_root, "logs", "record", "config.yaml")
        if not os.path.isfile(config_path):
            ap.error(f"resume configuration does not exist: {config_path}")
        rank0_print(f"[resume] configuration={config_path}")
    else:
        ap.error("--config is required unless --resume is provided")

    cfg = load_cfg(config_path)
    configured_init_from = (cfg.get("train", {}) or {}).get("init_from")
    if resume_dir and args.init_from:
        ap.error("--resume and --init-from are mutually exclusive")
    init_from = None
    # A resumed transfer run keeps train.init_from in its snapshot as provenance;
    # --resume must ignore it and restore the newer full training state instead.
    if not resume_dir and (args.init_from or configured_init_from):
        try:
            init_from = _resolve_init_from(args.init_from or configured_init_from)
        except FileNotFoundError as error:
            ap.error(str(error))
        cfg.setdefault("train", {})["init_from"] = init_from
    _infer_hand_supervision(cfg)
    tcfg = cfg["train"]
    reproducibility = _reproducibility_options(tcfg)
    if reproducibility is not None:
        _ensure_python_hash_seed(reproducibility["seed"])
        _configure_reproducibility(reproducibility)

    cfg["data"]["root"] = _resolve_data_root(cfg["data"]["root"])
    if cfg["model"].get("pretrained"):
        cfg["model"]["pretrained"] = _abs(cfg["model"]["pretrained"])
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps


    if args.inspect:
        cfg["model"]["pretrained"] = None
        cfg["model"]["_inspect_skip_pretrained"] = True
        model = build_model(cfg["model"])
        apply_freeze(model, cfg["model"].get("freeze"))
        print(module_inspector.summarize(model, title=cfg["model"]["name"]))
        print("\nFreezable module groups:")
        for n in module_inspector.list_freezable_names(model, depth=2):
            print("  -", n)
        return


    from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
    from accelerate.utils import broadcast_object_list
    from data import build_dataloader
    from losses import build_criterion
    from engine.optim import (
        build_optimizer,
        build_scheduler,
        capture_optimizer_group_runtime_config,
        restore_optimizer_group_runtime_config,
    )
    from engine.trainer import Trainer
    from logger import build_logger

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dataloader_config = None
    if reproducibility is not None:
        dataloader_config = DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=reproducibility["seed"],
        )
    accelerator = Accelerator(
        mixed_precision=tcfg.get("mixed_precision", "bf16"),
        gradient_accumulation_steps=int(tcfg.get("grad_accum", 1)),
        kwargs_handlers=[ddp_kwargs],
        dataloader_config=dataloader_config,
    )





    import re as _re
    def _san(s, dft):
        return _re.sub(r"[^0-9A-Za-z._-]+", "_", str(s)).strip("_") or dft
    ts = time.strftime("%Y%m%d_%H%M%S") if accelerator.is_main_process else None
    gpu = _gpu_tag() if accelerator.is_main_process else None
    box = [ts, gpu]
    broadcast_object_list(box, from_process=0)
    ts, gpu = _san(box[0], "0"), _san(box[1], "unknown")
    _resume_run = os.path.basename(os.path.dirname(os.path.normpath(resume_dir))) if resume_dir else None
    taskname = _san(os.environ.get("RUN_SERIES") or tcfg.get("run_name")
                    or os.environ.get("MLP_TASK_NAME") or _resume_run
                    or os.path.splitext(os.path.basename(config_path))[0], "task")
    out_dir = os.path.join(_REPO, "output", "model_train", gpu, ts, taskname)
    log_dir = os.path.join(out_dir, "logs")



    norm_cfg = cfg["data"].get("camera_translation_normalization", {}) or {}
    if bool(norm_cfg.get("enabled", False)):
        normalization_box = [None]
        if accelerator.is_main_process:
            try:
                from data.camera_normalization import resolve_global_camera_normalization
                normalization_box[0] = resolve_global_camera_normalization(
                    cfg["data"], log=rank0_print
                )
            except Exception as error:
                normalization_box[0] = {
                    "__error__": f"{type(error).__name__}: {error}"
                }
        broadcast_object_list(normalization_box, from_process=0)
        resolved_normalization = normalization_box[0]
        if resolved_normalization.get("__error__"):
            raise RuntimeError(
                "Camera normalization failed: " + resolved_normalization["__error__"]
            )
        norm_cfg["resolved"] = resolved_normalization
        cfg["data"]["camera_translation_normalization"] = norm_cfg

    if accelerator.is_main_process:
        os.makedirs(log_dir, exist_ok=True)



        record_dir = os.path.join(log_dir, "record")
        os.makedirs(record_dir, exist_ok=True)
        with open(os.path.join(record_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        _snapshot_code(record_dir)
    accelerator.wait_for_everyone()





    _lws = int(os.environ.get("LOCAL_WORLD_SIZE") or 0) or max(1, accelerator.num_processes)
    _node_rank = os.environ.get("GROUP_RANK") or str(accelerator.process_index // _lws)

    _logf = open(os.path.join(log_dir, f"node{_node_rank}.log"), "a", buffering=1)

    class _Tee:
        def __init__(self, real, f):
            self._real, self._f = real, f

        def write(self, s):
            self._real.write(s)
            self._f.write(s)

        def flush(self):
            self._real.flush()
            self._f.flush()

        def isatty(self):
            return self._real.isatty()

        def fileno(self):
            return self._real.fileno()

    sys.stdout = _Tee(sys.stdout, _logf)
    sys.stderr = _Tee(sys.stderr, _logf)

    rank0_print(f"[train] config={config_path}  gpu={gpu}  out_dir={out_dir}")
    if reproducibility is not None:
        rank0_print(
            f"[reproducibility] seed={reproducibility['seed']} "
            f"deterministic_algorithms={reproducibility['deterministic_algorithms']}"
        )



    monitor = None
    monitoring_options = _monitoring_options(tcfg)
    if monitoring_options is not None and accelerator.local_process_index == 0:
        try:
            from monitoring import ResourceMonitor
            mon_dir = os.path.join(out_dir, "monitoring", f"node{_node_rank}")
            monitor = ResourceMonitor(mon_dir, interval=monitoring_options["interval"])
            monitor.start()
        except Exception as e:
            print(f"[warning] Resource monitor could not start: {e}")

    model_cfg = cfg["model"]
    if init_from:
        # The full student checkpoint already contains the backbone and all heads.
        model_cfg = {**model_cfg, "pretrained": None, "_ckpt_provided": True}
    model = build_model(model_cfg)
    if init_from:
        from utils.weight_loader import load_state_dict_file
        state_dict = load_state_dict_file(init_from)
        model.load_state_dict(state_dict, strict=True)
        rank0_print(
            f"[init] loaded {len(state_dict)} tensors from {init_from}; "
            "optimizer state starts from scratch"
        )
        del state_dict
    apply_freeze(model, cfg["model"].get("freeze"))
    if accelerator.is_main_process:
        total_params = sum(parameter.numel() for parameter in model.parameters())
        trainable_params = sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        )
        rank0_print(
            f"[model] {cfg['model']['name']}: "
            f"params={total_params / 1e6:.1f}M "
            f"trainable={trainable_params / 1e6:.1f}M"
        )

    data_seed = reproducibility["seed"] if reproducibility is not None else None
    dataloader = build_dataloader(cfg["data"], seed=data_seed)
    criterion = build_criterion(cfg["loss"])
    optimizer = build_optimizer(model, cfg["optim"])
    optimizer_group_runtime_config = capture_optimizer_group_runtime_config(optimizer)






    import math
    ga = int(tcfg.get("grad_accum", 1))
    per_proc = len(dataloader) // max(1, accelerator.num_processes)
    steps_per_epoch = max(1, math.ceil(per_proc / ga))
    epochs = int(tcfg.get("epochs", 0))
    if epochs > 0:
        total_steps = epochs * steps_per_epoch
        rank0_print(
            f"[schedule] epochs={epochs} steps_per_epoch={steps_per_epoch} "
            f"total_steps={total_steps}"
        )
    else:
        total_steps = int(tcfg["max_steps"])



    scheduler = build_scheduler(optimizer, cfg["optim"], total_steps,
                                step_scale=accelerator.num_processes)

    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler)



    start_step = 0
    if resume_dir:
        accelerator.load_state(resume_dir)
        # Optimizer state_dict replaces arbitrary param-group keys. Restore the
        # current YAML's clipping mode/overrides instead of inheriting stale ones.
        restore_optimizer_group_runtime_config(
            optimizer, optimizer_group_runtime_config
        )
        import re as _re2
        m = _re2.search(r"step_(\d+)", os.path.basename(os.path.normpath(resume_dir)))
        if m:
            start_step = int(m.group(1))
            rank0_print(f"[resume] checkpoint={resume_dir} start_step={start_step}")
        else:
            rank0_print(f"[resume] checkpoint={resume_dir}; step could not be inferred")

    train_logger = build_logger(cfg, out_dir, accelerator, total=total_steps)
    try:
        Trainer(accelerator, model, criterion, dataloader,
                optimizer, scheduler, tcfg, out_dir, train_logger,
                start_step=start_step, steps_per_epoch=steps_per_epoch,
                batch_size=int(cfg["data"].get("batch_size", 0)),
                data_seed=data_seed).fit()
    finally:

        if monitor is not None:
            monitor.stop()


if __name__ == "__main__":
    main()
