#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import torch
from accelerate import skip_first_batches

from utils.logging import rank0_print
from engine.checkpoint import save_checkpoint


def _build_epoch_dataloader(dataloader, epoch, data_seed=None, skip_batches=0):
    loader = skip_first_batches(dataloader, skip_batches) if skip_batches else dataloader
    if data_seed is not None:
        generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.manual_seed(data_seed + epoch)
    if hasattr(loader, "set_epoch"):
        loader.set_epoch(epoch)
    return loader


def _grad_value_stats(params, clip_value: float):
    """Return pre/post value-clipping statistics without mutating gradients."""
    raw_sq = clipped_sq = clipped_count = max_abs = None
    total_count = 0
    for parameter in params:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        values = grad.coalesce().values() if grad.is_sparse else grad
        values = values.float()
        value_abs = values.abs()
        current_raw_sq = values.square().sum()
        current_clipped_sq = values.clamp(-clip_value, clip_value).square().sum()
        current_clipped_count = (value_abs > clip_value).sum()
        current_max = value_abs.max()
        raw_sq = current_raw_sq if raw_sq is None else raw_sq + current_raw_sq
        clipped_sq = (
            current_clipped_sq if clipped_sq is None else clipped_sq + current_clipped_sq
        )
        clipped_count = (
            current_clipped_count
            if clipped_count is None
            else clipped_count + current_clipped_count
        )
        max_abs = current_max if max_abs is None else torch.maximum(max_abs, current_max)
        total_count += values.numel()
    if raw_sq is None:
        return None
    return {
        "raw_norm": raw_sq.sqrt(),
        "clipped_norm": clipped_sq.sqrt(),
        "pre_max_abs": max_abs,
        "clipped_fraction": clipped_count.float() / max(1, total_count),
    }


class Trainer:
    def __init__(self, accelerator, model, criterion, dataloader,
                 optimizer, scheduler, cfg: dict, out_dir: str, logger,
                 start_step: int = 0, steps_per_epoch: int = 0,
                 batch_size: int = 0, data_seed=None):
        self.acc = accelerator
        self.model = model
        self.criterion = criterion
        self.dl = dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.out_dir = out_dir
        self.logger = logger
        self.max_steps = int(cfg.get("max_steps", 100))
        self.max_epochs = int(cfg.get("epochs", 0))
        self.log_every = int(cfg.get("log_every", 10))
        self.ckpt_every = int(cfg.get("ckpt_every", 0))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.grad_value_clip = float(cfg.get("grad_value_clip", 0.0))
        self.grad_accum = int(cfg.get("grad_accum", 1))
        self.start_step = int(start_step)
        self.steps_per_epoch = int(steps_per_epoch)
        self.data_seed = data_seed

        self.samples_per_step = int(batch_size) * self.grad_accum * self.acc.num_processes

        self.log_update_ratio = bool(cfg.get("log_update_ratio", True))
        self.log_diagnostics = bool(cfg.get("log_diagnostics", False))
        unwrap = getattr(self.acc, "unwrap_model", None)
        self._diagnostic_model = unwrap(model) if unwrap is not None else model
        self._validate_gradient_clipping()

    def _validate_gradient_clipping(self) -> None:
        norm_groups = []
        value_groups = []
        for index, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"g{index}")
            if float(group.get("grad_clip", self.grad_clip)) > 0:
                norm_groups.append(name)
            if float(group.get("grad_value_clip", self.grad_value_clip)) > 0:
                value_groups.append(name)
        if norm_groups and value_groups:
            raise ValueError(
                "[train]"
                f"[train]  {norm_groups}; {value_groups}."
            )
        self.gradient_clip_mode = (
            "norm" if norm_groups else "value" if value_groups else "disabled"
        )

    def _set_train_diagnostics(self, enabled: bool) -> None:
        setter = getattr(self._diagnostic_model, "set_train_diagnostics", None)
        if setter is not None:
            setter(enabled)

    def _reduce_gradient_sq(self, value: torch.Tensor) -> torch.Tensor:
        value = value.detach()
        if self.acc.num_processes > 1:
            value = self.acc.reduce(value, reduction="mean")
        return value

    def _loss_output_gradient_diagnostics(self, pred: dict) -> dict:
        """Measure actual weighted/scaled term gradients without touching .grad."""
        terms = pred.pop("_diagnostic_loss_terms", {})
        groups = []
        if "mano_param/orient_geo" in terms:
            groups.append(("mano_orient_geo", terms["mano_param/orient_geo"]))
        if "mano_param/pose_geo" in terms:
            groups.append(("mano_pose_geo", terms["mano_param/pose_geo"]))
        consistency = [
            value for name, value in terms.items()
            if name.startswith("camera_mano_consistency/") and "orient" in name
        ]
        if consistency:
            groups.append(("consistency_orient", sum(consistency)))

        logs = {}
        for name, term in groups:
            term = term / max(1, getattr(self, "grad_accum", 1))
            inputs = [("hand_output", pred.get("hand"))]
            if name == "consistency_orient":
                inputs.append(("camera_output", pred.get("pose_enc")))
            active = [(key, value) for key, value in inputs if value is not None]
            gradients = torch.autograd.grad(
                term,
                [value for _key, value in active],
                retain_graph=True,
                allow_unused=True,
            )
            total_sq = term.detach().new_zeros((), dtype=torch.float32)
            for (key, _value), gradient in zip(active, gradients):
                if gradient is None:
                    grad_sq = total_sq.new_zeros(())
                else:
                    grad_sq = gradient.detach().float().square().sum()
                grad_sq = self._reduce_gradient_sq(grad_sq)
                logs[f"diag/loss_grad/{name}/{key}_norm"] = grad_sq.sqrt()
                total_sq = total_sq + grad_sq
            logs[f"diag/loss_grad/{name}/total_output_norm"] = total_sq.sqrt()
        return logs

    def fit(self):
        self.model.train()


        step, done = self.start_step, False
        if self.steps_per_epoch > 0:
            epoch = step // self.steps_per_epoch
            step_in_epoch = step % self.steps_per_epoch
        else:
            epoch, step_in_epoch = 0, 0
        skip_batches = step_in_epoch * self.grad_accum
        use_epoch = self.max_epochs > 0
        limit = f"epochs={self.max_epochs}" if use_epoch else f"max_steps={self.max_steps}"
        rank0_print(f"[train]  {limit}."
                    f"[train]  {self.acc.num_processes}; {self.acc.mixed_precision}.")
        if self.start_step:
            self.logger.update(self.start_step)
            rank0_print(f"[train]  {self.start_step}."
                        f"[train]  {epoch}; {step_in_epoch}; {skip_batches}.")

        t_prev = time.perf_counter()
        step_t0 = t_prev
        data_s = 0.0
        while not done:

            loader = _build_epoch_dataloader(
                self.dl, epoch, data_seed=self.data_seed, skip_batches=skip_batches
            )
            skip_batches = 0
            for batch in loader:
                data_s += time.perf_counter() - t_prev
                with self.acc.accumulate(self.model):
                    diagnostic_step = (
                        self.log_diagnostics
                        and self.acc.sync_gradients
                        and (step + 1 == 1 or (step + 1) % self.log_every == 0)
                    )
                    self._set_train_diagnostics(diagnostic_step)
                    pred = self.model(batch)
                    pred["_diagnostics_enabled"] = diagnostic_step
                    loss, logs = self.criterion(pred, batch)
                    diagnostic_logs = (
                        self._loss_output_gradient_diagnostics(pred)
                        if diagnostic_step else {}
                    )
                    self.acc.backward(loss)




                    do_sync = self.acc.sync_gradients
                    will_log = do_sync and (step + 1 == 1 or (step + 1) % self.log_every == 0)
                    need_ur = will_log and self.log_update_ratio and self.acc.is_main_process
                    value_clip_stats = {}
                    if do_sync and self.gradient_clip_mode == "value":
                        # Unlike Accelerator.clip_grad_norm_, PyTorch's value clip does
                        # not unscale fp16 gradients automatically.
                        self.acc.unscale_gradients(self.optimizer)
                        for gi, group in enumerate(self.optimizer.param_groups):
                            value_clip = float(group.get(
                                "grad_value_clip", self.grad_value_clip
                            ))
                            if value_clip <= 0:
                                continue
                            name = group.get("name", f"g{gi}")
                            if will_log:
                                stats = _grad_value_stats(group["params"], value_clip)
                                if stats is not None:
                                    value_clip_stats[name] = {
                                        **stats,
                                        "limit": value_clip,
                                    }
                            torch.nn.utils.clip_grad_value_(
                                group["params"], value_clip
                            )
                    grad_norms = None
                    grad_norms_after_value_clip = None
                    if do_sync and self.gradient_clip_mode == "norm":




                        grad_norms = {}
                        for gi, g in enumerate(self.optimizer.param_groups):

                            max_norm = float(g.get("grad_clip", self.grad_clip))
                            if max_norm <= 0:
                                continue
                            gn = self.acc.clip_grad_norm_(g["params"], max_norm)
                            if gn is not None:
                                name = g.get("name", f"g{gi}")
                                grad_norms[name] = gn
                    elif will_log and value_clip_stats:
                        grad_norms = {
                            name: stats["raw_norm"]
                            for name, stats in value_clip_stats.items()
                        }
                        grad_norms_after_value_clip = {
                            name: stats["clipped_norm"]
                            for name, stats in value_clip_stats.items()
                        }
                    prev = None
                    if need_ur:

                        prev = [[p.detach().clone() for p in g["params"]]
                                for g in self.optimizer.param_groups]
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                if self.acc.sync_gradients:
                    step += 1
                    self.logger.update(1)
                    step_s = time.perf_counter() - step_t0
                    if step == 1 or step % self.log_every == 0:
                        metrics = {k: v.item() for k, v in logs.items()}
                        metrics.update(
                            {k: float(v) for k, v in diagnostic_logs.items()}
                        )

                        metrics["time/step_s"] = step_s
                        metrics["time/data_s"] = data_s
                        metrics["time/data_frac"] = data_s / step_s if step_s > 0 else 0.0
                        if self.samples_per_step > 0 and step_s > 0:
                            metrics["io/samples_per_s"] = self.samples_per_step / step_s


                        if grad_norms:
                            total_sq = 0.0
                            for name, gn in grad_norms.items():
                                v = float(gn)
                                metrics[f"opt/grad_norm/{name}"] = v
                                total_sq += v * v
                            metrics["opt/grad_norm"] = total_sq ** 0.5
                        if grad_norms_after_value_clip and value_clip_stats:
                            for name, stats in value_clip_stats.items():
                                base = f"opt/grad_value_clip/{name}"
                                metrics[f"{base}/limit"] = float(stats["limit"])
                                metrics[f"{base}/pre_max_abs"] = float(stats["pre_max_abs"])
                                metrics[f"{base}/clipped_fraction"] = float(
                                    stats["clipped_fraction"]
                                )
                                metrics[
                                    f"opt/grad_norm_after_value_clip/{name}"
                                ] = float(grad_norms_after_value_clip[name])

                        if self.acc.is_main_process:
                            with torch.no_grad():
                                sq = None
                                for g in self.optimizer.param_groups:
                                    for p in g["params"]:
                                        s = p.detach().float().pow(2).sum()
                                        sq = s if sq is None else sq + s
                                if sq is not None:
                                    metrics["opt/param_norm"] = float(sq.sqrt())

                                for gi, g in enumerate(self.optimizer.param_groups):
                                    name = g.get("name", f"g{gi}")
                                    metrics[f"lr/{name}"] = float(g["lr"])
                                    if prev is not None:
                                        dn = None
                                        dd = None
                                        for p, p0 in zip(g["params"], prev[gi]):
                                            a = (p.detach() - p0).float().pow(2).sum()
                                            b = p0.float().pow(2).sum()
                                            dn = a if dn is None else dn + a
                                            dd = b if dd is None else dd + b
                                        if dn is not None:
                                            den = float(dd.sqrt())
                                            metrics[f"update_ratio/{name}"] = float(dn.sqrt()) / den if den > 0 else 0.0
                        self.logger.log(step, metrics)
                    if self.ckpt_every and step % self.ckpt_every == 0:
                        save_checkpoint(self.acc, self.out_dir, step)
                        self.logger.info(f"[train]  {step}.")

                    step_t0 = time.perf_counter()
                    data_s = 0.0

                    if not use_epoch and step >= self.max_steps:
                        done = True
                        break
                t_prev = time.perf_counter()
            if done:
                break

            epoch += 1
            if use_epoch:
                rank0_print(f"[train]  {epoch}; {self.max_epochs}; {step}.")
                if epoch >= self.max_epochs:
                    done = True
        self.acc.wait_for_everyone()
        save_checkpoint(self.acc, self.out_dir, step)
        self.logger.close()
        rank0_print("[train]")
