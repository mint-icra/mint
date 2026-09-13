#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify that the visualization panel matches the training Criterion exactly.

The check builds random camera/MANO sequences, applies the same clip slicing and
first-frame rebasing as training, and compares every displayed module term plus
the weighted total against a direct Criterion call.  It covers camera,
hand-presence, image-space hand, MANO, camera/MANO consistency, and optional
camera translation normalization.

Usage:
  PY=python
  $PY eval/model_effect/visualization/tools/check_metrics_win_consistency.py \
      --input model_train/configs/volcano/lingbotmap_distill_full.yaml --frames 37
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


_HERE = Path(__file__).resolve().parent
_PKG_ROOT = _HERE.parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))
_REPO = _HERE.parents[3]
_MODEL_TRAIN = _REPO / "model_train"
for path in (str(_MODEL_TRAIN), str(_MODEL_TRAIN / "_vendor")):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.insert(0, path)


_MANO_CAT = ("transl_cam", "orient6d", "pose6d", "betas")
_DIMS = {"transl_cam": 3, "orient6d": 6, "pose6d": 90, "betas": 10}


def _rand_pose_enc(rng, length):
    pose = np.zeros((length, 9), np.float32)
    pose[:, :3] = rng.standard_normal((length, 3))
    quaternion = rng.standard_normal((length, 4)).astype(np.float32)
    pose[:, 3:7] = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    pose[:, 7:] = 0.5 + 0.1 * rng.standard_normal((length, 2))
    return pose


def _rand_hands(rng, length):
    hands = {
        hand: {
            column: rng.standard_normal((length, _DIMS[column])).astype(np.float32)
            for column in _MANO_CAT
        }
        for hand in ("left", "right")
    }
    for hand in hands.values():
        hand["transl_cam"][:, :2] *= 0.03
        hand["transl_cam"][:, 2] = 0.5 + 0.05 * np.abs(hand["transl_cam"][:, 2])
    return hands


def _hands_to_218(hands, length):
    return np.concatenate(
        [np.concatenate([hands[hand][column] for column in _MANO_CAT], axis=-1)
         for hand in ("left", "right")],
        axis=-1,
    )[:length].astype(np.float32)


def _loss_log_key(logs, group_name, term_name):
    key = f"loss/{group_name}/{term_name}"
    return f"{key}_norm" if f"{key}_norm" in logs else key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(_MODEL_TRAIN / "configs" / "volcano" / "lingbotmap_distill_full.yaml"),
        help="Training config whose real loss/data sections are verified.",
    )
    parser.add_argument("--frames", type=int, default=37)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()

    import torch
    import yaml
    from losses import build_criterion
    from visualization.render import metrics as metrics_module

    cfg = yaml.safe_load(Path(args.input).read_text(encoding="utf-8")) or {}
    loss_cfg = {
        "loss": cfg.get("loss", {}) or {},
        "model": cfg.get("model", {}) or {},
        "data": cfg.get("data", {}) or {},
    }
    clip_len = int(loss_cfg["data"].get("clip_len", 8) or 8)
    stride = int(loss_cfg["data"].get("clip_stride", 1) or 1)
    length = int(args.frames)
    rng = np.random.default_rng(args.seed)

    gt_pose = _rand_pose_enc(rng, length)
    pred_pose = _rand_pose_enc(rng, length)
    gt_hands = _rand_hands(rng, length)
    pred_hand = _hands_to_218(_rand_hands(rng, length), length)
    kept = rng.random((length, 2)) > 0.25
    kpt21_gt = rng.standard_normal((length, 2, 21, 3)).astype(np.float32)
    camera_std = np.array([0.02, 0.03, 0.04], dtype=np.float32)
    raw = {
        "cam_pose_enc": gt_pose,
        "hands": gt_hands,
        "kept": kept,
        "kpt21_gt": kpt21_gt,
        "camera_trans_std": camera_std,
        "episode_index": 0,
        "frames": np.zeros((length, 4, 4, 3), np.uint8),
    }
    pred = {
        "pose_enc": pred_pose,
        "hand": pred_hand,
        "hand_presence_logits": np.ones((length, 2), np.float32),
    }

    result = metrics_module.frame_metrics(raw, pred, loss_cfg)
    assert result is not None, "frame_metrics returned None"

    starts, use_len, _clipwin = metrics_module._clip_starts(length, clip_len, stride)
    indices = np.stack([np.arange(start, start + use_len) for start in starts])
    pred_batch = {
        "pose_enc": torch.from_numpy(
            metrics_module._rebase_clips_to_first(pred_pose[indices])
        ),
        "hand": torch.from_numpy(pred_hand[indices]),
        "hand_presence_logits": torch.ones(len(starts), use_len, 2),
    }
    gt_218 = _hands_to_218(gt_hands, length)
    batch = {
        "gt_pose_enc": torch.from_numpy(
            metrics_module._rebase_clips_to_first(gt_pose[indices])
        ),
        "hand_gt": torch.from_numpy(gt_218[indices]),
        "hand_kept": torch.from_numpy(kept[indices].astype(np.float32)),
        "hand_valid": torch.ones(len(starts), dtype=torch.bool),
        "mano_gt_valid": torch.ones(len(starts), dtype=torch.bool),
        "kpt21_gt": torch.from_numpy(kpt21_gt[indices]),
        "kpt21_gt_valid": torch.ones(len(starts), dtype=torch.bool),
        "image_hw": tuple(int(value) for value in raw["frames"].shape[1:3]),
    }
    normalization_enabled = bool(
        (loss_cfg["data"].get("camera_translation_normalization", {}) or {}).get("enabled")
    )
    trans_scales = metrics_module._camera_trans_scales(
        raw, loss_cfg["data"], clip_len, len(starts)
    )
    batch.update(trans_scales)

    criterion = build_criterion(loss_cfg["loss"])
    with torch.no_grad():
        expected_total, expected_logs = criterion(pred_batch, batch)

    expected = {}
    for group_name, loss_module in criterion.losses:
        for term_name, _function, _weight in loss_module.terms:
            log_key = _loss_log_key(expected_logs, group_name, term_name)
            expected[f"{group_name}__{term_name}"] = float(expected_logs[log_key])
    expected["total"] = float(expected_total)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = _REPO / "output" / "unit_test" / "lingbotmap" / "metrics_win_consistency" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    def report(message=""):
        print(message)
        lines.append(message)

    report(f"[config] {args.input}")
    report(
        f"[windows] clip_len={clip_len} stride={stride} T={length} -> "
        f"n_clips={len(starts)} clipwin={result['clipwin']}"
    )
    report("[groups] " + " | ".join(
        f"{group['label']} terms={len(group['terms'])} available={group['available']}"
        for group in result["groups"]
    ))
    reported_scales = {
        key: value[0].detach().cpu().tolist() for key, value in trans_scales.items()
    }
    report(f"[normalization] enabled={normalization_enabled} scales={reported_scales}")
    report("")
    report(f"{'term':<48}{'panel':>14}{'criterion':>14}{'abs diff':>14}  match")

    ok_all = True
    panel_keys = [term["key"] for group in result["groups"] for term in group["terms"]]
    if set(panel_keys) != set(expected) - {"total"}:
        ok_all = False
        report(f"term key mismatch: panel={sorted(panel_keys)} expected={sorted(set(expected)-{'total'})}")

    for key in panel_keys + ["total"]:
        panel_value = result["mean"].get(key)
        expected_value = expected.get(key)
        if panel_value is None or expected_value is None:
            ok_all = False
            report(f"{key:<48}{'missing':>42}")
            continue
        difference = abs(float(panel_value) - expected_value)
        matched = difference <= args.tol * max(1.0, abs(expected_value))
        ok_all = ok_all and matched
        report(
            f"{key:<48}{float(panel_value):>14.6f}{expected_value:>14.6f}"
            f"{difference:>14.2e}  {'yes' if matched else 'NO'}"
        )
        if key != "total" and len(result["per_frame"].get(key, [])) != length:
            ok_all = False
            report(f"  invalid per-frame length for {key}")

    report("")
    verdict = "PASS: panel matches training Criterion" if ok_all else "FAIL: mismatch detected"
    report(f"[result] {verdict}")
    report_path = out_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[out] {report_path}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
