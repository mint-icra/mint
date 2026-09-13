#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相机轨迹目检:跑一段序列 → Umeyama-Sim3 对齐 → 指标 + 轨迹图 + npz。

复用 benchmark 的 predictor / 数据集适配器 / extrinsics 头,口径与 run.py 完全一致,
只是额外把对齐后的预测中心与 GT 中心画出来,方便肉眼判断轨迹对不对。

用法:
    PY=python
    $PY eval/model_effect/benchmark/tools/traj_check.py --input P0001_10a27bf7 --max-frames 512
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_BENCH = Path(__file__).resolve().parents[1]                 # tools/ -> benchmark/
_ME = _BENCH.parent                                          # model_effect/
_REPO = _ME.parents[1]                                       # <repo>
for p in (str(_ME), str(_REPO)):
    if p not in sys.path:
        sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description="HOT3D 相机轨迹目检(对齐后 pred vs GT)")
    ap.add_argument("--input", nargs="+", required=True, help="HOT3D 序列名,如 P0001_10a27bf7")
    ap.add_argument("--dataset", default="hot3d")
    ap.add_argument("--config", default=str(
        _REPO / "configs" / "training" / "mint_step2.yaml"))
    ap.add_argument("--ckpt", default=None, help="不给=用 config 里的预训练骨干(骨干+相机头本就冻结)")
    ap.add_argument("--data-root", default=str(_REPO / "data" / "benchmark"))
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--stride", type=int, default=1, help="抽帧步长,>1 可在同样显存下覆盖更长时间跨度")
    ap.add_argument("--windowed", action="store_true", help="退回按 clip_len 分窗(会窗间断裂)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import os
    os.environ.setdefault("HOT3D_SEQS", ",".join(args.input))
    os.environ.setdefault("HOT3D_HANDS", "left")              # 轨迹与手无关,只取一条避免重复 GT

    import benchmark.datasets  # noqa: F401  触发注册
    import benchmark.heads     # noqa: F401
    from benchmark.core.registry import DATASETS, HEADS
    from benchmark.predictor import StudentPredictor

    out_dir = Path(args.out or (
        _REPO / "output" / "eval" / "traj_check" /
        datetime.now().strftime("%Y%m%d_%H%M%S")))
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = DATASETS.get(args.dataset)(args.data_root)
    head = HEADS.get("extrinsics")()
    predictor = StudentPredictor(args.config, ckpt=args.ckpt,
                                 single_forward=not args.windowed, image_workers=8)

    summary = []
    for gt in ds.iter_sequences(max_seqs=len(args.input),
                                max_frames=args.max_frames * args.stride):
        name = gt.seq_id
        paths = list(gt.image_paths)[::args.stride][:args.max_frames]
        gt_w2c = np.asarray(gt.extrinsic_w2c, np.float64)[::args.stride][:args.max_frames]

        t0 = time.perf_counter()
        pred = predictor.predict(paths, hw=tuple(gt.hw))
        dt = time.perf_counter() - t0
        print(f"[{name}] {len(paths)} 帧前向 {dt:.1f}s ({dt / max(1, len(paths)) * 1e3:.1f} ms/帧)",
              flush=True)

        # 与 extrinsics 头完全同一套对齐/指标代码
        gt_sub = type(gt)(seq_id=name, image_paths=paths, hw=gt.hw,
                          intrinsic=gt.intrinsic, extrinsic_w2c=gt_w2c,
                          capability=gt.capability, meta=gt.meta)
        aligned = head.align(head.extract(pred), gt_sub)
        metrics = head.metrics(aligned, gt_sub)
        print(f"[{name}] " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                       for k, v in metrics.items()), flush=True)

        pc, gc = aligned["pc"], aligned["gc"]
        err = np.linalg.norm(pc - gc, axis=1)
        np.savez_compressed(out_dir / f"{name.replace('#', '_')}_traj.npz",
                            pred_centers=pc, gt_centers=gc, err=err,
                            pred_c2w=pred.extrinsic_c2w, gt_w2c=gt_w2c,
                            **{k: v for k, v in metrics.items()})
        _plot(pc, gc, err, metrics, name, out_dir / f"{name.replace('#', '_')}_traj.png")
        summary.append({"seq": name, "frames": len(paths), "forward_s": dt, **metrics})

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[out] {out_dir}", flush=True)


def _plot(pc, gc, err, metrics, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9))
    title = (f"{name}  |  n={metrics['n']}  ATE_RMSE={metrics['ATE_RMSE']:.4f} m  "
             f"RPE_t={metrics['RPE_t']:.4f} m  RPE_rot={metrics['RPE_rot_deg']:.3f}°  "
             f"sim3_scale={metrics['scale']:.4f}")
    fig.suptitle(title, fontsize=13)

    ax = fig.add_subplot(2, 3, 1, projection="3d")
    ax.plot(*gc.T, color="tab:green", lw=1.4, label="GT")
    ax.plot(*pc.T, color="tab:red", lw=1.0, label="pred (Sim3 aligned)")
    ax.scatter(*gc[0], color="k", s=25, marker="o")
    ax.set_title("3D"); ax.legend(fontsize=8)
    _equal3d(ax, gc)

    for i, (a, b, lbl) in enumerate(((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))):
        ax = fig.add_subplot(2, 3, i + 2)
        ax.plot(gc[:, a], gc[:, b], color="tab:green", lw=1.4, label="GT")
        ax.plot(pc[:, a], pc[:, b], color="tab:red", lw=1.0, label="pred")
        ax.scatter(gc[0, a], gc[0, b], color="k", s=25)
        ax.set_title(lbl); ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=.3)
        if i == 0:
            ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 5)
    ax.plot(err, color="tab:blue", lw=1.0)
    ax.axhline(float(np.sqrt((err ** 2).mean())), color="tab:orange", ls="--",
               lw=1.0, label="RMSE")
    ax.set_title("逐帧 ATE (m)"); ax.set_xlabel("frame"); ax.grid(alpha=.3); ax.legend(fontsize=8)

    ax = fig.add_subplot(2, 3, 6)
    for k, c in zip(range(3), ("tab:red", "tab:green", "tab:blue")):
        ax.plot(gc[:, k], color=c, lw=1.2, label=f"GT {'XYZ'[k]}")
        ax.plot(pc[:, k], color=c, lw=1.0, ls="--", alpha=.75)
        ax.set_title("各轴位移 (实线 GT / 虚线 pred)")
    ax.set_xlabel("frame"); ax.grid(alpha=.3); ax.legend(fontsize=7, ncol=3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[plot] {path}", flush=True)


def _equal3d(ax, pts):
    center = (pts.max(0) + pts.min(0)) / 2
    radius = float(np.max(pts.max(0) - pts.min(0))) / 2 or 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


if __name__ == "__main__":
    main()
