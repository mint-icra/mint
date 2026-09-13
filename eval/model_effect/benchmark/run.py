#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""benchmark CLI 入口:训练好的学生 ckpt → data/benchmark 逐 head 量化指标 → report.json/md。

库编排在 engine.py(run_benchmark),能力清单在 engine.capabilities;viewer 也复用 run_benchmark
(见 visualization/viewer/store.py),保证 CLI 与网页行为一致。推理与几何解码复用本目录 predictor。

用法:
    PY=python
    $PY eval/model_effect/benchmark/run.py --ckpt output/model_train/<ts>/step_XXXX
    $PY eval/model_effect/benchmark/run.py --ckpt <step_*> --heads extrinsics,intrinsics \
        --datasets sintel --seq-start 20 --seq-end 53 --max-frames 50
    $PY eval/model_effect/benchmark/run.py --datasets sintel --max-seqs 1 --max-frames 8   # smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_BENCH = Path(__file__).resolve().parent
_REPO = _BENCH.parents[2]                      # benchmark -> model_effect -> eval -> <repo>
for p in (str(_BENCH.parent), str(_REPO)):     # 使 `import benchmark....` 可用
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.core.engine import run_benchmark, _default_out   # noqa: E402
from benchmark.core.registry import DATASETS, HEADS             # noqa: E402
import benchmark.datasets  # noqa: E402,F401  (触发数据集注册,供 --help 列名/校验)
import benchmark.heads     # noqa: E402,F401  (触发头注册)


def main():
    all_heads = HEADS.keys()
    all_ds = DATASETS.keys()
    ap = argparse.ArgumentParser(description="学生模型 benchmark 逐头量化评测")
    ap.add_argument("--ckpt", default=None,
                    help="accelerate save_state 目录(output/model_train/<ts>/step_*)或权重文件;不给则 smoke")
    ap.add_argument("--config", default=str(
        _REPO / "configs" / "training" / "mint_step2.yaml"),
                    help="模型结构 + size_hw(窗口=训练 clip_len,predictor 自动推断)")
    ap.add_argument("--data-root", default=str(_REPO / "data" / "benchmark"))
    ap.add_argument("--heads", default="all", help=f"逗号分隔或 all;可选 {all_heads}")
    ap.add_argument("--datasets", default="all", help=f"逗号分隔或 all;可选 {all_ds}")
    ap.add_argument("--max-seqs", type=int, default=None,
                    help="兼容参数：从 --seq-start 起最多取多少条")
    ap.add_argument("--seq-start", type=int, default=0,
                    help="每数据集序列范围起点（含，默认 0）")
    ap.add_argument("--seq-end", type=int, default=None,
                    help="每数据集序列范围终点（不含）；如 20/53 表示 [20,53)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument(
        "--dataset-selection-json", default="{}",
        help="按数据集抽样配置 JSON（sample_count/max_frames/sampling/seed）",
    )
    ap.add_argument(
        "--hand-coverage-test-segments", type=int,
        default=int(os.environ.get("HAND_COVERAGE_TEST_SEGMENTS", "437")),
        help="相机系手部覆盖率非官方 split 的确定性片段数；0=全部（默认 437）",
    )
    ap.add_argument(
        "--hand-coverage-split-seed", type=int,
        default=int(os.environ.get("HAND_COVERAGE_SPLIT_SEED", "42")),
        help="相机系手部覆盖率片段选择 seed（默认 42）",
    )
    ap.add_argument("--windowed", action="store_true",
                    help="长序列按训练 clip_len 分窗独立前向；coverage-aware 81帧协议始终强制单次前向")
    ap.add_argument(
        "--window-batch-size", type=int,
        default=int(os.environ.get("BENCH_WINDOW_BATCH_SIZE", "4")),
        help="分窗模式一次并行前向的窗口数(默认4；OOM自动减半)")
    ap.add_argument(
        "--image-workers", type=int,
        default=int(os.environ.get("BENCH_IMAGE_WORKERS", "8")),
        help="每个GPU进程并发读取JPEG的线程数(默认8；输出顺序不变)")
    ap.add_argument(
        "--hand-mode", choices=("hard", "blend", "smooth"), default="hard",
        help="手部拼窗/后处理模式；smooth=重叠融合后进行相机系 UKF+RTS 平滑（默认 hard）")
    ap.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs", "max-autotune"),
        default=None,
        help="可选 torch.compile 模式；固定分窗推荐 reduce-overhead（含 CUDA Graph）",
    )
    ap.add_argument(
        "--fp8-mode", choices=("dynamic",), default=None,
        help="可选 torchao FP8 dynamic activation + weight，仅量化 aggregator 大 Linear",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="lingbotmap",
                    help="推理模型(inference.registry 注册名);非 lingbotmap 走 registry 构建引擎")
    ap.add_argument("--shard-index", type=int, default=None,
                    help="多卡分片评测:本分片序号(0..shard-count-1);配合 --shard-count 与外层 CUDA_VISIBLE_DEVICES")
    ap.add_argument("--shard-count", type=int, default=None, help="多卡分片评测:分片总数(=参与卡数)")
    args = ap.parse_args()

    try:
        dataset_selection = json.loads(args.dataset_selection_json or "{}")
    except json.JSONDecodeError as exc:
        ap.error(f"--dataset-selection-json 不是有效 JSON: {exc}")

    os.environ["HAND_COVERAGE_TEST_SEGMENTS"] = str(args.hand_coverage_test_segments)
    os.environ["HAND_COVERAGE_SPLIT_SEED"] = str(args.hand_coverage_split_seed)

    # 学生推理引擎(懒加载:import torch 等重依赖只在此发生)
    from benchmark.predictor import StudentPredictor
    single_forward = not args.windowed
    if args.model == "lingbotmap":
        predictor = StudentPredictor(
            args.config,
            ckpt=args.ckpt,
            single_forward=single_forward,
            window_batch_size=args.window_batch_size,
            image_workers=args.image_workers,
            hand_mode=args.hand_mode,
            compile_mode=args.compile_mode,
            fp8_mode=args.fp8_mode,
        )
    else:   # 其它模型(如 vggt):复用 registry 构建引擎再包 StudentPredictor,与 viewer 语义一致
        from inference.registry import get_predictor
        engine = get_predictor(
            args.model, config=args.config, ckpt=args.ckpt,
            compile_mode=args.compile_mode, fp8_mode=args.fp8_mode,
        )
        predictor = StudentPredictor.from_engine(
            engine,
            ckpt=args.ckpt,
            single_forward=single_forward,
            window_batch_size=args.window_batch_size,
            image_workers=args.image_workers,
            hand_mode=args.hand_mode,
        )
    print(f"[run] model={args.model} ckpt={args.ckpt} single_forward={single_forward} "
          f"window_batch={args.window_batch_size} image_workers={args.image_workers} "
          f"hand_mode={args.hand_mode} compile={args.compile_mode or 'off'} "
          f"fp8={args.fp8_mode or 'off'} shard={args.shard_index}/{args.shard_count} "
          "coverage_protocol=forced-single-forward", flush=True)

    # 多卡分片时:向父进程回报进度(父进程解析 stdout 聚合)。控制行:
    #   [DSINIT] ds=a,b,c            —— 本分片将跑的数据集顺序(父进程据此列进度行)
    #   [SHARD] done=X total=Y       —— 本分片总体已完成/总序列(粗粒度总进度)
    #   [DS] <ds>|<done>|<total>     —— 本分片某数据集的序列 done/total(父进程按数据集跨卡聚合)
    #   [LIVE] {...}                 —— 当前卡正在处理的序列/阶段，父进程原地更新固定状态行
    #   [RESULT] {...}               —— 单 head/序列结果，父进程立即重算滚动均值
    out_dir = args.out or _default_out()
    _g = {"done": 0, "total": 0}                 # 本分片总体
    _ds = {}                                     # ds -> {"done":x, "total":y}

    def _event(tag, payload):
        def _json_value(value):
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(type(value).__name__)

        print(f"[{tag}] " + json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), default=_json_value,
        ), flush=True)

    def _on_progress(p):
        kind = p.get("kind")
        if kind == "init":
            print("[DSINIT] ds=" + ",".join(p.get("datasets") or []), flush=True)
        elif kind == "seqs":                     # 每数据集枚举完:记本分片该集序列总数
            ds = p.get("ds")
            _ds.setdefault(ds, {"done": 0, "total": 0})["total"] = int(p.get("seq_total") or 0)
            _g["total"] += int(p.get("seq_total") or 0)
            print(f"[DS] {ds}|{_ds[ds]['done']}|{_ds[ds]['total']}", flush=True)
        elif kind == "seq_done":                 # 每序列前向+逐头评测都结束后触发一次 → 计一条完成
            ds = p.get("ds")                      # (含推理异常的 error 序列;计在评完后,故 100% 才真表示全评完)
            d = _ds.setdefault(ds, {"done": 0, "total": 0})
            d["done"] += 1
            _g["done"] += 1
            print(f"[DS] {ds}|{d['done']}|{d['total']}", flush=True)
            print(f"[SHARD] done={_g['done']} total={_g['total']}", flush=True)
        if kind in {"dataset", "prepare", "predict", "predict_step", "evaluate", "seq_done"}:
            _event("LIVE", p)

    def _on_result(result):
        _event("RESULT", {
            "head": result.head,
            "dataset": result.dataset,
            "seq_id": result.seq_id,
            "status": result.status,
            "metrics": result.metrics,
            "note": result.note,
        })

    def _on_dataset_done(ds_name, subtree):
        # 本分片该数据集评完:局部结果落盘 _ds/<ds>.json,并告知父进程(全卡都完成后合并展开)
        import json
        import os
        dd = os.path.join(out_dir, "_ds")
        os.makedirs(dd, exist_ok=True)
        with open(os.path.join(dd, f"{ds_name}.json"), "w", encoding="utf-8") as f:
            json.dump({"ckpt": args.ckpt, "config": args.config,
                       "selection": {"seq_start": args.seq_start, "seq_end": args.seq_end,
                                     "max_frames": args.max_frames,
                                     "dataset_selection": dataset_selection,
                                     "hand_mode": args.hand_mode},
                       "heads": subtree}, f, ensure_ascii=False)
        print(f"[DSDONE] ds={ds_name}", flush=True)

    try:
        run_benchmark(predictor, heads=args.heads, datasets=args.datasets,
                      data_root=args.data_root, max_seqs=args.max_seqs, max_frames=args.max_frames,
                      seq_start=args.seq_start, seq_end=args.seq_end,
                      dataset_selection=dataset_selection, hand_mode=args.hand_mode,
                      ckpt=args.ckpt, config=args.config, out_dir=out_dir,
                      on_result=_on_result, on_progress=_on_progress,
                      on_dataset_done=_on_dataset_done,
                      shard_index=args.shard_index, shard_count=args.shard_count)
    except ValueError as e:
        ap.error(str(e))


if __name__ == "__main__":
    main()
