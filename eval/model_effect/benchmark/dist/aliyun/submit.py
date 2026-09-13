#!/usr/bin/env python3
"""Submit an Aliyun distributed Benchmark without starting the Viewer."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml


_REPO = Path(__file__).resolve().parents[5]
_MODEL_EFFECT = _REPO / "eval" / "model_effect"
if str(_MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(_MODEL_EFFECT))

from benchmark.dist.aliyun import AliyunBenchmarkManager, AliyunConfig, load_defaults  # noqa: E402


def _config(path: str | None, nnodes: int | None) -> AliyunConfig:
    raw = load_defaults().to_dict()
    if path:
        with Path(path).open(encoding="utf-8") as handle:
            document = yaml.safe_load(handle) or {}
        override = document.get("aliyun", document)
        if not isinstance(override, dict):
            raise ValueError("--aliyun-config 必须是映射，或包含 aliyun: 映射")
        raw.update(override)
    if nnodes is not None:
        raw["nnodes"] = nnodes
    return AliyunConfig.from_mapping(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="提交 Aliyun PAI-DLC 多节点 Benchmark")
    parser.add_argument("--ckpt", action="append", required=True,
                        help="checkpoint 文件或目录；可重复传入以依次比较多个模型")
    parser.add_argument("--config", required=True, help="模型结构 YAML（所有 checkpoint 共用）")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--heads", default="all")
    parser.add_argument("--seq-start", type=int, default=0)
    parser.add_argument("--seq-end", type=int, default=None)
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--nnodes", type=int, default=None,
                        help="覆盖 Aliyun 节点数；每节点默认使用 8 张卡")
    parser.add_argument("--aliyun-config", default=None,
                        help="覆盖 defaults.yaml 的 YAML；可写 aliyun: 顶层块")
    args = parser.parse_args()

    model_config = Path(args.config).resolve()
    if not model_config.is_file():
        parser.error(f"模型 config 不存在: {model_config}")
    models = []
    for ckpt_value in args.ckpt:
        checkpoint = Path(ckpt_value).resolve()
        if not checkpoint.is_dir() and not checkpoint.is_file():
            parser.error(f"checkpoint 不存在: {checkpoint}")
        models.append({
            "run": checkpoint.parent.name,
            "step": checkpoint.name,
            "ckpt": str(checkpoint),
            "tag": f"{checkpoint.parent.name}_{checkpoint.name}",
            "label": f"{checkpoint.parent.name} / {checkpoint.name}",
            "config": str(model_config),
            "model": "lingbotmap",
        })
    try:
        config = _config(args.aliyun_config, args.nnodes)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    manager = AliyunBenchmarkManager(_REPO)
    result = manager.start(
        models=models, datasets=args.datasets, heads=args.heads,
        max_seqs=args.max_seqs, max_frames=args.max_frames,
        seq_start=args.seq_start, seq_end=args.seq_end, config=config,
    )
    if not result.get("ok"):
        parser.error(result.get("error") or "提交失败")
    while True:
        status = manager.status()
        if status.get("job_id") or not status.get("running"):
            print(json.dumps({
                "ok": bool(status.get("job_id")),
                "job_id": status.get("job_id"),
                "job_status": status.get("job_status"),
                "out": status.get("out"),
                "error": status.get("error"),
            }, ensure_ascii=False, indent=2))
            raise SystemExit(0 if status.get("job_id") else 1)
        time.sleep(1)


if __name__ == "__main__":
    main()
