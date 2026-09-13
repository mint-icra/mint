# -*- coding: utf-8 -*-
"""评测结果汇总 → report.json(机器可读)+ report.md(Markdown 表,可贴 PR/飞书)。

结果树: head -> dataset -> {seqs: {seq_id: metrics}, mean: 聚合} / 或 status(skipped/not_implemented)。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SeqResult:
    """单序列在单 head 上的结果。status: evaluated / skipped / not_implemented / error。"""
    head: str
    dataset: str
    seq_id: str
    status: str
    metrics: Dict[str, float] = field(default_factory=dict)
    note: str = ""


class Report:
    def __init__(self, ckpt: Optional[str], config: str, selection: dict | None = None):
        self.ckpt = ckpt
        self.config = config
        self.selection = dict(selection or {})
        self.rows: List[SeqResult] = []
        self.timings: Optional[dict] = None      # engine 采集:{total_s, datasets:{ds:{total_s,predict_s,n_seqs,heads}}}

    def set_timings(self, timings: dict):
        self.timings = timings

    def add(self, r: SeqResult):
        self.rows.append(r)

    # ---- 聚合 ----
    def _tree(self) -> dict:
        tree: dict = {}
        for r in self.rows:
            h = tree.setdefault(r.head, {})
            d = h.setdefault(r.dataset, {"seqs": {}, "status_counts": {}})
            d["status_counts"][r.status] = d["status_counts"].get(r.status, 0) + 1
            if r.status == "evaluated":
                d["seqs"][r.seq_id] = r.metrics
            elif r.seq_id == "*":                      # 数据集级状态(缺数据/未实现)
                d["note"] = r.note
        # 每 (head,dataset) 求均值
        for head_name, head in tree.items():
            for dataset_name, d in head.items():
                add_metric_summaries(
                    d, head_name=head_name, dataset_name=dataset_name,
                )
        return tree

    def dataset_subtree(self, ds_name: str) -> dict:
        """取单个数据集的 {head: {ds: node}} 局部树(结构与 _tree 一致,便于前端复用同一渲染)。
        供 engine 每数据集评完时回调传出该集局部结果(多卡分片下父进程按数据集跨卡合并)。"""
        sub: dict = {}
        for r in self.rows:
            if r.dataset != ds_name:
                continue
            d = sub.setdefault(r.head, {}).setdefault(r.dataset, {"seqs": {}, "status_counts": {}})
            d["status_counts"][r.status] = d["status_counts"].get(r.status, 0) + 1
            if r.status == "evaluated":
                d["seqs"][r.seq_id] = r.metrics
            elif r.seq_id == "*":
                d["note"] = r.note
        for head_name, head in sub.items():
            for dataset_name, d in head.items():
                add_metric_summaries(
                    d, head_name=head_name, dataset_name=dataset_name,
                )
        return sub

    def dump(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        tree = self._tree()
        obj = {"ckpt": self.ckpt, "config": self.config, "selection": self.selection,
               "heads": tree, "timings": self.timings}
        jpath = os.path.join(out_dir, "report.json")
        with open(jpath, "w") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)
        mpath = os.path.join(out_dir, "report.md")
        with open(mpath, "w") as f:
            f.write(self._markdown(tree))
        print(f"[report] 写出:\n  {jpath}\n  {mpath}", flush=True)
        return jpath

    def _markdown(self, tree: dict) -> str:
        L = [f"# benchmark 报告", "", f"- ckpt: `{self.ckpt or '(smoke: config 预训练骨干)'}`",
             f"- config: `{self.config}`", ""]
        if self.selection:
            end = self.selection.get("seq_end")
            L.insert(4, f"- 序列范围: `[{self.selection.get('seq_start', 0)}, {end if end is not None else '全部'})`")
            dataset_selection = self.selection.get("dataset_selection") or {}
            if dataset_selection:
                details = ", ".join(
                    f"{name}={options.get('sample_count', '全部')}条"
                    + (f"×≤{options['max_frames']}帧" if options.get("max_frames") else "")
                    + (
                        f" ({options['split_version']}/{options['fixed_tier']},"
                        f" hash={options.get('split_hash', '')[:12]})"
                        if options.get("fixed_tier")
                        else f" (seed={options.get('seed', 42)})"
                    )
                    for name, options in sorted(dataset_selection.items())
                )
                L.insert(5, f"- 分数据集抽样: `{details}`")
        for head in sorted(tree):
            L.append(f"## {head}")
            for ds in sorted(tree[head]):
                d = tree[head][ds]
                mean = d.get("mean")
                if not mean:
                    sc = d.get("status_counts", {})
                    note = d.get("note", "")
                    tag = ", ".join(f"{k}×{v}" for k, v in sc.items())
                    L.append(f"- **{ds}**: — ({tag}{'; ' + note if note else ''})")
                    continue
                cols = list(mean)
                L.append(f"- **{ds}** (序列数 {len(d['seqs'])}):")
                protocol = d.get("protocol") or {}
                if protocol:
                    if protocol.get("sequence_mode"):
                        L.append(
                            f"  协议: `{protocol.get('name')}`，"
                            f"{protocol.get('evaluation_split')}，"
                            f"{protocol.get('alignment')}，{protocol.get('inference')}"
                        )
                    else:
                        split = "官方" if protocol.get("official_split") else "非官方"
                        clip_mode = "单次前向" if protocol.get("clip_single_forward") else "含分窗前向"
                        L.append(
                            f"  协议: `{protocol.get('name')}`，{split} split，"
                            f"{protocol.get('segment_frames')} 帧/段，{clip_mode}"
                        )
                L.append("")
                L.append("| seq | " + " | ".join(cols) + " |")
                L.append("|" + "---|" * (len(cols) + 1))
                for sid, m in d["seqs"].items():
                    L.append(f"| {sid} | " + " | ".join(_fmt(m.get(c)) for c in cols) + " |")
                by_side = d.get("mean_by_side")
                if by_side:
                    for group in ("overall", "left", "right"):
                        if group not in by_side:
                            continue
                        values = by_side[group]
                        L.append(
                            f"| **{group} mean** | "
                            + " | ".join(f"**{_fmt(values.get(c))}**" for c in cols)
                            + " |"
                        )
                else:
                    L.append("| **mean** | " + " | ".join(f"**{_fmt(mean[c])}**" for c in cols) + " |")
                reference = (d.get("reference") or {}).get("metrics")
                if reference:
                    L.append(
                        "| **public reference** | "
                        + " | ".join(f"**{_fmt(reference.get(c))}**" for c in cols)
                        + " |"
                    )
                    delta = d.get("delta_vs_reference") or {}
                    L.append(
                        "| **ours - public** | "
                        + " | ".join(f"**{_fmt(delta.get(c))}**" for c in cols)
                        + " |"
                    )
                L.append("")
        L += self._timings_md()
        return "\n".join(L) + "\n"

    def _timings_md(self) -> List[str]:
        """耗时小节:每数据集 前向/总 + 各 head evaluate 耗时(秒)。无计时则空。"""
        t = self.timings
        if not t or not t.get("datasets"):
            return []
        heads = sorted({h for d in t["datasets"].values() for h in (d.get("heads") or {})})
        L = ["## 耗时", "", f"- 总耗时: **{t.get('total_s', 0.0):.1f}s**", ""]
        cols = [
            "序列", "实际前向", "复用", "读图(s)", "预处理(s)",
            "GPU前向(s)", "预测总(s)", "数据集总(s)",
        ] + [f"{h} eval(s)" for h in heads]
        L.append("| dataset | " + " | ".join(cols) + " |")
        L.append("|" + "---|" * (len(cols) + 1))
        for ds, d in t["datasets"].items():
            hd = d.get("heads") or {}
            stages = d.get("predict_stages") or {}
            cells = [
                str(d.get("n_seqs", 0)),
                str(d.get("predict_calls", d.get("n_seqs", 0))),
                str(d.get("predict_cache_hits", 0)),
                f"{stages.get('image_load_s', 0.0):.2f}",
                f"{stages.get('preprocess_s', 0.0):.2f}",
                f"{stages.get('forward_s', 0.0):.2f}",
                f"{d.get('predict_s', 0.0):.2f}",
                f"{d.get('total_s', 0.0):.2f}",
            ]
            cells += [(f"{hd[h]:.2f}" if h in hd else "—") for h in heads]
            L.append(f"| {ds} | " + " | ".join(cells) + " |")
        L.append("")
        return L


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{float(v):.4f}"


def _nanmean(vals):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    return float(a.mean()) if a.size else None


def _mean_metrics(seqs: dict) -> dict:
    keys = sorted({key for metrics in seqs.values() for key in metrics})
    return {
        key: _nanmean([metrics.get(key) for metrics in seqs.values()])
        for key in keys
    }


def add_metric_summaries(
    node: dict,
    head_name: str | None = None,
    dataset_name: str | None = None,
) -> None:
    """Add the legacy overall mean plus explicit overall/left/right summaries."""
    seqs = node.get("seqs") or {}
    if not seqs:
        return
    if head_name == "hands_coverage":
        from .protocols.hand_coverage import aggregate_sequence_metrics

        node.update(aggregate_sequence_metrics(
            seqs, dataset_name=dataset_name or "hot3d_hand_coverage",
        ))
        node.pop("mean_by_side", None)
        return
    if head_name == "camera_trajectory":
        from .camera_trajectory.metrics import aggregate_trajectory_metrics

        node.update(aggregate_trajectory_metrics(
            seqs, dataset_name=dataset_name or "camera_hot3d",
        ))
        node.pop("mean_by_side", None)
        return
    node["mean"] = _mean_metrics(seqs)
    side_seqs = {
        side: {
            seq_id: metrics
            for seq_id, metrics in seqs.items()
            if str(seq_id).rsplit("#", 1)[-1] == side
        }
        for side in ("left", "right")
    }
    if any(side_seqs.values()):
        node["mean_by_side"] = {"overall": dict(node["mean"])}
        for side in ("left", "right"):
            if side_seqs[side]:
                node["mean_by_side"][side] = _mean_metrics(side_seqs[side])


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(str(type(o)))
