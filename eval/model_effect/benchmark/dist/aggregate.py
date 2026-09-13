# -*- coding: utf-8 -*-
"""合并多卡分片评测:BASE/gpu*/report.json → 合并 seqs、按 report.py 口径重算 mean(nanmean)、
汇总 status_counts,写出 BASE/report.json + report.md。分片仅按序列切,head/dataset 结构一致,直接并集。

可作库调用(viewer store 多卡评测结束后 `from benchmark.dist.aggregate import aggregate; aggregate(base)`),
也可命令行 `python aggregate.py <BASE>`。"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

_MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(_MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(_MODEL_EFFECT))

from benchmark.report import add_metric_summaries  # noqa: E402


def fmt(v):
    return "—" if v is None else f"{v:.4f}"


def _merge_timings(shards: list) -> dict | None:
    """Preserve both wall-clock estimates and summed GPU work across parallel shards."""
    timings = [item.get("timings") for item in shards if item.get("timings")]
    if not timings:
        return None
    merged = {
        "total_s": max(float(item.get("total_s", 0.0)) for item in timings),
        "gpu_total_s": sum(float(item.get("total_s", 0.0)) for item in timings),
        "shards": len(timings),
        "datasets": {},
    }
    dataset_names = {
        name for item in timings for name in (item.get("datasets") or {})
    }
    for name in sorted(dataset_names):
        parts = [
            item["datasets"][name]
            for item in timings
            if name in (item.get("datasets") or {})
        ]
        node = {
            "total_s": max(float(part.get("total_s", 0.0)) for part in parts),
            "gpu_total_s": sum(float(part.get("total_s", 0.0)) for part in parts),
            "predict_s": sum(float(part.get("predict_s", 0.0)) for part in parts),
            "predict_wall_s": max(float(part.get("predict_s", 0.0)) for part in parts),
            "n_seqs": sum(int(part.get("n_seqs", 0)) for part in parts),
            "predict_calls": sum(int(part.get("predict_calls", part.get("n_seqs", 0))) for part in parts),
            "predict_cache_hits": sum(int(part.get("predict_cache_hits", 0)) for part in parts),
            "predicted_frames": sum(int(part.get("predicted_frames", 0)) for part in parts),
            "reused_frames": sum(int(part.get("reused_frames", 0)) for part in parts),
            "predict_stages": {},
            "heads": {},
        }
        for part in parts:
            for key, value in (part.get("predict_stages") or {}).items():
                node["predict_stages"][key] = (
                    node["predict_stages"].get(key, 0.0) + float(value)
                )
            for key, value in (part.get("heads") or {}).items():
                node["heads"][key] = node["heads"].get(key, 0.0) + float(value)
        merged["datasets"][name] = node
    return merged


def merge_reports(shards: list) -> dict:
    """把多个分片 report dict(结构 {ckpt,config,heads:{head:{ds:node}}})按序列并集合并、重算 mean。
    分片仅按序列切、head/dataset 结构一致,直接并集。总合并(各卡完整 report)与单数据集跨卡合并
    (各卡 _ds/<ds>.json)复用同一逻辑。返回 merged dict;空输入返回 {heads:{}}。"""
    merged = None
    for d in shards:
        if merged is None:
            merged = {"ckpt": d.get("ckpt"), "config": d.get("config"),
                      "selection": d.get("selection") or {}, "heads": {}}
        for head, dss in (d.get("heads") or {}).items():
            H = merged["heads"].setdefault(head, {})
            for ds, node in dss.items():
                N = H.setdefault(ds, {"seqs": {}, "status_counts": {}})
                N["seqs"].update(node.get("seqs", {}))
                for k, c in node.get("status_counts", {}).items():
                    N["status_counts"][k] = N["status_counts"].get(k, 0) + c
                if node.get("note") and "note" not in N:
                    N["note"] = node["note"]
    if merged is None:
        return {"heads": {}}
    # 重算 mean(并集后,按 report.py 的 _nanmean 逐指标)
    for head, dss in merged["heads"].items():
        for ds, N in dss.items():
            seqs = N["seqs"]
            if not seqs:
                continue
            add_metric_summaries(N, head_name=head, dataset_name=ds)
    timings = _merge_timings(shards)
    if timings is not None:
        merged["timings"] = timings
    return merged


def merge_result_rows(shard_rows: dict, *, ckpt=None, config="", selection=None) -> dict:
    """Build a live report from per-shard ``[RESULT]`` events."""
    shards = []
    for rows in shard_rows.values():
        tree = {}
        for row in rows.values():
            required = {"head", "dataset", "seq_id", "status"}
            if not required.issubset(row):
                continue
            node = tree.setdefault(row["head"], {}).setdefault(
                row["dataset"], {"seqs": {}, "status_counts": {}},
            )
            status = row.get("status") or "error"
            node["status_counts"][status] = node["status_counts"].get(status, 0) + 1
            if status == "evaluated":
                node["seqs"][row["seq_id"]] = dict(row.get("metrics") or {})
            elif row.get("seq_id") == "*" and row.get("note"):
                node["note"] = row["note"]
        shards.append({
            "ckpt": ckpt,
            "config": config,
            "selection": dict(selection or {}),
            "heads": tree,
        })
    return merge_reports(shards) if shards else {
        "ckpt": ckpt,
        "config": config,
        "selection": dict(selection or {}),
        "heads": {},
    }


def aggregate(base) -> dict:
    """合并 base/gpu*/report.json → base/report.json + report.md,返回 merged dict。
    无任何分片 report 时抛 FileNotFoundError(上层可捕获给清晰报错,而非裸 assert)。"""
    base = Path(base)
    shards = sorted(glob.glob(str(base / "gpu*" / "report.json")))
    if not shards:
        raise FileNotFoundError(f"{base} 下无 gpu*/report.json,分片可能全部失败或被取消")
    merged = merge_reports([json.load(open(f, encoding="utf-8")) for f in shards])

    (base / "report.json").write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown
    L = [f"# benchmark 全量评测({len(shards)} 卡合并)\n", f"- ckpt: `{merged['ckpt']}`", f"- config: `{merged['config']}`"]
    selection = merged.get("selection") or {}
    if selection:
        end = selection.get("seq_end")
        L.append(f"- 序列范围: `[{selection.get('seq_start', 0)}, {end if end is not None else '全部'})`\n")
        dataset_selection = selection.get("dataset_selection") or {}
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
            L.append(f"- 分数据集抽样: `{details}`\n")
    else:
        L.append("")
    for head, dss in merged["heads"].items():
        for ds, N in dss.items():
            seqs, mean = N["seqs"], N.get("mean")
            L.append(f"\n## {head} / {ds}  (序列 {len(seqs)}, {N['status_counts']})\n")
            if not mean:
                continue
            cols = list(mean)
            protocol = N.get("protocol") or {}
            if protocol:
                if protocol.get("sequence_mode"):
                    L.append(
                        f"协议: `{protocol.get('name')}`，{protocol.get('evaluation_split')}，"
                        f"{protocol.get('alignment')}，{protocol.get('inference')}\n"
                    )
                else:
                    split = "官方" if protocol.get("official_split") else "非官方"
                    clip_mode = "单次前向" if protocol.get("clip_single_forward") else "含分窗前向"
                    L.append(
                        f"协议: `{protocol.get('name')}`，{split} split，"
                        f"{protocol.get('segment_frames')} 帧/段，{clip_mode}\n"
                    )
            L.append("| seq | " + " | ".join(cols) + " |")
            L.append("|" + "---|" * (len(cols) + 1))
            for sid in sorted(seqs):
                m = seqs[sid]
                L.append(f"| {sid} | " + " | ".join(fmt(m.get(c)) for c in cols) + " |")
            by_side = N.get("mean_by_side")
            if by_side:
                for group in ("overall", "left", "right"):
                    if group not in by_side:
                        continue
                    values = by_side[group]
                    L.append(
                        f"| **{group} mean** | "
                        + " | ".join(f"**{fmt(values.get(c))}**" for c in cols)
                        + " |"
                    )
            else:
                L.append("| **mean** | " + " | ".join(f"**{fmt(mean[c])}**" for c in cols) + " |")
            reference = (N.get("reference") or {}).get("metrics")
            if reference:
                L.append(
                    "| **public reference** | "
                    + " | ".join(f"**{fmt(reference.get(c))}**" for c in cols)
                    + " |"
                )
                delta = N.get("delta_vs_reference") or {}
                L.append(
                    "| **ours - public** | "
                    + " | ".join(f"**{fmt(delta.get(c))}**" for c in cols)
                    + " |"
                )
    timings = merged.get("timings")
    if timings:
        L.append(f"\n## 耗时\n\n- 多卡墙钟估计: **{timings['total_s']:.1f}s**")
        L.append("\n| dataset | GT序列 | 实际前向 | 复用 | 预测GPU累计(s) | 预测墙钟估计(s) | 数据集墙钟估计(s) |")
        L.append("|---|---:|---:|---:|---:|---:|---:|")
        for name, node in timings["datasets"].items():
            L.append(
                f"| {name} | {node['n_seqs']} | {node['predict_calls']} | "
                f"{node['predict_cache_hits']} | {node['predict_s']:.2f} | "
                f"{node['predict_wall_s']:.2f} | {node['total_s']:.2f} |"
            )
    (base / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    print(f"[aggregate] 合并 {len(shards)} 分片 → {base}/report.json + report.md")
    for head, dss in merged["heads"].items():
        for ds, N in dss.items():
            print(f"  {head}/{ds}: 序列={len(N['seqs'])} {N['status_counts']}")
            if N.get("mean"):
                print("    mean:", {k: (round(v, 2) if v is not None else None) for k, v in N["mean"].items()})
    return merged


if __name__ == "__main__":
    import sys
    aggregate(Path(sys.argv[1]))
