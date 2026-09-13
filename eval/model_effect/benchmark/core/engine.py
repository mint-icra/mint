# -*- coding: utf-8 -*-
"""benchmark 核心编排(库):
  - run_benchmark(...)  逐 dataset × head 评测,写 report.json/md;CLI(run.py)与 viewer(store.py)共用。
  - capabilities(...)   出「能力清单」(heads/datasets 的模态需求与提供、是否实现,+ 模型产出能力),
                        供选择面板做「功能↔��据集」三态联动,不加载权重。

编排逻辑:给定**已构造**的 predictor(可复用外部已加载引擎,避免重复加载权重),逐数据集×头
评测,能力不匹配/数据集未实现/模型不产该模态则跳过并记账。触发注册的 datasets/heads import
放在函数体内,使 `import benchmark` 门面不提前拉起重依赖(joblib/cv2)、也避免循环 import。
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_BENCH = Path(__file__).resolve().parents[1]   # core -> benchmark
_REPO = _BENCH.parents[2]                      # benchmark -> model_effect -> eval -> <repo>
for _p in (str(_BENCH.parent), str(_REPO)):    # 使 `import benchmark....` 可用
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmark.core.registry import DATASETS, HEADS          # noqa: E402
from benchmark.report import Report, SeqResult          # noqa: E402


def _prediction_group(seq, position: int):
    """Return a shared-prediction group, or a unique singleton for normal sequences."""
    group = (seq.meta or {}).get("prediction_group")
    return ("shared", str(group)) if group is not None else ("single", position)


def _grouped_shard(seqs: list, shard_index: int | None,
                   shard_count: int | None) -> list:
    """Keep shared-input GT rows together and greedily balance unique input frames."""
    if not shard_count or shard_count <= 1:
        return seqs
    index = int(shard_index or 0)
    count = int(shard_count)
    if index < 0 or index >= count:
        raise ValueError(f"shard_index={index} must be in [0, {count})")

    groups: list[dict] = []
    by_key = {}
    for position, seq in enumerate(seqs):
        key = _prediction_group(seq, position)
        group_index = by_key.get(key)
        if group_index is None:
            group_index = len(groups)
            by_key[key] = group_index
            groups.append({"members": [], "frames": 0, "order": group_index})
        group = groups[group_index]
        group["members"].append(seq)
        group["frames"] = max(group["frames"], len(seq.image_paths))

    loads = [0] * count
    assignments = [None] * len(groups)
    for group in sorted(groups, key=lambda item: (-item["frames"], item["order"])):
        target = min(range(count), key=lambda shard: (loads[shard], shard))
        assignments[group["order"]] = target
        loads[target] += group["frames"]
    return [
        seq
        for group, target in zip(groups, assignments)
        if target == index
        for seq in group["members"]
    ]


def _slice_sequence_groups(seqs: list, start: int = 0,
                           end: int | None = None) -> list:
    """Slice stable visual inputs while keeping shared-GT rows together."""
    groups: list[list] = []
    by_key = {}
    for position, seq in enumerate(seqs):
        key = _prediction_group(seq, position)
        group_index = by_key.get(key)
        if group_index is None:
            group_index = len(groups)
            by_key[key] = group_index
            groups.append([])
        groups[group_index].append(seq)
    selected = groups[int(start):end]
    return [seq for group in selected for seq in group]


def _prediction_cache_key(seq):
    """Validate the cheap group tag against frame boundaries before reusing output."""
    group = (seq.meta or {}).get("prediction_group")
    if group is None or not seq.image_paths:
        return None
    return (
        str(group), tuple(seq.hw), len(seq.image_paths),
        str(seq.image_paths[0]), str(seq.image_paths[-1]),
    )


def _default_out() -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return str(_REPO / "output" / "eval" / "benchmark" / ts)


def capabilities(config_path: str | None = None) -> dict:
    """能力清单:给面板判「功能↔数据集」三态用,不加载权重。
      heads:    [{name, required_gt:[...], implemented}]
      datasets: [{name, capability:[...], implemented}]
      model_capability: 模型实际产出的模态(恒含 extrinsic/intrinsic;hand/depth 按 config 头是否开启推断)
    """
    import benchmark.datasets  # noqa: F401  (触发数据集注册)
    import benchmark.heads     # noqa: F401  (触发头注册)

    heads = [{"name": n,
              "required_gt": sorted(HEADS.get(n).required_gt),
              "implemented": bool(getattr(HEADS.get(n), "implemented", True))}
             for n in HEADS.keys()]
    datasets = [{"name": n,
                 "capability": sorted(DATASETS.get(n).capability),
                 "implemented": bool(getattr(DATASETS.get(n), "implemented", True))}
                for n in DATASETS.keys()]

    # 模型产出能力:pose_enc 恒解出 extrinsic + intrinsic;hand/depth 看该 run 的 config 是否开对应头。
    model_cap = {"extrinsic", "intrinsic"}
    if config_path and os.path.isfile(config_path):
        try:
            import yaml
            cfg = yaml.safe_load(open(config_path)) or {}
            m = cfg.get("model") or {}
            if m.get("hand_head"):
                model_cap.add("hand")
                if m.get("enable_hand_presence"):
                    model_cap.add("hand_coverage")
            if m.get("depth_head") or m.get("enable_depth"):
                model_cap.add("depth")
        except Exception:
            pass
    return {"heads": heads, "datasets": datasets, "model_capability": sorted(model_cap)}


def dataset_sizes(data_root=None, datasets="all") -> dict:
    """统计每个数据集的**序列条数 / 总帧数**(不加载模型,仅枚举 GT),供面板在跑前显示规模。
      返回 {ds_name: {n_seqs:int|None, n_frames:int|None, note:str}};缺数据/未实现/枚举失败 →
      计数 None + note 说明。枚举会读 GT(逐序列),有一定耗时 → 上层放后台线程 + 缓存,别阻塞 UI。
    """
    import benchmark.datasets  # noqa: F401  (触发数据集注册)

    data_root = data_root or str(_REPO / "data" / "benchmark")
    names = list(DATASETS.keys()) if datasets == "all" else [d.strip() for d in datasets.split(",")]
    out = {}
    for n in names:
        klass = DATASETS.get(n)
        if klass is None:
            out[n] = {"n_seqs": None, "n_frames": None, "note": "未知数据集"}
            continue
        if not bool(getattr(klass, "implemented", True)):
            out[n] = {"n_seqs": None, "n_frames": None, "note": "数据集未实现"}
            continue
        try:
            ds = klass(data_root)
            n_seqs, n_frames = ds.count_sequences()  # 廉价计数(重集已重写为只走目录);默认精确走 iter
            out[n] = {
                "n_seqs": n_seqs,
                "n_frames": n_frames,
                "note": str(getattr(ds, "data_source_note", "") or ""),
                "source_root": str(getattr(ds, "root", "") or ""),
            }
        except (FileNotFoundError, NotImplementedError) as e:
            out[n] = {"n_seqs": None, "n_frames": None, "note": str(e)[:120]}
        except Exception as e:  # noqa: BLE001  单集枚举异常不拖垮整体
            out[n] = {"n_seqs": None, "n_frames": None, "note": f"枚举失败: {str(e)[:100]}"}
    return out


def run_benchmark(predictor, *, heads="all", datasets="all", data_root=None,
                  max_seqs=None, max_frames=None, seq_start=0, seq_end=None,
                  dataset_selection=None, hand_mode="hard",
                  ckpt=None, config="",
                  out_dir=None, on_result=None, should_cancel=None, on_progress=None,
                  shard_index=None, shard_count=None, on_dataset_done=None) -> str:
    """核心编排:给定**已构造**的 predictor(可复用外部已加载引擎,避免重复加载权重),逐数据集×头
    评测并写 report。CLI 与 viewer 共用此函数,保证行为一致。

    heads/datasets:'all' 或逗号分隔名(未知名抛 ValueError)。
    seq_start/seq_end:每个数据集按稳定输入顺序取左闭右开区间 [start,end)，再做多卡分片；
        共享同一视频的多条 GT（如 HOT3D 左/右手）视为一个输入，不会被范围或分片拆开。
        max_seqs 保留兼容，等价于从 seq_start 起最多取 max_seqs 个输入。
    dataset_selection:按数据集配置确定性抽样数量、seed 和单条帧上限。
    on_result(SeqResult):每条结果回调(上层实时更新进度用)。
    should_cancel()->bool:每序列前向前查一次,True 则提前收尾(已算结果照常落盘)。
    on_progress(dict):分步进度回调,字段 kind∈{init,dataset,seqs,seq,predict,evaluate,seq_done},
        含 ds/ds_i/ds_total/seq_i/seq_total/seq_id/stage,供上层画确定进度条与「到哪一步」。
        seq_done 在单序列前向+逐头评测都结束后触发恰好一次(含推理异常的 error 序列),供上层按
        「已评完序列数」计进度——故到 100% 才真表示全部序列评完(而非最后一条刚开始前向)。
    shard_index/shard_count:多卡分片评测用。共享输入的 GT 行保持同卡，再按唯一输入帧数贪心均衡；
        head/dataset 结构不变，各分片 report 可由 dist/aggregate 按序列并集合并。
    返回输出目录(含 report.json / report.md)。
    """
    import benchmark.datasets  # noqa: F401  (触发数据集注册)
    import benchmark.heads     # noqa: F401  (触发头注册)
    from benchmark.datasets.base import normalize_dataset_selection

    all_heads, all_ds = HEADS.keys(), DATASETS.keys()
    heads_sel = all_heads if heads == "all" else [h.strip() for h in heads.split(",")]
    ds_sel = (
        [name for name in all_ds if getattr(DATASETS.get(name), "default_enabled", True)]
        if datasets == "all" else [d.strip() for d in datasets.split(",")]
    )
    for h in heads_sel:
        if h not in all_heads:
            raise ValueError(f"未知 head {h!r};可选 {all_heads}")
    for d in ds_sel:
        if d not in all_ds:
            raise ValueError(f"未知 dataset {d!r};可选 {all_ds}")

    range_start = int(seq_start or 0)
    if range_start < 0:
        raise ValueError("seq_start 必须 >= 0")
    range_end = int(seq_end) if seq_end is not None else None
    if max_seqs is not None:
        max_end = range_start + int(max_seqs)
        range_end = max_end if range_end is None else min(range_end, max_end)
    if range_end is not None and range_end <= range_start:
        raise ValueError("序列范围必须满足 seq_end > seq_start（左闭右开）")
    dataset_selection = normalize_dataset_selection(dataset_selection)

    data_root = data_root or str(_REPO / "data" / "benchmark")
    out_dir = out_dir or _default_out()
    head_objs = [HEADS.get(h)() for h in heads_sel]
    report = Report(
        ckpt=ckpt,
        config=config,
        selection={"seq_start": range_start, "seq_end": range_end,
                   "max_frames": max_frames,
                   "dataset_selection": dataset_selection,
                   "hand_mode": str(hand_mode or "hard")},
    )

    def _add(r):
        report.add(r)
        if on_result is not None:
            on_result(r)

    def _prog(**kw):
        if on_progress is not None:
            on_progress(kw)

    def _ds_done(ds_name):
        # 每数据集评完(含跳过/枚举失败)触发一次:传出该集局部结果 {head:{ds:node}},供上层
        # 按数据集增量展示/多卡分片跨卡合并。放在各 continue 与正常结束点,保证每集恰好一次。
        if on_dataset_done is not None:
            try:
                on_dataset_done(ds_name, report.dataset_subtree(ds_name))
            except Exception:  # noqa: BLE001  回调异常不该影响评测主流程
                pass

    ds_sel = list(ds_sel)
    ds_total = len(ds_sel)
    _prog(kind="init", ds_total=ds_total, datasets=list(ds_sel), heads=list(heads_sel))
    print(f"[run] heads={heads_sel} datasets={ds_sel}\n[run] out={out_dir}", flush=True)
    # 计时:predict(模型前向,每序列一次,该集所有头共用)与 evaluate(每头一次)分开归账,
    # 直接对应「每个数据集上每个任务测评了多久」;跳过的数据集只记 total_s(枚举耗时)。
    timings = {"total_s": 0.0, "datasets": {}}
    _t_run0 = time.perf_counter()
    cancelled = False
    for ds_i, ds_name in enumerate(ds_sel, 1):
        if should_cancel is not None and should_cancel():
            cancelled = True
            break
        _t_ds0 = time.perf_counter()
        dst = {
            "total_s": 0.0,
            "predict_s": 0.0,
            "n_seqs": 0,
            "predict_calls": 0,
            "predict_cache_hits": 0,
            "predicted_frames": 0,
            "reused_frames": 0,
            "predict_stages": {},
            "heads": {},
        }
        _prog(kind="dataset", ds=ds_name, ds_i=ds_i, ds_total=ds_total, stage="枚举序列")
        ds = DATASETS.get(ds_name)(data_root)
        ds.set_benchmark_selection(dataset_selection.get(ds_name))
        dataset_max_frames = ds.benchmark_max_frames(max_frames)
        # 数据集能力若与所有选中头都不匹配,直接整集跳过(省得推理)
        usable = [h for h in head_objs if set(h.required_gt).issubset(ds.capability)]
        if not usable:
            for h in head_objs:
                _add(SeqResult(h.name, ds_name, "*", "skipped",
                               note=f"数据集 {ds_name} 能力 {sorted(ds.capability)} 不含头需求"))
            dst["total_s"] = time.perf_counter() - _t_ds0
            timings["datasets"][ds_name] = dst
            _ds_done(ds_name)
            continue
        try:
            pre_sharded = False
            pre_sharded_total = None
            shard_iter = None
            shard_method = getattr(ds, "iter_sequences_for_shard", None)
            if shard_count and shard_count > 1 and shard_method is not None:
                shard_iter = shard_method(
                    int(shard_index or 0), int(shard_count),
                    max_seqs=None, max_frames=dataset_max_frames,
                    seq_start=range_start, seq_end=range_end,
                )
            if shard_iter is not None:
                pre_sharded = True
                count_method = getattr(ds, "count_sequences_for_shard", None)
                if count_method is not None:
                    pre_sharded_total = count_method(
                        int(shard_index or 0), int(shard_count),
                        max_seqs=None, max_frames=dataset_max_frames,
                        seq_start=range_start, seq_end=range_end,
                    )
                seqs = shard_iter if pre_sharded_total is not None else list(shard_iter)
            else:
                seqs = list(ds.iter_sequences(
                    max_seqs=range_end, max_frames=dataset_max_frames
                ))
                seqs = _slice_sequence_groups(seqs, range_start, range_end)
        except Exception as e:  # noqa: BLE001  单数据集枚举异常不该中断整轮评测,降级成该集级结果
            if isinstance(e, NotImplementedError):
                st, note = "not_implemented", str(e)
            elif isinstance(e, FileNotFoundError):
                st, note = "skipped", str(e)               # 缺数据:优雅降级
            else:
                import traceback
                traceback.print_exc()                       # 完整堆栈落服务端控制台,便于定位
                st, note = "error", f"枚举异常 {type(e).__name__}: {str(e)[:150]}"
            for h in head_objs:
                _add(SeqResult(h.name, ds_name, "*", st, note=note))
            dst["total_s"] = time.perf_counter() - _t_ds0
            timings["datasets"][ds_name] = dst
            _ds_done(ds_name)
            continue

        # 多卡按“唯一输入帧组”分片：例如 HOT3D 左右手 GT 留在同一卡，共用一次模型预测。
        # 以每组最大帧数作代价做贪心均衡，避免只按序列条数导致长视频集中到某张卡。
        if not pre_sharded:
            seqs = _grouped_shard(seqs, shard_index, shard_count)
        seq_total = int(pre_sharded_total) if pre_sharded_total is not None else len(seqs)
        _prog(kind="seqs", ds=ds_name, ds_i=ds_i, ds_total=ds_total, seq_total=seq_total)
        set_dataset = getattr(predictor, "set_benchmark_dataset", None)
        if set_dataset is not None:
            set_dataset(ds_name)
        cached_key = cached_prediction = None
        def _prepared_sequences():
            iterator = iter(seqs)
            for position in range(1, seq_total + 1):
                _prog(kind="prepare", ds=ds_name, ds_i=ds_i, ds_total=ds_total,
                      seq_i=position, seq_total=seq_total, stage="读取 GT / 准备输入")
                try:
                    yield next(iterator)
                except StopIteration:
                    return
                except Exception as e:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    for head in head_objs:
                        _add(SeqResult(
                            head.name, ds_name, "*", "error",
                            note=f"准备输入异常 {type(e).__name__}: {str(e)[:150]}",
                        ))
                    return

        for seq_i, seq in enumerate(_prepared_sequences(), 1):
            if should_cancel is not None and should_cancel():
                cancelled = True
                break
            n_frames = len(seq.image_paths)
            _prog(kind="predict", ds=ds_name, ds_i=ds_i, ds_total=ds_total,
                  seq_i=seq_i, seq_total=seq_total, seq_id=seq.seq_id, stage="模型前向",
                  n_frames=n_frames)
            # 窗口级子进度:引擎分窗前向逐窗回调(done/total 窗),透给上层画单序列内进度条。
            def _on_step(done, total, _si=seq_i, _sid=seq.seq_id, _nf=n_frames):
                _prog(kind="predict_step", ds=ds_name, ds_i=ds_i, ds_total=ds_total,
                      seq_i=_si, seq_total=seq_total, seq_id=_sid, stage="模型前向",
                      win_done=int(done), win_total=int(total), n_frames=_nf)
            cache_key = _prediction_cache_key(seq)
            if cache_key is not None and cache_key == cached_key:
                pred = cached_prediction
                dst["predict_cache_hits"] += 1
                dst["reused_frames"] += n_frames
                _on_step(1, 1)
                print(
                    f"[run] 复用预测 {seq.seq_id}: {n_frames} 帧 "
                    f"group={cache_key[0]}",
                    flush=True,
                )
            else:
                _t_p0 = time.perf_counter()
                try:
                    pred = predictor.predict(seq.image_paths, hw=seq.hw, on_step=_on_step)
                except Exception as e:  # noqa: BLE001  单序列推理异常不中断
                    import traceback
                    traceback.print_exc()
                    dst["predict_s"] += time.perf_counter() - _t_p0
                    cached_key = cached_prediction = None
                    for h in head_objs:
                        _add(SeqResult(h.name, ds_name, seq.seq_id, "error",
                                       note=f"推理异常 {type(e).__name__}: {str(e)[:150]}"))
                    _prog(kind="seq_done", ds=ds_name, ds_i=ds_i, ds_total=ds_total,
                          seq_i=seq_i, seq_total=seq_total, seq_id=seq.seq_id)
                    continue
                dst["predict_s"] += time.perf_counter() - _t_p0
                dst["predict_calls"] += 1
                dst["predicted_frames"] += n_frames
                for name, value in ((pred.meta or {}).get("timings") or {}).items():
                    if isinstance(value, (int, float)) and name.endswith("_s"):
                        stages = dst["predict_stages"]
                        stages[name] = stages.get(name, 0.0) + float(value)
                cached_key = cache_key
                cached_prediction = pred if cache_key is not None else None
            dst["n_seqs"] += 1
            _prog(kind="evaluate", ds=ds_name, ds_i=ds_i, ds_total=ds_total,
                  seq_i=seq_i, seq_total=seq_total, seq_id=seq.seq_id, stage="逐头评测",
                  n_frames=n_frames)
            for h in head_objs:
                if not seq.has(h.required_gt):
                    miss = sorted(set(h.required_gt) - set(seq.capability))
                    _add(SeqResult(h.name, ds_name, seq.seq_id, "skipped",
                                   note=f"缺模态 {miss}"))
                    continue
                # 头还需模型确产出该模态(depth 未开时 pred 无 depth → not_implemented)
                if not set(h.required_gt).issubset(pred.capability):
                    miss = sorted(set(h.required_gt) - set(pred.capability))
                    _add(SeqResult(h.name, ds_name, seq.seq_id, "skipped",
                                   note=f"模型不产出 {miss}(学生未开该头)"))
                    continue
                _t_e0 = time.perf_counter()
                status, metrics, note = h.evaluate(pred, seq)
                dst["heads"][h.name] = dst["heads"].get(h.name, 0.0) + (time.perf_counter() - _t_e0)
                _add(SeqResult(h.name, ds_name, seq.seq_id, status, metrics=metrics, note=note))
            _prog(kind="seq_done", ds=ds_name, ds_i=ds_i, ds_total=ds_total,
                  seq_i=seq_i, seq_total=seq_total, seq_id=seq.seq_id)   # 该序列全评完 → 计一条完成
        dst["total_s"] = time.perf_counter() - _t_ds0
        timings["datasets"][ds_name] = dst
        _ds_done(ds_name)
        if cancelled:
            break

    timings["total_s"] = time.perf_counter() - _t_run0
    report.set_timings(timings)
    # stdout 摘要:总耗时 + 每数据集(总/前向/序列数)
    print(f"[run] 耗时 总 {timings['total_s']:.1f}s", flush=True)
    for _dn, _dt in timings["datasets"].items():
        print(
            f"[run]   {_dn}: 总 {_dt['total_s']:.1f}s 预测 {_dt['predict_s']:.1f}s "
            f"序列 {_dt['n_seqs']} 实际前向 {_dt['predict_calls']} "
            f"复用 {_dt['predict_cache_hits']}",
            flush=True,
        )
    if cancelled:
        print("[run] 已取消:落盘已完成部分", flush=True)
    report.dump(out_dir)
    return out_dir
