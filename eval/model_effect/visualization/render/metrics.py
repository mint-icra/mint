#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""逐帧 loss 计算(网页端"逐帧误差"面板用)——**复用训练侧 loss + 训练同一切窗**。

推理面板显示的 loss 必须与训练**同一套计算、同样的数量、同样的切窗**,故本模块:

  1) 数量/公式一致:直接用训练框架 model_train 的 Criterion,按所选 ckpt 的 config
     `loss` 段实例化,遍历它**实际启用的全部模块与 term**
     (不再写死 7 项;打开/删掉 config 的 term,面板行随之增减),每项用的就是训练那个
     term 函数、组权重×项权重。
  2) 切窗一致:训练按 config `data.clip_len`/`clip_stride` 把序列切成定长窗、逐窗算
     (data/lingbotmap/lerobot_v3.py:80-84,stride==clip_len 时窗不重叠、尾部不足一窗丢弃)。
     本模块把整段按同一规则切成 clips、堆成 batch [n_clips, clip_len, ...] **一次**喂
     Criterion 返回的 total 与各 term 标量(logs)即权威均值/total,与训练该
     切窗逐字一致(时序项 vel/acc 不再跨窗多算)。逐帧曲线则对每个 clip 内滑窗、写回全局帧。
  3) 输出归属:按 camera / hand_presence / image_hand / mano_param / mano_joint /
     camera_mano_consistency 返回
     `groups`,并从 episode 所属数据根加载与训练相同的 camera translation scale。

展示值一律保留训练 loss 的原单位:geo 为弧度、MANO joint 为米、归一化 camera translation
为无量纲值。geo 的度数由前端在括号中补显(纯展示,不改 loss)。逐帧头 w-1 帧(历史不足)
记 None;手部按 kept 掩码无效帧记 None;
不属于任何 clip 的尾部/短段帧记 None。

边界:整段帧数 T < clip_len 时训练本不覆盖该段,回退整段单窗 [1,T] 计算并置 clipwin=False
(前端标注「短于 clip_len,非训练切窗」)。
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import numpy as np

# 训练框架根(与 inference/engine.py 同法挂 sys.path,使 losses 包可 import)。
# render/metrics.py -> render -> visualization -> model_effect -> eval -> <repo>
_REPO_DIR = Path(__file__).resolve().parents[4]
_MODEL_TRAIN = _REPO_DIR / "model_train"


def _ensure_model_train_on_path() -> None:
    """挂 model_train 根 + 内联 _vendor,使训练 Criterion 可 import。"""
    for p in (str(_MODEL_TRAIN), str(_MODEL_TRAIN / "_vendor")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# 手部 GT cat 顺序(须与 hand[218]→[2,109] 布局一致:transl3+orient6+pose90+betas10)。
_MANO_CAT = ("transl_cam", "orient6d", "pose6d", "betas")
_CAMERA_SCALE_CTX_KEYS = {
    "trans_std",  # Old run compatibility.
    "position_std",
    "velocity_std",
    "acceleration_std",
}
_INITIAL_HAND_TERMS = {"initial_orient_6d_l1", "initial_pose_6d_l1"}


def _loss_config_for_outputs(loss_cfg: dict, pred: dict) -> tuple[dict, list[str]]:
    """Drop only terms whose required auxiliary model output is unavailable."""
    resolved = copy.deepcopy(loss_cfg)
    if pred.get("_hand_refine_initial") is not None:
        return resolved, []
    skipped = []
    for group_key in ("mano_param", "hand_loss"):
        group = resolved.get(group_key)
        if not isinstance(group, dict):
            continue
        terms = group.get("terms")
        if not isinstance(terms, list):
            continue
        kept = []
        for term in terms:
            name = term.get("name") if isinstance(term, dict) else None
            if name in _INITIAL_HAND_TERMS:
                skipped.append(f"{group_key}/{name}")
            else:
                kept.append(term)
        group["terms"] = kept
    return resolved, skipped


def _round5(x):
    return None if (x is None or not np.isfinite(x)) else round(float(x), 5)


def _jsonsafe(seq) -> list:
    """逐帧 list → JSON 安全(NaN/None → None,前端显示 —),保留 None。"""
    return [None if (x is None or not np.isfinite(x)) else round(float(x), 5) for x in seq]


def _win_len(name: str) -> int:
    """term 逐帧滑窗长度:acc(二阶差)→3, vel(一阶差)→2, 其余点态→1(与 term 内 shape 判断一致)。"""
    return 3 if "_acc_" in name else 2 if "_vel_" in name else 1


def _temporal_order(value, S: int) -> int | None:
    """Return 0/1/2 for frame, pair, or triplet tensors in a length-S clip."""
    if value.dim() < 2:
        return None
    order = S - value.shape[1]
    return order if order in (0, 1, 2) else None


def _slice_ctx(ctx: dict, t: int, w: int, S: int) -> dict:
    """把 ctx(单个 clip,batch 维已选为 1)每个「按帧」张量沿时间轴切出 [t-w+1, t] 最小窗。

    帧、pair、triplet 张量长度分别为 S/S-1/S-2，对应取 w/w-1/w-2 个；
    其余张量原样保留。s=t-w+1(调用方保证 t>=w-1 → s>=0)。"""
    import torch
    s = t - w + 1
    out = {}
    for k, v in ctx.items():
        order = (
            _temporal_order(v, S)
            if k not in _CAMERA_SCALE_CTX_KEYS and torch.is_tensor(v)
            else None
        )
        if order is not None:
            n = max(0, w - order)
            out[k] = v[:, s:s + n]
        else:
            out[k] = v
    return out


def _select_clip(ctx: dict, c: int, S: int) -> dict:
    """从整批 ctx(batch 维=n_clips)里取出第 c 个 clip → batch 维=1 的子 ctx。

    「按帧」张量(shape[1]∈{S, S-1, S-2})沿 batch 维切 [c:c+1];标量/常量原样。"""
    import torch
    out = {}
    for k, v in ctx.items():
        if k in _CAMERA_SCALE_CTX_KEYS and torch.is_tensor(v):
            out[k] = v[c:c + 1]
        elif torch.is_tensor(v) and _temporal_order(v, S) is not None:
            out[k] = v[c:c + 1]
        else:
            out[k] = v
    return out


def _per_frame_clipped(ctx, fn, w: int, S: int, starts, T: int, valid_full=None) -> list:
    """逐帧调同一 term 函数得 [T] 值，并用 vmap 合并数千次小 Torch 调用。

    对每个 clip c(起点 starts[c]、长 S):clip 内第 j 帧(j>=w-1,历史足)写回全局帧 starts[c]+j;
    头 w-1 帧记 None;valid_full[g] 为假的全局帧记 None;不属任何 clip 的帧保持 None。
    窗重叠(stride<clip_len)时同一全局帧被后一个 clip 覆盖(取最后写入值);均值/total 不受此影响。"""
    import torch

    out = [None] * T
    positions, samples = [], []
    for c, s in enumerate(starts):
        cctx = _select_clip(ctx, c, S)
        for j in range(w - 1, S):
            g = s + j
            if g >= T:
                break
            if valid_full is not None and not bool(valid_full[g]):
                continue
            positions.append(g)
            samples.append(_slice_ctx(cctx, j, w, S))
    if not samples:
        return out

    # BCE's fused reduction changes the last float32 bit under vmap on some Torch
    # builds; keep it scalar so the displayed five-decimal curve remains identical.
    if getattr(fn, "__name__", "") == "presence_bce":
        values = [float(fn(sample).item()) for sample in samples]
        for frame_index, value in zip(positions, values):
            out[frame_index] = value
        return out

    tensor_keys = [key for key, value in samples[0].items() if torch.is_tensor(value)]
    try:
        stacked = [torch.stack([sample[key] for sample in samples], dim=0)
                   for key in tensor_keys]
        constants = {
            key: value for key, value in samples[0].items() if key not in tensor_keys
        }

        def _one(*values):
            local = dict(constants)
            local.update(zip(tensor_keys, values))
            return fn(local)

        values = torch.vmap(_one)(*stacked).detach().cpu().tolist()
    except (RuntimeError, NotImplementedError):
        # Third-party/custom terms may use an operation without a vmap rule.
        values = [float(fn(sample).item()) for sample in samples]
    for frame_index, value in zip(positions, values):
        out[frame_index] = float(value)
    return out


def _clip_starts(T: int, clip_len: int, stride: int):
    """整段 T 帧按训练规则切窗:起点 0,stride,2*stride,… ≤ T-clip_len。

    返回 (starts, use_len, clipwin):T>=clip_len 时 clipwin=True、use_len=clip_len;
    T<clip_len(训练不覆盖)时回退整段单窗 starts=[0]、use_len=T、clipwin=False。"""
    if T >= clip_len:
        starts = list(range(0, T - clip_len + 1, max(1, stride)))
        return starts, clip_len, True
    return [0], T, False


def _rebase_clips_to_first(pe_nl9: np.ndarray) -> np.ndarray:
    """逐 clip 把相机 pose_enc[n,L,9](absT+quaR_xyzw+FoV,w2c)重锚到各自**窗首帧**。

    与训练 lerobot_v3._rebase_pose_enc_to_first 同一 SE(3) 变换(E'=E·E_0⁻¹),
    使相机 loss 面板在「窗首帧系」上算,和训练逐字一致;FoV 不变,手不涉及。"""
    import torch
    from lingbot_map.utils.rotation import mat_to_quat, quat_to_mat   # path 由 _ensure_model_train_on_path 挂好
    x = torch.from_numpy(np.ascontiguousarray(pe_nl9)).float()        # [n,L,9]
    T, quat, fov = x[..., :3], x[..., 3:7], x[..., 7:]
    R = quat_to_mat(quat)                                             # [n,L,3,3]
    R0t = R[:, 0].transpose(-1, -2)[:, None]                          # [n,1,3,3] R_0ᵀ
    Rn = R @ R0t                                                      # R_i·R_0ᵀ [n,L,3,3]
    Tn = T - torch.einsum("nlij,nj->nli", Rn, T[:, 0])               # t_i - R'_i·t_0 [n,L,3]
    qn = mat_to_quat(Rn)                                              # [n,L,4] xyzw
    return torch.cat([Tn, qn, fov], dim=-1).numpy().astype(np.float32)


def _camera_trans_scales(raw: dict, data_cfg: dict, clip_len: int, n_clips: int):
    """Build the same global position/velocity/acceleration scales as training."""
    norm_cfg = data_cfg.get("camera_translation_normalization", {}) or {}
    if not bool(norm_cfg.get("enabled", False)):
        return {}

    import torch
    resolved = norm_cfg.get("resolved")
    if isinstance(resolved, dict):
        mapping = {
            "camera_trans_position_std": "position_std_m",
            "camera_trans_velocity_std": "velocity_std_m_per_frame",
            "camera_trans_acceleration_std": "acceleration_std_m_per_frame2",
        }
        scales = {}
        for batch_key, resolved_key in mapping.items():
            scale = torch.as_tensor(resolved.get(resolved_key), dtype=torch.float32)
            if tuple(scale.shape) != (3,):
                raise RuntimeError(f"resolved.{resolved_key} 应为 [3],实际 {tuple(scale.shape)}")
            scales[batch_key] = scale.reshape(1, 3).expand(n_clips, -1).clone()
        return scales

    # 旧 run 的 config 没有 resolved，继续读取原来的单数据集 position std；速度/加速度
    # 复用它以逐字保持旧训练行为。
    from losses.normalization import load_camera_translation_std
    explicit = raw.get("camera_trans_std")
    if explicit is not None:
        scale = torch.as_tensor(explicit, dtype=torch.float32)
    else:
        root = raw.get("dataset_root")
        if root is None:
            configured = data_cfg.get("root")
            roots = configured if isinstance(configured, list) else [configured]
            roots = [item.get("root") if isinstance(item, dict) else item for item in roots]
            roots = [item for item in roots if item]
            if len(roots) != 1:
                raise RuntimeError(
                    "启用了相机平移归一化,但 episode 未携带 dataset_root,无法从多数据集配置选择统计"
                )
            root = roots[0]
        root = Path(root)
        if not root.is_absolute():
            root = _REPO_DIR / root
        scale = load_camera_translation_std(
            root,
            clip_len,
            norm_cfg.get("filename", "camera_translation_normalization.json"),
        )
    if tuple(scale.shape) != (3,):
        raise RuntimeError(f"camera_trans_std 应为 [3],实际 {tuple(scale.shape)}")
    expanded = scale.reshape(1, 3).expand(n_clips, -1).clone()
    return {"camera_trans_std": expanded}


def _episode_data_cfg(raw: dict, data_cfg: dict) -> dict:
    """Apply a matching data.root dict override for the episode being visualized."""
    roots = data_cfg.get("root")
    dataset_root = raw.get("dataset_root")
    if not isinstance(roots, list) or dataset_root is None:
        return data_cfg
    target = Path(dataset_root).resolve()
    for item in roots:
        if not isinstance(item, dict) or not item.get("root"):
            continue
        candidate = Path(item["root"])
        if not candidate.is_absolute():
            candidate = _REPO_DIR / candidate
        if candidate.resolve() == target:
            return {**data_cfg, **item}
    return data_cfg


def frame_metrics(raw: dict, pred: dict, loss_cfg: dict, hw=None, decode=None) -> dict | None:
    """逐帧 + 均值/加权 total,由训练 Criterion 按训练切窗计算(与训练一致)。

    raw:      lerobot_io.load_episode_raw 产物(需 cam_pose_enc / hands / kept)。
    pred:     predictor.predict 产物({'pose_enc':[N,9], 'hand':[N,218]?,
              'hand_presence_logits':[N,2]?})。
    loss_cfg: ckpts.load_loss_cfg 产物 {loss, model, data}(loss 段=组权重+terms;
              model 段=头开关/结构;data 段=clip_len/clip_stride)。
    hw:       原始图像的 (H,W),image_hand 投影 loss 需要;缺省时从 raw.frames 推断。
    decode:   兼容旧签名,不再使用(旋转解码由训练 loss 内部处理)。
    GT 相机编码或预测缺失时返回 None(该 episode 无对比意义)。
    """
    gt_pe = raw.get("cam_pose_enc")
    if gt_pe is None or "pose_enc" not in pred:
        return None

    _ensure_model_train_on_path()
    import torch
    from losses import build_criterion

    lc = (loss_cfg or {}).get("loss", {}) or {}
    mc = (loss_cfg or {}).get("model", {}) or {}
    dc = _episode_data_cfg(raw, (loss_cfg or {}).get("data", {}) or {})
    if not lc:
        return None
    clip_len = int(dc.get("clip_len", 8) or 8)
    stride = int(dc.get("clip_stride", 1) or 1)

    pe = np.asarray(pred["pose_enc"], np.float32)
    gt = np.asarray(gt_pe, np.float32)
    T = int(min(len(pe), len(gt)))
    if T == 0:
        return None

    # ---- 头启用:由 config 判定(与训练一致);available=该 episode 数据能否算出该头 loss ----
    cam_enabled = bool(mc.get("backbone", {}).get("enable_camera", True))
    cam_available = True                                  # 已保证 gt_pe/pose_enc 存在
    hand_enabled = bool(mc.get("enable_hand", False))
    pred_hand_available = bool(hand_enabled and "hand" in pred)
    presence_enabled = bool(mc.get("enable_hand_presence", False))
    pred_presence_available = bool(
        presence_enabled
        and np.asarray(pred.get("hand_presence_logits", [])).shape == (T, 2)
    )
    mano_gt_available = bool(raw.get("hands"))
    kp21_gt_available = raw.get("kpt21_gt") is not None

    # ---- 训练切窗:整段切成 clips → gather 成 batch [n, use_len, ...] 一次喂 ----
    starts, use_len, clipwin = _clip_starts(T, clip_len, stride)
    n_clips = len(starts)
    idx = np.stack([np.arange(s, s + use_len) for s in starts])          # [n, use_len]

    # 逐窗重锚到窗首帧(与训练 lerobot_v3._rebase_pose_enc_to_first 同系):相机 loss 训练时算在
    # 窗首帧系,面板须对 pred/GT 两侧同样重锚才能逐字对齐(仅相机 pose_enc;手为逐帧相机系,不涉及)。
    pred_t = {"pose_enc": torch.from_numpy(_rebase_clips_to_first(pe[idx]))}   # [n,use_len,9]
    batch_t = {"gt_pose_enc": torch.from_numpy(_rebase_clips_to_first(gt[idx]))}
    if hw is None:
        frames = np.asarray(raw.get("frames", []))
        if frames.ndim >= 3:
            hw = frames.shape[1:3]
    if hw is not None:
        batch_t["image_hw"] = tuple(int(value) for value in hw)
    initial = np.asarray(pred.get("_hand_refine_initial", []), np.float32)
    if initial.shape == (T, 218):
        pred_t["_hand_refine_initial"] = torch.from_numpy(initial[idx])
    kept = np.asarray(raw["kept"], bool)[:T]                              # [T,2]
    batch_t["hand_kept"] = torch.from_numpy(kept[idx].astype(np.float32))
    if pred_presence_available:
        presence_logits = np.asarray(pred["hand_presence_logits"], np.float32)[:T]
        pred_t["hand_presence_logits"] = torch.from_numpy(presence_logits[idx])
    trans_scales = _camera_trans_scales(raw, dc, clip_len, n_clips)
    batch_t.update(trans_scales)
    base_valid = pair_valid = None
    if pred_hand_available:
        ph = np.asarray(pred["hand"], np.float32)[:T]                    # [T,218]
        pred_t["hand"] = torch.from_numpy(ph[idx])                       # [n,use_len,218]
        if mano_gt_available:
            hands = raw["hands"]
            hg = np.concatenate([
                np.concatenate([np.asarray(hands[side][column], np.float32)
                                for column in _MANO_CAT], axis=-1)
                for side in ("left", "right")
            ], axis=-1)[:T]
            batch_t["hand_gt"] = torch.from_numpy(hg[idx])
        else:
            batch_t["hand_gt"] = torch.zeros(n_clips, use_len, 218)
        if kp21_gt_available:
            batch_t["kpt21_gt"] = torch.from_numpy(
                np.asarray(raw["kpt21_gt"], np.float32)[:T][idx]
            )
        else:
            batch_t["kpt21_gt"] = torch.zeros(n_clips, use_len, 2, 21, 3)
        batch_t["mano_gt_valid"] = torch.full((n_clips,), mano_gt_available)
        batch_t["kpt21_gt_valid"] = torch.full((n_clips,), kp21_gt_available)
        batch_t["hand_valid"] = torch.full(
            (n_clips,), mano_gt_available or kp21_gt_available
        )
        base_valid = kept.any(axis=1)                                   # [T] 该帧有有效手
        pair_valid = np.zeros(T, dtype=bool)                            # [T] 相邻帧同手都有效
        pair_valid[1:] = (kept[1:] & kept[:-1]).any(axis=1)

    runtime_lc, skipped_terms = _loss_config_for_outputs(lc, pred_t)
    if skipped_terms:
        print(
            "[metrics] 跳过缺少模型辅助输出的逐帧 loss: "
            + ", ".join(skipped_terms),
            flush=True,
        )
    criterion = build_criterion(runtime_lc)

    per: dict = {}       # 逐帧训练 loss 值(geo 仍为弧度)
    mean: dict = {}      # 均值展示值(权威,取训练整批返回)
    weight: dict = {}    # 有效权重=组权重×项权重
    cpf: dict = {}       # 逐帧加权贡献(弧度基)
    cmn: dict = {}       # 均值加权贡献(权威)
    total_pf = np.zeros(T, dtype=np.float64)   # 逐帧加权 total(累加各 term 贡献,缺项按 0)
    total_mean = 0.0

    def _emit(group_prefix, name, eff, logs, log_key, vals_rad):
        """把某 term 的逐帧值(loss 原单位)/权威均值写进各展示 dict,并累加 total。返回该 term 的 key。

        值/均值一律按 loss 原单位存(旋转 geo 项即弧度),与加权/占比/total 同口径;
        前端按 term 的 deg 标记在值列括号补度数(纯展示,不改数值)。"""
        key = f"{group_prefix}__{name}"
        for t, v in enumerate(vals_rad):
            if v is not None:
                total_pf[t] += eff * v
        per[key] = list(vals_rad)
        cpf[key] = [None if v is None else eff * v for v in vals_rad]
        weight[key] = round(float(eff), 5)
        mv = logs[log_key].item()
        mean[key] = _round5(mv)
        cmn[key] = _round5(eff * mv)
        return key

    def _term_meta(group_prefix, name, eff, normalized=False):
        return {"key": f"{group_prefix}__{name}", "name": name,
                "weight": round(float(eff), 5), "deg": ("geo" in name),
                "normalized": bool(normalized)}

    hand_head = mc.get("hand_head", {}) or {}
    hh_name = hand_head.get("name", "?")
    hh_bits = [f"dim={hand_head['dim']}"] if hand_head.get("dim") else []
    if hand_head.get("num_queries"):
        hh_bits.append(f"q={hand_head['num_queries']}")
    if hand_head.get("num_iterations"):
        hh_bits.append(f"it={hand_head['num_iterations']}")
    cam_iters = mc.get("backbone", {}).get("camera_num_iterations")
    group_display = {
        "camera": ("camera_head", "相机输出 · camera", f"iters={cam_iters}" if cam_iters is not None else ""),
        "fov": ("fov_head", "相机输出 · FoV", "independent head"),
        "hand_presence": ("hand_presence_head", "手部输出 · 左右手存在性", "2-query BCE"),
        "image_hand": ("hand_head", "手部输出 · 2D 重投影", "ViDiHand EPE-p"),
        "mano_param": ("hand_head", f"手部输出 · MANO 参数({hh_name})", " ".join(hh_bits)),
        "mano_joint": ("hand_head", "手部输出 · MANO 21点", "derived"),
        "camera_mano_consistency": ("camera_hand", "相机↔MANO 一致性", "cross-head"),
    }

    # ---- 权威均值/total:Criterion 整批一次 + 每个模块复用其 ctx/term 生成逐帧值 ----
    groups = []
    with torch.no_grad():
        total, criterion_logs = criterion(pred_t, batch_t)
        total_mean = float(total.item())
        for group_name, loss_module in criterion.losses:
            needs_hand = group_name in {
                "image_hand", "mano_param", "mano_joint", "camera_mano_consistency"
            }
            term_names = {name for name, _function, _weight in loss_module.terms}
            needs_mano_gt = (
                group_name in {"image_hand", "mano_param"}
                or bool(term_names & {
                    "rootrel_mpjpe", "abs_mpjpe", "rootrel_vel_mpjpe",
                    "world_trans_l1", "world_orient_geo",
                    "world_trans_vel_l1", "world_orient_vel_geo",
                })
            )
            needs_kpt21_gt = any("kp21" in name or name == "betas_reg" for name in term_names)
            if group_name == "camera":
                available = cam_available
                enabled = cam_enabled
            elif group_name == "fov":
                available = cam_available
                enabled = cam_enabled
            elif group_name == "hand_presence":
                available = pred_presence_available
                enabled = presence_enabled
            elif group_name == "image_hand":
                available = pred_hand_available and pred_presence_available
                enabled = hand_enabled and presence_enabled
            elif group_name == "camera_mano_consistency":
                available = pred_hand_available
                enabled = cam_enabled and hand_enabled
            else:
                available = pred_hand_available
                enabled = hand_enabled
            available = available and (not needs_mano_gt or mano_gt_available)
            available = available and (not needs_kpt21_gt or kp21_gt_available)

            terms_meta = []
            ctx = loss_module._ctx(pred_t, batch_t) if available else None
            for name, function, term_weight in loss_module.terms:
                effective_weight = loss_module.weight * term_weight
                normalized = (
                    group_name == "camera"
                    and bool(trans_scales)
                    and name in {"trans_l1", "trans_vel_l1", "trans_acc_l1"}
                )
                terms_meta.append(
                    _term_meta(group_name, name, effective_weight, normalized=normalized)
                )
                key = f"{group_name}__{name}"
                if not available:
                    weight[key] = round(float(effective_weight), 5)
                    mean[key] = None
                    continue

                valid = None
                if needs_hand:
                    valid = pair_valid if "_vel_" in name else base_valid
                values = _per_frame_clipped(
                    ctx, function, _win_len(name), use_len, starts, T, valid_full=valid
                )
                log_key = f"loss/{group_name}/{name}"
                if log_key not in criterion_logs and f"{log_key}_norm" in criterion_logs:
                    log_key = f"{log_key}_norm"
                _emit(
                    group_name,
                    name,
                    effective_weight,
                    criterion_logs,
                    log_key,
                    values,
                )

            head, label, arch = group_display[group_name]
            groups.append({
                "head": head,
                "label": label,
                "enabled": enabled,
                "available": available,
                "requires_hand": needs_hand,
                "weight": round(float(loss_module.weight), 5),
                "arch": arch,
                "terms": terms_meta,
            })

    # total:逐帧=各 term 加权贡献之和(权威均值取训练整批返回,保证均值列=训练值)。
    per["total"] = list(total_pf)
    cpf["total"] = list(total_pf)
    mean["total"] = _round5(total_mean)
    cmn["total"] = _round5(total_mean)

    return {
        "per_frame": {k: _jsonsafe(v) for k, v in per.items()},
        "mean": mean,
        "weight": weight,
        "contrib": {"per_frame": {k: _jsonsafe(v) for k, v in cpf.items()}, "mean": cmn},
        "groups": groups,
        "has_hand": pred_hand_available and (mano_gt_available or kp21_gt_available),
        "skipped_terms": skipped_terms,
        "nframes": int(T),
        "clip_len": clip_len, "stride": stride, "n_clips": int(n_clips), "clipwin": bool(clipwin),
    }
