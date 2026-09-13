#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 MuJoCo 在世界坐标系里查看双手 MANO 网格 + 相机轨迹（对标 HaWoR 世界系动图）。

复用本包已有解算链，不重造：lerobot GT 列 或 学生模型预测 → 世界系双手 verts + 相机外参
→ 交 render/mujoco_scene.py 用 MuJoCo 离屏渲染成 mp4（或 --interactive 交互查看）。

数据源：
  · 默认 GT（不给 --ckpt）：raw 的教师 GT 手 + GT 相机（compare.gt_to_world）。
  · 预测（给 --ckpt 或 --source pred）：模型 hand[T,218] + pose_enc → 预测手×预测相机（端到端世界输出）。

用法（使用新项目的 mint 环境）：
    PY=python
    # GT 世界系（无需 ckpt，先跑通/看数据）
    $PY eval/model_effect/visualization/mujoco_view.py --input <lerobot_v3 目录> --episode 0 --max-frames 64
    # 模型预测的世界系输出
    $PY eval/model_effect/visualization/mujoco_view.py --input <lerobot_v3 目录> --ckpt <step_* 目录> --episode 0
输出：output/eval/<model>/mujoco_view/<时间戳>/world.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))          # visualization/
_PKG_ROOT = os.path.dirname(_HERE)                          # model_effect/（包根）
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
REPO_DIR = Path(_HERE).resolve().parents[2]                 # visualization -> model_effect -> eval -> <repo>


def _first_valid(kept_col) -> int:
    import numpy as np
    idx = np.where(np.asarray(kept_col, dtype=bool))[0]
    return int(idx[0]) if len(idx) else 0


def _rot_align(a, b):
    """返回把单位向量 a 旋到单位向量 b 的旋转矩阵 (3,3)（Rodrigues）。"""
    import numpy as np
    a = np.asarray(a, float) / (np.linalg.norm(a) + 1e-9)
    b = np.asarray(b, float) / (np.linalg.norm(b) + 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:                       # 已同向/反向
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _upright_rotation(cam_c2w, hand_pts, up_mode):
    """求把数据世界系旋成 MuJoCo z-up 的旋转 (3,3)。

    up_mode='auto'：**先假设数据世界系已经重力对齐**（lightwheel / hot3d 这类导出都是），
    在 ±x/±y/±z 六个候选轴上打分选竖直轴；全不合格才退回「手关节 PCA 平面法线」。
    x/y/z：指定该世界轴为 up；none：不旋。

    为什么不再默认用 PCA：手关节点云的奇异值比实测是 [1, 0.42, 0.37]——后两轴几乎简并，
    「最小方差方向」根本不定。在 lightwheel ep0 上它估出的 up 偏离真竖直约 41°，
    渲出来手是「立」在桌上而不是掌心朝下平铺，第三人称四视角全错。
    """
    import numpy as np
    if up_mode == "none":
        return np.eye(3)
    if up_mode in ("x", "y", "z"):
        up = np.asarray({"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}[up_mode], float)
    else:
        up = _detect_gravity_axis(cam_c2w, hand_pts)
    return _rot_align(up, [0, 0, 1])


def _detect_gravity_axis(cam_c2w, hand_pts):
    """在 ±x/±y/±z 里挑最像「竖直向上」的世界轴；都不合格则退回 PCA 平面法线。

    判据全部来自「头戴相机 + 桌面操作」这个场景的物理先验：
      1) 相机俯角落在 [5°,60°]（人低头看手）——最主要，不满足直接淘汰；
      2) 相机中心高于手部 95 分位（头在手上方）——不满足直接淘汰；
      3) 手部点云沿该轴伸展最小（手大致贴着一个水平面活动）——用于在多个候选间排序。
    """
    import numpy as np
    c2w = np.asarray(cam_c2w, float)
    pts = np.asarray(hand_pts, float).reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    fwd = c2w[:, :3, 2].mean(0)                        # OpenCV +z 前向的平均光轴
    fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
    C = c2w[:, :3, 3]
    span_all = float(np.linalg.norm(np.percentile(pts, 98, axis=0) - np.percentile(pts, 2, axis=0))) + 1e-9

    best, best_score = None, None
    for axis in np.concatenate([np.eye(3), -np.eye(3)], axis=0):
        pitch = -np.degrees(np.arcsin(np.clip(float(np.dot(fwd, axis)), -1.0, 1.0)))   # 正=俯视
        if not (5.0 <= pitch <= 60.0):
            continue
        if float(np.dot(C.mean(0), axis)) <= float(np.percentile(pts @ axis, 95)):     # 头须在手上方
            continue
        h = pts @ axis
        span = float(np.percentile(h, 98) - np.percentile(h, 2)) / span_all            # 越扁越像水平面
        score = -span
        if best_score is None or score > best_score:
            best, best_score = axis, score
    if best is not None:
        return best
    # 兜底：手关节平面法线（PCA 最小方差方向），符号与相机自身「上」对齐。不稳，仅在
    # 六轴判据全不满足（非低头操作类数据 / 世界系未重力对齐）时使用。
    c = pts.mean(0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    up = vt[2]
    cam_up = (-c2w[:, :3, 1]).mean(0)
    if np.dot(up, cam_up) < 0:
        up = -up
    return up


def _ego_viewport(K, W, H, rw, rh, fovy_deg, mode):
    """算 ego 渲染图里「与真实相机**完全同视野**」的裁剪窗口，返回 ((x0,x1,y0,y1), (out_w,out_h))。

    为什么需要裁：MuJoCo 相机只有一个 `fovy`（竖直视场），隐含**方形像素 + 主点居中**，
    **水平视野完全由画布宽高比决定**。所以用 1280x720 的画布去渲 1920x1456 的数据时，
    几何投影本身是对的（fx=fy、主点居中、位姿逐帧一致，已实测验证），但**画幅更宽**——
    水平 FoV 117.4° vs 真实 101.3°，左右各多出约 165px 真实 ego 视频里根本没有的内容。
    要拿 ego 渲染图和真实 ego 视频逐像素对照，就得先裁到真实画幅。

    换算：渲染图等效焦距 fr = (rh/2)/tan(fovy/2)（方形像素）；真实相机像素 u 对应方向
    tanθ = (u-cx)/fx，落到渲染图的 rw/2 + fr·tanθ。四条边（u=0/W、v=0/H）即为窗口。
    主点不居中时窗口自然偏移，这里一并处理（只用 K[1,1] 反解 fovy 会静默丢掉这个信息）。
    mode='wide' 返回整幅（旧行为：视野更大，能看到画面外的手/物，但与真实 ego 对不齐）。
    """
    import numpy as np
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if abs(fx - fy) > 0.01 * max(fx, fy):        # MuJoCo 无法表达非方形像素，只能告警
        print(f"[mujoco_view][WARN] 数据 fx={fx:.2f} ≠ fy={fy:.2f}（非方形像素），"
              f"MuJoCo 相机只有单个 fovy → 水平方向存在 {abs(fx / fy - 1) * 100:.1f}% 比例偏差，无法消除。")
    if mode == "wide":
        return (0, int(rw), 0, int(rh)), (int(rw), int(rh))
    fr = (rh / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)          # 渲染画布等效焦距（px）
    x0, x1 = rw / 2.0 - fr * (cx / fx), rw / 2.0 + fr * ((W - cx) / fx)
    y0, y1 = rh / 2.0 - fr * (cy / fy), rh / 2.0 + fr * ((H - cy) / fy)
    if x0 < -0.5 or x1 > rw + 0.5 or y0 < -0.5 or y1 > rh + 0.5:
        print(f"[mujoco_view][WARN] 真实相机视野超出渲染画布，ego 边缘会被裁掉；"
              f"把 --width/--height 的宽高比调到接近 {W}/{H}={W / H:.3f} 即可避免。")
    ix0, ix1 = int(np.ceil(max(0.0, x0))), int(np.floor(min(float(rw), x1)))
    iy0, iy1 = int(np.ceil(max(0.0, y0))), int(np.floor(min(float(rh), y1)))
    ix1 -= (ix1 - ix0) % 2                                        # H.264 要求偶数边长
    iy1 -= (iy1 - iy0) % 2
    return (ix0, ix1, iy0, iy1), (ix1 - ix0, iy1 - iy0)


def _pick_ghost_frames(track, cur, max_n, min_gap):
    """从 [0, cur) 里按世界位移最小间距抽出至多 max_n 个历史帧索引（由旧到新）。

    与相机锥体的抽稀同思路：位移不够就不新增，手基本不动的片段自然只留很少几个，
    不会把十几个残影叠成一坨；位移大时沿轨迹排开，接近 HaWoR 那串手。
    """
    import numpy as np
    if cur <= 0 or max_n <= 0:
        return []
    idxs = [0]
    for i in range(1, cur):
        if np.linalg.norm(track[i] - track[idxs[-1]]) >= min_gap:
            idxs.append(i)
    if len(idxs) > max_n:                                  # 超上限沿链等距抽稀
        idxs = list(np.array(idxs)[np.unique(np.linspace(0, len(idxs) - 1, max_n).astype(int))])
    return [int(i) for i in idxs]


def main():
    ap = argparse.ArgumentParser(description="MuJoCo 世界系双手 + 相机轨迹查看")
    ap.add_argument("--input", required=True, help="lerobot v3 数据集目录")
    ap.add_argument("--model", default="lingbotmap", help="推理模型（inference.registry 注册名）")
    ap.add_argument("--config", default=None, help="训练 config；不给用该模型默认")
    ap.add_argument("--ckpt", default=None, help="学生 ckpt（step_* 目录/权重）；给了则默认走预测")
    ap.add_argument("--source", choices=["gt", "pred"], default=None,
                    help="数据源；默认：给 --ckpt 则 pred，否则 gt")
    ap.add_argument("--out", default=None, help="输出目录（默认 output/eval/<model>/mujoco_view/<ts>）")
    ap.add_argument("--episode", type=int, default=0, help="选第几个 episode（列表序号）")
    ap.add_argument("--max-frames", type=int, default=64, help="最多帧数")
    ap.add_argument("--window", type=int, default=None, help="分窗前向窗口（默认=ckpt 训练 clip_len）")
    ap.add_argument("--fps", type=float, default=20.0, help="输出回放帧率")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--azimuth", type=float, default=140.0)
    ap.add_argument("--elevation", type=float, default=-22.0,
                    help="第三人称俯仰角（MuJoCo 约定：负=从上往下俯视，默认 -22=贴着地台侧看操作）")
    ap.add_argument("--zoom", type=float, default=1.35,
                    help="第三人称取景留白系数（>1 相机更远、画面更松；1.0=包围球刚好贴满画面）")
    ap.add_argument("--up", choices=["auto", "x", "y", "z", "none"], default="auto",
                    help="数据世界系的 up 轴：auto=在 ±x/±y/±z 里自动检测已重力对齐的竖直轴"
                         "（判据：相机低头俯角 5~60°、头在手上方、手部沿该轴最扁），"
                         "检测不出才退回手关节 PCA 平面法线")
    ap.add_argument("--ghost", type=int, default=0,
                    help="手部运动残影数量（沿轨迹留一串由淡到实的手，对标 HaWoR）；默认 0=关闭，"
                         "手位移大的序列可开 6 左右")
    ap.add_argument("--allow-missing-video", action="store_true",
                    help="ego 视频缺失/软链断时仍能出图（仅 --source gt；GT 渲染不需要图像帧，"
                         "分辨率改从 meta/info.json 的 feature shape 取）")
    ap.add_argument("--no-floor", action="store_true", help="不画地面")
    ap.add_argument("--cube", type=float, default=0.0,
                    help="在垫布上放一个边长 N 米的参照方块（纯视觉、无碰撞、不参与物理，手会穿过它）。"
                         "用来核对①尺度：已知边长的方块 vs 手，一眼看出场景大小对不对；"
                         "②世界系：它在世界里完全静止，若画面里跟着手/相机漂说明解算有问题。"
                         "0=不放；常用 0.05（魔方大小）")
    ap.add_argument("--cube-pos", type=float, nargs=3, default=None,
                    metavar=("RIGHT", "FWD", "UP"),
                    help="方块相对手部活动中心的偏移（米），按『操作者坐标系』：右 / 前 / 上。"
                         "不给=自动摆到手部活动范围**前方、不碰手**（推荐）；"
                         "想放手后面用负的前向，如 --cube-pos 0 -0.2 0")
    ap.add_argument("--views", default="front,right,left,back,ego",
                    help="逗号分隔视角（各出一个 world_<view>.mp4）。第三人称 front/right/left/back "
                         "四个方位都相对『操作者前向』(相机平均光轴投影到地面)定，垫布也对齐该方向 → 画面整齐一致；"
                         "front=站在操作者对面看(手正面，与 ego 左右镜像)、back=操作者背后过肩看(与 ego 同侧)、"
                         "right/left=操作者右/左侧；ego=真实头戴相机第一人称；top=正对地台俯视。"
                         "另兼容旧名 orbit_a/orbit_b。")
    ap.add_argument("--ego-floor", choices=["ground", "billboard"], default="ground",
                    help="ego 第一人称的地台形态。ground（默认）=真实世界水平地面，"
                         "地面透视随相机正确变化、垫布与方块留在世界原位 → **ego 与 front/top 是同一个"
                         "世界，可直接对照**；billboard=旧行为，地台转成正对相机、铺满画面的背景板"
                         "（背景干净无天空，但地面没有透视、物体像浮空，且板跟着相机转、与第三人称对不上）")
    ap.add_argument("--ego-fov", choices=["exact", "wide"], default="exact",
                    help="ego 的**画幅**。exact（默认）=按数据内参 K 裁成与真实相机**完全一样的视野**"
                         "（可与真实 ego 视频逐像素对照；输出分辨率随之变化）；"
                         "wide=保留整幅渲染画布（MuJoCo 水平视野由画布宽高比决定，通常比真实相机更宽，"
                         "能看到真实 ego 拍不到的手/物，但对不齐）")
    ap.add_argument("--interactive", action="store_true", help="起 mujoco.viewer 交互回放（需显示环境）")
    args = ap.parse_args()

    import numpy as np
    from visualization.reproj_core import lerobot_io, mano, geometry as geom
    from visualization.render import compare, draw, mujoco_scene

    mano.ensure_mano_weights()
    source = args.source or ("pred" if args.ckpt else "gt")

    ds_dir = lerobot_io.find_dataset(Path(args.input))
    if ds_dir is None:
        print(f"[ERROR] --input 不是 lerobot v3 数据集（找不到 meta/info.json）：{args.input}")
        sys.exit(1)
    if args.allow_missing_video and source == "pred":
        print("[ERROR] --allow-missing-video 只能配 --source gt：模型预测必须吃真实图像帧。")
        sys.exit(1)
    eps = lerobot_io.discover_episodes(ds_dir, require_video=not args.allow_missing_video)
    if not (0 <= args.episode < len(eps)):
        print(f"[ERROR] --episode {args.episode} 越界（共 {len(eps)} 个）。")
        sys.exit(1)
    raw = lerobot_io.load_episode_raw(eps[args.episode], max_frames=args.max_frames,
                                      allow_missing_video=args.allow_missing_video)
    T = raw["T"]
    H, W = raw["hw"]
    hf = raw.get("hand_frame", "world")
    print(f"[mujoco_view] source={source} episode={raw['episode_index']} T={T} hand_frame={hf}")

    # ---------------- 解算：世界系双手 verts + 相机轨迹 + 有效性 ----------------
    if source == "gt":
        world = compare.gt_to_world(raw)
        cam_c2w = np.asarray(raw["cam_c2w"], dtype=np.float64)
        kept = np.asarray(raw["kept"], dtype=bool)
        K = np.asarray(raw["K"], dtype=np.float64)                        # ego 相机内参（含 fovy）
    else:
        from inference.registry import get_predictor
        predictor = get_predictor(args.model, config=args.config, ckpt=args.ckpt, window=args.window)
        pred = predictor.predict(raw["frames"])
        if "hand" not in pred:
            print("[ERROR] 模型未输出 hand（enable_hand 关？），无可渲染网格。")
            sys.exit(1)
        cam_c2w, K = geom.decode_camera_pose_enc(pred["pose_enc"], H, W)   # 预测相机 + 内参
        cam_c2w = np.asarray(cam_c2w, dtype=np.float64)
        world = compare.hands_to_world(compare.pred_hand_to_schema(pred["hand"]), cam_c2w, hf)
        pk = compare.predicted_presence(pred, T)
        kept = pk if pk is not None else np.ones((T, 2), dtype=bool)

    if world["left"].get("verts") is None or world["right"].get("verts") is None:
        print("[ERROR] 当前数据只有 21 点(无 MANO)，本入口只渲网格。请用带 MANO 的数据/模型。")
        sys.exit(1)

    # 把数据世界系旋成 MuJoCo z-up（否则水平地面托不住手）。
    raw_hand_pts = np.concatenate([world["left"]["joints"].reshape(-1, 3),
                                   world["right"]["joints"].reshape(-1, 3)], axis=0)
    Rup = _upright_rotation(cam_c2w, raw_hand_pts, args.up)
    _up_world = Rup.T @ np.array([0.0, 0.0, 1.0])          # 该竖直方向在原数据世界系里的朝向
    print(f"[mujoco_view] up={args.up} → 竖直轴(数据世界系) = {np.round(_up_world, 3)}")
    for s in ("left", "right"):
        world[s]["verts"] = world[s]["verts"] @ Rup.T
        world[s]["joints"] = world[s]["joints"] @ Rup.T
    cam_c2w = cam_c2w.copy()
    cam_c2w[:, :3, :3] = Rup[None] @ cam_c2w[:, :3, :3]
    cam_c2w[:, :3, 3] = cam_c2w[:, :3, 3] @ Rup.T

    # 把整个场景平移到世界原点附近（减去手运动的稳健中心）。数据世界系常离原点很远(~1m)，
    # 而 MuJoCo 方向光的 shadow map / 深度缓冲在偏远处会失准（掠射 shadow acne、free 相机异常）；
    # 移到原点后 shadow、地台、取景都在可控的小范围里，阴影干净、行为稳定。
    _rot_pts = np.concatenate([world["left"]["joints"].reshape(-1, 3),
                               world["right"]["joints"].reshape(-1, 3)], axis=0)
    _rot_pts = _rot_pts[np.all(np.isfinite(_rot_pts), axis=1)]
    scene_origin = np.median(_rot_pts, axis=0) if len(_rot_pts) else np.zeros(3)
    for s in ("left", "right"):
        world[s]["verts"] = world[s]["verts"] - scene_origin
        world[s]["joints"] = world[s]["joints"] - scene_origin
    cam_c2w[:, :3, 3] = cam_c2w[:, :3, 3] - scene_origin

    vl, vr = world["left"]["verts"], world["right"]["verts"]           # (T,778,3) world(z-up, 已居中)
    faces_right, faces_left = draw.get_faces()
    # ego 竖直 FoV 由内参反解。注意只用得上 K[1,1]：MuJoCo 相机只有单个 fovy（方形像素、主点居中），
    # 水平视野由画布宽高比决定 → K[0,0]/cx/cy 的信息全靠 _ego_viewport 在渲染后裁窗口补回来。
    fovy_deg = float(np.degrees(2.0 * np.arctan((H / 2.0) / float(K[1, 1]))))
    ego_crop, ego_size = _ego_viewport(K, W, H, args.width, args.height, fovy_deg, args.ego_fov)

    # ---------------- 搭 MuJoCo 场景（初始顶点取首个有效帧）----------------
    scene = mujoco_scene.HandWorldScene(
        faces_left, faces_right,
        vl[_first_valid(kept[:, 0])], vr[_first_valid(kept[:, 1])],
        width=args.width, height=args.height, floor=not args.no_floor,
        n_ghost=max(0, int(args.ghost)), cube_size=max(0.0, float(args.cube)),
    )
    hand_pts = np.concatenate([
        world["left"]["joints"].reshape(-1, 3), world["right"]["joints"].reshape(-1, 3),
    ], axis=0)
    # 落地高度必须用「网格顶点」而非关节：关节在手内部，网格表面(掌背/指腹)比关节更低，
    # 若按关节定地台，地台会卡在关节与网格之间 → 网格穿出地台。取有效帧的顶点。
    vert_pts = np.concatenate([
        vl[kept[:, 0]].reshape(-1, 3), vr[kept[:, 1]].reshape(-1, 3),
    ], axis=0)
    # 操作者前向 = 相机平均光轴(OpenCV +z)投影到地面(XY)的方位角。垫布对齐它、四个第三人称方位也以它为基准。
    _fwd = cam_c2w[:, :3, 2]                                            # (T,3) 各帧相机光轴(world)
    _fwd = _fwd[np.all(np.isfinite(_fwd), axis=1)]
    _fwd_mean = _fwd.mean(0) if len(_fwd) else np.array([1.0, 0.0, 0.0])
    _fwd_xy = _fwd_mean[:2]
    fwd_az = float(np.arctan2(_fwd_xy[1], _fwd_xy[0])) if np.linalg.norm(_fwd_xy) > 1e-6 else 0.0
    scene.place_floor(vert_pts, margin=0.008, fwd_az=fwd_az)
    # 参照方块与地台**同口径**落地（同一批顶点、同一个 margin），底面正好贴在垫布上。
    if args.cube > 0:
        _cpos = scene.place_cube(vert_pts, offset=args.cube_pos, fwd_az=fwd_az, margin=0.008)
        scene.extend_mat_for_cube()                # 垫布扩到托得住方块（否则方块悬在垫布外像踩空）
        _how = "手动" if args.cube_pos is not None else "自动前置(不碰手)"
        print(f"[mujoco_view] 参照方块 边长={args.cube:.3f}m {_how} 中心(世界)={np.round(_cpos, 3)}")
    # 取景：对标 HaWoR「手在下、紫色相机锥体串在上」，故关键点集要把**相机轨迹**也算进去
    # （原来只框垫布范围，头戴相机在手前上方 0.3m 开外，锥体全被裁在画外：实测紫色像素 front/back 为 0）。
    #   关键点 = 手部顶点稳健包围盒(1/99 分位) ∪ 相机轨迹包围盒 ∪ 垫布四角 → 取其 AABB 8 角；
    #   lookat 的 z 偏向手（0.55/0.45），否则重心被相机抬高、手掉出画面下沿；
    #   distance 由 autofit_camera 按 free 相机 fovy 几何反解，不再用装不下的经验倍数。
    _vp = vert_pts[np.all(np.isfinite(vert_pts), axis=1)]
    _cp = cam_c2w[:, :3, 3]
    _cp = _cp[np.all(np.isfinite(_cp), axis=1)]
    _mc, _mh = scene._mat_center, scene._mat_half
    _mat_corners = np.array([[_mc[0] + sx * _mh[0], _mc[1] + sy * _mh[1], _mc[2]]
                             for sx in (-1, 1) for sy in (-1, 1)])
    _key_parts = [np.percentile(_vp, [1, 99], axis=0),
                  np.percentile(_cp, [1, 99], axis=0), _mat_corners]
    if args.cube > 0:                      # 方块也框进画面（--cube-pos 偏移较大时才起作用）
        _cc = np.asarray(scene.model.body_pos[scene._cube_bid], dtype=float)
        _ch = float(args.cube) / 2.0
        _key_parts.append(np.array([_cc - _ch, _cc + _ch]))
    _key = np.concatenate(_key_parts, axis=0)
    _lo, _hi = _key.min(0), _key.max(0)
    _fit = np.array([[x, y, z] for x in (_lo[0], _hi[0])
                     for y in (_lo[1], _hi[1]) for z in (_lo[2], _hi[2])])
    _look = (_lo + _hi) / 2.0
    _look[2] = float(_vp[:, 2].mean()) * 0.55 + float(_cp[:, 2].mean()) * 0.45
    scene.autofit_camera(_fit, azimuth=args.azimuth, elevation=args.elevation,
                         lookat=_look, margin=max(1.0, float(args.zoom)))

    if args.interactive:
        _run_interactive(scene, vl, vr, kept, cam_c2w, T, args.fps)
        scene.close()
        return

    out_dir = Path(args.out) if args.out else REPO_DIR / "output" / "eval" / args.model / \
        "mujoco_view" / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 视角表：第三人称给 (方位角, 仰角)；ego 特判走固定相机。
    # ⚠ MuJoCo free 相机的 cam.azimuth 是**视线朝向**的方位角，不是相机所在方位——
    #   实测 azimuth=0 时相机在 lookat 的 -x 侧、朝 +x 看。所以「相机站在方位 A」要写 azimuth=A+180。
    #   （早先按「相机方位」写，导致 front/back、left/right 四个方位全反了 180°。）
    # 四方位都相对『操作者前向』az0 定：
    #   front=站在操作者对面看向他(手正面，与 ego 左右镜像)、back=站在他背后过肩看(与 ego 同侧)、
    #   right/left=站在他右/左侧。操作者右向 = az0-90（前向顺时针 90°，与 place_floor 的 ex 一致）。
    az0 = float(np.degrees(fwd_az))
    view_angles = {
        "front": (az0 + 180.0, args.elevation),   # 相机在 az0 侧
        "back":  (az0,         args.elevation),   # 相机在 az0+180 侧
        "right": (az0 + 90.0,  args.elevation),   # 相机在 az0-90 侧（操作者右）
        "left":  (az0 - 90.0,  args.elevation),   # 相机在 az0+90 侧（操作者左）
        # 正对地台俯视：视线朝操作者前向压下去 → 画面上方=操作者前方，与 ego 同向好对照。
        "top":   (az0,         -88.0),
        # 兼容旧名（固定世界方位）。
        "orbit_a": (args.azimuth, args.elevation),
        "orbit_b": (args.azimuth + 110.0, args.elevation),
    }
    views = [v.strip() for v in args.views.split(",") if v.strip()]
    bad = [v for v in views if v != "ego" and v not in view_angles]
    if bad:
        print(f"[ERROR] 未知视角 {bad}；可选：{list(view_angles) + ['ego']}")
        sys.exit(1)

    if "ego" in views:
        _hf = 2.0 * np.degrees(np.arctan(ego_size[0] / 2.0 / ((args.height / 2.0)
                                                              / np.tan(np.radians(fovy_deg) / 2.0))))
        print(f"[mujoco_view] ego 画幅={args.ego_fov} 输出 {ego_size[0]}x{ego_size[1]} "
              f"(fovy={fovy_deg:.2f}° 水平FoV={_hf:.2f}°)；真实相机 {W}x{H} "
              f"fx={float(K[0, 0]):.2f} fy={float(K[1, 1]):.2f} "
              f"c=({float(K[0, 2]):.1f},{float(K[1, 2]):.1f})")
    # ego 与第三人称的输出分辨率可能不同（exact 下 ego 被裁到真实画幅），writer 各自按实际尺寸开。
    writers = {v: draw.H264PipeWriter(out_dir / f"world_{v}.mp4", float(args.fps),
                                      ego_size if v == "ego" else (args.width, args.height))
               for v in views}
    # 残影抽帧用的位移度量：双手腕世界位置拼一起。最小间距按手部半径派生，手位移小的片段
    # 自然只抽出很少几个（叠一坨没意义），位移大时沿轨迹排开 —— 与相机锥体的抽稀同思路。
    n_ghost = max(0, int(args.ghost))
    wrist = np.concatenate([world["left"]["joints"][:, 0], world["right"]["joints"][:, 0]], axis=-1)
    ghost_gap = max(0.06, float(scene._hand_rad) * 0.9)
    has_third_person = any(v != "ego" for v in views)

    step = max(1, T // 10)
    for i in range(T):
        scene.bake_frame(vl[i], vr[i], bool(kept[i, 0]), bool(kept[i, 1]))   # 每帧只摆一次网格
        gi = _pick_ghost_frames(wrist, i, n_ghost, ghost_gap) if (n_ghost and has_third_person) else []
        for v in views:
            if v == "ego":
                scene.hide_ghosts()                               # 第一人称是实时画面，不留残影
                if args.ego_floor == "billboard":
                    scene.set_floor_billboard(cam_c2w[i], fovy_deg)   # 旧行为：正对相机的背景板
                else:
                    scene.set_floor_ego_ground(cam_c2w[i], fovy_deg)  # 默认：真实世界水平地面
                scene.set_ego(cam_c2w[i], fovy_deg)
                img = scene.render_ego()
                _x0, _x1, _y0, _y1 = ego_crop                     # 裁到真实相机画幅（wide 时=整幅）
                img = img[_y0:_y1, _x0:_x1]
            else:
                if n_ghost:
                    scene.set_ghosts([vl[k] if kept[k, 0] else None for k in gi],
                                     [vr[k] if kept[k, 1] else None for k in gi])
                scene.set_floor_ground()                          # 地台→世界水平地面
                az, el = view_angles[v]
                img = scene.render_free(az, el, cam_c2w, i)
            writers[v].write(np.ascontiguousarray(img[:, :, ::-1]))          # RGB→BGR
        if (i + 1) % step == 0 or i + 1 == T:
            print(f"[mujoco_view] 渲染 {i + 1}/{T}", flush=True)
    for w in writers.values():
        w.close()
    scene.close()
    print(f"[mujoco_view] 完成 → {out_dir}  视角: {', '.join(f'world_{v}.mp4' for v in views)}")


def _run_interactive(scene, vl, vr, kept, cam_c2w, T, fps):
    """mujoco.viewer 交互回放（循环播放；仅动画手部网格，相机可鼠标环视）。"""
    import mujoco.viewer
    dt = 1.0 / max(1.0, fps)
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        i = 0
        while viewer.is_running():
            scene._set_visible("left", bool(kept[i, 0]))
            scene._set_visible("right", bool(kept[i, 1]))
            if kept[i, 0]:
                scene._bake("left", vl[i])
            if kept[i, 1]:
                scene._bake("right", vr[i])
            viewer.sync()
            time.sleep(dt)
            i = (i + 1) % T


if __name__ == "__main__":
    main()
