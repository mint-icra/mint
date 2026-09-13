# -*- coding: utf-8 -*-
"""Aria FISHEYE624 鱼眼去畸变(rectify → 针孔),供 HOT3D 数据集适配器把鱼眼帧转成针孔帧。

HOT3D 是 Aria FISHEYE624 强鱼眼;学生模型在针孔 lerobot 上训练。**去畸变成针孔后模型 in-distribution,
相机系手部精度 85→28mm**。故 benchmark 评 HOT3D 一律走针孔(鱼眼禁用于测评)。

约定(实证):
  · FISHEYE624 15 参在 camera_models.json 的 camera-rgb 条(projectionParams:f,cx,cy,6 径向 k,2 切向 p,4 薄棱镜 s)。
  · 存储图相对传感器 90°CW:(u,v)→(col=H-1-v, row=u)。
  · 针孔(upright)cam = R90 @ sensor cam,R90=[[0,-1,0],[1,0,0],[0,0,1]];标准针孔 u=f*x/z+cx(cx=cy=W/2)。
  · rectify remap:针孔像素→upright 射线→R90ᵀ→sensor 射线→FISHEYE624→存储图像素,cv2.remap 采样。
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

R90 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)   # sensor→upright


def load_fisheye624(cam_json) -> dict:
    """读 camera_models.json 的 camera-rgb 条 → FISHEYE624 参数 dict。"""
    rgb = next(c for c in json.loads(Path(cam_json).read_text()) if c["label"] == "camera-rgb")
    assert "FISHEYE624" in rgb["projectionModelType"], f"非 FISHEYE624: {rgb['projectionModelType']}"
    P = rgb["projectionParams"]
    return dict(f=P[0], cx=P[1], cy=P[2], k=P[3:9], p=P[9:11], s=P[11:15],
                H=int(rgb["imageHeight"]), W=int(rgb["imageWidth"]))


def fe_uv(pc, cam):
    """sensor cam 点 (N,3) → sensor 像素 (u,v)(FISHEYE624 正投影,不含 90°CW)。"""
    f, cx, cy, k, p, s = cam["f"], cam["cx"], cam["cy"], cam["k"], cam["p"], cam["s"]
    x, y, z = pc[:, 0], pc[:, 1], np.maximum(pc[:, 2], 1e-9)
    aa, bb = x / z, y / z
    r = np.sqrt(aa * aa + bb * bb); th = np.arctan(r); t2 = th * th
    rad = th * (1 + t2 * (k[0] + t2 * (k[1] + t2 * (k[2] + t2 * (k[3] + t2 * (k[4] + t2 * k[5]))))))
    sc = np.where(r > 1e-9, rad / np.maximum(r, 1e-9), 1.0)
    xr, yr = sc * aa, sc * bb
    x2, y2, xy = xr * xr, yr * yr, xr * yr; rr = x2 + y2
    tx = 2 * p[0] * xy + p[1] * (rr + 2 * x2); ty = p[0] * (rr + 2 * y2) + 2 * p[1] * xy
    return f * (xr + tx + s[0] * rr + s[1] * rr * rr) + cx, f * (yr + ty + s[2] * rr + s[3] * rr * rr) + cy


def pinhole_f(cam, out_w: int) -> float:
    """针孔焦距(按输出边长缩放鱼眼 focal)。"""
    return cam["f"] * (out_w / cam["W"])


def build_remap(cam, out_w: int):
    """针孔(upright,out_w×out_w,cx=cy=out_w/2)→存储鱼眼图 的 cv2.remap 表 (map_x,map_y)。"""
    f_lin = pinhole_f(cam, out_w); c = out_w / 2.0
    up, vp = np.meshgrid(np.arange(out_w), np.arange(out_w))
    dirs = np.stack([(up - c) / f_lin, (vp - c) / f_lin, np.ones_like(up, float)], -1).reshape(-1, 3)
    sensor = (R90.T @ dirs.T).T
    u, v = fe_uv(sensor, cam)
    return ((cam["H"] - 1 - v).reshape(out_w, out_w).astype(np.float32),
            u.reshape(out_w, out_w).astype(np.float32))


def rectify_image(img_bgr, map_x, map_y):
    """单帧鱼眼图 → 针孔图(cv2.remap)。"""
    return cv2.remap(img_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def rectify_to_cache(img_paths, cam, out_w: int, cache_dir: Path) -> list:
    """把鱼眼帧逐张去畸变缓存为针孔 jpg,返回针孔图路径列表(已存在则跳过)。"""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    mx, my = build_remap(cam, out_w)
    out = []
    for p in img_paths:
        dst = cache_dir / Path(p).name
        if not dst.exists():
            cv2.imwrite(str(dst), rectify_image(cv2.imread(str(p)), mx, my))
        out.append(str(dst))
    return out
