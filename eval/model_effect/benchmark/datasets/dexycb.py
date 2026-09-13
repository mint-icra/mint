# -*- coding: utf-8 -*-
"""DexYCB 适配器 ✅:第三视角手物抓取,供 intrinsics + hands 两头。

落盘 data/benchmark/hand_pose/dexycb/ 下:
  <subject>(YYYYMMDD-subject-NN)/<seq>(YYYYMMDD_HHMMSS)/<serial>(相机号)/color_*.jpg + labels_*.npz
  calibration/intrinsics/<serial>_640x480.yml(color K)、mano_<calib>/mano.yml(betas)
每「序列×相机」一条 GTSequence(该相机连续帧、内参恒定、joint_3d 在该相机系)。

GT:labels_*.npz['joint_3d'] = 相机系公制米、OpenPose 21 序(0=wrist);手不在画面帧 DexYCB 标 -1(非 NaN),
本适配器只保留有效帧(21 关节均 Z>0),故 image_paths/hand_joints_3d 都是手在画面的连续有效帧。
⚠ 内参 yml 尾部 extrinsics 用 !!python/tuple 标签(safe_load 会挂),故截断到该行前再解析。
"""
from __future__ import annotations

import glob
import os
from typing import Iterator

import numpy as np
import yaml

from ..core.registry import DATASETS
from ..core.schema import HAND, INTRINSIC, GTSequence
from .base import DatasetAdapter

_HW = (480, 640)
# DEXYCB_SERIALS 环境变量:逗号分隔的相机号,只评这些视角;不设=全 8 视角。
_ONLY_SERIALS = set(filter(None, os.environ.get("DEXYCB_SERIALS", "").split(",")))


def _read_K(yml_path: str) -> np.ndarray:
    """读 calibration/intrinsics/<serial>_640x480.yml 的 color 内参 → K[3,3]。
    尾部 extrinsics 带 !!python/tuple 标签会让 safe_load 抛错,截断到该行前再解析。"""
    buf = []
    with open(yml_path) as f:
        for ln in f:
            if ln.startswith("extrinsics"):
                break
            buf.append(ln)
    c = yaml.safe_load("".join(buf))["color"]
    return np.array([[c["fx"], 0.0, c["ppx"]],
                     [0.0, c["fy"], c["ppy"]],
                     [0.0, 0.0, 1.0]], np.float64)


def _read_betas(mano_yml: str):
    with open(mano_yml) as f:
        return list(yaml.safe_load(f)["betas"])


@DATASETS.register("dexycb")
class DexYCBAdapter(DatasetAdapter):
    name = "dexycb"
    root_rel = "hand_pose/dexycb"
    capability = {INTRINSIC, HAND}

    def _visual_entries(self) -> list[dict]:
        cached = getattr(self, "_visual_entries_cache", None)
        if cached is not None:
            return cached
        calib = os.path.join(self.root, "calibration")
        if not os.path.isdir(calib):
            raise FileNotFoundError(
                f"未找到 DexYCB(缺 {calib});先下载解压 subject-* + calibration")
        entries = []
        subjects = sorted(d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d)) and "subject-" in d)
        if not subjects:
            raise FileNotFoundError(f"{self.root} 下无 <date>-subject-NN 目录")
        for subj in subjects:
            subj_dir = os.path.join(self.root, subj)
            for seq in sorted(os.listdir(subj_dir)):
                meta_f = os.path.join(subj_dir, seq, "meta.yml")
                if not os.path.isfile(meta_f):
                    continue
                with open(meta_f) as handle:
                    meta = yaml.safe_load(handle)
                if "right" not in (meta.get("mano_sides") or []):
                    continue
                for serial in meta.get("serials", []):
                    if _ONLY_SERIALS and serial not in _ONLY_SERIALS:
                        continue
                    cam_dir = os.path.join(subj_dir, seq, serial)
                    colors = sorted(glob.glob(os.path.join(cam_dir, "color_*.jpg")))
                    K_yml = os.path.join(calib, "intrinsics", f"{serial}_640x480.yml")
                    if colors and os.path.isfile(K_yml):
                        sid = f"{subj}/{seq}/{serial}"
                        entries.append({"seq_id": sid, "label": sid,
                                        "frame_count": len(colors), "cam_dir": cam_dir})
        self._visual_entries_cache = entries
        return entries

    def list_visual_sequences(self) -> list[dict]:
        return [{key: item[key] for key in ("seq_id", "label", "frame_count")}
                for item in self._visual_entries()]

    def load_visual_sequence(self, seq_id: str, max_frames=None) -> dict:
        entry = next((item for item in self._visual_entries()
                      if item["seq_id"] == seq_id), None)
        if entry is None:
            raise KeyError(f"DexYCB 中不存在序列 {seq_id!r}")
        paths = []
        colors = sorted(glob.glob(os.path.join(entry["cam_dir"], "color_*.jpg")))
        for color in colors:
            number = os.path.basename(color)[len("color_"):-len(".jpg")]
            label = os.path.join(entry["cam_dir"], f"labels_{number}.npz")
            if not os.path.isfile(label):
                continue
            joints = np.load(label)["joint_3d"][0]
            if np.isfinite(joints).all() and (joints[:, 2] > 0).all():
                paths.append(color)
                if max_frames and len(paths) >= max_frames:
                    break
        if not paths:
            raise FileNotFoundError(f"DexYCB 序列 {seq_id!r} 没有测评有效帧")
        return {"seq_id": seq_id, "label": seq_id, "image_paths": paths,
                "hw": _HW, "frame_count": len(paths), "fps": 30.0,
                "source_path": entry["cam_dir"]}

    def count_sequences(self):
        """廉价计数(面板显示规模用):只走 目录 + meta.yml + glob 原图,**不**逐帧 np.load 标注
        (iter_sequences 为筛有效帧对每帧读 labels_*.npz,极慢)。选序列条件与 iter 一致(右手序列、
        serial 白名单、colors+K 齐全);n_frames 用原图张数估计(未按标注有效性过滤,仅作规模)。"""
        calib = os.path.join(self.root, "calibration")
        if not os.path.isdir(calib):
            raise FileNotFoundError(
                f"未找到 DexYCB(缺 {calib});先下载解压 subject-* + calibration")
        subjects = sorted(d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d)) and "subject-" in d)
        if not subjects:
            raise FileNotFoundError(f"{self.root} 下无 <date>-subject-NN 目录")
        n_seqs = n_frames = 0
        for subj in subjects:
            subj_dir = os.path.join(self.root, subj)
            for seq in sorted(os.listdir(subj_dir)):
                meta_f = os.path.join(subj_dir, seq, "meta.yml")
                if not os.path.isfile(meta_f):
                    continue
                with open(meta_f) as f:
                    meta = yaml.safe_load(f)
                if "right" not in (meta.get("mano_sides") or []):
                    continue
                for serial in meta.get("serials", []):
                    if _ONLY_SERIALS and serial not in _ONLY_SERIALS:
                        continue
                    cam_dir = os.path.join(subj_dir, seq, serial)
                    colors = glob.glob(os.path.join(cam_dir, "color_*.jpg"))
                    K_yml = os.path.join(calib, "intrinsics", f"{serial}_640x480.yml")
                    if not colors or not os.path.isfile(K_yml):
                        continue
                    n_seqs += 1
                    n_frames += len(colors)
        return n_seqs, n_frames

    def iter_sequences(self, max_seqs=None, max_frames=None) -> Iterator[GTSequence]:
        calib = os.path.join(self.root, "calibration")
        if not os.path.isdir(calib):
            raise FileNotFoundError(
                f"未找到 DexYCB(缺 {calib});先下载解压 subject-* + calibration")
        subjects = sorted(d for d in os.listdir(self.root)
                          if os.path.isdir(os.path.join(self.root, d)) and "subject-" in d)
        if not subjects:
            raise FileNotFoundError(f"{self.root} 下无 <date>-subject-NN 目录")

        n = 0
        for subj in subjects:
            subj_dir = os.path.join(self.root, subj)
            for seq in sorted(os.listdir(subj_dir)):
                meta_f = os.path.join(subj_dir, seq, "meta.yml")
                if not os.path.isfile(meta_f):
                    continue
                with open(meta_f) as f:
                    meta = yaml.safe_load(f)
                if "right" not in (meta.get("mano_sides") or []):
                    continue                                    # DexYCB 单手;只评右手序列
                calib_id = (meta.get("mano_calib") or [None])[0]
                betas = None
                if calib_id:
                    mano_yml = os.path.join(calib, f"mano_{calib_id}", "mano.yml")
                    if os.path.isfile(mano_yml):
                        betas = _read_betas(mano_yml)

                for serial in meta.get("serials", []):
                    if _ONLY_SERIALS and serial not in _ONLY_SERIALS:
                        continue                            # 只评 DEXYCB_SERIALS 指定的相机视角
                    cam_dir = os.path.join(subj_dir, seq, serial)
                    colors = sorted(glob.glob(os.path.join(cam_dir, "color_*.jpg")))
                    K_yml = os.path.join(calib, "intrinsics", f"{serial}_640x480.yml")
                    if not colors or not os.path.isfile(K_yml):
                        continue
                    K = _read_K(K_yml)
                    imgs, joints = [], []
                    for cf in colors:                           # 遍历全部帧,只收有效帧(手在画面)
                        num = os.path.basename(cf)[len("color_"):-len(".jpg")]
                        lb = os.path.join(cam_dir, f"labels_{num}.npz")
                        if not os.path.isfile(lb):
                            continue
                        j = np.load(lb)["joint_3d"][0].astype(np.float64)   # [21,3] 相机系米
                        if not (np.isfinite(j).all() and (j[:, 2] > 0).all()):
                            continue                            # DexYCB 无标注帧标 -1(Z<0),跳过
                        imgs.append(cf); joints.append(j)
                        if max_frames and len(imgs) >= max_frames:
                            break
                    if not imgs:
                        continue
                    yield GTSequence(
                        seq_id=f"{subj}/{seq}/{serial}",
                        image_paths=imgs, hw=_HW,
                        intrinsic=K,
                        hand_joints_3d=np.stack(joints),        # [S,21,3] 均有效帧
                        hand_valid=np.ones(len(imgs), bool),    # [S] 已过滤,全 True
                        capability=self.capability,
                        meta={"dataset": "dexycb", "subject": subj, "seq": seq,
                              "serial": serial, "mano_side": "right", "betas": betas},
                    )
                    n += 1
                    if max_seqs and n >= max_seqs:
                        return
