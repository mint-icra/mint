# -*- coding: utf-8 -*-
"""intrinsics 头 ✅:相机内参精度。分辨率归一后 fx/fy 相对误差% / FoV° / 主点 px。

对齐:内参与分辨率绑定,pred(在 pred.hw)与 GT(在 gt.hw)先各自归一到「焦距/边长」再比,
FoV 本身分辨率无关。同行口径:GeoCalib / Perspective Fields。

⚠ 学生 K 主点固定取图像中心(K_from_fov),故主点误差恒近 0、无区分度,只作记录。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ..core.registry import HEADS
from ..core.schema import INTRINSIC, GTSequence, Prediction
from .base import HeadEvaluator


def _first_K(K) -> np.ndarray:
    K = np.asarray(K, np.float64)
    return K if K.ndim == 2 else K[0]     # 恒定内参取首帧


def _fov_deg(fx, W) -> float:
    return float(np.degrees(2.0 * np.arctan(W / (2.0 * fx))))


@HEADS.register("intrinsics")
class IntrinsicsHead(HeadEvaluator):
    name = "intrinsics"
    required_gt = {INTRINSIC}

    def extract(self, pred: Prediction) -> Any:
        return {"K": _first_K(pred.intrinsic), "hw": tuple(pred.hw)}

    def align(self, item: Dict, gt: GTSequence):
        # 归一 = 与分辨率解绑;这里不改数据,只把 pred/gt 的 (K,hw) 并列,metrics 里各自按边长归一。
        return {"pK": item["K"], "pHW": item["hw"],
                "gK": _first_K(gt.intrinsic), "gHW": tuple(gt.hw)}

    def metrics(self, a: Dict, gt: GTSequence) -> Dict[str, float]:
        pK, (pH, pW) = a["pK"], a["pHW"]
        gK, (gH, gW) = a["gK"], a["gHW"]
        # 归一焦距(焦距/边长,尺度无关)相对误差 %
        pfx_n, pfy_n = pK[0, 0] / pW, pK[1, 1] / pH
        gfx_n, gfy_n = gK[0, 0] / gW, gK[1, 1] / gH
        fx_err = abs(pfx_n - gfx_n) / max(abs(gfx_n), 1e-9) * 100.0
        fy_err = abs(pfy_n - gfy_n) / max(abs(gfy_n), 1e-9) * 100.0
        # FoV(分辨率无关)绝对误差(度)
        fov_x_err = abs(_fov_deg(pK[0, 0], pW) - _fov_deg(gK[0, 0], gW))
        fov_y_err = abs(_fov_deg(pK[1, 1], pH) - _fov_deg(gK[1, 1], gH))
        # 主点偏移(归一到边长的 px 差;学生恒中心,仅记录)
        cx_off = abs(pK[0, 2] / pW - gK[0, 2] / gW) * gW
        cy_off = abs(pK[1, 2] / pH - gK[1, 2] / gH) * gH
        return {"fx_relerr_pct": fx_err, "fy_relerr_pct": fy_err,
                "fov_x_deg": fov_x_err, "fov_y_deg": fov_y_err,
                "cx_off_px": cx_off, "cy_off_px": cy_off}
