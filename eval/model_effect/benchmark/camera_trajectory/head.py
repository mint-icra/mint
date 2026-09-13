# -*- coding: utf-8 -*-
"""Dedicated full-sequence camera trajectory evaluation head."""
from __future__ import annotations

import numpy as np

from ..core.registry import HEADS
from ..core.schema import EXTRINSIC, GTSequence, Prediction
from ..heads.base import HeadEvaluator
from .metrics import trajectory_metrics


@HEADS.register("camera_trajectory")
class CameraTrajectoryHead(HeadEvaluator):
    name = "camera_trajectory"
    required_gt = {EXTRINSIC}

    def extract(self, pred: Prediction):
        timings = dict((pred.meta or {}).get("timings") or {})
        return {
            "c2w": np.asarray(pred.extrinsic_c2w, np.float64),
            "forward_s": timings.get("forward_s"),
        }

    def align(self, item, gt: GTSequence):
        return item

    def metrics(self, item, gt: GTSequence):
        gt_c2w = np.linalg.inv(np.asarray(gt.extrinsic_w2c, np.float64))
        metrics = trajectory_metrics(
            item["c2w"], gt_c2w, forward_seconds=item.get("forward_s"),
        )
        metrics["_truncated"] = float(bool((gt.meta or {}).get("truncated_for_benchmark")))
        metrics["_full_frames"] = float((gt.meta or {}).get("full_sequence_frames", len(gt_c2w)))
        return metrics
