import sys
from pathlib import Path

import numpy as np


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.core.schema import EXTRINSIC, HAND, GTSequence  # noqa: E402
from benchmark.heads.extrinsics import ExtrinsicsHead  # noqa: E402
from benchmark.heads.hands_world import HandsWorldHead  # noqa: E402


def _c2w(centers):
    matrices = np.tile(np.eye(4, dtype=np.float64), (len(centers), 1, 1))
    matrices[:, :3, 3] = np.asarray(centers, np.float64)
    return matrices


def _trajectory(frames=8):
    time = np.arange(frames, dtype=np.float64)
    return np.stack([0.05 * time ** 2, 0.1 * time, 0.02 * time], axis=-1)


def test_extrinsics_reports_fitted_and_original_metric_scale_ate():
    gt_c2w = _c2w(_trajectory())
    pred_c2w = _c2w(2.0 * _trajectory())
    gt = GTSequence(
        seq_id="camera", image_paths=["frame"] * len(gt_c2w), hw=(8, 8),
        extrinsic_w2c=np.linalg.inv(gt_c2w), capability={EXTRINSIC},
    )

    head = ExtrinsicsHead()
    metrics = head.metrics(head.align(pred_c2w, gt), gt)

    np.testing.assert_allclose(metrics["scale"], 0.5, atol=1e-10)
    assert metrics["ATE_RMSE"] < 1e-10
    assert metrics["ATE_RMSE_metric"] > 0.05


def test_world_hand_reports_camera_scale_fitted_and_original_scale_variants():
    frames = 8
    gt_c2w = _c2w(_trajectory(frames))
    pred_c2w = _c2w(2.0 * _trajectory(frames))
    offsets = np.zeros((frames, 21, 3), dtype=np.float64)
    offsets[:, :, 0] = np.linspace(0.0, 0.08, 21)
    offsets[:, :, 1] = np.linspace(0.0, 0.04, 21)
    gt_world = offsets + gt_c2w[:, None, :3, 3]
    gt = GTSequence(
        seq_id="hand#right", image_paths=["frame"] * frames, hw=(8, 8),
        extrinsic_w2c=np.linalg.inv(gt_c2w), hand_joints_3d_world=gt_world,
        hand_valid=np.ones(frames, dtype=bool), capability={HAND, EXTRINSIC},
        meta={"mano_side": "right", "cam2world_slam": gt_c2w},
    )

    head = HandsWorldHead()
    aligned = head.align({
        "cam": {"right": offsets},
        "c2w_poseenc": pred_c2w,
    }, gt)
    metrics = head.metrics(aligned, gt)

    np.testing.assert_allclose(metrics["camera_scale_poseenc"], 0.5, atol=1e-10)
    assert metrics["RTE_poseenc_scaled"] < 1e-10
    assert metrics["RTE_poseenc"] > 0.1
    assert metrics["Accel_poseenc_scaled"] < 1e-10
    assert metrics["Accel_poseenc"] > 0.1
