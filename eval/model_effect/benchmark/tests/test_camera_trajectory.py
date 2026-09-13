import json
import sys
from pathlib import Path

import numpy as np


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.camera_trajectory.datasets import (  # noqa: E402
    ARCTICCameraTrajectoryDataset,
    HOT3DCameraTrajectoryDataset,
)
from benchmark.camera_trajectory.metrics import (  # noqa: E402
    aggregate_trajectory_metrics,
    trajectory_metrics,
)
from benchmark.camera_trajectory.import_external_results import (  # noqa: E402
    evaluate_method_dataset,
)
from benchmark.camera_trajectory.rerun_external import (  # noqa: E402
    audit_shard_outputs,
    balanced_shards,
    recover_cached_moge_timings,
)
from benchmark.core.schema import EXTRINSIC  # noqa: E402
from benchmark.predictor import StudentPredictor  # noqa: E402


def _poses(centers, angles=None):
    centers = np.asarray(centers, np.float64)
    angles = np.zeros(len(centers)) if angles is None else np.asarray(angles, np.float64)
    result = np.tile(np.eye(4, dtype=np.float64), (len(centers), 1, 1))
    for index, angle in enumerate(angles):
        cosine, sine = np.cos(angle), np.sin(angle)
        result[index, :3, :3] = (
            (cosine, -sine, 0.0),
            (sine, cosine, 0.0),
            (0.0, 0.0, 1.0),
        )
    result[:, :3, 3] = centers
    return result


def _write_sequence(root: Path, dataset: str, name: str, frames: int):
    directory = root / dataset / name
    images = directory / "images"
    images.mkdir(parents=True)
    for index in range(frames):
        (images / f"{index:06d}.jpg").write_bytes(b"jpg")
    centers = np.stack([np.arange(frames), np.zeros(frames), np.zeros(frames)], axis=1)
    np.savez(directory / "gt.npz", c2w=_poses(centers), K=np.eye(3), hw=np.array([8, 12]))
    (directory / "meta.json").write_text(json.dumps({
        "frames": frames, "hw": [8, 12], "fps": 30.0, "gt_source": "unit-test",
    }))


def test_metrics_match_known_sim3_and_keep_metric_scale_error():
    time = np.arange(10, dtype=np.float64)
    centers = np.stack([0.03 * time ** 2, 0.1 * time, 0.02 * time], axis=1)
    gt = _poses(centers, angles=0.01 * time)
    pred = _poses(2.0 * centers, angles=0.01 * time)

    metrics = trajectory_metrics(pred, gt, forward_seconds=2.0)

    np.testing.assert_allclose(metrics["scale"], 0.5, atol=1e-12)
    assert metrics["ATE_mm"] < 1e-8
    assert metrics["ATE_S_mm"] > 100.0
    assert metrics["RPE_T_S_mm"] > metrics["RPE_T_mm"]
    assert metrics["ATE_S_pct"] > metrics["ATE_pct"]
    np.testing.assert_allclose(metrics["path_scale"], 0.5, atol=1e-12)
    np.testing.assert_allclose(metrics["FPS"], 5.0)


def test_protocol_aggregation_is_sequence_equal_but_fps_is_corpus_weighted():
    seqs = {
        "short": {"ATE_mm": 10.0, "n_frames": 10.0, "_forward_s": 1.0},
        "long": {"ATE_mm": 30.0, "n_frames": 90.0, "_forward_s": 3.0},
    }

    result = aggregate_trajectory_metrics(seqs, "camera_hot3d")

    np.testing.assert_allclose(result["mean"]["ATE_mm"], 20.0)
    np.testing.assert_allclose(result["mean"]["FPS"], 25.0)
    assert result["counts"] == {
        "sequences": 2, "frames": 100,
        "truncated_sequences": 0, "degenerate_sequences": 0,
    }
    assert result["protocol"]["alignment"] == "whole-sequence Umeyama Sim(3)"


def test_protocol_aggregation_exposes_no_scale_comparison_medians():
    seqs = {
        "a": {"ATE_S_mm": 30.0, "RPE_T_S_mm": 6.0, "RPE_R_deg": 0.3,
              "ATE_S_pct": 3.0, "n_frames": 10.0},
        "b": {"ATE_S_mm": 10.0, "RPE_T_S_mm": 2.0, "RPE_R_deg": 0.1,
              "ATE_S_pct": 1.0, "n_frames": 10.0},
        "c": {"ATE_S_mm": 20.0, "RPE_T_S_mm": 4.0, "RPE_R_deg": 0.2,
              "ATE_S_pct": 2.0, "n_frames": 10.0},
    }

    result = aggregate_trajectory_metrics(seqs, "camera_hot3d")

    assert result["mean"]["ATE_S_mm"] == 20.0
    assert result["mean"]["ATE_S_median_mm"] == 20.0
    assert result["mean"]["RPE_T_S_median_mm"] == 4.0
    assert result["mean"]["RPE_R_median_deg"] == 0.2
    assert result["protocol"]["comparison_alignment"].endswith("fixed scale=1")


def test_static_prefix_is_degenerate_instead_of_fake_zero_ate():
    static = _poses(np.zeros((4, 3)))
    moving_prediction = _poses(np.stack([
        np.arange(4), np.zeros(4), np.zeros(4),
    ], axis=1))

    metrics = trajectory_metrics(moving_prediction, static, forward_seconds=1.0)

    assert metrics["degenerate"] == 1.0
    assert "ATE_mm" not in metrics
    assert metrics["FPS"] == 4.0


def test_dataset_prefers_configured_portable_root_and_marks_prefix(tmp_path, monkeypatch):
    _write_sequence(tmp_path, "hot3d_val", "P0001_demo", 5)
    _write_sequence(tmp_path, "arctic_val", "s05_box_use_01", 4)
    monkeypatch.setenv("CAMERA_TRAJECTORY_ROOT", str(tmp_path))

    hot3d = HOT3DCameraTrajectoryDataset(str(tmp_path / "unrelated"))
    arctic = ARCTICCameraTrajectoryDataset(str(tmp_path / "unrelated"))
    assert hot3d.count_sequences() == (1, 5)
    assert arctic.count_sequences() == (1, 4)

    sequence = next(hot3d.iter_sequences(max_frames=3))
    assert sequence.seq_id == "P0001_demo"
    assert sequence.num_frames == 3
    assert sequence.capability == {EXTRINSIC}
    assert sequence.meta["truncated_for_benchmark"] is True
    assert sequence.meta["full_sequence_frames"] == 5


def test_camera_dataset_shards_are_disjoint_and_frame_balanced(tmp_path, monkeypatch):
    for index, frames in enumerate((5, 8, 13, 21)):
        _write_sequence(tmp_path, "hot3d_val", f"P{index:04d}_demo", frames)
    monkeypatch.setenv("CAMERA_TRAJECTORY_ROOT", str(tmp_path))
    adapter = HOT3DCameraTrajectoryDataset(str(tmp_path))

    shards = [set(adapter._names_for_shard(index, 2)) for index in range(2)]

    assert shards[0].isdisjoint(shards[1])
    assert shards[0] | shards[1] == set(adapter._sequence_names())


def test_camera_protocol_forces_windowed_prediction_without_loading_model():
    predictor = StudentPredictor.__new__(StudentPredictor)
    predictor.single_forward = True
    predictor._dataset_name = "camera_hot3d"
    assert predictor.effective_single_forward is False
    predictor._dataset_name = "camera_arctic"
    assert predictor.effective_single_forward is False
    predictor._dataset_name = "hot3d_hand_coverage"
    assert predictor.effective_single_forward is True


def test_external_rerun_shards_are_disjoint_and_balance_frames():
    records = [
        {"path": f"seq-{index}", "dataset": "hot3d_val",
         "sequence": f"seq-{index}", "frames": frames}
        for index, frames in enumerate((100, 90, 80, 70, 60, 50))
    ]

    shards = balanced_shards(records, 2)

    paths = [{record["path"] for record in shard} for shard in shards]
    assert paths[0].isdisjoint(paths[1])
    assert paths[0] | paths[1] == {record["path"] for record in records}
    loads = [sum(record["frames"] for record in shard) for shard in shards]
    assert max(loads) - min(loads) <= 10


def test_external_worker_audits_exact_output_shape(tmp_path):
    shard = [{"path": "unused", "dataset": "hot3d_val",
              "sequence": "P0001_demo", "frames": 5}]
    destination = tmp_path / "demo/camera_pose/hot3d_val"
    destination.mkdir(parents=True)
    np.savez(destination / "P0001_demo.npz", c2w=_poses(np.zeros((4, 3))))

    issues = audit_shard_outputs("demo", shard, tmp_path)

    assert issues[0]["kind"] == "invalid_output"
    assert "expected=(5, 4, 4)" in issues[0]["detail"]


def test_external_rerun_recovers_cached_moge_timing_from_first_log(tmp_path):
    pred_root = tmp_path / "pred"
    log_root = tmp_path / "logs"
    destination = pred_root / "megasam/camera_pose/hot3d_val/P0001_demo.npz"
    destination.parent.mkdir(parents=True)
    log_root.mkdir()
    np.savez(
        destination, c2w=_poses(np.zeros((5, 3))), frames=np.int64(5),
        seconds=np.float64(10.0), variant="megasam:moge2_depth+full_ba",
        stage_seconds=json.dumps({"moge2": 0.0, "slam": 10.0}),
    )
    (log_root / "megasam_gpu0.log").write_text(
        "[1/1] hot3d_val/P0001_demo  5 frames focal=1.0\n"
        "[moge] P0001_demo finished (3.5s)\n"
        "[moge] P0001_demo cached, skip\n",
        encoding="utf-8",
    )

    summary = recover_cached_moge_timings(pred_root, log_root, ("megasam",))

    assert summary["missing"] == []
    assert summary["repaired"][0]["moge2_seconds"] == 3.5
    with np.load(destination, allow_pickle=False) as archive:
        assert float(archive["seconds"]) == 13.5
        assert json.loads(str(archive["stage_seconds"]))["moge2"] == 3.5
        assert str(archive["timing_recovered_from"]) == "first_pass_worker_log"


def test_external_import_requires_complete_frame_exact_predictions(tmp_path):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    _write_sequence(data_root, "hot3d_val", "P0001_demo", 5)
    destination = pred_root / "demo" / "camera_pose" / "hot3d_val"
    destination.mkdir(parents=True)
    gt = np.load(data_root / "hot3d_val/P0001_demo/gt.npz")["c2w"]
    np.savez(
        destination / "P0001_demo.npz", c2w=gt, frames=np.int64(5),
        seconds=np.float64(1.0), metric_scale=np.bool_(True),
        method="demo", variant="unit-test",
    )

    result = evaluate_method_dataset(pred_root, data_root, "demo", "hot3d_val")

    assert result["status"] == "complete"
    assert result["coverage"] == {
        "sequences": 1, "expected_sequences": 1,
        "frames": 5, "expected_frames": 5,
    }
    assert result["aggregate"]["mean"]["ATE_mm"] == 0.0


def test_external_import_hides_fps_when_cached_stage_timing_is_missing(tmp_path):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    _write_sequence(data_root, "hot3d_val", "P0001_demo", 5)
    destination = pred_root / "megasam" / "camera_pose" / "hot3d_val"
    destination.mkdir(parents=True)
    gt = np.load(data_root / "hot3d_val/P0001_demo/gt.npz")["c2w"]
    np.savez(
        destination / "P0001_demo.npz", c2w=gt, frames=np.int64(5),
        seconds=np.float64(2.0), metric_scale=np.bool_(True),
        method="megasam",
        variant="megasam:moge2_depth+full_ba",
        stage_seconds=json.dumps({"moge2": 0.0, "slam": 2.0}),
    )

    result = evaluate_method_dataset(pred_root, data_root, "megasam", "hot3d_val")

    assert result["status"] == "complete"
    assert result["aggregate"]["mean"]["ATE_mm"] == 0.0
    assert result["aggregate"]["mean"]["FPS"] is None
    assert result["timing"] == {
        "status": "incomplete", "incomplete_sequences": ["P0001_demo"],
    }


def test_external_import_requires_egopipeline_geocalib_timing(tmp_path):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    _write_sequence(data_root, "hot3d_val", "P0001_demo", 5)
    destination = pred_root / "egopipeline" / "camera_pose" / "hot3d_val"
    destination.mkdir(parents=True)
    gt = np.load(data_root / "hot3d_val/P0001_demo/gt.npz")["c2w"]
    np.savez(
        destination / "P0001_demo.npz", c2w=gt, frames=np.int64(5),
        seconds=np.float64(3.0), metric_scale=np.bool_(True),
        method="egopipeline",
        variant="egopipeline:geocalib_intr+moge2_depth+full_ba",
        stage_seconds=json.dumps({"geocalib": 0.0, "moge2": 1.0, "slam": 2.0}),
    )

    result = evaluate_method_dataset(
        pred_root, data_root, "egopipeline", "hot3d_val",
    )

    assert result["status"] == "complete"
    assert result["aggregate"]["mean"]["ATE_mm"] == 0.0
    assert result["aggregate"]["mean"]["FPS"] is None
    assert result["timing"] == {
        "status": "incomplete", "incomplete_sequences": ["P0001_demo"],
    }


def test_external_import_rejects_method_variant_mismatch(tmp_path):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    _write_sequence(data_root, "hot3d_val", "P0001_demo", 5)
    destination = pred_root / "droid_slam_official/camera_pose/hot3d_val"
    destination.mkdir(parents=True)
    gt = np.load(data_root / "hot3d_val/P0001_demo/gt.npz")["c2w"]
    np.savez(
        destination / "P0001_demo.npz", c2w=gt, frames=np.int64(5),
        seconds=np.float64(1.0), metric_scale=np.bool_(False),
        method="droid_slam_official",
        variant="droid_slam:no_depth+no_global_ba",
    )

    result = evaluate_method_dataset(
        pred_root, data_root, "droid_slam_official", "hot3d_val",
    )

    assert result["status"] == "incomplete"
    assert result["issues"][0]["kind"] == "invalid_prediction"
    assert "expected prefix='droid_slam_official:'" in result["issues"][0]["detail"]


def test_external_import_rejects_segmented_global_trajectory(tmp_path):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    _write_sequence(data_root, "hot3d_val", "P0001_demo", 5)
    destination = pred_root / "hawor" / "camera_pose" / "hot3d_val"
    destination.mkdir(parents=True)
    gt = np.load(data_root / "hot3d_val/P0001_demo/gt.npz")["c2w"]
    np.savez(
        destination / "P0001_demo.npz", c2w=gt, frames=np.int64(5),
        seconds=np.float64(1.0), metric_scale=np.bool_(True),
        method="hawor",
        variant="hawor_official:test", align_protocol="per_part_1000",
        part_bounds=np.array([[0, 3], [3, 5]]),
    )

    result = evaluate_method_dataset(pred_root, data_root, "hawor", "hot3d_val")

    assert result["status"] == "unsupported_global_trajectory"
    assert "aggregate" not in result


def test_segmented_method_still_requires_complete_coverage(tmp_path):
    data_root = tmp_path / "data"
    pred_root = tmp_path / "pred"
    _write_sequence(data_root, "hot3d_val", "P0001_demo", 5)
    _write_sequence(data_root, "hot3d_val", "P0002_missing", 4)
    destination = pred_root / "hawor" / "camera_pose" / "hot3d_val"
    destination.mkdir(parents=True)
    gt = np.load(data_root / "hot3d_val/P0001_demo/gt.npz")["c2w"]
    np.savez(
        destination / "P0001_demo.npz", c2w=gt, frames=np.int64(5),
        seconds=np.float64(1.0), metric_scale=np.bool_(True),
        method="hawor",
        variant="hawor_official:test", align_protocol="per_part_1000",
        part_bounds=np.array([[0, 3], [3, 5]]),
    )

    result = evaluate_method_dataset(pred_root, data_root, "hawor", "hot3d_val")

    assert result["status"] == "incomplete"
    assert result["coverage"]["sequences"] == 1
    assert result["issues"][0]["sequence"] == "P0002_missing"
