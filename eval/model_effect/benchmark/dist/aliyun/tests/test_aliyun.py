from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


MODEL_EFFECT = Path(__file__).resolve().parents[4]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.dist.aliyun.config import AliyunConfig, load_defaults  # noqa: E402
from benchmark.dist.aliyun.dlc import (DlcClient, build_submit_args,  # noqa: E402
                                      remote_command, submission_name)
from benchmark.dist.aliyun.manager import (AliyunBenchmarkManager,  # noqa: E402
                                           _progress_with_timing)
from benchmark.dist.aliyun import worker as worker_module  # noqa: E402
from benchmark.dist.aliyun.worker import (build_shard_command,  # noqa: E402
                                          node_rank, run_request)
from benchmark.dist.aggregate import merge_result_rows  # noqa: E402


def test_defaults_use_all_eight_gpus_per_node():
    config = load_defaults()
    assert config.nnodes == 2
    assert config.gpus_per_node == 8
    assert config.world_size == 16
    assert config.region == "cn-hangzhou"
    assert config.image == "registry.example.com/project/benchmark:latest"
    assert config.job_name == "mint_benchmark"


@pytest.mark.parametrize("key,value", [
    ("nnodes", 0),
    ("gpus_per_node", 9),
    ("worker_memory", "1800"),
    ("cpfs_uri", "https://example.com/data"),
    ("repo_dir", "relative/path"),
])
def test_config_rejects_unsafe_or_out_of_range_values(key, value):
    raw = load_defaults().to_dict()
    raw[key] = value
    with pytest.raises(ValueError):
        AliyunConfig.from_mapping(raw)


def test_submit_command_contains_requested_resources(tmp_path):
    raw = load_defaults().to_dict()
    raw.update(repo_dir=str(tmp_path), nnodes=4, gpus_per_node=8)
    config = AliyunConfig.from_mapping(raw)
    request = tmp_path / "request.json"
    args = build_submit_args(config, request, display_name="unit-eval")

    assert args[args.index("--workers") + 1] == "4"
    assert args[args.index("--worker_gpu") + 1] == "8"
    assert args[args.index("--worker_image") + 1] == config.image
    assert args[args.index("--data_source_uris") + 1] == f"{config.cpfs_uri}::/benchmark-data/"
    command = args[args.index("--command") + 1]
    assert command == remote_command(config, request)
    assert str(request) in command
    assert "worker.py" in command
    assert submission_name(config, "20260806_120000") == "mint_benchmark"


def test_dlc_job_detail_accepts_cli_notice_after_json():
    client = object.__new__(DlcClient)
    client.config = load_defaults()
    client._run = lambda _args: '{"JobId":"dlc12345678","Status":"Running"}\n[OK] notice'
    assert client.get_job("dlc12345678")["Status"] == "Running"


def test_node_rank_and_global_shard_command(tmp_path):
    assert node_rank("dlcabc-master-0", {}) == 0
    assert node_rank("dlcabc-worker-0", {}) == 1
    assert node_rank("dlcabc-worker-6", {}) == 7
    assert node_rank("ignored", {"NODE_RANK": "3"}) == 3

    request = {
        "selection": {
            "heads": "hands,hands_world", "datasets": "hot3d",
            "max_seqs": None, "max_frames": 30, "seq_start": 20, "seq_end": 53,
            "dataset_selection": {
                "hot3d": {"sampling": "diverse", "sample_count": 6, "seed": 42},
            },
        },
    }
    model = {"ckpt": "/data/run/step_1", "config": "/data/run/config.yaml"}
    command = build_shard_command(
        request, model, global_rank=9, world_size=16, output_dir=tmp_path,
    )
    assert command[command.index("--shard-index") + 1] == "9"
    assert command[command.index("--shard-count") + 1] == "16"
    assert command[command.index("--max-frames") + 1] == "30"
    encoded = command[command.index("--dataset-selection-json") + 1]
    assert json.loads(encoded)["hot3d"]["sample_count"] == 6
    assert "--windowed" in command
    assert command[command.index("--hand-mode") + 1] == "hard"


def test_remote_progress_aggregates_nodes_and_local_gpus(tmp_path):
    progress_dir = tmp_path / "progress" / "model_01"
    progress_dir.mkdir(parents=True)
    rows = [
        {
            "index": 0, "node": 0, "local_gpu": 0, "done": 2, "total": 4,
            "ds_order": ["hot3d"],
            "datasets": {"hot3d": {"done": 2, "total": 4, "finished": False}},
            "results": {"row-1": {
                "head": "hands_world", "dataset": "hot3d", "seq_id": "seq-1",
                "status": "evaluated", "metrics": {"PA_MPJPE": 4.5},
            }},
        },
        {
            "index": 8, "node": 1, "local_gpu": 0, "done": 4, "total": 4,
            "ds_order": ["hot3d"],
            "datasets": {"hot3d": {"done": 4, "total": 4, "finished": True}},
        },
    ]
    for row in rows:
        (progress_dir / f"shard_{row['index']:03d}.json").write_text(
            json.dumps(row), encoding="utf-8",
        )
    snapshot = AliyunBenchmarkManager._progress_snapshot({
        "out": str(tmp_path),
        "models": [{"label": "m1"}],
        "progress": {"model_index": 1, "model_total": 1},
    })
    assert snapshot["done"] == 6
    assert snapshot["total"] == 8
    assert snapshot["frac"] == pytest.approx(0.75)
    assert [(row["node"], row["local_gpu"]) for row in snapshot["gpus"]] == [(0, 0), (1, 0)]
    assert snapshot["datasets"]["hot3d"]["done"] == 6
    assert all("results" not in row for row in snapshot["gpus"])
    report = merge_result_rows(snapshot["_live_result_rows"])
    assert report["heads"]["hands_world"]["hot3d"]["mean"]["PA_MPJPE"] == 4.5


def test_remote_progress_exposes_elapsed_total_and_remaining(monkeypatch):
    monkeypatch.setattr("benchmark.dist.aliyun.manager.time.time", lambda: 160.0)
    running = _progress_with_timing(
        {"suite_frac": 0.25}, {"running": True, "started_at": 100.0, "finished_at": None},
    )
    assert running["elapsed_s"] == 60.0
    assert running["estimated_total_s"] == 240.0
    assert running["remaining_s"] == 180.0

    finished = _progress_with_timing(
        {"suite_frac": 1.0}, {"running": False, "started_at": 100.0, "finished_at": 145.0},
    )
    assert finished["elapsed_s"] == 45.0
    assert finished["estimated_total_s"] == 45.0
    assert finished["remaining_s"] == 0.0


def test_all_cached_models_do_not_initialize_or_submit_dlc(tmp_path, monkeypatch):
    report_path = tmp_path / "cached" / "report.json"
    report_path.parent.mkdir()
    report_path.write_text(json.dumps({"heads": {}}), encoding="utf-8")
    raw = load_defaults().to_dict()
    raw["repo_dir"] = str(tmp_path)
    config = AliyunConfig.from_mapping(raw)

    def fail_client(*args, **kwargs):
        raise AssertionError("all-cache request must not initialize DLC")

    monkeypatch.setattr("benchmark.dist.aliyun.manager.DlcClient", fail_client)
    manager = AliyunBenchmarkManager(tmp_path)
    result = manager.start(
        models=[{
            "run": "run", "step": "step_1", "ckpt": str(tmp_path / "step_1"),
            "config": str(tmp_path / "config.yaml"), "tag": "run_step_1",
            "label": "run / step_1", "status": "completed", "cache_hit": True,
            "report_path": str(report_path),
        }],
        datasets="hot3d", heads="hands_world", seq_start=0, seq_end=None,
        config=config,
    )

    assert result["ok"] is True
    assert result["submitted"] is False
    assert result["cached"] == 1
    status = manager.status()
    assert status["running"] is False
    assert status["models"][0]["report"] == {"heads": {}}
    assert "未提交 DLC" in status["log"][0]


def test_single_node_worker_fans_out_and_aggregates_without_gpus(tmp_path, monkeypatch):
    output = tmp_path / "remote-output"
    request = {
        "version": 1,
        "output_dir": str(output),
        "aliyun": {"nnodes": 1, "gpus_per_node": 2},
        "selection": {
            "heads": "hands", "datasets": "hot3d", "max_seqs": None,
            "max_frames": None, "seq_start": 0, "seq_end": None,
        },
        "models": [{
            "run": "run", "step": "step_1", "ckpt": "/data/run/step_1",
            "config": "/data/run/config.yaml", "tag": "run_step_1",
            "label": "run / step_1", "out_name": "model_01_run_step_1",
            "status": "pending",
        }],
        "barrier_timeout_seconds": 5,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    def fake_shard(_request, model, *, node, local_gpu, world_size, model_index,
                   output_dir, progress_dir):
        assert (node, world_size, model_index) == (0, 2, 1)
        shard = output_dir / f"gpu{local_gpu:03d}"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "report.json").write_text(json.dumps({
            "ckpt": model["ckpt"], "config": model["config"], "heads": {},
        }), encoding="utf-8")
        return 0

    monkeypatch.setattr(worker_module, "_run_shard", fake_shard)
    assert run_request(request_path, forced_node_rank=0) == 0
    assert (output / "model_01_run_step_1" / "gpu000" / "report.json").is_file()
    assert (output / "model_01_run_step_1" / "gpu001" / "report.json").is_file()
    final = json.loads((output / "remote_state.json").read_text(encoding="utf-8"))
    assert final["running"] is False
    assert final["models"][0]["status"] == "completed"
    assert (output / "comparison.json").is_file()


def test_remote_worker_resolves_and_runs_auto_ukf_placeholder(tmp_path, monkeypatch):
    output = tmp_path / "remote-output"
    raw_report = tmp_path / "raw-report.json"
    raw_report.write_text(json.dumps({"heads": {}}), encoding="utf-8")
    request = {
        "version": 1,
        "output_dir": str(output),
        "aliyun": {"nnodes": 1, "gpus_per_node": 1},
        "selection": {
            "heads": "hands", "datasets": "hot3d", "max_seqs": None,
            "max_frames": None, "seq_start": 0, "seq_end": None,
            "reuse_cache": False, "auto_ukf_best": True,
        },
        "models": [{
            "run": "run", "step": "step_1", "ckpt": "/data/run/step_1",
            "config": "/data/run/config.yaml", "tag": "run_step_1",
            "label": "run / step_1", "out_name": "model_01_raw",
            "status": "completed", "cache_hit": True,
            "report_path": str(raw_report),
            "benchmark_signature": {"inference": {"hand_mode": "hard"}},
            "benchmark_signature_key": "raw-key",
        }, {
            "run": "", "step": "", "ckpt": None, "tag": "auto_ukf_best",
            "label": "UKF pending", "out_name": "model_02_auto_ukf",
            "status": "pending", "variant": "ukf", "hand_mode": "smooth",
            "auto_select_ukf": True,
        }],
        "barrier_timeout_seconds": 5,
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    def fake_shard(_request, model, *, node, local_gpu, world_size, model_index,
                   output_dir, progress_dir):
        assert model_index == 2
        assert model["variant"] == "ukf"
        assert model["hand_mode"] == "smooth"
        shard = output_dir / f"gpu{local_gpu:03d}"
        shard.mkdir(parents=True, exist_ok=True)
        (shard / "report.json").write_text(json.dumps({
            "ckpt": model["ckpt"], "config": model["config"], "heads": {},
        }), encoding="utf-8")
        return 0

    monkeypatch.setattr(worker_module, "_run_shard", fake_shard)
    assert run_request(request_path, forced_node_rank=0) == 0
    final = json.loads((output / "remote_state.json").read_text(encoding="utf-8"))
    assert final["models"][1]["status"] == "completed"
    assert final["models"][1]["variant"] == "ukf"
    assert "UKF融合" in final["models"][1]["label"]
