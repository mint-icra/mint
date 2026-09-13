import json
import sys
from pathlib import Path

import pytest


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark import cache  # noqa: E402


def _report(ckpt, *, facc=0.9, mpjpe=10.0, same_split=False):
    return {
        "ckpt": str(ckpt), "config": "config.yaml",
        "selection": {"seq_start": 0, "seq_end": None, "max_frames": None},
        "heads": {
            "hands_coverage": {
                "hot3d_hand_coverage": {
                    "seqs": {"clip-1": {"FAcc": facc, "MPJPE-p": mpjpe}},
                    "status_counts": {"evaluated": 1},
                    "mean": {"FAcc": facc, "MPJPE-p": mpjpe},
                    "protocol": {"reference_same_split": same_split},
                    "reference": {"metrics": {"FAcc": 0.85, "MPJPE-p": 11.0}},
                },
            },
        },
    }


def _signature(tmp_path, *, max_frames=None, dataset_selection=None, hand_mode="hard"):
    ckpt = tmp_path / "models" / "run" / "step_00001000"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "model.safetensors").write_bytes(b"weights")
    config = tmp_path / "models" / "run" / "logs" / "record" / "config.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("model: test\n", encoding="utf-8")
    return ckpt, cache.build_signature(
        ckpt=ckpt, config=config, heads="hands_coverage",
        datasets="hot3d_hand_coverage", seq_start=0, seq_end=None,
        max_frames=max_frames, dataset_selection=dataset_selection,
        hand_mode=hand_mode,
    )


def test_exact_successful_report_is_reused(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmark"
    monkeypatch.setattr(cache, "BENCHMARK_ROOT", benchmark_root)
    ckpt, signature = _signature(tmp_path)
    report_path = benchmark_root / "run-1" / "model-1" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(_report(ckpt)), encoding="utf-8")

    cache.register_cached_report(signature, report_path)
    hit = cache.find_cached_report(signature, benchmark_root)

    assert hit is not None
    assert hit["report_path"] == str(report_path.resolve())
    assert hit["report"]["ckpt"] == str(ckpt)
    _, changed = _signature(tmp_path, max_frames=81)
    assert cache.find_cached_report(changed, benchmark_root) is None
    _, changed_sampling = _signature(tmp_path, dataset_selection={
        "hot3d_hand_coverage": {"sampling": "diverse", "sample_count": 32, "seed": 42},
    })
    assert cache.find_cached_report(changed_sampling, benchmark_root) is None
    _, ukf_signature = _signature(tmp_path, hand_mode="smooth")
    assert cache.signature_key(ukf_signature) != cache.signature_key(signature)
    assert cache.find_cached_report(ukf_signature, benchmark_root) is None


def test_remote_aliyun_report_is_found_without_recursive_cache_scan(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmark"
    monkeypatch.setattr(cache, "BENCHMARK_ROOT", benchmark_root)
    ckpt, signature = _signature(tmp_path)
    report_path = benchmark_root / "aliyun" / "job-1" / "model-1" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(_report(ckpt)), encoding="utf-8")

    cache.register_cached_report(signature, report_path)
    hit = cache.find_cached_report(signature, benchmark_root)

    assert hit is not None
    assert hit["report_path"] == str(report_path.resolve())


def test_partial_or_error_report_is_not_cached(tmp_path, monkeypatch):
    benchmark_root = tmp_path / "benchmark"
    monkeypatch.setattr(cache, "BENCHMARK_ROOT", benchmark_root)
    ckpt, signature = _signature(tmp_path)
    report = _report(ckpt)
    report["heads"]["hands_coverage"]["hot3d_hand_coverage"]["status_counts"]["error"] = 1
    report_path = benchmark_root / "run-1" / "report.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="不完整"):
        cache.register_cached_report(signature, report_path, report)


def test_step_log_records_every_completed_model_without_copying_sequences(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    benchmark_root = tmp_path / "benchmark"
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)
    models = []
    for index, (facc, mpjpe, same_split) in enumerate(((0.9, 10.0, True), (0.8, 9.0, False)), 1):
        ckpt = model_root / f"run-{index}" / f"step_{index:08d}"
        ckpt.mkdir(parents=True)
        report_path = benchmark_root / f"model-{index}" / "report.json"
        report_path.parent.mkdir(parents=True)
        report = _report(ckpt, facc=facc, mpjpe=mpjpe, same_split=same_split)
        report_path.write_text(json.dumps(report), encoding="utf-8")
        signature = {"model": index}
        models.append({
            "status": "completed", "ckpt": str(ckpt), "step": ckpt.name,
            "label": f"m{index}", "report": report, "report_path": str(report_path),
            "benchmark_signature": signature,
            "benchmark_signature_key": cache.signature_key(signature),
        })

    written = cache.publish_step_benchmark_logs(models, {"datasets": "hot3d_hand_coverage"})

    assert len(written) == 2
    first = json.loads(Path(written[0]).read_text(encoding="utf-8"))
    assert first["official_sota_metrics"]
    assert any(row["metric"] == "FAcc" for row in first["local_best_metrics"])
    node = first["result"]["heads"]["hands_coverage"]["hot3d_hand_coverage"]
    assert "seqs" not in node
    index = json.loads((Path(models[0]["ckpt"]).parent / "logs" /
                        f"benchmark_{models[0]['step']}_custom.json").read_text(encoding="utf-8"))
    assert index["version"] == 2
    assert first["record_id"] in index["records"]


def test_step_log_writes_non_winner_and_keeps_identical_events(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)
    ckpt = model_root / "run" / "step_00001000"
    ckpt.mkdir(parents=True)
    report = _report(ckpt, facc=0.1, mpjpe=99.0, same_split=False)
    signature = {"model": "no-winner"}
    model = {
        "status": "completed", "ckpt": str(ckpt), "step": ckpt.name,
        "label": "no winner", "report": report,
        "report_path": str(tmp_path / "benchmark" / "report.json"),
        "benchmark_signature": signature,
        "benchmark_signature_key": cache.signature_key(signature),
    }

    first = cache.publish_step_benchmark_logs([model], {"dataset_selection": {}})
    second = cache.publish_step_benchmark_logs([model], {"dataset_selection": {}})

    assert len(first) == len(second) == 1
    assert first[0] != second[0]
    assert Path(first[0]).is_file() and Path(second[0]).is_file()
    index = json.loads((ckpt.parent / "logs" /
                        f"benchmark_{ckpt.name}_custom.json").read_text())
    assert len(index["records"]) == 2
    records = cache.list_benchmark_history("run", ckpt.name)
    assert len(records) == 2
    assert cache.list_benchmark_history() == records
    loaded = cache.load_benchmark_history("run", records[0]["record_id"])
    assert loaded["model"]["historical"] is True
    assert loaded["model"]["report"]["heads"]


def test_ukf_step_log_has_distinct_timestamped_name(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)
    ckpt = model_root / "run" / "step_00002000"
    ckpt.mkdir(parents=True)
    signature = {"model": "ukf"}
    model = {
        "status": "completed", "ckpt": str(ckpt), "step": ckpt.name,
        "label": "ukf", "variant": "ukf", "hand_mode": "smooth",
        "report": _report(ckpt), "report_path": str(tmp_path / "report.json"),
        "benchmark_signature": signature,
        "benchmark_signature_key": cache.signature_key(signature),
    }

    written = cache.publish_step_benchmark_logs([model], {
        "dataset_selection": {"hot3d": {"fixed_tier": "half"}},
    })

    assert len(written) == 1
    assert Path(written[0]).name.startswith("benchmark_step_00002000_ukf_50pct_")
    event = json.loads(Path(written[0]).read_text())
    assert event["variant"] == "ukf"
    assert event["hand_mode"] == "smooth"
    assert event["sampling_tier"] == "half"
    assert (ckpt.parent / "logs" / "benchmark_step_00002000_50pct.json").is_file()


def test_partial_completed_model_does_not_get_history_log(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)
    ckpt, signature = _signature(tmp_path)
    report = _report(ckpt)
    report["heads"]["hands_coverage"]["hot3d_hand_coverage"]["status_counts"]["error"] = 1
    model = {
        "status": "completed", "ckpt": str(ckpt), "step": ckpt.name,
        "report": report, "benchmark_signature": signature,
        "benchmark_signature_key": cache.signature_key(signature),
    }

    assert cache.publish_step_benchmark_logs([model], {}) == []
    assert not list((ckpt.parent / "logs").glob("benchmark_*.json"))


def test_history_rejects_path_traversal(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    model_root.mkdir()
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)

    with pytest.raises(ValueError, match="超出"):
        cache.list_benchmark_history("../outside")
    with pytest.raises(ValueError, match="record_id"):
        cache.load_benchmark_history("run", "../../record")


def test_legacy_step_index_remains_selectable_and_detects_ukf(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)
    ckpt = model_root / "run" / "step_00003000"
    logs = ckpt.parent / "logs"
    ckpt.mkdir(parents=True)
    logs.mkdir()
    report = _report(ckpt)
    payload = {
        "version": 1, "step": ckpt.name, "checkpoint": str(ckpt),
        "benchmarks": {
            "legacy-key": {
                "recorded_at": "2026-08-07T12:00:00+08:00",
                "source_report": str(tmp_path / "model_ukf" / "report.json"),
                "selection": {"dataset_selection": {}},
                "result": cache._report_summary(report),
            },
        },
    }
    (logs / f"{ckpt.name}_benchmark.json").write_text(json.dumps(payload), encoding="utf-8")

    records = cache.list_benchmark_history("run", ckpt.name)

    assert len(records) == 1
    assert records[0]["variant"] == "ukf"
    loaded = cache.load_benchmark_history("run", records[0]["record_id"])
    assert loaded["model"]["variant"] == "ukf"


def test_new_sampling_tier_does_not_absorb_untiered_legacy_index(tmp_path, monkeypatch):
    model_root = tmp_path / "models"
    monkeypatch.setattr(cache, "MODEL_TRAIN_ROOT", model_root)
    ckpt = model_root / "run" / "step_00004000"
    logs = ckpt.parent / "logs"
    ckpt.mkdir(parents=True)
    logs.mkdir()
    report = _report(ckpt)
    legacy = {
        "version": 1, "step": ckpt.name, "checkpoint": str(ckpt),
        "benchmarks": {
            "quarter-key": {
                "selection": {"dataset_selection": {"hot3d": {"fixed_tier": "quarter"}}},
                "result": cache._report_summary(report),
            },
        },
    }
    (logs / f"benchmark_{ckpt.name}.json").write_text(json.dumps(legacy), encoding="utf-8")
    signature = {"model": "half"}
    model = {
        "status": "completed", "ckpt": str(ckpt), "step": ckpt.name,
        "report": report, "benchmark_signature": signature,
        "benchmark_signature_key": cache.signature_key(signature),
    }

    cache.publish_step_benchmark_logs([model], {
        "dataset_selection": {"hot3d": {"fixed_tier": "half"}},
    })

    index = json.loads((logs / f"benchmark_{ckpt.name}_50pct.json").read_text())
    assert "quarter-key" not in index["benchmarks"]
    assert len(index["records"]) == 1
