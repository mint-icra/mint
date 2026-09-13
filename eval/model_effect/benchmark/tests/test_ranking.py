import sys
from pathlib import Path


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.ranking import resolve_auto_ukf_model, select_best_model  # noqa: E402


def _model(label, *, ate, rte, facc):
    return {
        "label": label, "status": "completed", "report": {"heads": {
            "extrinsics": {"hot3d": {"mean": {"ATE_RMSE": ate}}},
            "hands_world": {"hot3d": {"mean": {"RTE_poseenc_scaled": rte}}},
            "hands_coverage": {"hot3d_hand_coverage": {"mean": {"FAcc": facc}}},
        }},
    }


def test_best_model_uses_unit_free_average_quality():
    models = [
        _model("balanced", ate=0.3, rte=5.0, facc=0.6),
        _model("bad-two-columns", ate=0.1, rte=9.0, facc=0.3),
        _model("best-average", ate=0.2, rte=1.0, facc=0.9),
    ]

    winner, ranking = select_best_model(models)

    assert winner["label"] == "best-average"
    assert ranking["metrics"] == 3
    assert ranking["candidates"] == 3


def test_single_model_is_selected_without_comparable_metrics():
    model = {"label": "only", "status": "completed", "report": {"heads": {}}}
    winner, ranking = select_best_model([model])
    assert winner is model
    assert ranking == {"score": 0.0, "metrics": 0, "candidates": 1}


def test_ukf_variant_keeps_checkpoint_but_changes_inference_signature():
    model = _model("raw", ate=0.2, rte=2.0, facc=0.8)
    model.update({
        "run": "run", "step": "step_1", "ckpt": "/tmp/run/step_1",
        "config": "/tmp/config.yaml", "model": "lingbotmap", "tag": "run_step_1",
        "benchmark_signature": {"checkpoint": {"path": "/tmp/run/step_1"},
                                "inference": {"hand_mode": "hard"}},
        "benchmark_signature_key": "raw-key",
    })

    ukf = resolve_auto_ukf_model([model], reuse_cache=False)

    assert ukf["ckpt"] == model["ckpt"]
    assert ukf["variant"] == "ukf"
    assert ukf["hand_mode"] == "smooth"
    assert ukf["benchmark_signature"]["inference"]["hand_mode"] == "smooth"
    assert ukf["benchmark_signature_key"] != model["benchmark_signature_key"]
