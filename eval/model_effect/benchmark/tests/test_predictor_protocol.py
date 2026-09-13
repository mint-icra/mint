import sys
from pathlib import Path


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.predictor import StudentPredictor  # noqa: E402


def _predictor(single_forward=False):
    predictor = StudentPredictor.__new__(StudentPredictor)
    predictor.single_forward = single_forward
    predictor._dataset_name = ""
    return predictor


def test_coverage_protocol_forces_single_forward_without_changing_hot3d():
    predictor = _predictor(single_forward=False)

    predictor.set_benchmark_dataset("hot3d")
    assert predictor.effective_single_forward is False

    predictor.set_benchmark_dataset("hot3d_hand_coverage")
    assert predictor.effective_single_forward is True

    predictor.set_benchmark_dataset("arctic_hand_coverage")
    assert predictor.effective_single_forward is True


def test_base_single_forward_remains_enabled_for_all_datasets():
    predictor = _predictor(single_forward=True)

    predictor.set_benchmark_dataset("hot3d")
    assert predictor.effective_single_forward is True
