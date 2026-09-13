import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from inference.acceleration import (  # noqa: E402
    COMPILE_MODES,
    _eligible_fp8_linear,
    normalize_optional_mode,
    prepare_allocator_for_compile,
)
from inference.engine import StudentEngine  # noqa: E402


def test_optional_acceleration_modes_normalize_off_values():
    assert normalize_optional_mode(None, choices=COMPILE_MODES, name="compile") is None
    assert normalize_optional_mode("off", choices=COMPILE_MODES, name="compile") is None
    assert (
        normalize_optional_mode(
            "REDUCE-OVERHEAD", choices=COMPILE_MODES, name="compile"
        )
        == "reduce-overhead"
    )
    with pytest.raises(ValueError, match="Unsupported compile"):
        normalize_optional_mode("fastest", choices=COMPILE_MODES, name="compile")


def test_cudagraph_compile_removes_expandable_segments_before_cuda(monkeypatch):
    monkeypatch.setenv(
        "PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8,expandable_segments:True"
    )
    real_torch = sys.modules.get("torch")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False)),
    )
    try:
        assert prepare_allocator_for_compile("reduce-overhead") is True
    finally:
        if real_torch is not None:
            monkeypatch.setitem(sys.modules, "torch", real_torch)
    assert (
        "expandable_segments"
        not in __import__("os").environ["PYTORCH_CUDA_ALLOC_CONF"]
    )


def test_fp8_filter_only_selects_large_aligned_aggregator_linears():
    eligible = torch.nn.Linear(1024, 4096)
    small = torch.nn.Linear(128, 128)
    unaligned = torch.nn.Linear(1024, 218)

    assert _eligible_fp8_linear(eligible, "backbone.aggregator.blocks.0.mlp.fc1")
    assert not _eligible_fp8_linear(small, "backbone.aggregator.small")
    assert not _eligible_fp8_linear(unaligned, "backbone.aggregator.output")
    assert not _eligible_fp8_linear(eligible, "hand_head.layers.0")


def test_compile_window_batch_keeps_local_batch_one():
    engine = StudentEngine.__new__(StudentEngine)
    engine.model = object()
    engine.models = [object()] * 4

    engine.compile_mode = None
    assert engine._effective_window_batch_size(8) == 8

    engine.compile_mode = "reduce-overhead"
    assert engine._effective_window_batch_size(1) == 1
    assert engine._effective_window_batch_size(4) == 4
    assert engine._effective_window_batch_size(8) == 4

    engine.models = [object()]
    assert engine._effective_window_batch_size(4) == 1
