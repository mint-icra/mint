import sys
from pathlib import Path


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.datasets.arctic import ARCTICHandCoverageAdapter  # noqa: E402
from benchmark.datasets.base import (  # noqa: E402
    FIXED_SOTA_SPLIT_VERSION,
    deterministic_diverse_sample,
    fixed_split_manifest,
    normalize_dataset_selection,
)
from benchmark.datasets.hot3d import HOT3DAdapter, HOT3DHandCoverageAdapter  # noqa: E402


def test_diverse_sample_is_deterministic_and_round_robins_groups():
    items = [(group, index) for group in "abcd" for index in range(5)]

    first = deterministic_diverse_sample(items, 4, lambda item: item[0], seed=42)
    repeated = deterministic_diverse_sample(items, 4, lambda item: item[0], seed=42)

    assert first == repeated
    assert len({item[0] for item in first}) == 4


def test_fixed_sota_tiers_are_exact_and_nested():
    manifest = fixed_split_manifest()
    expected = {
        "hot3d": (6, 23, 45, 90),
        "arctic_hand_coverage": (24, 76, 151, 302),
        "hot3d_hand_coverage": (32, 110, 219, 437),
    }
    for dataset, counts in expected.items():
        tiers = manifest["datasets"][dataset]["tiers"]
        values = [set(tiers[name]) for name in ("minimum", "quarter", "half", "full")]
        assert tuple(map(len, values)) == counts
        assert values[0] < values[1] < values[2] < values[3]

    normalized = normalize_dataset_selection({
        "hot3d": {
            "sampling": "diverse", "sample_count": 6,
            "fixed_tier": "minimum", "split_version": FIXED_SOTA_SPLIT_VERSION,
        },
    })
    assert normalized["hot3d"]["split_hash"] == manifest["content_sha256"]


def test_hot3d_world_selection_happens_before_sharding(tmp_path, monkeypatch):
    adapter = HOT3DAdapter(str(tmp_path))
    names = [f"P{participant:04d}_video{video}" for participant in range(1, 5) for video in range(3)]
    monkeypatch.setattr(adapter, "_sequence_names", lambda max_seqs=None: names[:max_seqs])
    monkeypatch.setattr(adapter, "_frame_count", lambda _name: 2000)
    adapter.set_benchmark_selection({
        "sampling": "diverse", "sample_count": 6, "max_frames": 600, "seed": 42,
    })

    shards = [adapter._names_for_shard(index, 2, max_frames=600) for index in range(2)]
    selected = adapter._benchmark_names()

    assert len(selected) == 6
    assert set(shards[0]).isdisjoint(shards[1])
    assert set(shards[0]) | set(shards[1]) == set(selected)
    assert adapter.benchmark_max_frames(None) == 600


def test_coverage_adapters_sample_complete_clips_across_sources(tmp_path, monkeypatch):
    entries = [(f"source-{source}", clip) for source in range(8) for clip in range(10)]
    for adapter in (
        ARCTICHandCoverageAdapter(str(tmp_path)),
        HOT3DHandCoverageAdapter(str(tmp_path)),
    ):
        monkeypatch.setattr(adapter, "_selected_segments", lambda entries=entries: entries)
        adapter.set_benchmark_selection({"sampling": "diverse", "sample_count": 16, "seed": 42})
        selected = adapter._benchmark_segments()

        assert len(selected) == 16
        assert len({source for source, _clip in selected}) == 8
