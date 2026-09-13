#!/usr/bin/env python3
"""Freeze exact nested SOTA benchmark subsets from the current protocol pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path


_BENCHMARK = Path(__file__).resolve().parents[1]
_MODEL_EFFECT = _BENCHMARK.parent
_REPO = _MODEL_EFFECT.parents[1]
if str(_MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(_MODEL_EFFECT))

from benchmark.datasets.arctic import ARCTICHandCoverageAdapter  # noqa: E402
from benchmark.datasets.base import (  # noqa: E402
    FIXED_SOTA_SPLIT_VERSION,
    deterministic_diverse_sample,
)
from benchmark.datasets.hot3d import HOT3DAdapter, HOT3DHandCoverageAdapter  # noqa: E402


MINIMUM_COUNTS = {
    "hot3d": 6,
    "arctic_hand_coverage": 24,
    "hot3d_hand_coverage": 32,
}


def _tiers(items, *, minimum: int, group_key, item_id) -> dict[str, list[str]]:
    counts = {
        "minimum": minimum,
        "quarter": math.ceil(len(items) * 0.25),
        "half": math.ceil(len(items) * 0.5),
        "full": len(items),
    }
    tiers = {
        name: [item_id(item) for item in deterministic_diverse_sample(
            items, count, group_key=group_key, seed=42,
        )]
        for name, count in counts.items()
    }
    sets = {name: set(values) for name, values in tiers.items()}
    assert sets["minimum"] <= sets["quarter"] <= sets["half"] <= sets["full"]
    assert all(len(tiers[name]) == count for name, count in counts.items())
    return tiers


def build_manifest(data_root: Path) -> dict:
    hot3d = HOT3DAdapter(str(data_root))._sequence_names()
    arctic = ARCTICHandCoverageAdapter(str(data_root))._selected_segments()
    hot3d_coverage = HOT3DHandCoverageAdapter(str(data_root))._selected_segments()
    datasets = {
        "hot3d": {
            "unit": "source_video",
            "tiers": _tiers(
                hot3d, minimum=MINIMUM_COUNTS["hot3d"],
                group_key=lambda name: name.split("_", 1)[0], item_id=str,
            ),
        },
        "arctic_hand_coverage": {
            "unit": "81_frame_clip",
            "tiers": _tiers(
                arctic, minimum=MINIMUM_COUNTS["arctic_hand_coverage"],
                group_key=lambda entry: entry[0],
                item_id=lambda entry: f"{entry[0]}#seg{entry[1]:04d}",
            ),
        },
        "hot3d_hand_coverage": {
            "unit": "81_frame_clip",
            "tiers": _tiers(
                hot3d_coverage, minimum=MINIMUM_COUNTS["hot3d_hand_coverage"],
                group_key=lambda entry: entry[0],
                item_id=lambda entry: f"{entry[0]}#seg{entry[1]:04d}",
            ),
        },
    }
    encoded = json.dumps(datasets, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return {
        "version": FIXED_SOTA_SPLIT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "seed_used_to_freeze": 42,
        "content_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
        "datasets": datasets,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=_REPO / "data" / "benchmark")
    parser.add_argument(
        "--out", type=Path,
        default=_BENCHMARK / "splits" / f"{FIXED_SOTA_SPLIT_VERSION}.json",
    )
    args = parser.parse_args()
    manifest = build_manifest(args.data_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out} · sha256={manifest['content_sha256']}")


if __name__ == "__main__":
    main()
