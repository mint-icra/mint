import sys
from pathlib import Path

import numpy as np


MODEL_EFFECT = Path(__file__).resolve().parents[2]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.core.engine import run_benchmark  # noqa: E402
from benchmark.core.registry import DATASETS, HEADS  # noqa: E402
from benchmark.core.schema import GTSequence, Prediction, VideoFrameRef  # noqa: E402
from benchmark.datasets.base import DatasetAdapter  # noqa: E402
from benchmark.predictor import _load_preprocessed_rgb  # noqa: E402


def test_video_frames_decode_directly_without_jpeg_cache(tmp_path):
    import cv2

    video = tmp_path / "frames.mp4"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 64),
    )
    assert writer.isOpened()
    for bgr in ((0, 0, 220), (0, 220, 0), (220, 0, 0)):
        writer.write(np.full((64, 64, 3), bgr, np.uint8))
    writer.release()

    values, hw = _load_preprocessed_rgb(
        [VideoFrameRef(str(video), 2), VideoFrameRef(str(video), 0)],
        (32, 32), workers=4,
    )

    assert hw == (64, 64)
    assert tuple(values.shape) == (2, 3, 32, 32)
    assert float(values[0, 2].mean()) > 0.75
    assert float(values[1, 0].mean()) > 0.75
    assert list(tmp_path.glob("*.jpg")) == []


def test_engine_reports_total_before_streaming_expensive_sequences(tmp_path):
    dataset_name = "_test_streaming_dataset"
    head_name = "_test_streaming_head"
    trace = []

    @DATASETS.register(dataset_name)
    class StreamingDataset(DatasetAdapter):
        capability = set()

        def set_benchmark_selection(self, options):
            super().set_benchmark_selection(options)
            trace.append(("selection", dict(options or {})))

        def iter_sequences_for_shard(self, *args, **kwargs):
            assert kwargs["max_frames"] == 7
            for index in range(2):
                trace.append(f"build-{index}")
                yield GTSequence(
                    seq_id=f"seq-{index}", image_paths=[f"frame-{index}.jpg"],
                    hw=(8, 8), capability=set(), meta={"index": index},
                )

        def count_sequences_for_shard(self, *args, **kwargs):
            assert kwargs["max_frames"] == 7
            trace.append("count")
            return 2

    @HEADS.register(head_name)
    class StreamingHead:
        name = head_name
        required_gt = set()

        def evaluate(self, pred, gt):
            return "evaluated", {"score": float(gt.meta["index"])}, ""

    class Predictor:
        def set_benchmark_dataset(self, dataset):
            pass

        def predict(self, image_paths, hw=None, on_step=None):
            if on_step:
                on_step(1, 1)
            frames = len(image_paths)
            return Prediction(
                pose_enc=np.zeros((frames, 9), np.float32),
                extrinsic_c2w=np.tile(np.eye(4), (frames, 1, 1)),
                intrinsic=np.tile(np.eye(3), (frames, 1, 1)),
                hw=hw, capability=set(),
            )

    def progress(event):
        trace.append(event["kind"])

    run_benchmark(
        Predictor(), heads=head_name, datasets=dataset_name,
        data_root=str(tmp_path), out_dir=str(tmp_path / "report"),
        shard_index=0, shard_count=2, on_progress=progress,
        dataset_selection={dataset_name: {"sampling": "diverse", "sample_count": 2,
                                          "max_frames": 7, "seed": 42}},
    )

    assert ("selection", {"sampling": "diverse", "sample_count": 2,
                           "max_frames": 7, "seed": 42}) in trace
    assert trace.index("count") < trace.index("seqs")
    assert trace.index("seqs") < trace.index("build-0")
    assert trace.index("prepare") < trace.index("build-0")
