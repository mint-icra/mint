"""Read ordinary images and frame references through one benchmark input API."""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .core.schema import VideoFrameRef


_VIDEO_READERS: OrderedDict[str, object] = OrderedDict()
_MAX_VIDEO_READERS = 4


def _video_reader(path: str):
    key = str(path)
    reader = _VIDEO_READERS.pop(key, None)
    if reader is None:
        from decord import VideoReader

        reader = VideoReader(key, num_threads=4)
    _VIDEO_READERS[key] = reader
    while len(_VIDEO_READERS) > _MAX_VIDEO_READERS:
        _VIDEO_READERS.popitem(last=False)
    return reader


def read_rgb_frames(sources, executor=None) -> list[np.ndarray]:
    """Decode a homogeneous video batch at once; retain threaded JPEG reads."""
    import cv2

    values = list(sources)
    if not values:
        return []
    if all(isinstance(value, VideoFrameRef) for value in values):
        paths = {value.video_path for value in values}
        if len(paths) == 1:
            reader = _video_reader(values[0].video_path)
            indices = [int(value.frame_index) for value in values]
            if min(indices) < 0 or max(indices) >= len(reader):
                raise IndexError(
                    f"视频帧越界: {values[0].video_path}, 范围={min(indices)}..{max(indices)}, "
                    f"总帧数={len(reader)}"
                )
            return list(reader.get_batch(indices).asnumpy())

    def read_one(source):
        if isinstance(source, VideoFrameRef):
            reader = _video_reader(source.video_path)
            index = int(source.frame_index)
            if index < 0 or index >= len(reader):
                raise IndexError(f"视频帧越界: {source}")
            return np.asarray(reader[index].asnumpy())
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"读图失败: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    if executor is None or any(isinstance(value, VideoFrameRef) for value in values):
        return [read_one(value) for value in values]
    return list(executor.map(read_one, values))


def read_bgr_frame(source) -> np.ndarray:
    """Decode one source for visual error exports."""
    import cv2

    rgb = read_rgb_frames([source])[0]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
