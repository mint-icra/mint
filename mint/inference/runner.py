"""Command-line prediction runner."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .acceleration import COMPILE_MODES, FP8_MODES
from .engine import StudentEngine
from .video import read_video


def inference_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run MINT prediction-only inference on one video.")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="artifacts/inference", help="Output directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file or Accelerate checkpoint directory")
    parser.add_argument(
        "--config", default="configs/training/mint_step2.yaml"
    )
    parser.add_argument("--devices", default="auto", help="auto, cpu, a CUDA device, or a comma-separated list")
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument(
        "--window-batch-size", type=int, default=None,
        help="Independent windows per forward; compile keeps local batch=1 per GPU",
    )
    parser.add_argument("--camera-mode", choices=("chunked", "max_chunked", "streaming", "full"), default="chunked")
    parser.add_argument("--hand-mode", choices=("hard", "blend", "smooth"), default="smooth")
    parser.add_argument(
        "--compile-mode", choices=("off", *COMPILE_MODES), default="off",
        help="torch.compile mode; use auto for reduce-overhead on CUDA",
    )
    parser.add_argument(
        "--fp8-mode", choices=("off", *FP8_MODES), default="off",
        help="FP8 mode; auto enables dynamic FP8 on supported CUDA devices",
    )
    parser.add_argument(
        "--warmup-passes", type=int, default=0,
        help="Optional compile/CUDA Graph warmup passes before timed inference",
    )
    parser.add_argument("--render-mode", choices=("mesh", "skeleton", "mesh_skel"), default="mesh_skel")
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args(argv)

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    video = read_video(source, max_frames=args.max_frames, target_fps=args.target_fps)
    print(f"Decoded {len(video.frames_rgb)} frames at {video.fps:.2f} FPS from {source.name}")

    started = time.perf_counter()
    load_started = time.perf_counter()
    engine = StudentEngine(
        args.config,
        ckpt=args.checkpoint,
        devices=args.devices,
        window=args.window,
        compile_mode=args.compile_mode,
        fp8_mode=args.fp8_mode,
    )
    load_seconds = time.perf_counter() - load_started
    warmup = None
    if args.warmup_passes > 0 and engine.compile_mode is not None:
        warmup = engine.warmup_acceleration(
            window_batch_size=args.window_batch_size or engine.parallel_device_count,
            passes=args.warmup_passes,
        )
    inference_started = time.perf_counter()
    prediction = engine.predict(
        video.frames_rgb,
        cam_mode=args.camera_mode,
        window_batch_size=args.window_batch_size,
        hand_mode=args.hand_mode,
    )
    inference_seconds = time.perf_counter() - inference_started
    elapsed = time.perf_counter() - started

    arrays = {key: np.asarray(value) for key, value in prediction.items() if not key.startswith("_")}
    np.savez_compressed(output / "prediction.npz", **arrays)
    if not args.no_render:
        from mint.visualization.render import render_prediction

        render_prediction(
            video.frames_rgb,
            prediction,
            output / "prediction.mp4",
            video.fps,
            mode=args.render_mode,
        )
    summary = {
        "schema_version": 1,
        "input_name": source.name,
        "frames": len(video.frames_rgb),
        "fps": video.fps,
        "duration_seconds": round(elapsed, 4),
        "model_load_seconds": round(load_seconds, 4),
        "inference_seconds": round(inference_seconds, 4),
        "camera_mode": args.camera_mode,
        "hand_mode": args.hand_mode,
        "window_batch_size": args.window_batch_size,
        "acceleration": engine.acceleration_metadata,
        "warmup": warmup,
        "timings": prediction.get("_timings"),
        "rendered": not args.no_render,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Artifacts written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(inference_main())
