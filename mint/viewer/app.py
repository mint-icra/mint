"""Secure, prediction-only web viewer for approved egocentric samples."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from flask import Flask, abort, jsonify, request, send_file

from mint.inference.acceleration import COMPILE_MODES, FP8_MODES
from mint.inference.base import InferenceCancelled
from mint.inference.engine import StudentEngine
from mint.inference.video import read_video
from mint.visualization.render import render_prediction, trajectory_payload


PROJECT_DIR = Path(__file__).resolve().parents[2]
VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
DEFAULT_SAMPLES_DIR = PROJECT_DIR / "data" / "samples"
DEFAULT_ARTIFACTS_DIR = PROJECT_DIR / "artifacts" / "viewer"
DEFAULT_CHECKPOINT = PROJECT_DIR / "checkpoints" / "model.safetensors"
DEFAULT_CONFIG = (
    PROJECT_DIR / "configs" / "training" / "mint_step2.yaml"
)


@dataclass(frozen=True)
class Sample:
    id: str
    path: Path = field(repr=False)
    title: str
    collection: str
    size_mb: float

    def public(self) -> dict:
        return {"id": self.id, "title": self.title, "collection": self.collection, "size_mb": self.size_mb}


@dataclass
class Job:
    id: str
    sample_id: str
    status: str = "queued"
    stage: str = "Waiting"
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    frames: int | None = None
    fps: float | None = None
    artifacts: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    def public(self) -> dict:
        return {
            "id": self.id,
            "sample_id": self.sample_id,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "error": self.error,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "frames": self.frames,
            "fps": self.fps,
            "artifacts": dict(self.artifacts),
        }


def _display_name(path: Path) -> str:
    words = path.stem.replace("_", "-").split("-")
    return " ".join(word.upper() if word.lower() in {"ego4d", "egodex"} else word.title() for word in words)


class ViewerService:
    def __init__(
        self,
        samples_dir: Path,
        artifacts_dir: Path,
        checkpoint: str,
        config: str,
        devices: str,
        max_frames: int,
        target_fps: float,
        compile_mode: str = "auto",
        fp8_mode: str = "auto",
        window_batch_size: int | None = None,
        warmup_passes: int = 2,
    ) -> None:
        self.samples_dir = samples_dir.resolve()
        self.artifacts_dir = artifacts_dir.resolve()
        self.checkpoint = checkpoint
        self.config = config
        self.devices = devices
        self.max_frames = max_frames
        self.target_fps = target_fps
        self.compile_mode = compile_mode
        self.fp8_mode = fp8_mode
        self.window_batch_size = window_batch_size
        self.warmup_passes = max(0, int(warmup_passes))
        self.samples = self._scan_samples()
        self.jobs: dict[str, Job] = {}
        self._jobs_lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._engine: StudentEngine | None = None

    def _scan_samples(self) -> dict[str, Sample]:
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        samples = {}
        for path in sorted(self.samples_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            relative = path.relative_to(self.samples_dir)
            sample_id = hashlib.sha256(relative.as_posix().encode()).hexdigest()[:16]
            collection = relative.parts[0] if len(relative.parts) > 1 else "Samples"
            samples[sample_id] = Sample(
                id=sample_id,
                path=path,
                title=_display_name(path),
                collection=collection.replace("_", " ").replace("-", " ").title(),
                size_mb=round(path.stat().st_size / (1024 * 1024), 2),
            )
        return samples

    def get_sample(self, sample_id: str) -> Sample:
        sample = self.samples.get(sample_id)
        if sample is None:
            raise KeyError(sample_id)
        resolved = sample.path.resolve()
        if self.samples_dir not in resolved.parents:
            raise PermissionError("Sample escaped the configured directory.")
        return sample

    def _get_engine(self, job: Job) -> StudentEngine:
        with self._engine_lock:
            if self._engine is None:
                job.stage = "Loading checkpoint"
                job.progress = 0.12
                engine = StudentEngine(
                    self.config,
                    ckpt=self.checkpoint,
                    devices=self.devices,
                    compile_mode=self.compile_mode,
                    fp8_mode=self.fp8_mode,
                )
                if engine.compile_mode is not None and self.warmup_passes:
                    job.stage = "Optimizing model"
                    job.progress = 0.15
                    engine.warmup_acceleration(
                        window_batch_size=(
                            self.window_batch_size or engine.parallel_device_count
                        ),
                        passes=self.warmup_passes,
                    )
                self._engine = engine
            return self._engine

    def submit(self, sample_id: str) -> Job:
        self.get_sample(sample_id)
        job = Job(id=uuid.uuid4().hex, sample_id=sample_id)
        with self._jobs_lock:
            self.jobs[job.id] = job
        threading.Thread(target=self._run_job, args=(job,), name=f"mint-{job.id[:8]}", daemon=True).start()
        return job

    def _run_job(self, job: Job) -> None:
        sample = self.get_sample(job.sample_id)
        target = self.artifacts_dir / job.id
        target.mkdir(parents=True, exist_ok=False)
        try:
            job.status = "running"
            job.stage = "Decoding sample"
            job.progress = 0.04
            video = read_video(sample.path, max_frames=self.max_frames, target_fps=self.target_fps)
            job.frames = len(video.frames_rgb)
            job.fps = round(video.fps, 4)
            if job.cancel.is_set():
                raise InferenceCancelled("Cancelled before inference.")

            engine = self._get_engine(job)
            job.stage = "Running inference"

            def inference_progress(done: int, total: int) -> None:
                job.progress = 0.18 + 0.54 * done / max(total, 1)

            prediction = engine.predict(
                video.frames_rgb,
                on_step=inference_progress,
                cancel_check=job.cancel.is_set,
                cam_mode="chunked",
                window_batch_size=self.window_batch_size,
                hand_mode="smooth",
            )
            arrays = {key: np.asarray(value) for key, value in prediction.items() if not key.startswith("_")}
            np.savez_compressed(target / "prediction.npz", **arrays)

            job.stage = "Rendering prediction"

            def render_progress(done: int, total: int) -> None:
                if job.cancel.is_set():
                    raise InferenceCancelled("Cancelled during rendering.")
                job.progress = 0.74 + 0.24 * done / max(total, 1)

            render_prediction(
                video.frames_rgb,
                prediction,
                target / "prediction.mp4",
                video.fps,
                progress=render_progress,
            )
            trajectory = trajectory_payload(video.frames_rgb, prediction)
            (target / "trajectory.json").write_text(json.dumps(trajectory), encoding="utf-8")
            summary = {
                "schema_version": 1,
                "sample": sample.title,
                "frames": job.frames,
                "fps": job.fps,
                "ground_truth_used": False,
                "acceleration": engine.acceleration_metadata,
                "timings": prediction.get("_timings"),
            }
            (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            job.artifacts = {
                "video": f"/artifacts/{job.id}/prediction.mp4",
                "prediction": f"/artifacts/{job.id}/prediction.npz",
                "trajectory": f"/api/jobs/{job.id}/trajectory",
            }
            job.progress = 1.0
            job.stage = "Ready"
            job.status = "completed"
        except InferenceCancelled:
            job.status = "cancelled"
            job.stage = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            message = str(exc).replace(str(PROJECT_DIR), "<project>").replace(str(self.samples_dir), "<samples>")
            job.error = message[:1000]
            job.status = "failed"
            job.stage = "Failed"
        finally:
            job.finished_at = time.time()

    def get_job(self, job_id: str) -> Job:
        with self._jobs_lock:
            job = self.jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job


def create_app(service: ViewerService) -> Flask:
    static_dir = Path(__file__).with_name("static")
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    app.config.update(MAX_CONTENT_LENGTH=64 * 1024, JSON_SORT_KEYS=False)

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; media-src 'self'; "
            "style-src 'self'; script-src 'self'; connect-src 'self'"
        )
        return response

    @app.get("/")
    def index():
        return send_file(static_dir / "index.html")

    @app.get("/api/status")
    def status():
        return jsonify(
            {
                "name": "MINT",
                "mode": "Prediction only",
                "samples": len(service.samples),
                "model_loaded": service._engine is not None,
                "ground_truth": False,
                "acceleration": (
                    service._engine.acceleration_metadata
                    if service._engine is not None
                    else {
                        "compile_mode_requested": service.compile_mode,
                        "fp8_mode_requested": service.fp8_mode,
                    }
                ),
            }
        )

    @app.get("/api/samples")
    def samples():
        return jsonify([sample.public() for sample in service.samples.values()])

    @app.get("/media/<sample_id>")
    def media(sample_id: str):
        try:
            sample = service.get_sample(sample_id)
        except (KeyError, PermissionError):
            abort(404)
        return send_file(sample.path, conditional=True)

    @app.post("/api/jobs")
    def submit_job():
        payload = request.get_json(silent=True) or {}
        sample_id = str(payload.get("sample_id", ""))
        try:
            job = service.submit(sample_id)
        except KeyError:
            return jsonify({"error": "Unknown sample."}), 404
        return jsonify(job.public()), 202

    @app.get("/api/jobs/<job_id>")
    def get_job(job_id: str):
        try:
            return jsonify(service.get_job(job_id).public())
        except KeyError:
            abort(404)

    @app.delete("/api/jobs/<job_id>")
    def cancel_job(job_id: str):
        try:
            job = service.get_job(job_id)
        except KeyError:
            abort(404)
        job.cancel.set()
        return jsonify({"ok": True})

    @app.get("/api/jobs/<job_id>/trajectory")
    def trajectory(job_id: str):
        try:
            job = service.get_job(job_id)
        except KeyError:
            abort(404)
        path = service.artifacts_dir / job.id / "trajectory.json"
        if job.status != "completed" or not path.is_file():
            abort(404)
        return send_file(path, mimetype="application/json")

    @app.get("/artifacts/<job_id>/<name>")
    def artifact(job_id: str, name: str):
        if name not in {"prediction.mp4", "prediction.npz", "summary.json"}:
            abort(404)
        try:
            job = service.get_job(job_id)
        except KeyError:
            abort(404)
        path = service.artifacts_dir / job.id / name
        if job.status != "completed" or not path.is_file():
            abort(404)
        return send_file(path, conditional=True, as_attachment=name != "prediction.mp4")

    return app


def viewer_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Start the local MINT prediction viewer.")
    parser.add_argument("--samples", default=str(DEFAULT_SAMPLES_DIR), help="Approved sample directory")
    parser.add_argument("--artifacts", default=str(DEFAULT_ARTIFACTS_DIR), help="Generated artifact directory")
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="MINT checkpoint file or directory (default: checkpoints/model.safetensors)",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--devices", default="auto")
    parser.add_argument(
        "--compile-mode", choices=("off", *COMPILE_MODES), default="auto",
        help="Acceleration defaults to reduce-overhead on CUDA; use off to disable",
    )
    parser.add_argument(
        "--fp8-mode", choices=("off", *FP8_MODES), default="auto",
        help="Auto enables dynamic FP8 on CUDA capability >= 8.9",
    )
    parser.add_argument(
        "--window-batch-size", type=int, default=None,
        help="Defaults to one independent window per loaded GPU",
    )
    parser.add_argument(
        "--warmup-passes", type=int, default=2,
        help="Compile/CUDA Graph warmup passes when acceleration is enabled",
    )
    parser.add_argument("--max-frames", type=int, default=160)
    parser.add_argument("--target-fps", type=float, default=15.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args(argv)

    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.exists():
        parser.error(
            f"checkpoint not found: {checkpoint}. Run 'bash scripts/download_assets.sh' "
            "or pass --checkpoint PATH."
        )

    service = ViewerService(
        samples_dir=Path(args.samples).expanduser(),
        artifacts_dir=Path(args.artifacts).expanduser(),
        checkpoint=str(checkpoint),
        config=args.config,
        devices=args.devices,
        max_frames=args.max_frames,
        target_fps=args.target_fps,
        compile_mode=args.compile_mode,
        fp8_mode=args.fp8_mode,
        window_batch_size=args.window_batch_size,
        warmup_passes=args.warmup_passes,
    )
    app = create_app(service)
    url = f"http://127.0.0.1:{args.port}"
    print(f"MINT viewer: {url}")
    print(f"Approved samples: {len(service.samples)}")
    if args.host != "127.0.0.1":
        print("Warning: the viewer is exposed beyond localhost. Add authentication at your reverse proxy.")
    if args.open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(viewer_main())
