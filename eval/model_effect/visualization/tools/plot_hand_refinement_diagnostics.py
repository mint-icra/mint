#!/usr/bin/env python3
"""Plot final/initial wrist translation, refinement delta, and presence logits."""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve()
MODEL_EFFECT_ROOT = HERE.parents[2]
REPO_ROOT = HERE.parents[4]
if str(MODEL_EFFECT_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT_ROOT))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
PER_HAND = 109
SIDES = ("left", "right")


def _config_from_checkpoint(checkpoint: Path) -> Path:
    run_root = checkpoint if checkpoint.is_dir() else checkpoint.parent
    for _ in range(5):
        candidate = run_root / "logs" / "record" / "config.yaml"
        if candidate.is_file():
            return candidate
        run_root = run_root.parent
    raise FileNotFoundError(
        "Cannot find logs/record/config.yaml above the checkpoint; pass --config."
    )


def _load_frames(input_path: Path, episode: int, max_frames: int | None, fps: float):
    from visualization.reproj_core import lerobot_io

    if input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS:
        frames = lerobot_io.read_video_frames(
            str(input_path), fps=fps, max_frames=max_frames
        )
        return frames, input_path.stem

    dataset = lerobot_io.find_dataset(input_path)
    if dataset is None:
        raise FileNotFoundError(
            f"{input_path} is neither a video nor a LeRobot v3 dataset."
        )
    episodes = lerobot_io.discover_episodes(dataset)
    if not 0 <= episode < len(episodes):
        raise IndexError(f"episode {episode} is outside [0, {len(episodes)})")
    raw = lerobot_io.load_episode_raw(episodes[episode], max_frames=max_frames)
    return raw["frames"], f"episode_{int(raw['episode_index']):06d}"


def _extract_series(prediction: dict, side_index: int) -> dict[str, np.ndarray]:
    final = np.asarray(prediction["hand"], dtype=np.float32)
    initial = np.asarray(prediction["_hand_refine_initial"], dtype=np.float32)
    expected = (final.shape[0], 2 * PER_HAND)
    if final.shape != expected or initial.shape != expected:
        raise ValueError(
            f"Expected final/initial hand output {expected}, got "
            f"{final.shape}/{initial.shape}."
        )
    final_translation = final.reshape(-1, 2, PER_HAND)[:, side_index, :3]
    initial_translation = initial.reshape(-1, 2, PER_HAND)[:, side_index, :3]
    delta = final_translation - initial_translation
    logits = prediction.get("hand_presence_logits")
    if logits is None:
        presence = np.full(final.shape[0], np.nan, dtype=np.float32)
    else:
        presence = np.asarray(logits, dtype=np.float32)[:, side_index]
    def acceleration_norm(value: np.ndarray) -> np.ndarray:
        result = np.full(value.shape[0], np.nan, dtype=np.float32)
        if value.shape[0] >= 3:
            acceleration = value[2:] - 2.0 * value[1:-1] + value[:-2]
            result[1:-1] = np.linalg.norm(acceleration, axis=-1)
        return result

    return {
        "final": final_translation,
        "initial": initial_translation,
        "delta": delta,
        "delta_norm": np.linalg.norm(delta, axis=-1),
        "final_acc_norm": acceleration_norm(final_translation),
        "initial_acc_norm": acceleration_norm(initial_translation),
        "delta_acc_norm": acceleration_norm(delta),
        "presence_logit": presence,
    }


def _plot_side(series: dict[str, np.ndarray], side: str, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames = np.arange(len(series["final"]))
    colors = ("#d1495b", "#00798c", "#edae49")
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    for axis, key, title in (
        (axes[0], "final", "Final wrist translation"),
        (axes[1], "initial", "Initial wrist translation"),
    ):
        for component, color in enumerate(colors):
            axis.plot(
                frames, series[key][:, component], color=color, linewidth=1.1,
                label=("x", "y", "z")[component],
            )
        axis.set_ylabel("meters")
        axis.set_title(title)
        axis.legend(loc="upper right", ncol=3)
        axis.grid(alpha=0.2)

    for component, color in enumerate(colors):
        axes[2].plot(
            frames, series["delta"][:, component], color=color, linewidth=0.9,
            alpha=0.75, label=f"d{('x', 'y', 'z')[component]}",
        )
    axes[2].plot(
        frames, series["delta_norm"], color="#202c39", linewidth=1.6,
        label="L2 norm",
    )
    axes[2].set_ylabel("meters")
    axes[2].set_title("Refinement delta: final - initial")
    axes[2].legend(loc="upper right", ncol=4)
    axes[2].grid(alpha=0.2)

    axes[3].plot(frames, series["presence_logit"], color="#306b34", linewidth=1.2)
    axes[3].axhline(0.0, color="#8b2635", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("logit")
    axes[3].set_xlabel("frame")
    axes[3].set_title("Hand presence logit (decision threshold = 0)")
    axes[3].grid(alpha=0.2)

    peak = int(np.nanargmax(series["final_acc_norm"]))
    for axis in axes:
        axis.axvline(peak, color="#5c415d", linestyle=":", linewidth=1.0)
    fig.suptitle(
        f"{side} hand refinement diagnostics | max final acceleration frame={peak}, "
        f"final/initial={series['final_acc_norm'][peak]:.4f}/"
        f"{series['initial_acc_norm'][peak]:.4f} m/frame^2",
        fontsize=14,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _write_csv(series_by_side: dict[str, dict[str, np.ndarray]], output: Path) -> None:
    fields = ["frame"]
    for side in SIDES:
        fields.extend(
            f"{side}_{name}"
            for name in (
                "final_x_m", "final_y_m", "final_z_m",
                "initial_x_m", "initial_y_m", "initial_z_m",
                "delta_x_m", "delta_y_m", "delta_z_m", "delta_norm_m",
                "final_acc_norm_m_per_frame2", "initial_acc_norm_m_per_frame2",
                "delta_acc_norm_m_per_frame2",
                "presence_logit",
            )
        )
    frame_count = len(series_by_side["left"]["final"])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame in range(frame_count):
            row: dict[str, float | int] = {"frame": frame}
            for side in SIDES:
                values = series_by_side[side]
                for prefix in ("final", "initial", "delta"):
                    for component, axis in enumerate("xyz"):
                        row[f"{side}_{prefix}_{axis}_m"] = float(
                            values[prefix][frame, component]
                        )
                row[f"{side}_delta_norm_m"] = float(values["delta_norm"][frame])
                for name in ("final_acc_norm", "initial_acc_norm", "delta_acc_norm"):
                    row[f"{side}_{name}_m_per_frame2"] = float(values[name][frame])
                row[f"{side}_presence_logit"] = float(
                    values["presence_logit"][frame]
                )
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose whether wrist jumps originate before or during refinement."
    )
    parser.add_argument("--input", required=True, help="Video or LeRobot v3 directory")
    parser.add_argument("--ckpt", required=True, help="step_* directory or model weights")
    parser.add_argument("--config", default=None, help="Defaults to checkpoint config snapshot")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=96)
    parser.add_argument("--fps", type=float, default=30.0, help="Video sampling rate")
    parser.add_argument("--device", default=None)
    parser.add_argument("--window", type=int, default=None)
    parser.add_argument(
        "--cam-mode", choices=("chunked", "max_chunked", "streaming", "full"),
        default="max_chunked",
    )
    parser.add_argument(
        "--hand-mode", choices=("hard", "blend", "smooth"), default="hard",
        help="Use hard first to expose raw spikes; compare with blend/smooth separately.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.ckpt).resolve()
    config = Path(args.config).resolve() if args.config else _config_from_checkpoint(checkpoint)
    frames, input_tag = _load_frames(
        Path(args.input).resolve(), args.episode, args.max_frames, args.fps
    )

    from inference.engine import StudentEngine

    engine = StudentEngine(
        str(config), ckpt=str(checkpoint), device=args.device, window=args.window
    )
    prediction = engine.predict(
        frames, cam_mode=args.cam_mode, hand_mode=args.hand_mode
    )
    required = {"hand", "_hand_refine_initial"}
    missing = required - prediction.keys()
    if missing:
        raise RuntimeError(f"Checkpoint did not produce required outputs: {sorted(missing)}")

    output_dir = (
        Path(args.out).resolve()
        if args.out
        else REPO_ROOT / "output" / "eval" / "hand_refinement_diagnostics"
        / f"{checkpoint.parent.name}_{checkpoint.name}_{input_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    series_by_side = {
        side: _extract_series(prediction, side_index)
        for side_index, side in enumerate(SIDES)
    }
    for side, series in series_by_side.items():
        _plot_side(series, side, output_dir / f"{side}_diagnostics.png")
    _write_csv(series_by_side, output_dir / "diagnostics.csv")
    np.savez_compressed(output_dir / "prediction_outputs.npz", **{
        key: value for key, value in prediction.items() if isinstance(value, np.ndarray)
    })

    for side, series in series_by_side.items():
        peak = int(np.nanargmax(series["final_acc_norm"]))
        print(
            f"[{side}] max final acceleration: frame={peak}, "
            f"final={series['final_acc_norm'][peak]:.6f}, "
            f"initial={series['initial_acc_norm'][peak]:.6f}, "
            f"refinement_delta={series['delta_acc_norm'][peak]:.6f} m/frame^2, "
            f"delta_norm={series['delta_norm'][peak]:.6f} m, "
            f"presence_logit={series['presence_logit'][peak]:.4f}"
        )
    print(f"Wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
