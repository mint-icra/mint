import sys
from pathlib import Path

import numpy as np


MODEL_EFFECT = Path(__file__).resolve().parents[2]
REPO = Path(__file__).resolve().parents[4]
if str(MODEL_EFFECT) not in sys.path:
    sys.path.insert(0, str(MODEL_EFFECT))

from benchmark.heads.align import umeyama_sim3  # noqa: E402


def _apply(points, scale, rotation, translation):
    return (scale * (rotation @ points.T)).T + translation


def _fixture_trajectories(frames=180):
    phase = np.linspace(0.0, 2.4 * np.pi, frames)
    gt = np.stack([
        0.055 * phase + 0.035 * np.sin(phase),
        -0.045 * phase + 0.025 * np.cos(1.3 * phase),
        0.018 * phase + 0.020 * np.sin(0.7 * phase),
    ], axis=-1)

    angle = np.deg2rad(34.0)
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    expected_scale = 0.62
    translation = np.array([0.31, -0.24, 0.17])
    pred = ((rotation.T @ (gt - translation).T).T / expected_scale)

    # Sim(3) cannot absorb this local drift, so the aligned curves remain distinguishable.
    pred[:, 0] += 0.007 * np.sin(2.7 * phase)
    pred[:, 1] += 0.005 * np.cos(1.9 * phase)
    return gt, pred, expected_scale


def _equal_3d(ax, points):
    low, high = points.min(axis=0), points.max(axis=0)
    center = (low + high) * 0.5
    radius = float(np.max(high - low)) * 0.5 or 1.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def _write_three_panel(gt, pred_metric, pred_sim3, scale, metric_rmse,
                       sim3_rmse, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(19.2, 6.6), facecolor="#f6f2e8")
    fig.suptitle("Sim(3) camera trajectory alignment", fontsize=17,
                 fontweight="bold", y=.985)
    bounds = np.concatenate([gt, pred_metric], axis=0)
    panels = (
        ("1  Ground truth", None, "Reference trajectory in the GT world frame"),
        ("2  Prediction - original scale", pred_metric,
         f"SE(3) only  |  scale fixed at 1  |  RMSE {metric_rmse:.3f} m"),
        ("3  Prediction after Sim(3)", pred_sim3,
         f"scale x {scale:.3f}  |  RMSE {sim3_rmse:.3f} m"),
    )
    for index, (title, prediction, subtitle) in enumerate(panels, 1):
        ax = fig.add_subplot(1, 3, index, projection="3d", facecolor="#fffdf8")
        ax.set_proj_type("ortho")
        ax.plot(*gt.T, color="#147d92", lw=2.5, label="GT", zorder=3)
        if prediction is not None:
            ax.plot(*prediction.T, color="#e05a33", lw=1.8,
                    label="Prediction", zorder=4)
        ax.scatter(*gt[0], color="#183642", s=42, marker="o", label="start")
        ax.scatter(*gt[-1], color="#183642", s=48, marker="X", label="end")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
        ax.text2D(.5, .965, subtitle, transform=ax.transAxes, ha="center",
                  va="top", fontsize=10, color="#554f48")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.view_init(elev=25, azim=-58)
        ax.grid(True, alpha=.22)
        ax.legend(loc="lower left", fontsize=9, framealpha=.92)
        _equal_3d(ax, bounds)

    fig.text(
        .5, .025,
        "SE(3) removes world origin/orientation. Sim(3) also fits one global scale.",
        ha="center", fontsize=11, color="#554f48",
    )
    fig.subplots_adjust(left=.015, right=.985, bottom=.08, top=.83, wspace=.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def test_sim3_reduces_trajectory_error_and_writes_comparison():
    gt, pred, expected_scale = _fixture_trajectories()

    scale, rotation, translation = umeyama_sim3(pred, gt, with_scale=True)
    pred_sim3 = _apply(pred, scale, rotation, translation)
    _, metric_rotation, metric_translation = umeyama_sim3(
        pred, gt, with_scale=False,
    )
    pred_metric = _apply(pred, 1.0, metric_rotation, metric_translation)

    metric_rmse = float(np.sqrt(np.mean(np.sum((pred_metric - gt) ** 2, axis=1))))
    sim3_rmse = float(np.sqrt(np.mean(np.sum((pred_sim3 - gt) ** 2, axis=1))))

    np.testing.assert_allclose(scale, expected_scale, atol=0.01)
    assert sim3_rmse < metric_rmse * 0.2

    output = REPO / "output" / "eval" / "benchmark" / "tests" / \
        "sim3_alignment_three_panel.png"
    _write_three_panel(
        gt, pred_metric, pred_sim3, scale, metric_rmse, sim3_rmse, output,
    )
    assert output.is_file()
    assert output.stat().st_size > 0
