# Full-sequence camera trajectory benchmark

This package evaluates one global camera trajectory per complete video. It is
separate from `data/benchmark/hand_pose`, whose adapters evaluate hand pose or
fixed-length hand-coverage clips.

## Data contract

Set `CAMERA_TRAJECTORY_ROOT` or place the data below the benchmark data root:

```text
data/benchmark/camera_trajectory/
  hot3d_val/<sequence>/
    images/000000.jpg
    gt.npz                 # c2w[T,4,4], K[3,3], hw[2]
    meta.json
  arctic_val/<sequence>/
    images/000000.jpg
    gt.npz                 # c2w[T,4,4], K[3,3], hw[2]
    meta.json
```

The publisher does not distribute these datasets. Users are responsible for
obtaining them, complying with their licenses, and preparing rectified,
frame-aligned images and metric camera poses.

Data roots resolve in this order:

1. `CAMERA_TRAJECTORY_ROOT`
2. `<benchmark --data-root>/camera_trajectory`

## Protocol

- Datasets: `camera_hot3d`, `camera_arctic`
- Head: `camera_trajectory`
- Inference: training-length windows with adjacent-window SE(3) chaining
- Alignment: one Umeyama Sim(3) transform over the complete sequence
- Metrics: ATE, RPE translation/rotation, ATE-S, fitted/path-length scale,
  relative ATE, and forward FPS
- Aggregation: equal weight per sequence; corpus FPS is total frames divided by
  total forward seconds

Run the integrated benchmark with an environment that contains this project's
inference dependencies:

```bash
python eval/model_effect/benchmark/run.py \
  --ckpt output/model_train/<run>/step_<n> \
  --config output/model_train/<run>/logs/record/config.yaml \
  --heads camera_trajectory \
  --datasets camera_hot3d,camera_arctic \
  --windowed
```

## External baselines

`rerun_external.py` can coordinate DROID-SLAM, MegaSaM, HaWoR, InfiniteVGGT,
LingBot-Map, EgoPipeline, and project checkpoints. Their source, weights,
compiled extensions, and mutually incompatible environments are not bundled.
Configure these paths before running:

```bash
export CAMERA_BASELINE_CODE_ROOT=/path/to/camera_baselines
export CAMERA_BASELINE_ENVS_ROOT=/path/to/conda_envs
export CAMERA_BASELINE_RUN_ROOT=/path/to/results
export CAMERA_BASELINE_WORK_ROOT=/path/to/work
export CAMERA_BASELINE_TRAIN_RUN=/path/to/training/run
export MINT_REPO_ROOT="$PWD"
export MINT_PYTHON="$(command -v python)"

python eval/model_effect/benchmark/camera_trajectory/rerun_external.py launch
python eval/model_effect/benchmark/camera_trajectory/rerun_external.py status
```

The launcher records interpreter paths, sequence assignments, output roots,
and work roots in its manifest. It runs each method in its own environment and
uses frame-balanced, disjoint GPU shards.

After inference, audit and evaluate NPZ files using the same metrics as project
checkpoints:

```bash
PYTHONPATH=eval/model_effect python \
  eval/model_effect/benchmark/camera_trajectory/import_external_results.py \
  --data-root /path/to/camera_trajectory \
  --pred-root /path/to/results/pred \
  --output /path/to/results/metrics/baselines.json
```

The importer requires exact frame coverage, finite `[T,4,4]` camera poses, and
a successful whole-sequence metric. HaWoR outputs that use independently solved
pieces are marked unsupported for global-trajectory comparison rather than
being concatenated across unrelated origins or scales.
