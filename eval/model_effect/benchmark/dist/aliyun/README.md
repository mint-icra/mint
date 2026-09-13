# Aliyun distributed Benchmark

This directory owns the complete PAI-DLC execution path. Metric and dataset code stays in the
parent Benchmark package; Aliyun only distributes sequence shards and aggregates their reports.

## Layout

- `defaults.yaml`: non-secret PAI-DLC defaults exposed in the Viewer. Every field can be edited.
- `config.py`: typed validation and derived `world_size = nnodes * gpus_per_node`.
- `dlc.py`: credential loading plus submit/query/stop wrappers for the installed `dlc` CLI.
- `manager.py`: asynchronous Viewer lifecycle, DLC status polling, CPFS progress and report loading.
- `worker.py`: command executed on every DLC node; launches one `benchmark/run.py` shard per GPU.
- `submit.py`: standalone submission entrypoint for use without the Viewer.

Credentials remain on the submitting machine in a user-provided credential file. They are never
written to `request.json` or passed to the remote container.

The checked-in `defaults.yaml` contains placeholders. Configure the region, workspace, resource
quota, image, CPFS URI, repository path, resources, Conda environment, and credentials yourself.

The task output is shared through CPFS:

```text
output/eval/benchmark/aliyun/<timestamp>/
├── request.json
├── remote_state.json
├── comparison.json
├── progress/model_01/shard_000.json
├── barriers/model_01/node_000.json
└── model_01_<tag>/
    ├── gpu000/report.json
    ├── gpu001/report.json
    ├── ...
    ├── report.json
    └── report.md
```

Each global shard is unique across nodes:

```text
global_rank = node_rank * gpus_per_node + local_gpu
shard_count = nnodes * gpus_per_node
```

Node 0 waits for every node marker before calling the existing `dist/aggregate.py`. All nodes then
wait for the aggregation marker before loading the next model.

## CLI

```bash
python eval/model_effect/benchmark/dist/aliyun/submit.py \
  --ckpt /path/to/step_00100000 \
  --config /path/to/logs/record/config.yaml \
  --datasets hot3d \
  --heads hands,hands_world \
  --nnodes 2
```

Pass `--ckpt` more than once to compare multiple models in one DLC job. Use `--aliyun-config` for a
YAML override; omitted fields inherit `defaults.yaml`. The command returns after it resolves the
real DLC JobId. It does not wait for the evaluation to finish.
