#!/usr/bin/env bash
# 8 卡并行全量评测:27 个视频分到 8 卡(HOT3D_SEQS 白名单分片)，每个视频的左右手
# 共用一次预测；heads=hands,hands_world，--windowed 使用训练窗并做窗口 batch。各卡各自 --out，
# 跑完由 dist/aggregate.py 合并:
#   CKPT=... CFG=... BASE=... bash eval/model_effect/benchmark/dist/dispatch.sh
#   $PY eval/model_effect/benchmark/dist/aggregate.py "$BASE"
set -u
PY=${PY:-python}
# 仓库根按脚本自定位(不写死):dist -> benchmark -> model_effect -> eval -> <repo>
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
VAL=$REPO/data/benchmark/hand_pose/hot3d
CKPT=${CKPT:?需设 CKPT}
CFG=${CFG:?需设 CFG}
BASE=${BASE:?需设 BASE(输出根)}
cd "$REPO"
mkdir -p "$BASE"

seqs=($(ls -d "$VAL"/P0* | xargs -n1 basename | sort))
echo "[eval] 全 ${#seqs[@]} 序列 → 8 卡分片"
for g in 0 1 2 3 4 5 6 7; do
  shard=()
  for ((i=g; i<${#seqs[@]}; i+=8)); do shard+=("${seqs[i]}"); done
  [ ${#shard[@]} -eq 0 ] && continue
  csv=$(IFS=,; echo "${shard[*]}")
  echo "[eval] GPU$g: ${#shard[@]} 序列 = $csv"
  CUDA_VISIBLE_DEVICES=$g PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HOT3D_RECTIFY=1 HOT3D_SEQS="$csv" \
    $PY eval/model_effect/benchmark/run.py \
      --ckpt "$CKPT" --config "$CFG" \
      --datasets hot3d --heads hands,hands_world --windowed \
      --out "$BASE/gpu$g" \
      > "$BASE/_gpu$g.log" 2>&1 &
done
wait
echo "ALL_DONE"
