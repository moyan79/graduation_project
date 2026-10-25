#!/bin/bash
# usage: NPROC=4 [EXTRA_ARGS="..."] bash scripts/run_one_model_ddp.sh resnet [out_dir]
# multi-GPU DDP via torchrun. --batch_size is per-GPU (global = batch * NPROC).
set -e
MODEL=${1:-resnet}
OUT=${2:-outputs/diag_ddp_${MODEL}}
NPROC=${NPROC:-4}
EXTRA=${EXTRA_ARGS:-}
mkdir -p "$OUT"
if [ -n "$TRAIN_JSON" ] && [ -f "$TRAIN_JSON" ]; then
  MANIFEST_ARG="--train_json $TRAIN_JSON"
else
  MANIFEST_DIR=${MANIFEST_DIR:-${OPENMMSEC_ROOT}/jsons_v4}
  MANIFEST_ARG="--manifest_dir $MANIFEST_DIR"
fi
echo "[run-ddp] model=$MODEL nproc=$NPROC out=$OUT $MANIFEST_ARG $EXTRA"
torchrun --nproc_per_node=$NPROC --master_port=${MASTER_PORT:-29500} src/unified_train_diag.py \
  --model "$MODEL" $MANIFEST_ARG \
  --epochs 10 --batch_size 128 --lr 0.0004 --diag_every 20 --probe_every 1 \
  --out_dir "$OUT" $EXTRA 2>&1 | tee "$OUT/train.log"
# analysis runs on rank-0 outputs (only rank 0 wrote them)
python src/analyze_gradient.py --log "$OUT/grad_log.csv" --out "$OUT" || echo "[warn] analyze_gradient failed"
python src/analyze_rank.py --npz "$OUT/features_last.npz" --out "$OUT" || echo "[warn] analyze_rank failed"
# per-epoch trajectory (rank-collapse process evidence): one rank_report per epoch dump.
# traj/ dirs + traj.log are written automatically; aggregate across models later.
: > "$OUT/traj.log"
for f in "$OUT"/features_epoch*.npz; do
  [ -e "$f" ] || continue
  b=$(basename "$f" .npz)
  echo "== $b ==" >> "$OUT/traj.log"
  python src/analyze_rank.py --npz "$f" --out "$OUT/traj/$b" >> "$OUT/traj.log" 2>&1 \ || echo "[warn] analyze_rank failed"
    || echo "[warn] traj $b failed"
done
echo "[run-traj] done: $OUT/traj/*/rank_report.json (log: $OUT/traj.log)"
echo "[run-ddp] done: $OUT/rank_report.json $OUT/gradient_report.json"


