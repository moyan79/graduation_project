#!/bin/bash
# usage: bash scripts/run_one_model.sh resnet [out_dir]
# manifest selection:
#   1) if $TRAIN_JSON exists -> single-file v4.json mode
#   2) else use $MANIFEST_DIR (default $OPENMMSEC_ROOT/jsons_v4)
set -e
MODEL=${1:-resnet}
OUT=${2:-outputs/diag_${MODEL}}
EXTRA=${EXTRA_ARGS:-}
mkdir -p "$OUT"
if [ -n "$TRAIN_JSON" ] && [ -f "$TRAIN_JSON" ]; then
  echo "[run] model=$MODEL out=$OUT single-file=$TRAIN_JSON $EXTRA"
  python src/unified_train_diag.py --model "$MODEL" --train_json "$TRAIN_JSON" \
    --epochs 10 --batch_size 32 --probe_every 1 \
    --out_dir "$OUT" $EXTRA 2>&1 | tee "$OUT/train.log"
else
  MANIFEST_DIR=${MANIFEST_DIR:-${OPENMMSEC_ROOT}/jsons_v4}
  echo "[run] model=$MODEL out=$OUT manifest=$MANIFEST_DIR $EXTRA"
  python src/unified_train_diag.py --model "$MODEL" --manifest_dir "$MANIFEST_DIR" \
    --epochs 10 --batch_size 32 --probe_every 1 \
    --out_dir "$OUT" $EXTRA 2>&1 | tee "$OUT/train.log"
fi
python src/analyze_gradient.py --log "$OUT/grad_log.csv" --out "$OUT"
python src/analyze_rank.py --npz "$OUT/features_last.npz" --out "$OUT"
echo "[run] done: $OUT/rank_report.json $OUT/gradient_report.json"
