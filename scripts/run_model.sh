#!/bin/bash
# usage: [NPROC=4] [EXTRA_ARGS="..."] bash scripts/run_model.sh model [out_dir]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${1:-resnet}
NPROC=${NPROC:-1}
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NPROC must be a positive integer" >&2
  exit 1
fi
SETTINGS=$("$PYTHON_BIN" src/model_config.py --model "$MODEL" \
  --field category ddp.batch_size ddp.lr)
read -r CATEGORY BATCH_SIZE LR <<< "$SETTINGS"
PREFIX=diag
if [ "$NPROC" -gt 1 ]; then
  PREFIX=diag_ddp
fi
OUT=${2:-outputs/$CATEGORY/${PREFIX}_${MODEL}}
mkdir -p "$OUT"

if [ -n "${TRAIN_JSON:-}" ] && [ -f "$TRAIN_JSON" ]; then
  MANIFEST_ARGS=(--train_json "$TRAIN_JSON")
else
  MANIFEST_DIR=${MANIFEST_DIR:-${OPENMMSEC_ROOT:-}/jsons_v4}
  MANIFEST_ARGS=(--manifest_dir "$MANIFEST_DIR")
fi
TRAIN_ARGS=(--model "$MODEL" "${MANIFEST_ARGS[@]}" --out_dir "$OUT")
if [ "$NPROC" -gt 1 ]; then
  LAUNCH=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="$NPROC"
          --master_port="${MASTER_PORT:-29500}")
  TRAIN_ARGS+=(--batch_size "$BATCH_SIZE" --lr "$LR" --diag_every 20)
else
  LAUNCH=("$PYTHON_BIN")
fi
read -r -a EXTRA <<< "${EXTRA_ARGS:-}"
echo "[run] model=$MODEL nproc=$NPROC out=$OUT ${EXTRA_ARGS:-}"
"${LAUNCH[@]}" src/unified_train_diag.py "${TRAIN_ARGS[@]}" "${EXTRA[@]}" \
  2>&1 | tee "$OUT/train.log"

"$PYTHON_BIN" src/analyze_gradient.py --log "$OUT/grad_log.csv" --out "$OUT"
"$PYTHON_BIN" src/analyze_rank.py --npz "$OUT/features_last.npz" --out "$OUT"
: > "$OUT/traj.log"
for path in "$OUT"/features_epoch*.npz; do
  [ -e "$path" ] || continue
  name=$(basename "$path" .npz)
  echo "== $name ==" >> "$OUT/traj.log"
  "$PYTHON_BIN" src/analyze_rank.py --npz "$path" --out "$OUT/traj/$name" \
    >> "$OUT/traj.log" 2>&1
done
echo "[run] done: $OUT/rank_report.json $OUT/gradient_report.json"
