#!/bin/bash
# usage: [NPROC=1|4] bash scripts/run_all.sh [out_root] [only_model]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-4}
OUTROOT=${1:-outputs}
ONLY=${2:-}
BASE_EXTRA=${EXTRA_ARGS:-}
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NPROC must be a positive integer" >&2
  exit 1
fi
if [ -n "$ONLY" ]; then
  MODELS=("$ONLY")
else
  MODEL_LIST=$("$PYTHON_BIN" src/model_config.py --list)
  mapfile -t MODELS <<< "$MODEL_LIST"
fi

FAILED=0
for model in "${MODELS[@]}"; do
  SETTINGS=$("$PYTHON_BIN" src/model_config.py --model "$model" \
    --field category ddp.batch_size ddp.lr)
  read -r CATEGORY BATCH_SIZE LR <<< "$SETTINGS"
  PREFIX=diag
  EXTRA=$BASE_EXTRA
  if [ "$NPROC" -gt 1 ]; then
    PREFIX=diag_ddp
    EXTRA="--batch_size $BATCH_SIZE --lr $LR $BASE_EXTRA"
  fi
  OUT="$OUTROOT/$CATEGORY/${PREFIX}_${model}"
  echo "================ $model (nproc=$NPROC out=$OUT) ================"
  if ! EXTRA_ARGS="$EXTRA" NPROC="$NPROC" PYTHON_BIN="$PYTHON_BIN" \
    bash "$SCRIPT_DIR/run_model.sh" "$model" "$OUT"; then
    echo "[warn] $model failed, continue"
    FAILED=$((FAILED + 1))
  fi
done

"$PYTHON_BIN" tools/inspect_results.py --outputs "$OUTROOT" --model "${MODELS[@]}" \
  || echo "[warn] no experiment reports available"
if [ "$FAILED" -gt 0 ]; then
  echo "[run-all] failed models: $FAILED" >&2
  exit 1
fi
echo "[run-all] done"
