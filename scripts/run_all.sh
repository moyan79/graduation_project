#!/bin/bash
# usage: [NPROC=1|4] bash scripts/run_all.sh [--dry-run] [--pending] [--runnable-only] [out_root] [only_model]
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR/.."

PYTHON_BIN=${PYTHON_BIN:-python}
NPROC=${NPROC:-4}
DRY_RUN=0
SELECTION=()
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --pending|--runnable-only) SELECTION+=("$arg") ;;
    --*) echo "[error] unknown option: $arg" >&2; exit 1 ;;
    *) POSITIONAL+=("$arg") ;;
  esac
done
OUTROOT=${POSITIONAL[0]:-outputs}
ONLY=${POSITIONAL[1]:-}
if [ "${#POSITIONAL[@]}" -gt 2 ]; then
  echo "[error] expected [out_root] [only_model]" >&2
  exit 1
fi
BASE_EXTRA=${EXTRA_ARGS:-}
if [[ ! "$NPROC" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] NPROC must be a positive integer" >&2
  exit 1
fi
if [ -n "$ONLY" ]; then SELECTION+=(--model "$ONLY"); fi
SELECTION+=(--outputs "$OUTROOT")
# Always audit the full declaration set, including explicitly blocked models.
"$PYTHON_BIN" tools/forensichub_coverage.py --check "${SELECTION[@]}"
MODEL_LIST=$("$PYTHON_BIN" tools/forensichub_coverage.py --list "${SELECTION[@]}")
MODELS=()
if [ -n "$MODEL_LIST" ]; then mapfile -t MODELS <<< "$MODEL_LIST"; fi
if [ "$DRY_RUN" -eq 1 ]; then
  printf '[plan] %s\n' "${MODELS[@]}"
  "$PYTHON_BIN" tools/forensichub_coverage.py --check --require-runnable "${SELECTION[@]}"
  exit 0
fi

FAILED=0
BLOCKED=0
for model in "${MODELS[@]}"; do
  if ! "$PYTHON_BIN" tools/forensichub_coverage.py --require-runnable --model "$model" >/dev/null; then
    echo "[blocked] $model has an unresolved model/data adaptation; see the coverage report" >&2
    BLOCKED=$((BLOCKED + 1))
    continue
  fi
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
if [ "$FAILED" -gt 0 ] || [ "$BLOCKED" -gt 0 ]; then
  echo "[run-all] failed models: $FAILED; blocked models: $BLOCKED" >&2
  exit 1
fi
echo "[run-all] done"
