#!/bin/bash
# loop the 15 sub-domain models, one unified diag run each
set -e
MODELS="resnet efficientnet capsulenet segformer swin convnext recce sia iml-vit trufor univfd ffdn effort mesorch co-spy"
MANIFEST_DIR=${MANIFEST_DIR:-${OPENMMSEC_ROOT}/jsons_v4}
for m in $MODELS; do
  echo "================ $m ================"
  bash scripts/run_one_model.sh "$m" "outputs/diag_${m}" || echo "[warn] $m failed, continue"
done
echo "all done. summaries:"
for m in $MODELS; do
  echo "--- $m ---"
  cat "outputs/diag_${m}/rank_report.json" 2>/dev/null | head -12 || echo "no report"
  cat "outputs/diag_${m}/gradient_report.json" 2>/dev/null | head -12 || echo "no report"
done
