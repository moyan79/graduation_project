#!/bin/bash
# 全量 15 个子领域模型 DDP 拉通诊断（resnet 除外，已跑完）。
# 用法：
#   source env.sh
#   NPROC=4 bash scripts/run_all_ddp.sh [out_root] [only_model]
#   只跑某一个：bash scripts/run_all_ddp.sh outputs univfd
# 说明：
#   - batch/lr 按模型体量分组：CNN 组 128/4e-4（全局512对齐 resnet 正式跑），
#     ViT/mask/CLIP 组 64/2e-4（全局256，线性缩放）。
#   - 依赖：B组需 DeepfakeBench，D组需 imdlbenco；co-spy 的 datacomp 权重
#     若镜像拉不到会直接失败（放最后跑，失败不影响前面的）。
set -u
NPROC=${NPROC:-4}
OUTROOT=${1:-outputs}
ONLY=${2:-}
BASE_EXTRA=${EXTRA_ARGS:-}

# A组 timm CNN：efficientnet segformer swin convnext（无额外依赖）
# B组 DeepfakeBench yaml：capsulenet recce sia（需 pip install DeepfakeBench）
# C组 CLIP：univfd co-spy（ViT-L/14；co-spy 另需 datacomp_xl_s13b_b90k）
# D组 Mask2LabelWrapper：iml-vit trufor ffdn effort mesorch（需 pip install imdlbenco）
MODELS="efficientnet capsulenet segformer swin convnext recce iml-vit trufor univfd ffdn mesorch co-spy"

if [ -n "$ONLY" ]; then
  MODELS="$ONLY"
fi

run_one() {
  local m=$1 bs=$2 lr=$3
  local out="$OUTROOT/diag_ddp_${m}"
  echo "================ $m (bs=$bs lr=$lr out=$out) ================"
  EXTRA_ARGS="$BASE_EXTRA --batch_size $bs --lr $lr" \
    NPROC=$NPROC bash scripts/run_one_model_ddp.sh "$m" "$out" \
    || echo "[warn] $m failed, continue"
}

for m in $MODELS; do
  case "$m" in
    efficientnet|capsulenet|segformer|swin|convnext|recce|sia)
      run_one "$m" 128 0.0004 ;;
    iml-vit|trufor|univfd|ffdn|effort|mesorch|co-spy)
      run_one "$m" 64 0.0002 ;;
    resnet)
      run_one "$m" 128 0.0004 ;;
    *)
      echo "[warn] unknown model $m, skip" ;;
  esac
done

echo "==================== summaries ===================="
for m in $MODELS; do
  echo "--- $m ---"
  python3 -c "import json; r=json.load(open('$OUTROOT/diag_ddp_${m}/rank_report.json')); print({k: round(v,4) if isinstance(v,float) else v for k,v in r.items() if k in ('cumvar_top20','effective_rank','kappa_s0_s19','s0','s19','n','dim')})" 2>/dev/null || echo "no rank report"
  python3 -c "import json; g=json.load(open('$OUTROOT/diag_ddp_${m}/gradient_report.json')); print({k: (round(v,4) if isinstance(v,float) else v) for k,v in g.items()})" 2>/dev/null | head -5 || echo "no gradient report"
done
