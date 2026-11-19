# weights / 模型权重目录

2026-11-19 已在 A40 通过 HF 国内镜像新增 18 个图像权重文件（约 2.77 GB），复用 13 个已有文件，31 个权重均完成 SHA256 和路径校验。新增任务参数包括 CORE、EfficientDetector、FFD、Meso4、Meso4Inception、RecceDetector、UCF，保存在 `manual/hf_tasks/deepfakebench/`；Uniformer 保存在 `manual/backbones/`，timm/CLIP/OpenCLIP 权重使用 `hf/hub/` 缓存。记录见 `../reports/HF镜像权重下载记录_20261119.md`。Qwen 等生成式大模型和视频模型不纳入本项目；本轮未下载这些权重。

训练时自动下载到这里（`env.sh` 已指向），`/tmp` 重启会丢，不要放 /tmp。

| 子目录 | 说明 |
|---|---|
| `hf/` | HF Hub 缓存（`HF_HOME`，timm 新版 + transformers 共用） |
| `torch/` | torch.hub 缓存（`TORCH_HOME`，timm 旧版 checkpoint 用） |
| `manual/` | 手动 `hf download --local-dir` 归档，比如 `/tmp/test_resnet` 搬过来的 |

只有本文件进 git，权重本身全部忽略。新机器 `git clone` 后目录会自动建好（HF 首次下载时创建）。

当前领域权重放在 `manual/hf_tasks/`：`deepfakebench/` 下为 `spsl_best.pth`、`srm_best.pth`、`f3net_best.pth`、`xception_best.pth`、`capsule_best.pth`；`univfd/fc_weights.pth` 仅包含 UnivFD 分类头。来源清单和许可文件与权重放在同一归档目录。

IMDL 领域任务权重放在 `manual/imdl_tasks/`：`trufor_casiav2.pth`、`iml_vit_casiav2.pth` 来自 IMDLBenCo 官方 checkpoint 集合；`mesorch-98.pth` 来自 Mesorch 作者发布的 checkpoint。A40 从 Mesorch 原文件提取完整 `model` 字段到 `mesorch-98-model.pth`，逐张量核对一致；配置加载提取后的文件，原文件保留。
