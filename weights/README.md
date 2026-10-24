# weights / 模型权重目录

训练时自动下载到这里（`env.sh` 已指向），`/tmp` 重启会丢，不要放 /tmp。

| 子目录 | 说明 |
|---|---|
| `hf/` | HF Hub 缓存（`HF_HOME`，timm 新版 + transformers 共用） |
| `torch/` | torch.hub 缓存（`TORCH_HOME`，timm 旧版 checkpoint 用） |
| `manual/` | 手动 `hf download --local-dir` 归档，比如 `/tmp/test_resnet` 搬过来的 |

只有本文件进 git，权重本身全部忽略。新机器 `git clone` 后目录会自动建好（HF 首次下载时创建）。
