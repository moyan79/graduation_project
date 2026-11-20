# linear_probe-and-gradient

在 OpenMMSecV2 四域混合数据上训练检测模型，同时记录表征谱、有效秩和分域梯度冲突。这里的 probe 指冻结模型提取特征，不是单独训练线性分类器的严格 linear probe。

已归档的 HF 权重及校验结果见 [下载记录](reports/HF镜像权重下载记录_20261119.md)。

## 项目结构

项目与 `ForensicHub/`、`IMDLBenCo/` 同级。

```text
linear_probe-and-gradient/
├── configs/
│   ├── models.yaml            # 模型、分类、初始化、特征层与 DDP 参数
│   ├── experiment_scope.yaml  # 项目范围；排除 LLM/VLM 和时序视频，保留图像 CLIP
│   └── unified_diag.yaml      # 默认训练参数
├── src/
│   ├── model_config.py        # 共用配置读取，不导入模型依赖
│   ├── model_factory.py       # 模型构建和特征 hook
│   ├── unified_train_diag.py  # 训练、梯度记录和特征 dump
│   ├── analyze_rank.py        # 单次实验的 SVD 分析
│   ├── analyze_gradient.py    # 单次实验的梯度分析
│   ├── openmmsec_dataset.py   # Dataset 与预处理
│   ├── manifest.py            # 数据清单加载
│   └── offline_patch.py       # 在线/离线权重加载
├── scripts/
│   ├── run_model.sh           # 单模型，支持单卡与 DDP
│   └── run_all.sh             # 批量运行配置中的模型
├── tools/
│   ├── check_manifest.py      # 数据划分检查
│   ├── sync_forensichub.py    # 应用随仓库维护的 ForensicHub 修复
│   └── inspect_results.py     # 多模型结果、日志检查及 JSON 导出
├── patches/
│   └── forensichub-image-adapters.patch
├── docs/
│   └── 交接文档.md            # 随仓库同步的实验交接记录
├── reports/                   # 生成的汇总报告
├── weights/                   # 权重缓存
├── env.sh.example
└── requirements.txt
```

模型配置只有一份：修改 `configs/models.yaml` 后，训练入口、运行脚本和结果检查工具共同读取。`init` 和 `wrapper_init` 用于构建模型，`feature` 指定诊断层，`ddp` 指定多卡默认 batch 和学习率。

2026-11-19 的本地源码扫描共发现 124 个模型名称条目，包含 46 个骨干预处理变体、同方法另实现和特征骨干；这是源码声明/兼容导出的统计，不是 124 种独立算法或已验证的运行时注册数。用户已明确排除 Qwen 等生成式 LLM/VLM 及 8 个时序视频模型，规则见 [experiment_scope.yaml](configs/experiment_scope.yaml)：当前项目纳入 115 个名称条目，范围外 9 个条目。CLIPDetector、OpenCLIP、UnivFD、Effort 和 CO-SPY 继续作为图像模型保留；LSDA、UCF 的图像批次配对监督不归为时序视频。分类及官方跨域图像配置的 21 个基础目标 + 24 个变体见 [模型分类](reports/ForensicHub模型分类_20261119.md)。18 个模型有本项目完成记录，涉及 23 组实验；项目内另有 97 个条目没有完整实验对应记录，其中 19 个存在已知适配阻塞。这些条目数仍不能直接当作独立论文方法数或教师必做实验数。详细证据和权重来源见 [下载清单](reports/未完成模型与ckpt下载地址_20261119.md) 与 `reports/experiment_completion_20261119.json`。

```bash
# 全量覆盖审计，不导入模型依赖，也不训练
python tools/forensichub_coverage.py --check --json reports/forensichub_coverage.json

# 查看尚未完成且没有已知适配阻塞的候选；不会启动训练
bash scripts/run_all.sh --dry-run --pending --runnable-only outputs/all_models

# 下载构建所需权重后，逐模型检查可微损失、梯度和诊断特征；默认离线
python tools/preflight_models.py --model synthbuster dualnet --device cpu

# 接入任务权重时可显式指定加载子模块，并强制禁止无任务权重启动
EXTRA_ARGS="$EXTRA_ARGS --require_task_checkpoint --task_checkpoint /path/to/detector.pth --task_checkpoint_target base_model" \
  bash scripts/run_model.sh mvss outputs/domain_pretrained/imdl/diag_mvss

# 预检通过后运行剩余候选；独立输出根目录避免覆盖历史实验
bash scripts/run_all.sh --pending --runnable-only outputs/all_models
```

`configured` 表示已提供运行配置，不代表真实权重和 GPU 预检通过。项目内的 JPEG 辅助输入、身份/配对监督等适配缺口保留在 `blocked_reason`；全量运行遇到阻塞会返回非零状态。生成式 LLM/VLM 和时序视频模型属于范围外条目，不进入待下载、待训练或阻塞模型清单。`--runnable-only` 是明确选择无已知阻塞的子集，不能把其成功称为全部模型实验完成。新定位模型使用 `LabelOnlyMaskWrapper`，从原生图像预测或掩码概率计算图像标签损失；占位掩码的像素损失不参与更新。`norm: none` 保留 `[0,1]` 图像输入。

每次新训练保存 `run_metadata.json`，记录实际轮数、lr、seed、每卡 batch、输入尺寸、四域与任务权重加载目标。训练缺少可读域、没有训练 batch 或出现非有限 loss 时停止；秩报告或梯度分析失败也会让运行脚本返回失败。DIRE 的冻结扩散 checkpoint 已改为严格加载，FIDL 的 CLIP LoRA 和骨干模型显式导入已接通；这些修复仍需真实权重和 GPU 验证。

`--pending` 还会核对输出根目录内的新 `run_metadata.json`、有效 CSV、完成日志、最终权重及所需逐轮报告，完整实验才会从后续候选中排除。已有的手工验收记录保留证据来源，未同步到本机的远端原始产物不会被当作新验收。

`task_checkpoint` 指定领域检测权重，`task_checkpoint_target` 指定加载模块。文件缺失或参数不匹配时停止构建，不退回随机初始化。

本轮先通过 `https://hf-mirror.com` 在 A40 的 `/mnt/data3/tianxin/graduation_project/weights/` 准备项目内图像模型的 HF 资源，使用 `hf download` 或 Hub 下载接口。SparseViT 的 `Sense-X/uniformer_image/uniformer_base_in1k.pth`、实际 timm 配置指定的 HF 骨干、CLIPDetector/Effort 的原生 Transformers CLIP 等属于这一类。完整任务 checkpoint 与骨干初始化分别记录；自带 OpenAI CLIP `.pt` 加载器不能直接换成 Transformers 的 HF 文件。具体来源与格式限制见下载清单，下载结果由本轮实际归档记录确认。Qwen 等生成式大模型及视频模型均不下载。

## 环境与数据

```bash
cp env.sh.example env.sh
# 修改 env.sh 中的数据和权重路径
source env.sh
pip install -r requirements.txt
```

对应模型还需要安装同级 ForensicHub、IMDLBenCo 或 DeepfakeBench。

本项目通过 [依赖补丁](patches/forensichub-image-adapters.patch) 维护 DIRE 严格权重加载与冻结、HiFi 设备缓冲区、AIGC/FIDL 导入修复。拉取本仓库会带回补丁文件，同级 ForensicHub checkout 还需要执行 [同步脚本](tools/sync_forensichub.py) 应用修复。在本仓库根目录运行：

```bash
# 先只检查补丁兼容性
python tools/sync_forensichub.py --check

# 应用尚未应用的修复；已应用文件会识别并跳过
python tools/sync_forensichub.py

# ForensicHub 不在默认同级目录时指定 checkout
python tools/sync_forensichub.py --repo /path/to/ForensicHub
```

脚本先核对所有目标文件，再应用补丁，保留依赖 checkout 中其他修改。发现目标文件冲突时停止，由输出指出具体文件。

数据支持 `jsons_v4/` 下的 `train.json`、`id_test.json`，或包含划分信息的单个 `v4.json`：

```bash
python tools/check_manifest.py --manifest_dir "$MANIFEST_DIR"
# 单文件模式
python tools/check_manifest.py --train_json /path/to/v4.json
```

## 运行实验

```bash
# 单卡单模型，默认 NPROC=1
bash scripts/run_model.sh resnet

# 多卡单模型
NPROC=4 bash scripts/run_model.sh resnet

# 批量运行全部配置模型，默认 NPROC=4
bash scripts/run_all.sh

# 单卡批量运行
NPROC=1 bash scripts/run_all.sh

# 指定输出根目录，只跑一个模型
NPROC=4 bash scripts/run_all.sh outputs univfd

# 覆盖模型默认运行参数
EXTRA_ARGS="--epochs 5 --batch_size 16 --lr 0.00005" \
  NPROC=4 bash scripts/run_model.sh resnet
```

`run_model.sh` 的第二个参数可指定完整输出目录；`run_all.sh` 的第一个参数指定输出根目录。`PYTHON_BIN` 可指定 Python 解释器，`MASTER_PORT` 可指定 DDP 端口。

单卡训练使用 `configs/unified_diag.yaml` 的默认值；DDP 使用模型配置中的 batch 和学习率。命令行及 `EXTRA_ARGS` 优先于默认值。训练失败时单模型脚本返回非零状态；批量脚本继续其余模型，结束时报告失败数量。

查看当前支持的模型及配置：

```bash
python src/model_config.py --list
python src/model_config.py --model resnet
```

### 从领域检测权重开始训练

SPSL、SRM、F3Net、Xception 默认读取 `weights/manual/hf_tasks/deepfakebench/<model>_best.pth`，加载完整 DeepfakeBench 检测器。构建时保留原生结构，跳过随后会被完整 checkpoint 覆盖的 ImageNet 初始化；输入使用原配置的 `[0.5, 0.5, 0.5]` 均值和标准差。

CapsuleNet 和 UnivFD 通过参数选择任务权重，保留不传参数时的初始化行为：

```bash
EXTRA_ARGS="$EXTRA_ARGS --task_checkpoint weights/manual/hf_tasks/deepfakebench/capsule_best.pth" \
  NPROC=4 bash scripts/run_model.sh capsulenet outputs/domain_pretrained/deepfake/diag_ddp_capsulenet
EXTRA_ARGS="$EXTRA_ARGS --task_checkpoint weights/manual/hf_tasks/univfd/fc_weights.pth" \
  NPROC=4 bash scripts/run_model.sh univfd outputs/domain_pretrained/aigc/diag_ddp_univfd
```

CapsuleNet 加载完整检测器参数，并使用 Deepfake 归一化；VGG19 的构建缓存仍需可用。其诊断层为 `vgg_ext` 输出，经全局池化得到 256 维向量。SPSL、SRM、F3Net、Xception 取分类器前的 2048 维特征，SRM 使用 RGB 分类分支。

UnivFD 的文件只包含 `fc` 分类头，CLIP ViT-L/14 编码器继续冻结。它是固定表征对照，不能用来证明编码器在联合训练中发生表征塌陷。日志中的 `loaded task checkpoint=... strict=True` 表示任务权重加载成功。

TruFor、IML-ViT、Mesorch 默认加载 `weights/manual/imdl_tasks/` 下的领域任务权重，加载目标为 `Mask2LabelWrapper.base_model`：

| 模型 | 任务权重文件 |
|---|---|
| trufor | `trufor_casiav2.pth` |
| iml-vit | `iml_vit_casiav2.pth` |
| mesorch | `mesorch-98-model.pth` |

Mesorch 作者的 `mesorch-98.pth` 还包含优化器、scaler 和训练参数；A40 已提取其 `model` 字段为 `mesorch-98-model.pth`，逐张量核对一致，原文件保留。训练使用提取后的文件，继续通过 `weights_only=True` 严格加载。TruFor 构建仍需现有的 `noiseprint.pth`、`mit_b2.pth` 和配置文件；保留默认 phase 与冻结策略。

```bash
source env.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC=4
export EXTRA_ARGS="$EXTRA_ARGS --offline --epochs 10 --batch_size 64 --lr 0.0002 --image_size 256 --seed 42"
MASTER_PORT=29541 bash scripts/run_model.sh trufor outputs/domain_pretrained/imdl/diag_ddp_trufor
MASTER_PORT=29542 bash scripts/run_model.sh iml-vit outputs/domain_pretrained/imdl/diag_ddp_iml-vit
MASTER_PORT=29543 bash scripts/run_model.sh mesorch outputs/domain_pretrained/imdl/diag_ddp_mesorch
```

选择空闲 GPU，按顺序运行。2026-11-09 已在 A40 完成 CPU batch 2 的离线严格加载、前向和梯度检查；每卡 batch 64 的显存占用尚未实测。

## 输出目录

```text
outputs/
├── backbone/                 # 骨干预训练实验
├── detector_baseline/        # 未加载领域检测 checkpoint 的检测模型实验
│   ├── deepfake/
│   ├── imdl/
│   ├── aigc/
│   └── document/
└── domain_pretrained/        # 从领域检测 checkpoint 开始的实验
    ├── deepfake/
    ├── imdl/
    ├── aigc/
    └── document/
```

实验目录名为 `diag_<model>` 或 `diag_ddp_<model>`，保存日志、损失记录、逐轮特征和分析报告。顶层区分模型用途和权重来源；检测模型再按领域分类。旧检测模型仅加载骨干或组件权重的结果放在 `detector_baseline/`，不能仅根据模型名称放入 `domain_pretrained/`。运行时通过输出目录参数选择对应位置。

脚本在训练结束后生成 `rank_report.json`、`gradient_report.json`，并将逐轮秩报告写入 `traj/`。DDP 的梯度诊断只记录 rank 0 的数据 shard；特征通过各卡提取后汇总。

## 检查与汇总结果

```bash
python tools/inspect_results.py
python tools/inspect_results.py --view built --model resnet sia
python tools/inspect_results.py --view init --model segformer
python tools/inspect_results.py --view errors --limit 20
python tools/inspect_results.py --category imdl --view metrics
python tools/inspect_results.py --view trajectory --model trufor
python tools/inspect_results.py --export reports/results.json
```

默认读取项目内 `outputs/`；本地没有该目录时读取同级 `../outputs/`。也可通过 `--outputs /path/to/outputs` 指定目录。工具支持平铺和分类布局，并将 `traj/` 排除在模型计数之外。

| `--view` | 用途 |
|---|---|
| `summary` | 模型类别、报告状态、特征维度与核心指标 |
| `built` | 构建参数、batch 和特征 dump 日志 |
| `init` | 权重加载与初始化日志 |
| `errors` | 错误、失败与 OOM 日志 |
| `logs` | 常用诊断与警告日志 |
| `keys` | 报告字段和特征形状 |
| `metrics` | 完整秩、梯度指标和损失记录 |
| `trajectory` | 各 epoch 的特征谱指标 |
| `details` | 报告与日志的原始内容 |
| `files` | 文件数量、大小与类型统计 |

`reports_ready` 表示两份分析报告和损失记录齐全；`epochs` 显示实际记录轮数。初始化信息取自日志，汇总工具不根据模型名称推断预训练状态。

## 结果解释

累计解释方差较高、有效秩较低表示谱能量集中；是否随训练发生塌陷，需要对照初始化和逐轮变化。跨模型比较同时报告特征层、维度与冻结策略。UnivFD 当前冻结 CLIP 编码器，属于固定表征对照。

梯度报告保留每对域的平均余弦和负对齐比例，不仅依据总冲突率判断。SVD 主方向的正交性不作为共线性证据。

训练产物、权重缓存和生成的报告不提交到 Git。
