# linear_probe-and-gradient

在 OpenMMSecV2 四域混合数据上训练检测模型，同时记录表征谱、有效秩和分域梯度冲突。这里的 probe 指冻结模型提取特征，不是单独训练线性分类器的严格 linear probe。

## 项目结构

项目与 `ForensicHub/`、`IMDLBenCo/` 同级。

```text
linear_probe-and-gradient/
├── configs/
│   ├── models.yaml            # 模型、分类、初始化、特征层与 DDP 参数
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
│   └── inspect_results.py     # 多模型结果、日志检查及 JSON 导出
├── reports/                   # 生成的汇总报告
├── weights/                   # 权重缓存
├── env.sh.example
└── requirements.txt
```

模型配置只有一份：修改 `configs/models.yaml` 后，训练入口、运行脚本和结果检查工具共同读取。`init` 和 `wrapper_init` 用于构建模型，`feature` 指定诊断层，`ddp` 指定多卡默认 batch 和学习率。

## 环境与数据

```bash
cp env.sh.example env.sh
# 修改 env.sh 中的数据和权重路径
source env.sh
pip install -r requirements.txt
```

对应模型还需要安装同级 ForensicHub、IMDLBenCo 或 DeepfakeBench。

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

## 输出目录

```text
outputs/
├── backbone/   # resnet / efficientnet / segformer / swin / convnext
├── deepfake/   # capsulenet / recce / sia
├── imdl/       # iml-vit / trufor / mesorch
├── aigc/       # univfd / effort / co-spy
└── document/   # ffdn
```

实验目录名为 `diag_<model>` 或 `diag_ddp_<model>`，保存日志、损失记录、逐轮特征和分析报告。分类依据是模型类型，不代表已经加载原领域训练完成的检测权重。

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
