# linear_probe-and-gradient（与 ForensicHub 同级）

一次拉通训练，同时拿到两份证据。这里的 probe 指冻结训练后模型并分析特征谱，
不是单独训练线性分类器的严格 linear probe：

1. **表征秩塌陷**：在 id_test 上冻住 encoder 提特征，做 SVD，看前 20 维奇异谱 /
   累计方差 cumvar@20 / 有效秩 / 条件数 kappa。
2. **梯度冲突**：同一步内按域分别算梯度，看两两 cos(g_i,g_j)、
   冲突率 conflict_ratio = P(cos<0)、各域梯度范数是否失衡。

特征 hook 默认截取最后一个线性分类器之前的向量。IML-ViT 和 TruFor 使用
模型特定语义瓶颈：`predict_head.linear_fuse`（256 维）和
`decode_head.linear_fuse`（512 维），避免误抓 16 维辅助分支或 128 维检测头。

```
graduation_project/
  ForensicHub/                  # 同级仓库，不要移动
  linear_probe-and-gradient/    # 本项目（你在这里）
    src/unified_train_diag.py   # 主程序：训练 + 梯度记录 + 特征dump
    src/analyze_rank.py         # SVD 分析报告
    src/analyze_gradient.py     # 梯度冲突分析报告
    src/model_factory.py        # 15 个模型 -> ForensicHub 注册表
    src/openmmsec_dataset.py    # OpenMMSecV2 统一 Dataset
    src/manifest.py             # train.json / id_test.json / v4.json 加载
    tools/check_manifest.py     # 数据划分自检
    scripts/run_one_model.sh    # 跑单个模型
    scripts/run_all_models.sh   # 跑全部 15 个模型
```

## 0. 远程服务器环境准备（在服务器上跑）

```bash
cd <graduation_project>/linear_probe-and-gradient
cp env.sh.example env.sh  # 修改里面的 OPENMMSEC_ROOT
source env.sh
pip install -r requirements.txt
# 下面两个按需安装，对应哪些模型见注释：
pip install imdlbenco      # iml-vit / trufor / mesorch / effort 需要
pip install DeepfakeBench  # recce / sia / capsulenet 需要
```

`OPENMMSEC_ROOT` 支持两种目录结构：

- `$OPENMMSEC_ROOT/jsons_v4/train.json` + `id_test.json`（外加 `*_ood.json`），或
- 单个 `$OPENMMSEC_ROOT/v4.json`（list，或 dict 分 split），用 `--train_json` 指定。

## 1. 先检查数据划分是否正确

```bash
python tools/check_manifest.py --manifest_dir $OPENMMSEC_ROOT/jsons_v4
# 或：python tools/check_manifest.py --train_json $OPENMMSEC_ROOT/v4.json
```

正常情况：train 约 8.1 万张，id_test 约 8 千张，都包含 deepfake / aigc / imdl / doc 四个域。

## 2. 跑一个模型，一次拿两份证据

```bash
bash scripts/run_one_model.sh resnet
# 输出在 outputs/diag_resnet/ 里：
# train.log, grad_log.csv, features_epoch*.npz,
# features_last.npz, rank_report.json, gradient_report.json,
# svd_spectrum.png, grad_conflict_curve.png, ...
```

`--model` 可选：`resnet efficientnet capsulenet segformer swin convnext recce sia iml-vit trufor univfd ffdn effort mesorch co-spy`。

手动跑（和上面脚本等价）：

```bash
python src/unified_train_diag.py --model resnet --manifest_dir $OPENMMSEC_ROOT/jsons_v4 \
  --epochs 5 --batch_size 32 --diag_every 50 --probe_every 1 --out_dir outputs/diag_resnet
python src/analyze_gradient.py --log outputs/diag_resnet/grad_log.csv --out outputs/diag_resnet
python src/analyze_rank.py --npz outputs/diag_resnet/features_last.npz --out outputs/diag_resnet
```

## 2.5 多卡一起跑（DDP，服务器推荐）

单卡只是默认，`--batch_size` 是单卡 batch，多卡时全局 batch = batch × 卡数：

```bash
NPROC=4 bash scripts/run_one_model_ddp.sh resnet
# 等价于 torchrun --nproc_per_node=4 src/unified_train_diag.py --model resnet ...
```

* 梯度诊断只在 rank 0 上记（它那份 shard 的统计，结论一致，还省了同步开销）；
* 特征 dump 是各卡分片提完再 gather 拼起来，和单卡结果等价；
* 日志/CSV/NPZ/checkpoint 只有 rank 0 写，不会打架。
* 每个 epoch 结束会写一行 `epoch_loss.csv`（全局平均 loss、本 epoch 平均冲突率/平均余弦），训练健康情况直接看这个文件。
* 想调大学习率按线性缩放自己乘卡数即可（默认不自动缩放）。

## 3. 报告里的数字怎么看

- `rank_report.json`：`cumvar_top20` 较高且 `effective_rank` 较低、
  `kappa_s0_s19` 较大，说明谱能量集中、有效维度偏低。
  SVD 主方向彼此正交，`mean_abs_corr_top20_PC` 不应作为共线性证据。
- `gradient_report.json`：`mean_conflict_ratio > 0.5`，且多对域的平均 `cos < 0`、
  `neg_frac` 很高，就说明“域间梯度负对齐、更新相互抵消”。

每次运行都有 `train.log` + `grad_log.csv` 留底，“一次训练同时得出两个结论”是可审计的。
