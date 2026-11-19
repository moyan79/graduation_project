# 实验汇总报告

当前实验的日志、特征和分析报告放在 `outputs/`，通过统一检查工具查看或导出：

```bash
python tools/inspect_results.py --export reports/results.json
```

汇总从实际实验目录读取，不包含写在代码中的固定实验数值。

`outputs/` 和本目录内自动生成的报告不提交到 Git。手工核对的 `experiment_completion_20261119.json`、`未完成模型与ckpt下载地址_20261119.md` 与 `ForensicHub模型分类_20261119.md` 随项目保留，供完成状态、权重来源和分类口径追溯。

范围以 `configs/experiment_scope.yaml` 为准：源码统计 124 项，项目内 115 项，范围外 9 项为 Qwen 1 项及 8 个时序视频模型，均不下载、不训练；CLIP 等图像模型和图像配对监督方法保留。项目内完成 18 个模型、尚无完整记录 97 项、已知适配阻塞 19 项。分类文档保留历史来源总数，待补数量采用当前项目范围。HF 下载状态由 A40 归档记录确认。
