# 实验汇总报告

当前实验的日志、特征和分析报告放在 `outputs/`，通过统一检查工具查看或导出：

```bash
python tools/inspect_results.py --export reports/results.json
```

汇总从实际实验目录读取，不包含写在代码中的固定实验数值。

`outputs/` 和本目录内生成的报告不提交到 Git；本目录只保留使用说明。
