# -*- coding: utf-8 -*-
"""生成 ResNet101 拉通训练诊断实验报告 Word 文档（含配图）。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
FIGDIR = OUTDIR / "report_figs"
FIGDIR.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

# ---------- 数据（10 epoch DDP 正式跑） ----------
S_TOP20 = [487.88, 233.91, 176.99, 154.58, 94.21, 77.23, 73.00, 66.78,
           53.35, 48.42, 43.10, 40.92, 36.77, 35.45, 34.09, 32.67,
           30.75, 30.31, 28.81, 27.20]
TRAJ = [(1, 0.9431, 11.68), (3, 0.9564, 15.56), (5, 0.9463, 10.49),
        (7, 0.9722, 21.22), (9, 0.9647, 17.94)]
PAIRS = ["de-ai", "de-im", "de-do", "ai-im", "ai-do", "im-do"]
PAIR_MEAN = [-0.053, 0.058, 0.040, 0.167, -0.196, -0.210]
PAIR_NEG = [0.5375, 0.4219, 0.4062, 0.3125, 0.6625, 0.7750]
NAMES = {"de": "Deepfake", "ai": "AIGC", "im": "IMDL", "do": "文档"}

# 图1 奇异值谱
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(range(20), S_TOP20, marker="o", ms=4)
ax.set_xlabel("奇异值序号")
ax.set_ylabel("奇异值")
ax.set_title("终版特征 Top20 奇异值谱（s0=487.9，s19=27.2，落差约18倍）")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "fig1_spectrum.png", dpi=150)
plt.close(fig)

# 图2 cumvar@20 与 s0/s19 轨迹
ep = [t[0] for t in TRAJ]
cum = [t[1] for t in TRAJ]
kap = [t[2] for t in TRAJ]
fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
ax1.plot(ep, cum, marker="o", ms=5, label="cumvar@20（左轴）")
ax1.set_xlabel("epoch")
ax1.set_ylabel("cumvar@20")
ax1.set_ylim(0.93, 0.98)
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(ep, kap, marker="s", ms=5, color="tab:orange", label="s0/s19（右轴）")
ax2.set_ylabel("s0/s19")
ax2.set_ylim(8, 24)
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="lower right")
ax1.set_title("特征集中度随训练的变化（每2个epoch抽查）")
ax1.set_xticks(ep)
fig.tight_layout()
fig.savefig(FIGDIR / "fig2_traj.png", dpi=150)
plt.close(fig)

# 图3 域对梯度对齐
x = range(len(PAIRS))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.6, 3.8))
bars = a1.bar([str(i) for i in x], PAIR_MEAN, color=["tab:red" if v < 0 else "tab:blue" for v in PAIR_MEAN])
a1.axhline(0, color="k", lw=1)
a1.set_xticklabels(PAIRS, rotation=0)
a1.set_ylabel("平均 cos")
a1.set_title("六对域平均梯度余弦")
a1.grid(alpha=0.3, axis="y")
a2.bar([str(i) for i in x], PAIR_NEG, color="tab:orange")
a2.axhline(0.5, ls="--", color="k", lw=1)
a2.set_xticklabels(PAIRS, rotation=0)
a2.set_ylabel("cos<0 占比")
a2.set_title("六对域负对齐占比（虚线=0.5）")
a2.grid(alpha=0.3, axis="y")
fig.suptitle("域间梯度对齐情况（80个诊断步平均）")
fig.tight_layout()
fig.savefig(FIGDIR / "fig3_grad.png", dpi=150)
plt.close(fig)
print("figures done:", list(FIGDIR.glob("*.png")))
