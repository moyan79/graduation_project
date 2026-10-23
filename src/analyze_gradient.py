"""Aggregate grad_log.csv -> conflict summary + plots.

  python src/analyze_gradient.py --log outputs/diag_resnet/grad_log.csv --out outputs/diag_resnet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out", default=".")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(a.log)
    cos_cols = [c for c in df.columns if c.startswith("cos_")]
    rep = {
        "steps": int(len(df)),
        "mean_conflict_ratio": float(df["conflict_ratio"].mean()),
        "frac_steps_majority_conflict": float((df["conflict_ratio"] > 0.5).mean()),
        "mean_cos_overall": float(df["mean_cos"].mean()),
        "per_pair_mean_cos": {c: float(df[c].mean()) for c in cos_cols},
        "per_pair_neg_frac": {c: float((df[c] < 0).mean()) for c in cos_cols},
    }
    import json
    with open(out / "gradient_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["step"], df["conflict_ratio"], marker="o", ms=3)
        plt.axhline(0.5, ls="--")
        plt.xlabel("step")
        plt.ylabel("conflict ratio (frac pairs cos<0)")
        plt.title("Gradient conflict over unified training")
        plt.tight_layout()
        plt.savefig(out / "grad_conflict_curve.png", dpi=150)
        plt.figure()
        means = [df[c].mean() for c in cos_cols]
        plt.bar(range(len(cos_cols)), means)
        plt.xticks(range(len(cos_cols)), cos_cols, rotation=30, ha="right")
        plt.ylabel("mean cos(g_i, g_j)")
        plt.axhline(0, color="k", lw=1)
        plt.title("Mean per-domain gradient alignment")
        plt.tight_layout()
        plt.savefig(out / "grad_cos_bar.png", dpi=150)
        print(f"[grad] figures saved to {out}")
    except Exception as e:
        print(f"[grad] plot skipped ({e})")


if __name__ == "__main__":
    main()
