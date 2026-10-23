"""SVD rank-collapse analysis on dumped features.

  python src/analyze_rank.py --npz outputs/diag_resnet/features_last.npz --out outputs/diag_resnet

Outputs: rank_report.json + svd_spectrum.png + svd_cumsum.png
Metrics: top-20 singular values, cumvar@20, effective rank, kappa=s0/s19,
         mean |corr| of top-20 PCs (your "co-linearity" claim).
"""
import argparse
import json
from pathlib import Path

import numpy as np


def center(X):
    return X - X.mean(axis=0, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True)
    p.add_argument("--out", default=".")
    p.add_argument("--k", type=int, default=20)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(a.npz, allow_pickle=True)
    X = d["features"].astype(np.float64)
    X = center(X)
    # economy SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    tot = float((S ** 2).sum())
    cum = (S ** 2).cumsum() / (S ** 2).sum()
    k = min(a.k, len(S))
    p_ = S / S.sum()
    eff_rank = float(np.exp(-(p_ * np.log(p_ + 1e-12)).sum()))
    rep = {
        "n": int(X.shape[0]), "dim": int(X.shape[1]),
        "topk_singular": [float(x) for x in S[:k]],
        f"cumvar_top{k}": float(cum[k - 1]),
        "effective_rank": eff_rank,
        f"kappa_s0_s{k-1}": float(S[0] / (S[k - 1] + 1e-12)),
        "s0": float(S[0]), "s19": float(S[min(19, len(S)-1)]),
    }
    # PC collinearity proxy: mean |corr| among top-k PC scores
    Z = U[:, :k] * S[:k]
    C = np.corrcoef(Z.T)
    iu = np.triu_indices(k, 1)
    rep[f"mean_abs_corr_top{k}_PC"] = float(np.abs(C[iu]).mean())
    with open(out / "rank_report.json", "w") as f:
        json.dump(rep, f, indent=2)
    print(json.dumps(rep, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(S[:min(100, len(S))], marker="o", ms=3)
        plt.axvline(k - 1, ls="--")
        plt.xlabel("singular index")
        plt.ylabel("singular value")
        plt.title(f"SVD spectrum (cumvar@{k}={rep[f'cumvar_top{k}']:.3f}, eff_rank={eff_rank:.1f})")
        plt.tight_layout()
        plt.savefig(out / "svd_spectrum.png", dpi=150)
        plt.figure()
        plt.plot(np.arange(1, min(100, len(S)) + 1), cum[:min(100, len(S))], marker="o", ms=3)
        plt.axhline(0.9, ls="--")
        plt.xlabel("top-k")
        plt.ylabel("cumulative variance")
        plt.title("Cumulative explained variance")
        plt.tight_layout()
        plt.savefig(out / "svd_cumsum.png", dpi=150)
        print(f"[rank] figures saved to {out}")
    except Exception as e:
        print(f"[rank] plot skipped ({e})")


if __name__ == "__main__":
    main()
