"""One unified run -> two evidences (grad conflict + rank collapse).

  python src/unified_train_diag.py --model resnet --manifest_dir $OPENMMSEC_ROOT/jsons_v4 \
      --epochs 5 --batch_size 32 --diag_every 50 --probe_every 1 --out_dir outputs/diag_resnet

What it does:
  * unified training on OpenMMSecV2 (all 4 domains mixed in every batch);
  * every --diag_every steps: per-domain grads -> pairwise cosine / conflict ratio (grad_log.csv);
  * every --probe_every epochs (+ finally): dump frozen features on id_test (features_epoch*.npz).

Analyse afterwards:
  python src/analyze_gradient.py --log outputs/diag_resnet/grad_log.csv --out outputs/diag_resnet
  python src/analyze_rank.py --npz outputs/diag_resnet/features_last.npz --out outputs/diag_resnet
"""
import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS = Path(__file__).resolve()
sys.path.insert(0, str(THIS.parent))

import offline_patch  # noqa: F401
from manifest import DOMAIN_NAMES, DOMAIN2ID, find_manifests, load_samples
from openmmsec_dataset import OpenMMSecUnifiedDataset, collate_with_domain
from model_factory import MODEL_CFG, build_forensic_model, extract_features


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="resnet", help=f"one of {sorted(MODEL_CFG)}")
    p.add_argument("--manifest_dir", default=os.environ.get("MANIFEST_DIR", None))
    p.add_argument("--train_json", default=os.environ.get("TRAIN_JSON", None))
    p.add_argument("--id_test_json", default=os.environ.get("ID_TEST_JSON", None))
    p.add_argument("--openmmsec_root", default=os.environ.get("OPENMMSEC_ROOT", None))
    p.add_argument("--path_prefix_from", default="")
    p.add_argument("--path_prefix_to", default="")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--diag_every", type=int, default=50, help="per-domain grad diagnosis cadence (steps)")
    p.add_argument("--probe_every", type=int, default=1)
    p.add_argument("--probe_samples", type=int, default=2000, help="cap for feature dump")
    p.add_argument("--image_size", type=int, default=None, help="override model default")
    p.add_argument("--out_dir", default="outputs/diag_run")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def _flatten_grads(model):
    vs = []
    for pg in model.parameters():
        if pg.grad is not None:
            vs.append(pg.grad.detach().float().reshape(-1))
    if not vs:
        return None
    return torch.cat(vs)


def per_domain_grad_snapshots(model, batch, device):
    """Forward/backward once per domain present in the batch. Returns {dom: grad_vec}."""
    imgs, labs, doms = batch["image"].to(device), batch["label"].to(device), batch["domain"].to(device)
    B, _, H, W = imgs.shape
    snaps = {}
    for d in sorted(set(doms.tolist())):
        m = (doms == d)
        if m.sum().item() < 2:
            continue
        model.zero_grad(set_to_none=True)
        aux = dict(mask=torch.zeros(m.sum(), 1, H, W, device=device),
                   edge_mask=torch.zeros(m.sum(), 1, H, W, device=device),
                   dct=torch.zeros(m.sum(), 1, H, W, dtype=torch.long, device=device),
                   qt=torch.ones(m.sum(), 1, 8, 8, dtype=torch.long, device=device),
                   DCT_coef=torch.zeros(m.sum(), 1, H, W, dtype=torch.long, device=device),
                   qtables=torch.ones(m.sum(), 1, 8, 8, dtype=torch.long, device=device))
        try:
            out = model(image=imgs[m], label=labs[m], **aux)
        except TypeError:
            out = model(image=imgs[m], label=labs[m])
        loss = out["backward_loss"]
        if loss.dim() > 0:
            loss = loss.mean()
        loss.backward()
        g = _flatten_grads(model)
        if g is not None:
            snaps[d] = g.cpu()
        model.zero_grad(set_to_none=True)
    return snaps


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ms = find_manifests(a.manifest_dir, a.train_json, a.openmmsec_root)
    train_json, id_json = ms["train"], ms["id_test"]
    if a.id_test_json:
        id_json = Path(a.id_test_json)
    print(f"[diag] model={a.model} train={train_json} id_test={id_json}")

    train_samples = load_samples(train_json)
    id_samples = load_samples(id_json)
    print(f"[diag] train={len(train_samples)} id_test={len(id_samples)}")

    model, mcfg = build_forensic_model(a.model)
    img_size = a.image_size or mcfg["image_size"]
    norm = mcfg["norm"]
    device = torch.device(a.device)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[diag] built {mcfg['registry']} params={n_params/1e6:.1f}M img={img_size} norm={norm}")

    ds_train = OpenMMSecUnifiedDataset(train_samples, img_size, True, norm, ms["base"],
                                       a.openmmsec_root, a.path_prefix_from, a.path_prefix_to)
    ds_probe = OpenMMSecUnifiedDataset(id_samples, img_size, False, norm, ms["base"],
                                       a.openmmsec_root, a.path_prefix_from, a.path_prefix_to)
    # cap probe set balanced across domains
    if len(ds_probe) > a.probe_samples:
        rng = np.random.RandomState(a.seed)
        idx = rng.choice(len(ds_probe), a.probe_samples, replace=False)
        ds_probe = torch.utils.data.Subset(ds_probe, idx.tolist())

    dl_train = DataLoader(ds_train, batch_size=a.batch_size, shuffle=True,
                          num_workers=a.num_workers, pin_memory=True, drop_last=True,
                          collate_fn=collate_with_domain)
    dl_probe = DataLoader(ds_probe, batch_size=a.batch_size, shuffle=False,
                          num_workers=a.num_workers, pin_memory=True, collate_fn=collate_with_domain)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    grad_csv = out / "grad_log.csv"
    pairs = list(itertools.combinations(range(len(DOMAIN_NAMES)), 2))
    with open(grad_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "step", "conflict_ratio", "mean_cos",
                    *[f"cos_{DOMAIN_NAMES[i]}_vs_{DOMAIN_NAMES[j]}" for i, j in pairs],
                    *[f"norm_{d}" for d in DOMAIN_NAMES], "train_loss"])

    step, best = 0, 1e9
    for ep in range(a.epochs):
        model.train()
        for batch in dl_train:
            step += 1
            imgs, labs = batch["image"].to(device), batch["label"].to(device)
            B, _, H, W = imgs.shape
            # --- diagnostic per-domain grads (does NOT update weights) ---
            if step % a.diag_every == 1:
                snaps = per_domain_grad_snapshots(model, batch, device)
                if len(snaps) >= 2:
                    keys = sorted(snaps)
                    coss, norms = {}, {}
                    for k in keys:
                        norms[k] = float(snaps[k].norm().item())
                    for i, j in pairs:
                        if i in snaps and j in snaps:
                            c = float(F.cosine_similarity(snaps[i], snaps[j], dim=0).item())
                            coss[(i, j)] = c
                    vals = list(coss.values())
                    cr = float(np.mean([v < 0 for v in vals])) if vals else float("nan")
                    mc = float(np.mean(vals)) if vals else float("nan")
                    # real full-batch update afterwards, so re-zero here
                    with open(grad_csv, "a", newline="") as f:
                        w = csv.writer(f)
                        w.writerow([ep, step, f"{cr:.4f}", f"{mc:.4f}",
                                    *[f"{coss.get(p, float('nan')):.4f}" for p in pairs],
                                    *[f"{norms.get(i, float('nan')):.3e}" for i in range(4)], ""])
                    print(f"[ep{ep} s{step}] conflict={cr:.2f} mean_cos={mc:+.3f} " +
                          " ".join(f"{DOMAIN_NAMES[i][:2]}-{DOMAIN_NAMES[j][:2]}={coss.get((i,j),float('nan')):+.2f}" for i, j in pairs))
            # --- real unified update ---
            model.zero_grad(set_to_none=True)
            aux = dict(mask=torch.zeros(B, 1, H, W, device=device),
                       edge_mask=torch.zeros(B, 1, H, W, device=device),
                       dct=torch.zeros(B, 1, H, W, dtype=torch.long, device=device),
                       qt=torch.ones(B, 1, 8, 8, dtype=torch.long, device=device),
                       DCT_coef=torch.zeros(B, 1, H, W, dtype=torch.long, device=device),
                       qtables=torch.ones(B, 1, 8, 8, dtype=torch.long, device=device))
            try:
                outd = model(image=imgs, label=labs, **aux)
            except TypeError:
                outd = model(image=imgs, label=labs)
            loss = outd["backward_loss"]
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            opt.step()
            if step % 100 == 1:
                print(f"[ep{ep} s{step}] train_loss={loss.item():.4f}")

        # --- periodic frozen-feature dump for SVD / linear probe ---
        if (ep + 1) % a.probe_every == 0 or ep == a.epochs - 1:
            fb = extract_features(model, dl_probe, device)
            np.savez(out / f"features_epoch{ep}.npz", **fb,
                     domain_names=np.array(DOMAIN_NAMES))
            print(f"[diag] dumped features_epoch{ep}.npz {fb['features'].shape}")

    fb = extract_features(model, dl_probe, device)
    np.savez(out / "features_last.npz", **fb, domain_names=np.array(DOMAIN_NAMES))
    torch.save(model.state_dict(), out / "unified_model_last.pth")
    print(f"[diag] done. grad log: {grad_csv}")
    print(f"[diag] next: analyze_gradient.py + analyze_rank.py on {out}")


if __name__ == "__main__":
    main()




