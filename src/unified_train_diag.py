"""One unified run -> two evidences (grad conflict + rank collapse).

Single GPU:
  python src/unified_train_diag.py --model resnet --manifest_dir $OPENMMSEC_ROOT/jsons_v4 \
      --epochs 5 --batch_size 32 --diag_every 50 --probe_every 1 --out_dir outputs/diag_resnet

Multi GPU (DDP, recommended on the server):
  NPROC=4 bash scripts/run_one_model_ddp.sh resnet
  # == torchrun --nproc_per_node=4 src/unified_train_diag.py --model resnet ...

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
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32, help="per-GPU batch size")
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
    p.add_argument("--dist_url", default="env://", help="DDP init method (torchrun sets env)")
    return p.parse_args()


def init_dist(a):
    a.rank = int(os.environ.get("RANK", 0))
    a.world_size = int(os.environ.get("WORLD_SIZE", 1))
    a.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    a.distributed = a.world_size > 1
    if a.distributed:
        torch.cuda.set_device(a.local_rank)
        dist.init_process_group(backend="nccl", init_method=a.dist_url)
    return a


def is_main(a):
    return a.rank == 0


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
    """Forward/backward once per domain present in the batch. Returns {dom: grad_vec}.

    NOTE: under DDP this must run inside model.no_sync() (done by caller on rank 0),
    otherwise the diagnostic backward would allreduce while other ranks idle.
    """
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
    init_dist(a)
    if a.distributed:
        device = torch.device(f"cuda:{a.local_rank}")
    else:
        device = torch.device(a.device)
    torch.manual_seed(a.seed + a.rank)
    np.random.seed(a.seed + a.rank)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ms = find_manifests(a.manifest_dir, a.train_json, a.openmmsec_root)
    train_json, id_json = ms["train"], ms["id_test"]
    if a.id_test_json:
        id_json = Path(a.id_test_json)
    if is_main(a):
        print(f"[diag] model={a.model} train={train_json} id_test={id_json} "
              f"world={a.world_size} per_gpu_bs={a.batch_size} (global_bs={a.batch_size * a.world_size})")

    train_samples = load_samples(train_json)
    id_samples = load_samples(id_json)
    if is_main(a):
        print(f"[diag] train={len(train_samples)} id_test={len(id_samples)}")

    model, mcfg = build_forensic_model(a.model)
    img_size = a.image_size or mcfg["image_size"]
    norm = mcfg["norm"]
    model.to(device)
    if a.distributed:
        # find_unused_parameters=True: mask/backbone branches vary across forwards;
        # diagnosis tolerates the extra sync cost.
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[a.local_rank], find_unused_parameters=True)
        base_model = model.module
    else:
        base_model = model
    n_params = sum(p.numel() for p in base_model.parameters())
    if is_main(a):
        print(f"[diag] built {mcfg['registry']} params={n_params/1e6:.1f}M img={img_size} norm={norm}")

    ds_train = OpenMMSecUnifiedDataset(train_samples, img_size, True, norm, ms["base"],
                                       a.openmmsec_root, a.path_prefix_from, a.path_prefix_to)
    ds_probe = OpenMMSecUnifiedDataset(id_samples, img_size, False, norm, ms["base"],
                                       a.openmmsec_root, a.path_prefix_from, a.path_prefix_to)
    if len(ds_train) == 0 or len(ds_probe) == 0:
        raise SystemExit(
            f"[diag] empty dataset: train_kept={len(ds_train)} probe_kept={len(ds_probe)}. "
            "Image paths in the manifest do not resolve locally. "
            "Re-run with --path_prefix_from/--path_prefix_to (via EXTRA_ARGS), e.g. "
            'EXTRA_ARGS="--path_prefix_from /mnt/public/public_datasets/OpenMMSecV2/ '
            '--path_prefix_to /mnt/nas/public/public_datasets/OpenMMSecV2/" ...')
    # cap probe set (same indices on all ranks: same seed, no rank offset here)
    if len(ds_probe) > a.probe_samples:
        rng = np.random.RandomState(a.seed)
        idx = rng.choice(len(ds_probe), a.probe_samples, replace=False)
        ds_probe = torch.utils.data.Subset(ds_probe, idx.tolist())

    if a.distributed:
        train_sampler = DistributedSampler(ds_train, shuffle=True, drop_last=True)
        probe_sampler = DistributedSampler(ds_probe, shuffle=False, drop_last=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(ds_train)
        probe_sampler = None
    dl_train = DataLoader(ds_train, batch_size=a.batch_size, sampler=train_sampler,
                          num_workers=a.num_workers, pin_memory=True, drop_last=True,
                          collate_fn=collate_with_domain)
    dl_probe = DataLoader(ds_probe, batch_size=a.batch_size, sampler=probe_sampler,
                          shuffle=(probe_sampler is None),
                          num_workers=a.num_workers, pin_memory=True, collate_fn=collate_with_domain)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    grad_csv = out / "grad_log.csv"
    pairs = list(itertools.combinations(range(len(DOMAIN_NAMES)), 2))
    if is_main(a):
        with open(grad_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["epoch", "step", "conflict_ratio", "mean_cos",
                        *[f"cos_{DOMAIN_NAMES[i]}_vs_{DOMAIN_NAMES[j]}" for i, j in pairs],
                        *[f"norm_{d}" for d in DOMAIN_NAMES], "train_loss"])

    step = 0
    for ep in range(a.epochs):
        if a.distributed:
            train_sampler.set_epoch(ep)
        model.train()
        for batch in dl_train:
            step += 1
            imgs, labs = batch["image"].to(device), batch["label"].to(device)
            B, _, H, W = imgs.shape
            # --- diagnostic per-domain grads (rank 0 only, does NOT update weights) ---
            if step % a.diag_every == 1 and is_main(a):
                ctx = model.no_sync() if a.distributed else nullcontext()
                with ctx:
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
            # --- real unified update (all ranks, synced by DDP) ---
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
            if step % 100 == 1 and is_main(a):
                print(f"[ep{ep} s{step}] train_loss={loss.item():.4f}")

        # --- periodic frozen-feature dump for SVD / linear probe ---
        if (ep + 1) % a.probe_every == 0 or ep == a.epochs - 1:
            local_fb = extract_features(model, dl_probe, device)
            if a.distributed:
                gathered = [None] * a.world_size
                dist.all_gather_object(gathered, {k: v for k, v in local_fb.items()})
                if is_main(a):
                    import numpy as _np
                    fb = {k: _np.concatenate([g[k] for g in gathered]) for k in local_fb}
                    np.savez(out / f"features_epoch{ep}.npz", **fb,
                             domain_names=np.array(DOMAIN_NAMES))
                    print(f"[diag] dumped features_epoch{ep}.npz {fb['features'].shape}")
            else:
                np.savez(out / f"features_epoch{ep}.npz", **local_fb,
                         domain_names=np.array(DOMAIN_NAMES))
                print(f"[diag] dumped features_epoch{ep}.npz {local_fb['features'].shape}")

    local_fb = extract_features(model, dl_probe, device)
    if a.distributed:
        gathered = [None] * a.world_size
        dist.all_gather_object(gathered, {k: v for k, v in local_fb.items()})
        if is_main(a):
            import numpy as _np
            fb = {k: _np.concatenate([g[k] for g in gathered]) for k in local_fb}
            np.savez(out / "features_last.npz", **fb, domain_names=np.array(DOMAIN_NAMES))
            torch.save(base_model.state_dict(), out / "unified_model_last.pth")
    else:
        np.savez(out / "features_last.npz", **local_fb, domain_names=np.array(DOMAIN_NAMES))
        torch.save(base_model.state_dict(), out / "unified_model_last.pth")
    if a.distributed:
        dist.barrier()
        dist.destroy_process_group()
    if is_main(a):
        print(f"[diag] done. grad log: {grad_csv}")
        print(f"[diag] next: analyze_gradient.py + analyze_rank.py on {out}")


if __name__ == "__main__":
    main()
