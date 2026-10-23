"""Unified OpenMMSecV2 torch Dataset (image, label, domain)."""
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from manifest import DOMAIN2ID, resolve_image_path
except ImportError:
    from .manifest import DOMAIN2ID, resolve_image_path

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def build_transform(image_size, is_train, norm_type="imagenet"):
    mean = CLIP_MEAN if norm_type == "clip" else IMAGENET_MEAN
    std = CLIP_STD if norm_type == "clip" else IMAGENET_STD
    ops = []
    if is_train:
        ops.append(T.RandomHorizontalFlip(p=0.5))
    ops += [T.Resize((image_size, image_size)), T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(ops)


class OpenMMSecUnifiedDataset(Dataset):
    def __init__(self, samples, image_size=256, is_train=True, norm_type="imagenet",
                 manifest_dir=None, openmmsec_root=None, prefix_from="", prefix_to="",
                 skip_missing=True):
        self.image_size = image_size
        self.tf = build_transform(image_size, is_train, norm_type)
        self.openmmsec_root = openmmsec_root
        self.manifest_dir = manifest_dir
        self.prefix_from = prefix_from
        self.prefix_to = prefix_to
        # resolve + optional existence filter (do NOT crash on one bad path)
        kept = []
        missing = 0
        total = len(samples)
        for i, s in enumerate(samples):
            if i % 20000 == 0:
                print(f"[dataset] resolving {i}/{total} (is_train={is_train})", flush=True)
            rp = resolve_image_path(s["path"], manifest_dir or ".",
                                    openmmsec_root, prefix_from, prefix_to)
            if skip_missing and not os.path.exists(rp):
                missing += 1
                continue
            kept.append({**s, "_resolved": rp})
        self.samples = kept
        if missing:
            print(f"[dataset] skipped {missing} missing files, kept {len(kept)} (is_train={is_train})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            with Image.open(s["_resolved"]) as im:
                im = im.convert("RGB")
                img = self.tf(im)
        except Exception as e:
            # bad image (SICA starter also documents corrupted images): return neighbour
            print(f"[dataset] bad image {s['_resolved']}: {e}")
            return self.__getitem__((idx + 1) % len(self.samples))
        return {
            "image": img,
            "label": torch.tensor(float(s["label"]), dtype=torch.float),
            "domain": torch.tensor(DOMAIN2ID.get(s["domain"], 2), dtype=torch.long),
            "domain_name": s["domain"],
            "path": s["_resolved"],
        }


def collate_with_domain(batch):
    images = torch.stack([b["image"] for b in batch])
    labels = torch.stack([b["label"] for b in batch]).float()
    domains = torch.stack([b["domain"] for b in batch]).long()
    paths = [b["path"] for b in batch]
    names = [b["domain_name"] for b in batch]
    return {"image": images, "label": labels, "domain": domains,
            "domain_name": names, "path": paths}

