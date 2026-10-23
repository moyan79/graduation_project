"""OpenMMSecV2 manifest loading + domain normalisation.

Supports two layout cases on the remote server:
  1) jsons_v4 directory:  train.json / id_test.json / *_ood.json / total_ood_test.json
  2) single file v4.json: either a list of samples, or a dict {split: [...]}

Sample fields (robust to variants):
  path / image / file,  label (0 real / 1 fake),  domain / ori_dataset / source
"""
import json
import os
from pathlib import Path

DOMAIN_NAMES = ["deepfake", "aigc", "imdl", "doc"]
DOMAIN2ID = {d: i for i, d in enumerate(DOMAIN_NAMES)}


def _norm_domain(sample) -> str:
    for k in ("domain", "ori_dataset", "dataset", "source", "type"):
        if k in sample and sample[k]:
            s = str(sample[k]).lower()
            if "deepfake" in s or "ff++" in s or "faceforensic" in s:
                return "deepfake"
            if "aigc" in s or "genimage" in s or "diffusion" in s or "gan" in s or "coco" in s:
                # coco/genimage entries in OpenMMSecV2 are the AIGC split
                if "coco" in s or "gen" in s or "aigc" in s or "diffusion" in s:
                    return "aigc"
            if "iml" in s or "casia" in s or "coverage" in s or "nist" in s or "imd" in s or "manipulat" in s:
                return "imdl"
            if "doc" in s or "sroie" in s or "ostf" in s or "tamper" in s or "ic13" in s or "realtext" in s:
                return "doc"
    # fallback: infer from path string
    p = str(sample.get("path", sample.get("image", sample.get("file", "")))).lower()
    if "deepfake" in p or "faceforensic" in p or "ff-" in p:
        return "deepfake"
    if "aigc" in p or "genimage" in p or "adm_" in p or "biggan" in p or "stable_diffusion" in p or "coco" in p:
        return "aigc"
    if "doc" in p or "ostf" in p or "sroie" in p or "tamper" in p:
        return "doc"
    # default bucket used by SICA starter for leftovers
    return "imdl"


def _norm_label(sample) -> int:
    if "label" in sample:
        return int(sample["label"])
    for k in ("is_fake", "fake", "isfake"):
        if k in sample:
            return int(bool(sample[k]))
    return 0


def _norm_path(sample):
    for k in ("path", "image", "file", "img_path", "filename"):
        if k in sample and sample[k]:
            return str(sample[k])
    return ""


def resolve_image_path(raw_path, manifest_dir, openmmsec_root=None, prefix_from="", prefix_to=""):
    p = raw_path
    if prefix_from and p.startswith(prefix_from):
        p = prefix_to + p[len(prefix_from):]
    if os.path.isabs(p) and os.path.exists(p):
        return p
    # try relative to manifest dir
    cand = os.path.join(str(manifest_dir), p)
    if os.path.exists(cand):
        return cand
    # try relative to OPENMMSEC_ROOT
    if openmmsec_root:
        cand2 = os.path.join(str(openmmsec_root), p.lstrip("/"))
        if os.path.exists(cand2):
            return cand2
        # OpenMMSecV2 images live under OPENMMSEC_ROOT, manifest stores absolute
        # /mnt/... prefix from the release machine -> basename fallback search is
        # expensive, so just return the remapped absolute path and let the
        # Dataset skip missing files with a warning.
    return p


def load_samples(json_path, manifest_dir=None, openmmsec_root=None, prefix_from="", prefix_to=""):
    json_path = Path(json_path)
    manifest_dir = Path(manifest_dir) if manifest_dir else json_path.parent
    with open(json_path, "r") as f:
        data = json.load(f)
    # single v4.json may be {"train": [...], "id_test": [...]} or {"data": [...]}
    if isinstance(data, dict):
        for k in ("train", "data", "samples", "annotations", "v4"):
            if k in data and isinstance(data[k], list):
                data = data[k]
                break
        else:
            # maybe {split: [...]} -> take the longest list
            lists = [v for v in data.values() if isinstance(v, list)]
            if lists:
                data = max(lists, key=len)
            else:
                raise ValueError(f"Unsupported dict manifest format: {json_path}")
    out = []
    for s in data:
        if not isinstance(s, dict):
            continue
        out.append({
            "path": _norm_path(s),
            "label": _norm_label(s),
            "domain": _norm_domain(s),
            "raw": s,
        })
    return out


def find_manifests(manifest_dir=None, train_json=None, openmmsec_root=None):
    """Return dict {train, id_test} paths. Priority: explicit args > jsons_v4 dir > single v4.json."""
    if train_json and os.path.exists(str(train_json)):
        tp = Path(train_json)
        base = tp.parent
        cand_id = base / "id_test.json"
        return {"train": tp, "id_test": cand_id if cand_id.exists() else tp, "base": base}
    cands = []
    if manifest_dir:
        cands.append(Path(manifest_dir))
    if openmmsec_root:
        cands.append(Path(openmmsec_root) / "jsons_v4")
        cands.append(Path(openmmsec_root))
    for c in cands:
        if (c / "train.json").exists():
            idt = c / "id_test.json" if (c / "id_test.json").exists() else (c / "train.json")
            return {"train": c / "train.json", "id_test": idt, "base": c}
        if (c / "v4.json").exists():
            return {"train": c / "v4.json", "id_test": c / "v4.json", "base": c}
    raise FileNotFoundError(
        "Cannot find train.json / v4.json. Pass --train_json or --manifest_dir, "
        f"or set OPENMMSEC_ROOT. Tried: {cands}")
