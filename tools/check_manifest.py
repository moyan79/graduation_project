"""Quickly inspect OpenMMSecV2 manifests (counts per domain / label, path resolving).

  python tools/check_manifest.py --manifest_dir $OPENMMSEC_ROOT/jsons_v4
  python tools/check_manifest.py --train_json /path/to/v4.json
"""
import argparse
import collections
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from manifest import find_manifests, load_samples

p = argparse.ArgumentParser()
p.add_argument("--manifest_dir", default=os.environ.get("MANIFEST_DIR", None))
p.add_argument("--train_json", default=os.environ.get("TRAIN_JSON", None))
p.add_argument("--openmmsec_root", default=os.environ.get("OPENMMSEC_ROOT"))
p.add_argument("--limit", type=int, default=3)
a = p.parse_args()

ms = find_manifests(a.manifest_dir, a.train_json, a.openmmsec_root)
for k, v in ms.items():
    print(f"{k}: {v}")
train = load_samples(ms["train"])
print(f"train samples: {len(train)}")
print("domain x label:", collections.Counter((s["domain"], s["label"]) for s in train))
for s in train[:a.limit]:
    print(s["domain"], s["label"], s["path"], "exists=", os.path.exists(s["path"]))
idt = load_samples(ms["id_test"])
print(f"id_test samples: {len(idt)}")
print("domain x label:", collections.Counter((s["domain"], s["label"]) for s in idt))

