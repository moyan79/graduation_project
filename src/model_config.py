"""Load model definitions and training defaults without importing model code."""
import argparse
import json
from pathlib import Path
import sys

import yaml

try:
    from .forensichub_inventory import exclusion_reason, load_experiment_scope
except ImportError:
    from forensichub_inventory import exclusion_reason, load_experiment_scope


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
MODEL_CATEGORIES = ("backbone", "deepfake", "imdl", "aigc", "document")


def load_config(filename):
    path = CONFIG_DIR / filename
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a configuration mapping: {path}")
    return config


MODEL_CFG = load_config("models.yaml")
TRAIN_DEFAULTS = load_config("unified_diag.yaml")
EXPERIMENT_SCOPE = load_experiment_scope()


def get_exclusion_reason(model, config=None):
    return exclusion_reason(model, config, EXPERIMENT_SCOPE)

for model, config in MODEL_CFG.items():
    reason = get_exclusion_reason(model, config)
    if reason:
        raise ValueError(f"Model {model!r} is outside the experiment scope: {reason}")
    if config.get("category") not in MODEL_CATEGORIES:
        raise ValueError(f"Invalid category for model {model!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true", help="list configured model keys")
    selection.add_argument("--model", help="model key")
    parser.add_argument("--field", nargs="+", help="print scalar fields, e.g. category ddp.lr")
    args = parser.parse_args()
    if args.list:
        if args.field:
            parser.error("--field requires --model")
        print("\n".join(MODEL_CFG))
        return
    key = args.model.lower()
    reason = get_exclusion_reason(key)
    if reason:
        parser.error(f"Model {args.model!r} is outside the experiment scope: {reason}")
    if key not in MODEL_CFG:
        parser.error(f"Unknown model {args.model!r}; choose from {', '.join(MODEL_CFG)}")
    config = MODEL_CFG[key]
    if not args.field:
        print(json.dumps(config, ensure_ascii=False, indent=2))
        return
    values = []
    for field in args.field:
        value = config
        try:
            for part in field.split("."):
                value = value[part]
        except (KeyError, TypeError):
            parser.error(f"Unknown field {field!r} for model {key!r}")
        if not isinstance(value, (str, int, float, bool)):
            parser.error(f"Field {field!r} is not a scalar")
        values.append(str(value))
    print(" ".join(values))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
    main()
