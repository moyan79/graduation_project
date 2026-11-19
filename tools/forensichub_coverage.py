"""Audit every local ForensicHub model declaration against this experiment."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from forensichub_inventory import get_scoped_inventory
from model_config import EXPERIMENT_SCOPE, MODEL_CFG, get_exclusion_reason


def _completed_keys(path):
    if not path.is_file():
        return set()
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    return set(report.get("completed_model_keys", []))


def _local_completed(output_roots, expected_epochs):
    """Accept new runs only when their metadata and diagnostic products agree."""
    completed = {}
    for root in output_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("run_metadata.json"):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                requested = int(metadata["epochs_requested"])
                if (metadata.get("status") != "training_complete"
                        or int(metadata.get("epochs_completed", 0)) < max(expected_epochs, requested)):
                    continue
                run = path.parent
                if not (run / "unified_model_last.pth").is_file():
                    continue
                if "[diag] done." not in (run / "train.log").read_text(encoding="utf-8", errors="replace"):
                    continue
                with (run / "epoch_loss.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                epochs = {int(row["epoch"]) for row in rows
                          if int(row["steps"]) > 0 and math.isfinite(float(row["mean_train_loss"]))}
                if not set(range(requested)).issubset(epochs):
                    continue
                with (run / "grad_log.csv").open(encoding="utf-8", newline="") as stream:
                    if not any(math.isfinite(float(row["mean_cos"])) for row in csv.DictReader(stream)):
                        continue
                products = [run / "rank_report.json", run / "gradient_report.json"]
                cadence = max(1, int(metadata.get("probe_every", 1)))
                probe_epochs = {epoch for epoch in range(requested) if (epoch + 1) % cadence == 0}
                probe_epochs.add(requested - 1)
                products.extend(run / "traj" / f"features_epoch{epoch}" / "rank_report.json"
                                for epoch in probe_epochs)
                valid_products = True
                for product in products:
                    report = json.loads(product.read_text(encoding="utf-8"))
                    if not isinstance(report, dict) or not report:
                        valid_products = False
                        break
                    if product.name == "rank_report.json":
                        if (int(report.get("n", 0)) < 2 or int(report.get("dim", 0)) < 1
                                or not math.isfinite(float(report.get("effective_rank", "nan")))):
                            valid_products = False
                            break
                    elif (int(report.get("steps", 0)) < 1
                          or not math.isfinite(float(report.get("mean_cos_overall", "nan")))):
                        valid_products = False
                        break
                if not valid_products:
                    continue
                key = metadata["model"].lower()
                if key in MODEL_CFG and metadata.get("registry") == MODEL_CFG[key]["registry"]:
                    completed[key] = str(run.resolve())
            except (OSError, ValueError, KeyError, TypeError):
                continue  # partial, corrupt or mismatched products remain pending
    return completed


def coverage_report(completion_report=None, outputs=None, expected_epochs=10):
    inventory, excluded = get_scoped_inventory(scope=EXPERIMENT_SCOPE)
    completion = Path(completion_report or PROJECT / "reports" / "experiment_completion_20261119.json")
    completed = _completed_keys(completion)
    roots = [Path(path).resolve() for path in (outputs or [PROJECT / "outputs", PROJECT.parent / "outputs"])]
    local_completed = _local_completed(roots, expected_epochs)
    configured = {}
    duplicates = {}
    for key, config in MODEL_CFG.items():
        registry = config["registry"]
        if registry in configured:
            duplicates.setdefault(registry, [configured[registry]]).append(key)
        configured[registry] = key
    rows = []
    for registry, source in inventory.items():
        key = configured.get(registry)
        config = MODEL_CFG.get(key, {})
        assets = []
        for settings in (config.get("init", {}), config.get("wrapper_init", {})):
            for name, value in settings.items():
                if not isinstance(value, str) or not value:
                    continue
                if name.endswith(("_path", "_weights")) or name in ("weight_path", "cfg_file", "pretrained_path"):
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        path = PROJECT / path
                    assets.append({"field": name, "path": str(path.resolve()), "exists": path.is_file()})
        checkpoint = config.get("task_checkpoint")
        checkpoint_path = Path(checkpoint).expanduser() if checkpoint else None
        if checkpoint_path and not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT / checkpoint_path
        reason = config.get("blocked_reason", "")
        rows.append({"registry": registry, "model": key,
                     "category": source["category"], "source": str(source["source"]),
                     "declaration": source["declaration"],
                     "adaptation": "missing" if not key else "blocked" if reason else "configured",
                     "blocked_reason": reason,
                     "completion": "completed_by_record" if key in completed else
                                   "completed_by_artifacts" if key in local_completed else "pending_or_unconfirmed",
                     "local_completed_run": local_completed.get(key),
                     "task_checkpoint": str(checkpoint_path.resolve()) if checkpoint_path else None,
                     "task_checkpoint_exists": checkpoint_path.is_file() if checkpoint_path else False,
                     "required_assets": assets,
                     "missing_assets": [asset for asset in assets if not asset["exists"]]})
    missing = sorted(set(inventory) - set(configured))
    unknown = sorted(set(configured) - set(inventory))
    return {"scope": "independent-image forensic declarations; generative LLM/VLM and temporal video models excluded",
            "completion_report": str(completion), "completion_report_exists": completion.is_file(),
            "output_roots": [str(root) for root in roots],
            "counts": {"source_declared": len(inventory) + len(excluded),
                       "declared": len(inventory), "configured": len(MODEL_CFG),
                       "excluded": len(excluded),
                       "completed_by_record": sum(row["completion"] == "completed_by_record" for row in rows),
                       "completed_by_artifacts": sum(row["completion"] == "completed_by_artifacts" for row in rows),
                       "pending_or_unconfirmed": sum(not row["completion"].startswith("completed_") for row in rows),
                       "blocked": sum(row["adaptation"] == "blocked" for row in rows),
                       "missing": len(missing)},
            "excluded_models": [{"registry": name, "category": row["category"],
                                 "source": str(row["source"]), "reason": row["excluded_reason"]}
                                for name, row in excluded.items()],
            "missing_configurations": missing, "unknown_configurations": unknown,
            "duplicate_registries": duplicates, "models": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if any declaration lacks a unique configuration")
    parser.add_argument("--require-runnable", action="store_true", help="also fail when any selected model is blocked")
    parser.add_argument("--json", type=Path, help="write a full machine-readable inventory")
    parser.add_argument("--list", action="store_true", help="print selected configured model keys only")
    parser.add_argument("--pending", action="store_true", help="omit models completed in the explicit completion report")
    parser.add_argument("--runnable-only", action="store_true", help="explicitly select configs without blocked_reason")
    parser.add_argument("--model", nargs="+", help="select specific experiment keys")
    parser.add_argument("--completion-report", type=Path, help="an audited completion record")
    parser.add_argument("--outputs", type=Path, nargs="+", help="also inspect new run metadata and required products under these roots")
    parser.add_argument("--expected-epochs", type=int, default=10)
    args = parser.parse_args()
    if args.expected_epochs < 1:
        parser.error("--expected-epochs must be positive")
    report = coverage_report(args.completion_report, args.outputs, args.expected_epochs)
    if args.pending and not report["completion_report_exists"]:
        parser.error("--pending requires a completion report; absent local outputs do not prove an experiment is unfinished")
    rows = report["models"]
    if args.model:
        for key in args.model:
            reason = get_exclusion_reason(key)
            if reason:
                parser.error(f"Model {key!r} is outside the experiment scope: {reason}")
        unknown = set(args.model) - set(MODEL_CFG)
        if unknown:
            parser.error(f"Unknown model keys: {sorted(unknown)}")
        rows = [row for row in rows if row["model"] in args.model]
    if args.pending:
        rows = [row for row in rows if not row["completion"].startswith("completed_")]
    if args.runnable_only:
        rows = [row for row in rows if row["adaptation"] == "configured"]
    report["selected_models"] = [row["model"] for row in rows if row["model"]]
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.list:
        print("\n".join(report["selected_models"]))
    else:
        counts = report["counts"]
        print("[coverage] " + " ".join(f"{name}={value}" for name, value in counts.items()))
        for row in report["excluded_models"]:
            print(f"[excluded] {row['registry']}: {row['reason']}")
        for row in rows:
            if row["blocked_reason"]:
                print(f"[blocked] {row['model']}: {row['blocked_reason']}")
        if report["missing_configurations"]:
            print(f"[missing] {report['missing_configurations']}")
        if report["unknown_configurations"]:
            print(f"[unknown] {report['unknown_configurations']}")
        if report["duplicate_registries"]:
            print(f"[duplicate] {report['duplicate_registries']}")
    if args.check and (report["missing_configurations"] or report["unknown_configurations"] or report["duplicate_registries"]):
        raise SystemExit(1)
    if args.require_runnable and any(row["adaptation"] != "configured" for row in rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
