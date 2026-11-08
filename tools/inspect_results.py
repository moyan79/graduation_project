"""Inspect experiment outputs, logs, trajectories and export a JSON summary.

  python tools/inspect_results.py
  python tools/inspect_results.py --view built --model resnet
  python tools/inspect_results.py --outputs /path/to/outputs --view errors
"""
import argparse
from collections import Counter
import csv
import json
import math
import os
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.model_config import MODEL_CFG, MODEL_CATEGORIES

LOG_PATTERNS = {
    "built": r"\[diag\] (model=|built |dumped features_epoch0|feature_hook=)",
    "init": (
        r"random init|pretrain|Downloading:|lazy import|Loading '|Loading weights|"
        r"load_state_dict|loaded task checkpoint|fallback|fall back|missing keys|"
        r"unexpected keys|No checkpoint"
    ),
    "errors": (
        r"Traceback|\bError\b|not found|not registered|failed|Failure|"
        r"out of memory|\bnan\b|assert|exception"
    ),
    "logs": (
        r"\[diag\]|random init|pretrain|not found|Loading|load_state|"
        r"state_dict|warn|fallback|fall back"
    ),
}


def default_outputs():
    # 本地旧结果放在同级 outputs；项目内有结果时优先使用项目目录。
    for path in (PROJECT_ROOT / "outputs", PROJECT_ROOT.parent / "outputs"):
        if path.is_dir():
            return path
    return PROJECT_ROOT / "outputs"


def model_key(name):
    for prefix in ("diag_ddp_", "diag_"):
        if name.startswith(prefix):
            return name[len(prefix):].lower()
    return name.lower()


def experiment_group(directory, key, configs):
    for parent in directory.parents:
        if parent.name in MODEL_CATEGORIES:
            return parent.name
    return configs.get(key, {}).get("category", "unknown")


def find_experiments(root):
    # 找到实验目录后停止向下扫描，避免将 traj 下的逐轮报告当成新实验。
    experiments = []
    markers = {"rank_report.json", "gradient_report.json", "epoch_loss.csv", "train.log"}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        directory = Path(base)
        if markers.intersection(files) or directory.name.startswith(("diag_ddp_", "diag_")):
            experiments.append(directory)
            dirs[:] = []
    return sorted(experiments, key=lambda path: str(path.relative_to(root)))


def read_json(path, errors):
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        return data
    except (OSError, ValueError) as error:
        errors.append(f"{path.name}: {error}")
        return None


def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def read_losses(path, errors):
    if not path.is_file():
        return []
    rows = []
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream), start=2):
                epoch = number(row.get("epoch"))
                if epoch is None:
                    errors.append(f"{path.name}:{index}: invalid epoch")
                    continue
                steps = number(row.get("steps"))
                rows.append({
                    "epoch": int(epoch),
                    "steps": int(steps) if steps is not None else None,
                    "mean_train_loss": number(row.get("mean_train_loss")),
                    "mean_conflict_ratio": number(row.get("mean_conflict_ratio")),
                    "mean_cos": number(row.get("mean_cos")),
                })
    except (OSError, csv.Error) as error:
        errors.append(f"{path.name}: {error}")
    return sorted(rows, key=lambda row: row["epoch"])


def clean_log_line(line):
    return re.sub(r"^.*?\[rank\d+\]:\s*", "", line).strip()


def matching_logs(directory, view, limit):
    path = directory / "train.log"
    if not path.is_file():
        return []
    pattern = re.compile(LOG_PATTERNS[view], re.IGNORECASE)
    matches = []
    seen = set()
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = clean_log_line(line)
            if "it/s" in line or not pattern.search(line) or line in seen:
                continue
            seen.add(line)
            matches.append(line)
            if len(matches) >= limit:
                break
    return matches


def log_metadata(directory):
    path = directory / "train.log"
    metadata = {}
    if not path.is_file():
        return metadata
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            batch = re.search(r"world=(\d+) per_gpu_bs=(\d+) \(global_bs=(\d+)\)", line)
            if batch:
                metadata.update(zip(("world", "per_gpu_bs", "global_bs"), map(int, batch.groups())))
            built = re.search(r"\[diag\] built (\S+) params=([\d.]+)M img=(\d+) norm=(\w+)", line)
            if built:
                metadata.update(registry=built[1], params_m=float(built[2]),
                                image_size=int(built[3]), norm=built[4])
            hook = re.search(r"feature_hook[=:]\s*(.+)", line)
            if hook:
                metadata["feature_hook"] = hook[1].strip()
    return metadata


def read_trajectory(directory, errors):
    trajectory = []
    for path in (directory / "traj").glob("features_epoch*/rank_report.json"):
        epoch = re.search(r"features_epoch(\d+)$", path.parent.name)
        report = read_json(path, errors)
        if epoch and report:
            trajectory.append({"epoch": int(epoch[1]), "rank": report})
    return sorted(trajectory, key=lambda row: row["epoch"])


def rank_metric(report):
    keys = [key for key in (report or {}) if re.fullmatch(r"cumvar_top\d+", key)]
    if not keys:
        return None, None
    key = max(keys, key=lambda name: int(name[len("cumvar_top"):]))
    return key, report[key]


def collect_result(directory, configs):
    key = model_key(directory.name)
    errors = []
    rank = read_json(directory / "rank_report.json", errors)
    gradient = read_json(directory / "gradient_report.json", errors)
    losses = read_losses(directory / "epoch_loss.csv", errors)
    trajectory = read_trajectory(directory, errors)
    metadata = log_metadata(directory)
    metric_key, metric_value = rank_metric(rank)
    first = losses[0]["mean_train_loss"] if losses else None
    last = losses[-1]["mean_train_loss"] if losses else None
    change = None
    if first not in (None, 0) and last is not None:
        change = (last - first) / abs(first) * 100
    images = {
        name: str(directory / name)
        for name in (
            "svd_cumsum.png", "svd_spectrum.png", "grad_conflict_curve.png", "grad_cos_bar.png"
        )
        if (directory / name).is_file()
    }
    registry = metadata.get("registry", configs.get(key, {}).get("registry", key))
    return {
        "dir": directory.name, "path": str(directory), "model_key": key,
        "registry": registry, "category": experiment_group(directory, key, configs),
        "status": "reports_ready" if rank and gradient and losses and not errors else "partial",
        "errors": errors, **metadata,
        "feature_shape": (rank or {}).get("feat_shape"),
        "rank": rank, "rank_metric": metric_value,
        "rank_metric_name": metric_key.replace("cumvar_top", "cumvar@") if metric_key else None,
        "gradient": gradient, "losses": losses, "epochs_recorded": len(losses),
        "loss_change_pct": change,
        "trajectory": trajectory, "images": images,
        "initialization_logs": matching_logs(directory, "init", 30),
    }


def formatted(value):
    if value is None:
        return "-"
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def print_table(headers, rows):
    rows = [[formatted(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def show_summary(results):
    rows = []
    for result in results:
        rank, gradient = result["rank"] or {}, result["gradient"] or {}
        rows.append([
            result["model_key"], result["category"], result["status"], result["epochs_recorded"],
            rank.get("dim"), result["rank_metric"], rank.get("effective_rank"),
            gradient.get("mean_conflict_ratio"), gradient.get("mean_cos_overall"),
        ])
    print_table(
        ["model", "category", "status", "epochs", "dim", "cumvar",
         "eff_rank", "conflict", "mean_cos"],
        rows,
    )


def show_result(result, view, limit):
    directory = Path(result["path"])
    print(f"\n=== {result['model_key']} ({result['category']}) ===\n{directory}")
    for error in result["errors"]:
        print("[error]", error)
    if view in LOG_PATTERNS:
        for line in matching_logs(directory, view, limit):
            print(line)
        if not (directory / "train.log").is_file():
            print("NO LOG")
    elif view == "keys":
        for name in ("rank", "gradient"):
            print(name, list((result[name] or {}).keys()))
        rank = result["rank"] or {}
        print("n", rank.get("n"), "dim", rank.get("dim"), "feature_shape", result["feature_shape"])
    elif view == "metrics":
        print("RANK", json.dumps(result["rank"], ensure_ascii=False))
        print("GRAD", json.dumps(result["gradient"], ensure_ascii=False))
        print("LOSS", json.dumps(result["losses"], ensure_ascii=False))
    elif view == "trajectory":
        rows = []
        for row in result["trajectory"]:
            rank = row["rank"]
            metric_key, value = rank_metric(rank)
            count = int(metric_key[len("cumvar_top"):]) if metric_key else None
            kappa = rank.get(f"kappa_s0_s{count - 1}") if count else None
            rows.append([
                row["epoch"], rank.get("dim"), value, rank.get("effective_rank"), kappa,
            ])
        if rows:
            print_table(["epoch", "dim", "cumvar", "eff_rank", "kappa"], rows)
        else:
            print("NO TRAJECTORY")
        print("loss_change_pct", formatted(result["loss_change_pct"]))
    elif view == "details":
        for name in (
            "rank_report.json", "gradient_report.json", "epoch_loss.csv",
            "train.log", "grad_log.csv"
        ):
            path = directory / name
            print(f"\n-- {name} --")
            if not path.is_file():
                print("MISSING")
                continue
            with path.open(encoding="utf-8", errors="replace") as stream:
                for index, line in enumerate(stream):
                    if index >= limit:
                        print("...")
                        break
                    print(line.rstrip())


def show_files(results):
    count, size = 0, 0
    suffixes = Counter()
    for result in results:
        for base, dirs, files in os.walk(result["path"], followlinks=False):
            for name in files:
                path = Path(base) / name
                count += 1
                size += path.lstat().st_size
                suffixes[path.suffix.lower() or "(none)"] += 1
    print(f"experiments={len(results)} files={count} bytes={size} GB={size / 1024**3:.3f}")
    print(json.dumps(dict(sorted(suffixes.items())), indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--outputs", type=Path, default=default_outputs(),
                        help="experiment output root or one experiment")
    views = ("summary", *LOG_PATTERNS, "keys", "metrics", "trajectory", "details", "files")
    parser.add_argument("--view", default="summary", choices=views)
    parser.add_argument("--model", nargs="+", help="model keys or experiment directory names")
    parser.add_argument("--category", choices=MODEL_CATEGORIES, help="filter one model category")
    parser.add_argument("--limit", type=int, default=30,
                        help="maximum log/detail lines per experiment")
    parser.add_argument("--export", type=Path, help="write a fresh JSON summary to this path")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    root = args.outputs.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Outputs directory does not exist: {root}; pass --outputs explicitly")
    selected = {model_key(name) for name in args.model} if args.model else None
    directories = [
        path for path in find_experiments(root)
        if selected is None or model_key(path.name) in selected
    ]
    results = [collect_result(path, MODEL_CFG) for path in directories]
    if args.category:
        results = [result for result in results if result["category"] == args.category]
    if not results:
        parser.error("No matching experiment directories found")
    print("Outputs:", root)
    if args.view == "summary":
        show_summary(results)
    elif args.view == "files":
        show_files(results)
    else:
        for result in results:
            show_result(result, args.view, args.limit)
    if args.export:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print("WROTE", args.export.resolve())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
