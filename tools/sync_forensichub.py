"""Apply the project's ForensicHub fixes while preserving unrelated checkout edits."""
import argparse
from pathlib import Path
import re
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
PATCH = PROJECT / "patches" / "forensichub-image-adapters.patch"


def sync(repo, check_only=False):
    repo = repo.resolve()
    if not (repo / ".git").exists():
        raise ValueError(f"ForensicHub Git checkout not found: {repo}")
    chunks = re.split(r"(?=^diff --git )", PATCH.read_text(encoding="utf-8"), flags=re.MULTILINE)
    planned = []
    for chunk in filter(str.strip, chunks):
        match = re.match(r"diff --git a/(\S+) b/(\S+)", chunk)
        if not match or match[1] != match[2]:
            raise ValueError("Dependency patch must modify existing files without renames")
        name = match[2]
        target = (repo / name).resolve()
        if not target.is_relative_to(repo) or not name.startswith("ForensicHub/"):
            raise ValueError(f"Patch target outside the ForensicHub package: {name}")
        command = ["git", "-C", str(repo), "apply", "--ignore-space-change", "--ignore-whitespace"]
        forward = subprocess.run(command + ["--check", "-"], input=chunk,
                                 text=True, encoding="utf-8", capture_output=True)
        if forward.returncode == 0:
            planned.append((name, command, chunk))
            continue
        reverse = subprocess.run(command + ["--reverse", "--check", "-"], input=chunk,
                                 text=True, encoding="utf-8", capture_output=True)
        if reverse.returncode == 0:
            print(f"[dependency] already applied: {name}")
            continue
        # The A40 checkout already contains these additions with extra blank lines.
        if name == "ForensicHub/lazy_maps.py":
            source = target.read_text(encoding="utf-8")
            required = ("from ForensicHub.tasks.fidl.models import _lazy_model_map as _fidl_model_map",
                        "from ForensicHub.tasks.fidl.models import _lazy_postfunc_map as _fidl_postfunc_map",
                        "model_map.update(_fidl_model_map)",
                        "postfunc_map.update(_fidl_postfunc_map)")
            if all(line in source for line in required):
                print(f"[dependency] already applied: {name}")
                continue
        raise RuntimeError(f"ForensicHub has conflicting edits in {name}; no patches applied.\n"
                           f"{forward.stderr.strip()}")
    # All files are checked before applying any changes.
    for name, command, chunk in planned:
        if not check_only:
            subprocess.run(command + ["-"], input=chunk, text=True, encoding="utf-8", check=True)
        print(f"[dependency] {'ready' if check_only else 'applied'}: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=PROJECT.parent / "ForensicHub")
    parser.add_argument("--check", action="store_true", help="check compatibility without modifying files")
    args = parser.parse_args()
    sync(args.repo, args.check)


if __name__ == "__main__":
    main()
