"""Pre-distribution checks (raw coordinates and frame tables are never distributed).

1. Every release under ``releases/`` matches its MANIFEST (sha256, no extra / missing files)
   and is read-only.
2. The data root is not inside a git work tree (tracking coordinates never enter git).
3. This code repository tracks no data files (parquet, 7z, raw JSON).
4. ``derived_release/`` (if present, the directory meant for external distribution) contains no
   table with raw coordinate columns (x, y, z).

Exit code 1 on any problem.

Usage:
    uv run python scripts/check_release.py [--version v1.0]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

from nbacore import paths
from nbacore.release import verify_release

REPO = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = (".parquet", ".7z", ".json.gz")
COORD_COLUMNS = {"x", "y", "z"}


def in_git_work_tree(path: Path) -> bool:
    r = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, text=True
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", help="check only this release (default: all)")
    args = ap.parse_args()
    problems: list[str] = []

    rel_root = paths.releases_dir()
    versions = (
        [args.version]
        if args.version
        else sorted(
            p.name for p in rel_root.glob("v*") if p.is_dir() and not p.name.endswith(".partial")
        )
    )
    for v in versions:
        problems += [f"{v}: {p}" for p in verify_release(rel_root / v)]
        print(f"{v}: checked")

    if in_git_work_tree(paths.data_root()):
        problems.append(f"data root {paths.data_root()} is inside a git work tree")

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.split()
    problems += [f"data file tracked by git: {f}" for f in tracked if f.endswith(DATA_SUFFIXES)]
    problems += [f"raw JSON tracked by git: {f}" for f in tracked if f.startswith("data/")]

    derived = paths.data_root() / "derived_release"
    for p in sorted(derived.rglob("*.parquet")) if derived.exists() else []:
        cols = set(pq.read_schema(p).names)
        if cols & COORD_COLUMNS:
            problems.append(
                f"derived_release contains coordinates: {p} {sorted(cols & COORD_COLUMNS)}"
            )

    for p in problems:
        print("PROBLEM:", p)
    print("OK" if not problems else f"{len(problems)} problems")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
