"""Copy SportVU archives that other projects already downloaded into ``raw/sportvu``.

Copies (never moves, so the source projects keep theirs) every ``*.7z`` in the given
directories whose name is in the GitHub listing and that is not yet in ``raw/sportvu``, then
checks the copy against the listing's size and git blob SHA-1. Sources that fail the check are
reported and not copied.

Usage:
    uv run python scripts/import_archives.py <project A>/data/raw/sportvu <project B>/data/raw/sportvu
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from nbacore import paths
from nbacore.io.listing import load_listing, verify_archive


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="directories holding .7z archives")
    args = ap.parse_args()

    listing = {g["name"]: g for g in load_listing()}
    dest = paths.sportvu_dir()
    dest.mkdir(parents=True, exist_ok=True)
    n_copied = n_present = n_unknown = n_bad = 0
    for src_dir in map(Path, args.sources):
        for src in sorted(src_dir.glob("*.7z")):
            item = listing.get(src.name)
            if item is None:
                n_unknown += 1
                print(f"not in listing, skipped: {src}")
                continue
            out = dest / src.name
            if out.exists():
                if not verify_archive(out, item):
                    raise RuntimeError(f"existing archive fails verification: {out}")
                n_present += 1
                continue
            if not verify_archive(src, item):
                n_bad += 1
                print(f"source fails verification, not copied: {src}")
                continue
            tmp = out.with_suffix(".7z.part")
            shutil.copy2(src, tmp)
            if not verify_archive(tmp, item):
                tmp.unlink()
                raise RuntimeError(f"copy fails verification: {src}")
            tmp.replace(out)
            n_copied += 1
    n_total = len(list(dest.glob("*.7z")))
    print(
        f"copied {n_copied}, already present {n_present}, not in listing {n_unknown}, "
        f"bad source {n_bad}; raw/sportvu now holds {n_total} of {len(listing)} archives"
    )


if __name__ == "__main__":
    main()
