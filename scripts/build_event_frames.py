"""Distance snapshots at key event frames for every game of an L2 build.

Reads ``<build>/events/<game_id>.parquet`` and the L1 frames of the base release; writes
``<build>/event_frames/<game_id>.parquet`` (``nbacore.geometry.event_frame_distances``).

Usage:
    uv run python scripts/build_event_frames.py --l2-build scratch/l2_full [--base v1.0] [--workers 4]
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.geometry import event_frame_distances


def run(args: tuple[str, str, str]) -> tuple[str, int]:
    gid, base, l2 = args
    ev = pl.read_parquet(Path(l2) / "events" / f"{gid}.parquet")
    d = event_frame_distances(L.frames(gid, base), ev)
    d.write_parquet(Path(l2) / "event_frames" / f"{gid}.parquet", compression="zstd")
    return gid, d.height


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-build", required=True)
    ap.add_argument("--base", default="v1.0")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    l2 = paths.data_root() / args.l2_build
    (l2 / "event_frames").mkdir(exist_ok=True)
    gids = sorted(p.stem for p in (l2 / "events").glob("*.parquet"))
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        n = sum(h for _, h in ex.map(run, [(g, args.base, str(l2)) for g in gids]))
    print(f"{len(gids)} games, {n} snapshots, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
