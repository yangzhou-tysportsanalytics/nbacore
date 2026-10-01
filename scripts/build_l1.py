"""Build L1 (+ the L4 split) for a set of raw games.

Per game (``nbacore.build.process_archive``, same processing as ghost-defense
``scripts/phase1_clean.py`` at ed6d754): entity-level dedup, frame index, rosters, tracking
events, attack direction, raw-level diagnostics. Games run in parallel processes.

Outputs under ``<data root>/<out>/``:
    frames/<game_id>.parquet, frame_index/<game_id>.parquet
    rosters.parquet, tracking_events.parquet, attack_direction.parquet
    pbp.parquet                      typed play-by-play of the whole season
    games.parquet                    (only with --all) games table + split / fold / parity
    build_results.json               per-archive status, dedup / raw / direction diagnostics

Games: ``--config tiny|small|...`` reads ``raw/manifest_<config>.json`` (written by
``scripts/download_raw.py``); ``--all`` takes every archive of the GitHub listing.

Usage:
    uv run python scripts/build_l1.py --all --out scratch/l1_full --workers 4
    uv run python scripts/build_l1.py --config tiny --out scratch/l1_tiny
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

from nbacore import paths
from nbacore.build import process_archive
from nbacore.catalog import build_games
from nbacore.io import load_pbp
from nbacore.io.listing import load_listing
from nbacore.splits import assign_splits


def select_archives(config: str | None) -> list[Path]:
    raw = paths.sportvu_dir()
    if config is None:
        names = [g["name"] for g in load_listing()]
    else:
        manifest = paths.raw_dir() / f"manifest_{config}.json"
        names = json.loads(manifest.read_text(encoding="utf-8"))["archives"]
    missing = [n for n in names if not (raw / n).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} archives missing from {raw}: {missing[:3]}")
    return sorted(raw / n for n in names)


def write_games(results: list[dict], pbp: pl.DataFrame, out: Path) -> None:
    games, problems = build_games(results, pbp)
    games = assign_splits(games)
    games.write_parquet(out / "games.parquet")
    for p in problems:
        print("PROBLEM:", p)
    print(games.group_by("status").len().sort("status"))


def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--config", help="subset name with a raw/manifest_<config>.json")
    grp.add_argument("--all", action="store_true", help="every archive of the listing")
    ap.add_argument("--out", required=True, help="output dir, relative to the data root")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes (<= 6)")
    ap.add_argument("--force", action="store_true", help="rewrite existing per-game parquet")
    ap.add_argument(
        "--games-only",
        action="store_true",
        help="with --all: rebuild games.parquet from an existing build_results.json + pbp.parquet",
    )
    args = ap.parse_args()
    if not 1 <= args.workers <= 6:
        raise SystemExit("--workers must be in 1..6 (memory: ~1.5-2 GB per process)")

    t_start = time.time()
    out = paths.data_root() / args.out
    if args.games_only:
        if not args.all:
            raise SystemExit("--games-only needs --all")
        results = json.loads((out / "build_results.json").read_text(encoding="utf-8"))
        write_games(results, pl.read_parquet(out / "pbp.parquet"), out)
        return
    out.mkdir(parents=True, exist_ok=True)
    pbp = load_pbp(paths.pbp_csv())
    pbp_path = out / "pbp.parquet"
    pbp.write_parquet(pbp_path, compression="zstd")

    archives = select_archives(None if args.all else args.config)
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process_archive, a, out, pbp_path, args.force) for a in archives]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if r["status"] == "ok":
                d, dd = r["direction"], r["dedup"]
                msg = (
                    f"{r['game_id']}: frames {dd['raw_moments']}->{dd['unique_frames']} "
                    f"(dup {dd['moment_dup_fraction']:.3f}) gaps={dd['n_gaps']} "
                    f"consistency={d['overall_consistency']:.3f}"
                )
            else:
                msg = f"{r['archive_name']}: {r['status']} {r.get('error', '')[:200]}"
            print(f"[{i}/{len(archives)}] {msg} [{r['seconds']}s]", flush=True)

    results.sort(key=lambda r: r["archive_name"])
    ok = [r for r in results if r["status"] == "ok"]
    for name in ("rosters", "tracking_events", "attack_direction"):
        pl.concat([r["tables"][name] for r in ok]).sort(
            "game_id", maintain_order=True
        ).write_parquet(out / f"{name}.parquet")
    if args.all:
        write_games(results, pbp, out)
    report = [{k: v for k, v in r.items() if k != "tables"} for r in results]
    (out / "build_results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    n_status = pl.Series("status", [r["status"] for r in results]).value_counts().sort("count")
    print(n_status)
    print(f"wrote {len(ok)} games to {out} in {time.time() - t_start:.0f} s")


if __name__ == "__main__":
    main()
