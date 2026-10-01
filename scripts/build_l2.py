"""Build the L2 tables (events, handler, pbp alignment) from a published L1 release.

Per game (``nbacore.events.build.build_game_l2``), in parallel processes:
    handler/<game_id>.parquet   25 Hz handler table
    events/<game_id>.parquet    long event table
    _results/<game_id>.json     per-game counts and timing (``--resume`` skips games that have one)
Then:
    pbp_align.parquet           all games
    build_l2_results.json       per-game counts, timings, errors

Usage:
    uv run python scripts/build_l2.py --base v1.0 --out scratch/l2_full --workers 4 [--games small]
        [--resume]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.events.build import build_game_l2


def run_game(gid: str, base: str, out: str) -> dict:
    t0 = time.time()
    out_dir = Path(out)
    try:
        fr, fi = L.frames(gid, base), L.frame_index(gid, base)
        r = build_game_l2(fr, fi, L.pbp(base, game_id=gid), L.attack_direction(base, game_id=gid))
        r["handler"].write_parquet(out_dir / "handler" / f"{gid}.parquet", compression="zstd")
        r["events"].write_parquet(out_dir / "events" / f"{gid}.parquet", compression="zstd")
        r["pbp_align"].write_parquet(out_dir / "_pbp_align" / f"{gid}.parquet")
        res = {"game_id": gid, "status": "ok", "seconds": round(time.time() - t0, 1), **r["stats"]}
        (out_dir / "_results" / f"{gid}.json").write_text(json.dumps(res))
        return res
    except Exception as e:  # noqa: BLE001 - recorded per game
        return {
            "game_id": gid,
            "status": f"failed:{type(e).__name__}",
            "error": f"{e}\n{traceback.format_exc(limit=4)}",
            "seconds": round(time.time() - t0, 1),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v1.0", help="L1 release to read")
    ap.add_argument("--out", required=True, help="output dir, relative to the data root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--games", default="all", help="all | tiny | small (manifest subsets)")
    ap.add_argument("--resume", action="store_true", help="skip games already built in --out")
    args = ap.parse_args()
    if not 1 <= args.workers <= 6:
        raise SystemExit("--workers must be in 1..6")
    out = paths.data_root() / args.out
    for sub in ("handler", "events", "_pbp_align", "_results"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    games = L.games(args.base).filter(pl.col("status") == "ok")
    if args.games != "all":
        names = json.loads((paths.raw_dir() / f"manifest_{args.games}.json").read_text())[
            "archives"
        ]
        games = games.filter(pl.col("archive_name").is_in(names))
    gids = sorted(games["game_id"].to_list())
    results = []
    if args.resume:
        done = {p.stem for p in (out / "_results").glob("*.json")}
        results = [
            json.loads((out / "_results" / f"{g}.json").read_text()) for g in gids if g in done
        ]
        gids = [g for g in gids if g not in done]
        print(f"resume: {len(results)} done, {len(gids)} to build", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_game, g, args.base, str(out)) for g in gids]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            msg = (
                r["status"]
                if r["status"] != "ok"
                else f"{sum(n for _, n in r['n_events'])} events, dup uids {r['uid_duplicates']}"
            )
            print(f"[{i}/{len(gids)}] {r['game_id']}: {msg} [{r['seconds']}s]", flush=True)
    parts = sorted(
        out / "_pbp_align" / f"{r['game_id']}.parquet" for r in results if r["status"] == "ok"
    )
    pl.concat([pl.read_parquet(p) for p in parts]).sort("game_id", "event_num").write_parquet(
        out / "pbp_align.parquet"
    )
    (out / "build_l2_results.json").write_text(
        json.dumps(sorted(results, key=lambda r: r["game_id"]), indent=1)
    )
    bad = [r for r in results if r["status"] != "ok"]
    print(f"{len(results) - len(bad)} ok, {len(bad)} failed, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
