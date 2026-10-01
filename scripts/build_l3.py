"""Build the L3 tables (possession ledger, fouls, free throws, ghost_v1, shot clock, pbp →
possession map).

Inputs: an L1 release (``--base``, frames / frame_index / pbp / attack direction / games) and an
L2 build directory (``--l2``: events, handler, per-game pbp_align) — v1.1 content.

Per game (parallel processes), under ``<data root>/<out>/``:
    _ledger/<game_id>.parquet       ledger + details (``nbacore.ledger_detail``)
    _fouls/<game_id>.parquet        foul states (+ unix_ms from pbp_align)
    _free_throws/<game_id>.parquet  free throw links (+ unix_ms)
    _ghost_v1/<game_id>.parquet     ghost_v1 windows with window_uid / poss_uid
    _pbp_possession/<game_id>.parquet  every pbp row → poss_uid + role (v1.3)
    shot_clock/<game_id>.parquet    per-frame shot clock (observed or imputed, flag)
Then the concatenated ``ledger.parquet``, ``fouls.parquet``, ``free_throws.parquet``,
``ghost_v1.parquet``, ``pbp_possession.parquet`` and ``build_l3_results.json``.

Usage:
    uv run python scripts/build_l3.py --base v1.0 --l2 scratch/l2_full --out scratch/l3_full \
        --workers 3 [--games small]
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
from nbacore.fouls import foul_states, free_throw_links
from nbacore.ledger import (
    ledger_times,
    pbp_possession_estimate,
    pbp_possession_map,
    pbp_possessions,
    possession_estimate_aligned,
)
from nbacore.ledger_detail import possession_details
from nbacore.shot_clock import shot_clock_imputed, shot_clock_resets, shot_clock_table
from nbacore.views.ghost_v1 import SegmentConfig, ghost_v1_windows

PARTS = ("_ledger", "_fouls", "_free_throws", "_ghost_v1", "_pbp_possession", "shot_clock")


def run_game(gid: str, base: str, l2: str, out: str) -> dict:
    t0 = time.time()
    o, l2d = Path(out), Path(l2)
    try:
        pbp = L.pbp(base, game_id=gid)
        fi = L.frame_index(gid, base).sort("period", "unix_ms")
        fr = L.frames(gid, base)
        direction = L.attack_direction(base, game_id=gid)
        home = int(L.games(base).filter(pl.col("game_id") == gid)["home_team_id"][0])
        align = pl.read_parquet(l2d / "_pbp_align" / f"{gid}.parquet")
        events = pl.read_parquet(l2d / "events" / f"{gid}.parquet")
        handler = pl.read_parquet(
            l2d / "handler" / f"{gid}.parquet", columns=["period", "unix_ms", "control_team_id"]
        )
        shots = events.filter(pl.col("event_type") == "shot_release")
        t_of = align.select("event_num", "unix_ms")

        poss = pbp_possessions(pbp)
        led = ledger_times(poss, align, fi, direction, handler=handler, pbp_game=pbp)
        fouls = foul_states(pbp).join(t_of, on="event_num", how="left")
        fts = free_throw_links(pbp).join(t_of, on="event_num", how="left")
        win, rej = ghost_v1_windows(fr, fi, pbp, direction, home, led, shots, SegmentConfig())
        ledger = possession_details(led, pbp, align, fi, fr, events, fts, home, ghost_windows=win)

        ball = fr.filter(pl.col("player_id") == -1).select("period", "unix_ms", "x", "y")
        resets = shot_clock_resets(
            led,
            shots,
            events.filter(pl.col("event_type") == "inbound"),
            fouls,
            pbp,
            align,
            fi,
            ball,
            direction,
        )
        sc = shot_clock_table(fi, shot_clock_imputed(fi, resets))

        ledger.write_parquet(o / "_ledger" / f"{gid}.parquet")
        fouls.write_parquet(o / "_fouls" / f"{gid}.parquet")
        fts.write_parquet(o / "_free_throws" / f"{gid}.parquet")
        win.write_parquet(o / "_ghost_v1" / f"{gid}.parquet")
        pmap = pbp_possession_map(pbp, ledger)
        pmap.write_parquet(o / "_pbp_possession" / f"{gid}.parquet")
        sc.write_parquet(o / "shot_clock" / f"{gid}.parquet", compression="zstd")
        est = pbp_possession_estimate(pbp)
        return {
            "game_id": gid,
            "status": "ok",
            "seconds": round(time.time() - t0, 1),
            "n_possessions": ledger.height,
            "estimate": est,
            "estimate_aligned": possession_estimate_aligned(pbp, poss),
            "poss_uid_unique": ledger["poss_uid"].drop_nulls().is_unique().all(),
            "n_ghost_windows": win.height,
            "ghost_offense_match": int(win["offense_match"].fill_null(False).sum()),
            "ghost_rejects": rej,
            "n_fouls": fouls.height,
            "pbp_rows_unmapped": int(pmap["poss_uid"].is_null().sum()),
            "n_free_throws": fts.height,
            "ft_linked": int(fts["linked_foul_event_num"].is_not_null().sum()),
            "shot_clock_imputed_frac": float(sc["shot_clock_imputed"].mean()),
            "end_time_methods": dict(ledger.group_by("end_time_method").len().iter_rows()),
        }
    except Exception as e:  # noqa: BLE001 - recorded per game
        return {
            "game_id": gid,
            "status": f"failed:{type(e).__name__}",
            "error": f"{e}\n{traceback.format_exc(limit=6)}",
            "seconds": round(time.time() - t0, 1),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v1.0", help="L1 release to read")
    ap.add_argument("--l2", required=True, help="L2 build dir, relative to the data root")
    ap.add_argument("--out", required=True, help="output dir, relative to the data root")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games", default="all", help="all | tiny | small (manifest subsets)")
    args = ap.parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be in 1..4 (16 GB machine)")
    out = paths.data_root() / args.out
    l2 = paths.data_root() / args.l2
    for sub in PARTS:
        (out / sub).mkdir(parents=True, exist_ok=True)
    games = L.games(args.base).filter(pl.col("status") == "ok")
    if args.games != "all":
        names = json.loads((paths.raw_dir() / f"manifest_{args.games}.json").read_text())[
            "archives"
        ]
        games = games.filter(pl.col("archive_name").is_in(names))
    gids = sorted(games["game_id"].to_list())
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_game, g, args.base, str(l2), str(out)) for g in gids]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            msg = r["status"] if r["status"] != "ok" else f"{r['n_possessions']} possessions"
            print(f"[{i}/{len(gids)}] {r['game_id']}: {msg} [{r['seconds']}s]", flush=True)
    ok = {r["game_id"] for r in results if r["status"] == "ok"}
    for part, name in (
        ("_ledger", "ledger"),
        ("_fouls", "fouls"),
        ("_free_throws", "free_throws"),
        ("_ghost_v1", "ghost_v1"),
        ("_pbp_possession", "pbp_possession"),
    ):
        files = [out / part / f"{g}.parquet" for g in sorted(ok)]
        pl.concat([pl.read_parquet(p) for p in files], how="vertical_relaxed").write_parquet(
            out / f"{name}.parquet", compression="zstd"
        )
    (out / "build_l3_results.json").write_text(
        json.dumps(sorted(results, key=lambda r: r["game_id"]), indent=1, default=str)
    )
    bad = [r for r in results if r["status"] != "ok"]
    print(f"{len(results) - len(bad)} ok, {len(bad)} failed, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
