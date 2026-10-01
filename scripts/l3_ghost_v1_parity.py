"""L3 acceptance, part 1: ``nbacore.views.ghost_v1`` reproduces ghost-defense's possessions.

Runs ``ghost_v1`` on nbacore inputs and compares it, game by game and column by column, with
ghost-defense's own output (read only) ``<ghost>/data/processed/nbacore-v1.0/<set>/possessions/
<game_id>.parquet``. Reports rows only on one side, per-column mismatches, and the share of
windows whose t_start / t_end agree within one frame (acceptance: ≥ 99 %).

Usage:
    uv run python scripts/l3_ghost_v1_parity.py [--set small|all] [--games N] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore.views.ghost_v1 import SegmentConfig, ghost_v1

GHOST = Path("../ghost-defense/data/processed/nbacore-v1.0")
REPO = Path(__file__).resolve().parents[1]
KEY = "terminal_event_num"
FRAME_MS = 40


def run_game(args: tuple[str, str]) -> dict:
    gid, game_set = args
    ref = pl.read_parquet(GHOST / game_set / "possessions" / f"{gid}.parquet")
    home = L.games().filter(pl.col("game_id") == gid)["home_team_id"][0]
    ours, rej = ghost_v1(
        L.frames(gid),
        L.frame_index(gid),
        L.pbp(game_id=gid),
        L.attack_direction(game_id=gid),
        int(home),
        SegmentConfig(),
    )
    j = ref.join(ours, on=KEY, how="full", suffix="_ours", coalesce=True)
    only_ref = j.filter(pl.col("possession_id_ours").is_null()).height
    only_ours = j.filter(pl.col("possession_id").is_null()).height
    both = j.filter(
        pl.col("possession_id").is_not_null() & pl.col("possession_id_ours").is_not_null()
    )
    mism = {}
    for c in ref.columns:
        if c == KEY or f"{c}_ours" not in both.columns:
            continue
        a, b = both[c], both[f"{c}_ours"]
        diff = ~((a == b).fill_null(False) | (a.is_null() & b.is_null()))
        if diff.any():
            mism[c] = int(diff.sum())
    within = both.filter(
        ((pl.col("t_start") - pl.col("t_start_ours")).abs() <= FRAME_MS)
        & ((pl.col("t_end") - pl.col("t_end_ours")).abs() <= FRAME_MS)
    ).height
    return {
        "game_id": gid,
        "n_ref": ref.height,
        "n_ours": ours.height,
        "only_ref": only_ref,
        "only_ours": only_ours,
        "both": both.height,
        "within_1_frame": within,
        "mismatch_columns": mism,
        "rejects": rej,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="small")
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    games = sorted(p.stem for p in (GHOST / a.set / "possessions").glob("*.parquet"))
    games = games[: a.games] if a.games else games
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_game, [(g, a.set) for g in games]))
    n_ref = sum(r["n_ref"] for r in res)
    within = sum(r["within_1_frame"] for r in res)
    cols: dict[str, int] = {}
    for r in res:
        for c, k in r["mismatch_columns"].items():
            cols[c] = cols.get(c, 0) + k
    summary = {
        "set": a.set,
        "games": len(games),
        "ghost_windows": n_ref,
        "nbacore_windows": sum(r["n_ours"] for r in res),
        "only_ghost": sum(r["only_ref"] for r in res),
        "only_nbacore": sum(r["only_ours"] for r in res),
        "t_start_t_end_within_1_frame": within,
        "within_1_frame_frac": within / n_ref if n_ref else None,
        "mismatch_columns": cols,
        "games_with_differences": [
            r["game_id"] for r in res if r["only_ref"] or r["only_ours"] or r["mismatch_columns"]
        ],
    }
    print(json.dumps(summary, indent=1))
    out = REPO / "reports" / f"p3_ghost_v1_parity_{a.set}.json"
    out.write_text(json.dumps({**summary, "per_game": res}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
