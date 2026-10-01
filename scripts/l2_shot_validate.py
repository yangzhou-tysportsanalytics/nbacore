"""Validate L2 shot events (L2 acceptance: release method distribution, release − pbp quantiles).

* Parity: for ghost-defense possessions ending in a FG attempt, ``t_release_ms`` must equal
  ghost's ``t_terminal`` (same event_num).
* Method distribution, release − pbp quantiles, rim contact share, offensive rebound rate,
  shooter_inferred == pbp shooter.

Usage:
    uv run python scripts/l2_shot_validate.py [--games small] [--ghost-processed ...]
"""

from __future__ import annotations

import argparse

import polars as pl

import nbacore.load as L
from nbacore.events.handler import game_handler
from nbacore.events.shots import shot_events
from l2_pass_validate import game_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="small")
    ap.add_argument("--version", default="v1.0")
    ap.add_argument("--ghost-processed", default="../ghost-defense/data/processed")
    args = ap.parse_args()
    shots = []
    for gid in game_ids(args.games, args.version):
        fr = L.frames(gid, args.version)
        fi = L.frame_index(gid, args.version)
        h = game_handler(fr, fi)
        shots.append(
            shot_events(
                fr,
                fi,
                L.pbp(args.version, game_id=gid),
                L.attack_direction(args.version, game_id=gid),
                h,
            )
        )
    s = pl.concat(shots)
    n = s.height
    print(f"FG attempts: {n}; methods:")
    for m, c in s.group_by("method").len().sort("len", descending=True).iter_rows():
        print(f"  {m:12s} {c:6d} {c / n:.3f}")
    q = s["release_minus_pbp_s"].drop_nulls()
    print(
        "release - pbp (s) quantiles 5/25/50/75/95 %:",
        [round(q.quantile(x), 2) for x in (0.05, 0.25, 0.5, 0.75, 0.95)],
    )
    print(f"rim contact found: {s['t_rim_ms'].is_not_null().mean():.3f}")
    miss = s.filter(~pl.col("made") & pl.col("offensive_rebound").is_not_null())
    print(
        f"offensive rebound rate (misses with a control): {miss['offensive_rebound'].mean():.3f} (n={miss.height})"
    )
    ok = s.filter(pl.col("shooter_inferred_id").is_not_null())
    print(
        f"shooter_inferred == pbp shooter: {(ok['shooter_inferred_id'] == ok['shooter_id']).mean():.4f}"
    )
    try:
        g = pl.read_parquet(f"{args.ghost_processed}/possessions.parquet").filter(
            pl.col("terminal_type").is_in(["fg_made", "fg_missed"])
        )
    except FileNotFoundError:
        return
    j = g.join(
        s,
        left_on=["game_id", "terminal_event_num"],
        right_on=["game_id", "pbp_event_num"],
        how="left",
    )
    eq = (j["t_terminal"] == j["t_release_ms"]).fill_null(False)
    dup = (j["method"] == "duplicate_release").fill_null(False)
    print(
        f"parity with ghost t_terminal: {int(eq.sum())}/{j.height}; explained by late-inserted "
        f"pbp rows (duplicate_release, ghost reuses an earlier release): {int((~eq & dup).sum())}"
    )
    if not eq.all():
        print(
            j.filter(~eq)
            .select(
                "game_id",
                "terminal_event_num",
                "t_terminal",
                "t_release_ms",
                "shot_time_method",
                "method",
            )
            .head(10)
        )


if __name__ == "__main__":
    main()
