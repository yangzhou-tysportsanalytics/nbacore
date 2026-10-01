"""L3 ``event_possession.parquet``: every L2 event → the ledger possession it belongs to.

For each offensive-action event of the L2 build (``OFFENSIVE_TYPES``): the ledger possession
containing its key time (``t_contact_ms`` for screen candidates, the release − 1 ms for shots so
that a made shot belongs to the possession it ends, ``t_start_ms`` otherwise), that possession's
``poss_uid`` and offence, and ``offense_match`` = the event's team is the ledger offence.

Why: the L2 handler's controlling team is sometimes the defence for seconds at a time (e.g.
0021500286 Q1 2:50: a defender near the ball > 1 s before a blocked shot); candidates generated
then have screener and user on the defence. In a manual review of 224 screen candidates, those
with ``offense_match`` false were real screens 2 times in 20 judged (vs 82 / 171); season-wide
16.4 % of screen candidates. Consumers should filter candidates on ``offense_match``.

Usage:
    uv run python scripts/build_event_possession.py --l2 scratch/l2_full --l3 scratch/l3_full
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

from nbacore import paths

# events whose team_id is the team performing an offensive action (rim contacts, landings,
# flights, clock stops carry other team meanings and are left out)
OFFENSIVE_TYPES = [
    "screen_candidate",
    "cut_candidate",
    "drive_candidate",
    "postup_candidate",
    "pass",
    "handoff",
    "dribble",
    "possession_touch",
    "shot_release",
    "inbound",
]
SCHEMA = {
    "game_id": pl.Utf8,
    "event_uid": pl.Utf8,
    "event_type": pl.Utf8,
    "period": pl.Int8,
    "t_key_ms": pl.Int64,
    "team_id": pl.Int64,
    "poss_uid": pl.Utf8,
    "offense_team_id": pl.Int64,
    "offense_match": pl.Boolean,
}


def run(args: tuple[str, str, str]) -> pl.DataFrame:
    gid, l2, l3 = args
    led = (
        pl.scan_parquet(Path(l3) / "ledger.parquet")
        .filter((pl.col("game_id") == gid) & pl.col("t_start_ms").is_not_null())
        .select("period", pl.col("t_start_ms").alias("t_key_ms"), "poss_uid", "offense_team_id")
        .collect()
        .sort("period", "t_key_ms")
    )
    ev = pl.read_parquet(Path(l2) / "events" / f"{gid}.parquet")
    key = (
        pl.when(pl.col("event_type") == "screen_candidate")
        .then(pl.col("t_contact_ms"))
        .when(pl.col("event_type") == "shot_release")
        .then(pl.col("t_start_ms") - 1)
        .otherwise(pl.col("t_start_ms"))
    )
    ev = (
        ev.filter(pl.col("team_id").is_not_null() & pl.col("event_type").is_in(OFFENSIVE_TYPES))
        .select("game_id", "event_uid", "event_type", "period", key.alias("t_key_ms"), "team_id")
        .sort("period", "t_key_ms")
    )
    j = ev.join_asof(led, on="t_key_ms", by="period", strategy="backward", check_sortedness=False)
    return (
        j.with_columns((pl.col("team_id") == pl.col("offense_team_id")).alias("offense_match"))
        .select(list(SCHEMA))
        .cast(SCHEMA)
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", default="scratch/l2_full")
    ap.add_argument("--l3", default="scratch/l3_full")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    l2, l3 = paths.data_root() / a.l2, paths.data_root() / a.l3
    games = sorted(p.stem for p in (l2 / "events").glob("*.parquet"))
    with ProcessPoolExecutor(a.workers) as ex:
        parts = list(ex.map(run, [(g, str(l2), str(l3)) for g in games], chunksize=8))
    out = pl.concat(parts)
    out.write_parquet(l3 / "event_possession.parquet", compression="zstd")
    s = (
        out.group_by("event_type")
        .agg(
            pl.len().alias("n"),
            (~pl.col("offense_match").fill_null(False)).mean().alias("mismatch"),
        )
        .sort("n", descending=True)
    )
    print(out.height, "rows")
    print(s)


if __name__ == "__main__":
    main()
