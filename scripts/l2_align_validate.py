"""Validate pbp alignment and clock stops (L2 acceptance: stop residuals vs pbp fouls, timeouts,
violations).

Usage:
    uv run python scripts/l2_align_validate.py [--games small] [--version v1.0]
"""

from __future__ import annotations

import argparse

import polars as pl

import nbacore.load as L
from nbacore.events.clock_stop import clock_stops
from nbacore.events.flight import ball_flights
from nbacore.events.handler import game_handler
from nbacore.events.passes import mark_after_made_fg, mark_shots, passes
from nbacore.events.pbp_align import align_game
from nbacore.events.shots import shot_events
from l2_pass_validate import game_ids

NAMES = {1: "FG made", 2: "FG missed", 3: "free throw", 4: "rebound", 5: "turnover", 6: "foul",
         7: "violation", 8: "substitution", 9: "timeout", 10: "jump ball", 12: "period start",
         13: "period end", 18: "replay"}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="small")
    ap.add_argument("--version", default="v1.0")
    args = ap.parse_args()
    al, st, short_all = [], [], {}
    for gid in game_ids(args.games, args.version):
        v = args.version
        fr, fi = L.frames(gid, v), L.frame_index(gid, v)
        pb = L.pbp(v, game_id=gid)
        h = game_handler(fr, fi)
        fl = ball_flights(h, fr)
        sh = shot_events(fr, fi, pb, L.attack_direction(v, game_id=gid), h)
        ps = mark_shots(mark_after_made_fg(passes(h, fl, fr, fi), sh, h), sh)
        stops, short = clock_stops(fr, fi)
        for k, n in short.items():
            short_all[k] = short_all.get(k, 0) + n
        a, s2, _ = align_game(pb, fi, sh, stops, fl, ps)
        al.append(a)
        st.append(s2)
    a, s = pl.concat(al), pl.concat(st)
    print(
        f"pbp events {a.height}; stops {s.height} (exact onset {s['onset_exact'].mean():.3f}); "
        f"freezes shorter than 0.2 s by frames: {dict(sorted(short_all.items()))}"
    )
    t = (
        a.group_by("msg_type", "method")
        .len()
        .with_columns(pl.col("msg_type").replace_strict(NAMES, default=None).alias("name"))
        .sort("msg_type", "len", descending=[False, True])
    )
    with pl.Config(tbl_rows=60):
        print(t)
        print(
            a.group_by("msg_type")
            .agg(
                pl.len(),
                (pl.col("anchor_quality") >= 2).mean().alias("tracking_event"),
                (pl.col("anchor_quality") == 0).mean().alias("unlocated"),
            )
            .sort("msg_type")
        )
    for msg in (6, 9, 7):
        r = a.filter((pl.col("msg_type") == msg) & (pl.col("method") == "clock_stop"))["residual_s"]
        if r.len():
            print(
                f"{NAMES[msg]:10s} residual (pbp s - clock at stop onset) quantiles 5/50/95 %: "
                f"{[round(r.quantile(x), 2) for x in (0.05, 0.5, 0.95)]} (n={r.len()})"
            )
    fts = a.filter(pl.col("msg_type") == 3)
    print(f"free throws anchored on a FT flight: {(fts['method'] == 'ft_flight').mean():.3f}")
    reb = a.filter(pl.col("msg_type") == 4)
    print(
        f"rebounds on a tracking event: {(reb['anchor_quality'] >= 2).mean():.3f} "
        f"(control {(reb['method'] == 'control').mean():.3f}, "
        f"FT dead ball {(reb['method'] == 'ft_dead_ball').mean():.3f})"
    )
    to = a.filter(pl.col("msg_type") == 5)
    print(f"turnovers anchored on an interception: {(to['method'] == 'pass').mean():.3f}")
    print(s.group_by("stop_cause_hint").len().sort("len", descending=True))


if __name__ == "__main__":
    main()
