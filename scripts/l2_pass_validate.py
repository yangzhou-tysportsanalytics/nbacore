"""Validate L2 passes on a set of games (L2 acceptance: assist coverage).

* Assist coverage: for every made FG with an assist in pbp (msg 1, PLAYER2 > 0), is there a
  completed pass or handoff from the assister to the shooter whose catch lies within the game
  clock window [pbp second - 1 s, pbp second + 12 s] (clock counts down; pbp is ~2 s late)?
* Interceptions vs pbp steals (turnovers with a PLAYER2) per game.
* Counts by kind, uid uniqueness.

Usage:
    uv run python scripts/l2_pass_validate.py [--games small|tiny|all] [--version v1.0]
"""

from __future__ import annotations

import argparse
import json

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.events.flight import ball_flights
from nbacore.events.handler import game_handler
from nbacore.events.passes import mark_after_made_fg, mark_shots, passes
from nbacore.events.shots import shot_events


def game_ids(which: str, version: str) -> list[str]:
    if which == "all":
        return L.games(version).filter(pl.col("status") == "ok")["game_id"].to_list()
    names = json.loads((paths.raw_dir() / f"manifest_{which}.json").read_text())["archives"]
    g = L.games(version).filter(pl.col("archive_name").is_in(names))
    return sorted(g["game_id"].to_list())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="small")
    ap.add_argument("--version", default="v1.0")
    args = ap.parse_args()
    tot = {
        "assists": 0,
        "covered": 0,
        "steals": 0,
        "intercepted": 0,
        "rows": 0,
        "dup_uid": 0,
        "dead": 0,
    }
    kinds: dict = {}
    for gid in game_ids(args.games, args.version):
        fr = L.frames(gid, args.version)
        fi = L.frame_index(gid, args.version)
        h = game_handler(fr, fi)
        p = passes(h, ball_flights(h, fr), fr, fi)
        sh = shot_events(
            fr,
            fi,
            L.pbp(args.version, game_id=gid),
            L.attack_direction(args.version, game_id=gid),
            h,
        )
        p = mark_shots(mark_after_made_fg(p, sh, h), sh)
        tot["is_shot"] = tot.get("is_shot", 0) + int(p["is_shot"].sum())
        tot["after_made"] = tot.get("after_made", 0) + int(p["after_made_fg"].sum())
        tot["rows"] += p.height
        tot["dup_uid"] += p.height - p["event_uid"].n_unique()
        for k, n in p.group_by("kind").len().iter_rows():
            kinds[k] = kinds.get(k, 0) + n
        pc = p.filter(pl.col("completed")).join(
            fi.select("period", "unix_ms", "game_clock"),
            left_on=["period", "t_catch_ms"],
            right_on=["period", "unix_ms"],
            how="left",
        )
        pb = L.pbp(args.version, game_id=gid)
        ast = pb.filter((pl.col("msg_type") == 1) & (pl.col("p2_id") > 0))
        for a in ast.iter_rows(named=True):
            hit = pc.filter(
                (pl.col("period") == a["period"])
                & (pl.col("passer_id") == a["p2_id"])
                & (pl.col("receiver_id") == a["p1_id"])
                & (pl.col("game_clock") >= a["pctime_sec"] - 1)
                & (pl.col("game_clock") <= a["pctime_sec"] + 12)
            )
            tot["assists"] += 1
            tot["covered"] += hit.height > 0
        tot["steals"] += pb.filter((pl.col("msg_type") == 5) & (pl.col("p2_id") > 0)).height
        tot["intercepted"] += int(p["intercepted"].sum())
        tot["dead"] += int(p["dead_ball"].sum())
    print(f"games={args.games}  rows={tot['rows']}  duplicate uids={tot['dup_uid']}  kinds={kinds}")
    print(
        f"assist coverage: {tot['covered']}/{tot['assists']} = {tot['covered'] / tot['assists']:.4f}"
    )
    print(
        f"intercepted passes {tot['intercepted']} vs pbp steals {tot['steals']}; dead-ball transfers {tot['dead']} (after made FG {tot.get('after_made', 0)}); shots mistaken for passes {tot.get('is_shot', 0)}"
    )


if __name__ == "__main__":
    main()
