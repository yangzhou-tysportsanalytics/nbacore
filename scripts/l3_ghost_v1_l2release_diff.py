"""Per-window causes of the ghost_v1 vs ghost_v1_l2release differences.

For every game with a window in only one of ``ghost_v1`` / ``ghost_v1_l2release`` (release
``--version``), ``segment_game`` is run twice with ``trace`` (original release rule and
``ShotTimeConfig(shooter_only=True)``); for each differing ``window_uid`` the outcome of both runs
(accepted / reject reason), the window times and the previous terminal event are written to
``reports/l3_ghost_v1_l2release_diff.csv`` with a summary in ``..._diff.json``. Times are unix ms;
``*_old`` = original rule, ``*_new`` = corrected release.

Usage:
    uv run python scripts/l3_ghost_v1_l2release_diff.py [--version v1.4]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore.possession.shots import ShotTimeConfig
from nbacore.views.ghost_v1 import SegmentConfig, segment_game

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.4")
    args = ap.parse_args()
    v = args.version
    old = L.ghost_v1(v)
    new = L.ghost_v1(v, release="l2")
    diff = pl.concat(
        [
            old.join(new, on="window_uid", how="anti")
            .select("game_id", "window_uid")
            .with_columns(pl.lit("dropped").alias("change")),
            new.join(old, on="window_uid", how="anti")
            .select("game_id", "window_uid")
            .with_columns(pl.lit("added").alias("change")),
        ]
    )
    rows = []
    for gid in diff["game_id"].unique().sort().to_list():
        fr, fi = L.frames(gid, v), L.frame_index(gid, v).sort("period", "unix_ms")
        pbp, d = L.pbp(v, game_id=gid), L.attack_direction(v, game_id=gid)
        home = int(L.games(v).filter(pl.col("game_id") == gid)["home_team_id"][0])
        tr = {}
        for tag, cfg in (
            ("old", SegmentConfig()),
            ("new", SegmentConfig(shot_cfg=ShotTimeConfig(shooter_only=True))),
        ):
            t: list[dict] = []
            segment_game(fr, fi, pbp, d, home, cfg, trace=t)
            tr[tag] = {r["event_num"]: r for r in t}
        for uid, change in (
            diff.filter(pl.col("game_id") == gid).select("window_uid", "change").iter_rows()
        ):
            ev = int(uid.split(":")[1])
            o, n = tr["old"].get(ev, {}), tr["new"].get(ev, {})
            rec = {
                "game_id": gid,
                "window_uid": uid,
                "change": change,
                "terminal_type": o.get("terminal_type"),
            }
            for k in (
                "outcome",
                "prev_event_num",
                "w_start",
                "t_terminal",
                "t_cross",
                "t_start",
                "t_end",
                "cropped",
            ):
                rec[f"{k}_old"], rec[f"{k}_new"] = o.get(k), n.get(k)
            # previous terminal event and how its time moved
            pe = n.get("prev_event_num")
            po, pn = tr["old"].get(pe, {}), tr["new"].get(pe, {})
            rec["prev_terminal_shift_ms"] = (
                pn["t_terminal"] - po["t_terminal"]
                if pn.get("t_terminal") is not None and po.get("t_terminal") is not None
                else None
            )
            rows.append(rec)
    out = pl.DataFrame(rows, infer_schema_length=None).sort("change", "game_id", "window_uid")
    out.write_csv(REPO / "reports" / "l3_ghost_v1_l2release_diff.csv")
    reason = (
        pl.when(pl.col("change") == "dropped")
        .then(pl.col("outcome_new"))
        .otherwise(pl.col("outcome_old"))
    )
    summ = {
        "version": v,
        "dropped": out.filter(pl.col("change") == "dropped").height,
        "added": out.filter(pl.col("change") == "added").height,
        "dropped_by_new_reject_reason": dict(
            out.filter(pl.col("change") == "dropped").group_by(reason).len().iter_rows()
        ),
        "added_by_old_reject_reason": dict(
            out.filter(pl.col("change") == "added").group_by(reason).len().iter_rows()
        ),
        "prev_event_changed": out.filter(
            pl.col("prev_event_num_old") != pl.col("prev_event_num_new")
        ).height,
        "prev_terminal_moved": out.filter(
            pl.col("prev_terminal_shift_ms").fill_null(0) != 0
        ).height,
        "own_terminal_moved": out.filter(
            pl.col("t_terminal_old") != pl.col("t_terminal_new")
        ).height,
    }
    (REPO / "reports" / "l3_ghost_v1_l2release_diff.json").write_text(
        json.dumps(summ, indent=1), encoding="utf-8"
    )
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
