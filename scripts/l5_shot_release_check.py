"""Shot release plausibility (v1.3 plan: late releases).

For a sample of tracked games: ``shot_events`` with the v1.3 L2 config vs the v1.2 release, the
distance between the ball and the pbp shooter at the release (> 10 ft = implausible: the ball is
already in flight) and the ball height, by method. Writes ``reports/l5_shot_release_check.json``.

Usage:
    uv run python scripts/l5_shot_release_check.py [--every 60] [--n 10]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore.events.build import CONFIGS
from nbacore.events.handler import game_handler
from nbacore.events.shots import shot_events

REPO = Path(__file__).resolve().parents[1]


def _geom(s: pl.DataFrame, fr: pl.DataFrame, t: str) -> pl.DataFrame:
    ball = fr.filter(pl.col("player_id") == -1).select(
        pl.col("unix_ms").alias(t), pl.col("x").alias("_bx"), pl.col("y").alias("_by"), "z"
    )
    ply = fr.filter(pl.col("player_id") > 0).select(
        pl.col("unix_ms").alias(t),
        pl.col("player_id").alias("shooter_id"),
        pl.col("x").alias("_sx"),
        pl.col("y").alias("_sy"),
    )
    return (
        s.join(ball, on=t, how="left")
        .join(ply, on=[t, "shooter_id"], how="left")
        .with_columns(
            ((pl.col("_bx") - pl.col("_sx")) ** 2 + (pl.col("_by") - pl.col("_sy")) ** 2)
            .sqrt()
            .alias("dist_ft"),
            pl.col("z").alias("z_ft"),
        )
        .drop("_bx", "_by", "_sx", "_sy", "z")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()
    gids = sorted(L.games("v1.2").filter(pl.col("status") == "ok")["game_id"].to_list())
    gids = gids[:: args.every][: args.n]
    old, new = [], []
    for gid in gids:
        fr, fi = L.frames(gid, "v1.2"), L.frame_index(gid, "v1.2")
        o = L.events(gid, "shot_release", "v1.2").select(
            "game_id", "pbp_event_num", "shooter_id", "t_release_ms", "method"
        )
        n = shot_events(
            fr,
            fi,
            L.pbp("v1.2", game_id=gid),
            L.attack_direction("v1.2", game_id=gid),
            game_handler(fr, fi),
            CONFIGS["shot"],
        ).select("game_id", "pbp_event_num", "shooter_id", "t_release_ms", "method")
        old.append(_geom(o, fr, "t_release_ms"))
        new.append(_geom(n, fr, "t_release_ms"))
    o, n = pl.concat(old), pl.concat(new)
    j = o.join(n, on=["game_id", "pbp_event_num"], suffix="_new")
    shift = j["t_release_ms"] - j["t_release_ms_new"]

    def summ(d: pl.DataFrame) -> dict:
        v = d.filter(pl.col("dist_ft").is_not_null())
        return {
            "shots": d.height,
            "late_gt10ft": float((v["dist_ft"] > 10).mean()),
            "gt6ft": float((v["dist_ft"] > 6).mean()),
            "median_z_ft": float(v["z_ft"].median()),
            "by_method": {
                m: {"n": k, "late_gt10ft": lt}
                for m, k, lt in v.group_by("method")
                .agg(pl.len(), (pl.col("dist_ft") > 10).mean())
                .sort("len", descending=True)
                .iter_rows()
            },
        }

    stats = {
        "games": gids,
        "config": "ShotTimeConfig(shooter_only=True)",
        "v1.2": summ(o),
        "v1.3": summ(n),
        "unchanged_frac": float((shift == 0).mean()),
        "moved_earlier": int((shift > 0).sum()),
        "moved_later": int((shift < 0).sum()),
        "shift_ms_quantiles_of_moved": {
            str(q): float(shift.filter(shift != 0).quantile(q)) for q in (0.1, 0.5, 0.9)
        },
    }
    (REPO / "reports" / "l5_shot_release_check.json").write_text(
        json.dumps(stats, indent=1), encoding="utf-8"
    )
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
