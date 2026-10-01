"""Tests for ball-derived shot release detection (synthetic parabola)."""

from __future__ import annotations

import numpy as np
import polars as pl

from nbacore.court import HOOP_LEFT
from nbacore.io.raw import BALL_ID, FRAME_SCHEMA
from nbacore.possession.shots import ShotTimeConfig, refine_shot_release

T0 = 1_447_287_043_000
OFF = 100
SHOOTER, BIG, PASSER = 1000, 1001, 1002


def _entities(rows):
    cols = [c for c in FRAME_SCHEMA if c != "game_id"]
    data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    data["game_id"] = ["0021500001"] * len(rows)
    return pl.DataFrame(data, schema=FRAME_SCHEMA)


def _scene():
    """Shooter holds the ball at (25, 25) until t=2.0 s, then a 1.0 s arc into the hoop; a big
    man stands under the rim the whole time; the passer had the ball until t=0.6 s."""
    rows = []
    hx, hy = HOOP_LEFT
    for i in range(125):  # 5 s at 25 Hz
        t = i * 0.04
        u = T0 + 40 * i
        gc = 700.9 - t
        if t < 0.6:  # passer has it at (35, 25)
            bx, by, bz = 35.0, 25.0, 4.0
        elif t < 0.9:  # pass in flight
            f = (t - 0.6) / 0.3
            bx, by, bz = 35.0 - 10.0 * f, 25.0, 6.0
        elif t < 2.0:  # shooter holds / dribbles
            bx, by, bz = 25.0, 25.0, 3.0 + 0.5 * np.sin(20 * t)
        elif t < 3.0:  # arc: release z=8 at t=2, apex ~15, rim at t=3 with z=10
            f = t - 2.0
            bx, by = 25.0 + (hx - 25.0) * f, 25.0
            bz = 8.0 + 18.0 * f - 16.0 * f * f  # z(1)=10
        else:  # through the net and down
            bx, by, bz = hx, hy, max(0.0, 10.0 - 30.0 * (t - 3.0))
        rows.append((1, i, 1, u, gc, None, BALL_ID, BALL_ID, bx, by, bz))
        rows.append((1, i, 1, u, gc, None, OFF, SHOOTER, 25.0, 25.5, 0.0))
        rows.append((1, i, 1, u, gc, None, OFF, BIG, hx + 1.0, hy + 1.0, 0.0))
        rows.append((1, i, 1, u, gc, None, OFF, PASSER, 35.0, 25.0, 0.0))
    return _entities(rows)


def test_refine_release_rim():
    ent = _scene()
    t_pbp = T0 + 3800  # pbp clock is late by ~0.8 s relative to the rim moment
    t_rel, shooter, method = refine_shot_release(ent, t_pbp, OFF, HOOP_LEFT, ShotTimeConfig())
    assert method == "rim"
    assert shooter == SHOOTER  # not the big man standing under the rim
    # release: last in-hand frame at/before the rim moment -> ball leaves 3 ft of shooter at
    # t = 2 + 3/ (hoop distance per second) ~ 2.15 s
    assert T0 + 2000 <= t_rel <= T0 + 2200


def test_refine_release_respects_t_min_and_falls_back():
    ent = _scene()
    t_pbp = T0 + 3800
    # previous terminal after the shot -> no rim contact allowed -> fallback (apex is also excluded)
    t_rel, shooter, method = refine_shot_release(
        ent, t_pbp, OFF, HOOP_LEFT, ShotTimeConfig(), t_min=T0 + 3500
    )
    assert method == "pbp" and t_rel == t_pbp and shooter is None
    # ball never reaches the rim: apex fallback
    ent2 = ent.with_columns(
        pl.when((pl.col("team_id") == BALL_ID) & (pl.col("unix_ms") >= T0 + 2800))
        .then(pl.lit(30.0, dtype=pl.Float32))
        .otherwise(pl.col("x"))
        .alias("x")
    )
    t_rel2, shooter2, method2 = refine_shot_release(ent2, t_pbp, OFF, HOOP_LEFT, ShotTimeConfig())
    assert method2 == "apex" and shooter2 == SHOOTER
    assert T0 + 2000 <= t_rel2 <= T0 + 2200
