"""v1.6 fallbacks of refine_shot_release when neither a rim contact nor an apex is found."""

from __future__ import annotations

import numpy as np
import polars as pl

from nbacore.io.raw import BALL_ID
from nbacore.possession.shots import ShotTimeConfig, refine_shot_release

HOOP = np.array([88.75, 25.0])
SHOOTER, TEAM = 10, 1


def _entities(ball_z: float, t_hold_until: int, t0: int = 0, t1: int = 12_000) -> pl.DataFrame:
    """Ball held by the shooter at (60, 25) until ``t_hold_until``, then 10 ft away (a low
    blocked shot: never near the rim, never above ``ball_z``)."""
    rows = []
    for t in range(t0, t1, 40):
        held = t <= t_hold_until
        rows.append({"unix_ms": t, "team_id": BALL_ID, "player_id": -1,
                     "x": 60.0 if held else 70.0, "y": 25.0, "z": ball_z})  # fmt: skip
        rows.append({"unix_ms": t, "team_id": TEAM, "player_id": SHOOTER,
                     "x": 60.5, "y": 25.0, "z": 0.0})  # fmt: skip
    return pl.DataFrame(rows)


def test_hand_fallback_for_a_low_blocked_shot():
    ent = _entities(ball_z=6.0, t_hold_until=8_000)
    base = ShotTimeConfig(shooter_only=True)
    assert refine_shot_release(ent, 9_000, TEAM, HOOP, base, shooter_id=SHOOTER)[2] == "pbp"
    cfg = ShotTimeConfig(shooter_only=True, hand_fallback=True)
    t, sid, method = refine_shot_release(ent, 9_000, TEAM, HOOP, cfg, shooter_id=SHOOTER)
    assert (t, sid, method) == (8_000, SHOOTER, "hand")


def test_wide_window_finds_an_apex_more_than_5_s_before_the_pbp_time():
    ent = _entities(ball_z=6.0, t_hold_until=1_000).with_columns(
        pl.when((pl.col("team_id") == BALL_ID) & pl.col("unix_ms").is_between(1_200, 1_800))
        .then(12.0 - (pl.col("unix_ms") - 1_500).abs() / 300)
        .otherwise(pl.col("z"))
        .alias("z")
    )
    t_pbp = 8_000  # apex at 1.5 s: 6.5 s before the pbp time
    base = ShotTimeConfig(shooter_only=True)
    assert refine_shot_release(ent, t_pbp, TEAM, HOOP, base, shooter_id=SHOOTER)[2] == "pbp"
    cfg = ShotTimeConfig(shooter_only=True, wide_before_s=10.0)
    t, _, method = refine_shot_release(ent, t_pbp, TEAM, HOOP, cfg, shooter_id=SHOOTER)
    assert method.endswith("_wide") and method.startswith("apex")
    assert t <= 1_500
