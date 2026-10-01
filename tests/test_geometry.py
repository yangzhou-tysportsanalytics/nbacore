"""Tests for nbacore.geometry and nbacore.shot_clock on synthetic frames."""

from __future__ import annotations

import polars as pl

from nbacore.geometry import ball_distances, pair_distances, player_speeds
from nbacore.shot_clock import shot_clock_filled
from test_events_handler import T0, A, B, D, E, _frames, _seq


def test_pair_and_ball_distances_null_on_identity_merge():
    # frame 0: A and D at the same x, y (identity merge); frame 1: apart
    spec = [
        (
            T0,
            (10.0, 10.0, 3.0),
            {A: (11.0, 10.0), B: (20.0, 10.0), D: (11.0, 10.0), E: (30.0, 10.0)},
        ),
        (
            T0 + 40,
            (10.0, 10.0, 3.0),
            {A: (11.0, 10.0), B: (20.0, 10.0), D: (14.0, 10.0), E: (30.0, 10.0)},
        ),
    ]
    f, _ = _frames(spec)
    d = pair_distances(f)
    assert d.height == 8  # 2 frames x 2 x 2 pairs
    f0 = d.filter(pl.col("unix_ms") == T0)
    assert f0.filter((pl.col("id_a") == A) & (pl.col("id_b") == D))["dist_ft"].to_list() == [None]
    assert f0.filter((pl.col("id_a") == B) & (pl.col("id_b") == E))["dist_ft"].to_list() == [10.0]
    f1 = d.filter(pl.col("unix_ms") == T0 + 40)
    assert f1.filter((pl.col("id_a") == A) & (pl.col("id_b") == D))["dist_ft"].to_list() == [3.0]
    b = ball_distances(f).filter(pl.col("unix_ms") == T0)
    assert b.filter(pl.col("player_id") == A)["ball_dist_ft"].to_list() == [None]
    assert b.filter(pl.col("player_id") == B)["ball_dist_ft"].to_list() == [10.0]


def test_player_speed():
    spec = _seq(
        25, T0, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (10.0 + 0.2 * i, 10.0), D: (30.0, 10.0)}
    )
    f, _ = _frames(spec)
    s = player_speeds(f).filter(pl.col("player_id") == A)["speed_fts"].drop_nulls()
    assert abs(s[12] - 5.0) < 1e-3  # 0.2 ft per 40 ms


def test_shot_clock_fill_methods():
    spec = _seq(10, T0, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (15.0, 10.0)})
    f, _ = _frames(spec)
    from nbacore.clean.dedup import build_frame_index

    # observed 20 s, then null with a running clock at > 24 s game clock, then frozen clock
    f = f.with_columns(
        pl.when(pl.col("unix_ms") < T0 + 120)
        .then(20.0)
        .otherwise(None)
        .cast(pl.Float32)
        .alias("shot_clock"),
        pl.when(pl.col("unix_ms") < T0 + 240)
        .then(pl.col("game_clock"))
        .otherwise(pl.lit(699.0))
        .cast(pl.Float32)
        .alias("game_clock"),
    )
    s = shot_clock_filled(build_frame_index(f))
    assert s["fill_method"].to_list()[:3] == ["observed"] * 3
    assert s["fill_method"].to_list()[3] == "missing"
    assert s["fill_method"].to_list()[-1] == "carry_forward" and s["shot_clock_filled"][-1] == 20.0


def test_event_frame_distances():
    from nbacore.geometry import event_frame_distances

    spec = [
        (
            T0,
            (10.0, 10.0, 3.0),
            {A: (11.0, 10.0), B: (20.0, 10.0), D: (11.0, 13.0), E: (30.0, 10.0)},
        )
    ]
    f, _ = _frames(spec)
    ev = pl.DataFrame(
        {
            "period": [1],
            "event_type": ["shot_release"],
            "t_start_ms": [T0],
            "t_end_ms": [None],
            "team_id": [100],
            "event_uid": ["u"],
        },  # fmt: skip
        schema={
            "period": pl.Int8,
            "event_type": pl.Utf8,
            "t_start_ms": pl.Int64,
            "t_end_ms": pl.Int64,
            "team_id": pl.Int64,
            "event_uid": pl.Utf8,
        },  # fmt: skip
    )
    r = event_frame_distances(f, ev).row(0, named=True)
    assert r["off_ids"] == [A, B] and r["def_ids"] == [D, E]
    assert r["off_def_dist_ft"][0] == 3.0  # A-D
    assert r["ball_dist_off_ft"][0] == 1.0 and not r["xy_shared"]
