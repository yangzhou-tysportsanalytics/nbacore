"""Tests for dribbles, post-ups and inbounds on synthetic frames."""

from __future__ import annotations

import math

import polars as pl

from nbacore.clean.dedup import build_frame_index
from nbacore.events.build import possession_touches
from nbacore.events.dribble import dribbles
from nbacore.events.flight import ball_flights
from nbacore.events.handler import game_handler
from nbacore.events.motion import postups
from nbacore.events.passes import inbounds, passes
from test_events_handler import T0, A, B, D, _frames
from test_events_screens import _direction


def test_dribbles_counted_per_touch():
    # A holds the ball 3 s at (30, 25); the ball bounces every 0.6 s between 0.4 ft and 3.4 ft
    spec = []
    for i in range(75):
        z = 0.4 + 3.0 * abs(math.sin(math.pi * i / 15))
        spec.append((T0 + 40 * i, (30.3, 25.0, z), {A: (30.0, 25.0), D: (35.0, 25.0)}))
    f, fi = _frames(spec)
    h = game_handler(f, fi)
    d, tch = dribbles(f, possession_touches(h, f), fi)
    assert d.height == 4  # minima at i = 15, 30, 45, 60 (i = 0 and 75 are the window edges)
    assert set(d["handler_id"]) == {A} and (d["prominence_ft"] >= 1.0).all()
    assert tch.filter(pl.col("handler_id") == A)["n_dribbles"].to_list() == [4]


def test_postup():
    # A holds at 10 ft from the left hoop, nearly still, D between him and the basket for 1.6 s
    spec = [
        (
            T0 + 40 * i,
            (15.0, 25.5, 3.0),
            {A: (15.0 + 0.001 * i, 25.0), D: (12.5, 25.0), B: (40.0, 40.0)},
        )
        for i in range(40)
    ]
    f, fi = _frames(spec)
    pu = postups(f, game_handler(f, fi), _direction())
    r = pu.row(0, named=True)
    assert (r["player_id"], r["def_id"]) == (A, D) and r["duration_s"] >= 1.5
    assert r["def_between_frac"] == 1.0 and 9 < r["mean_dist_basket_ft"] < 11


def test_inbound_after_stoppage():
    # A holds the ball out of bounds (y = -1) with the clock stopped, passes to B inside
    spec = [(T0 + 40 * i, (40.0, -0.8, 4.0), {A: (40.0, -1.0), B: (40.0, 15.0)}) for i in range(25)]
    spec += [
        (T0 + 1000 + 40 * i, (40.0, 4.0 + i * 0.7, 6.0), {A: (40.0, -1.0), B: (40.0, 15.0)})
        for i in range(15)
    ]
    spec += [
        (T0 + 1600 + 40 * i, (40.0, 15.2, 4.0), {A: (40.0, -1.0), B: (40.0, 15.0)})
        for i in range(25)
    ]
    f, _ = _frames(spec)
    f = f.with_columns(
        pl.when(pl.col("unix_ms") < T0 + 1640)
        .then(pl.lit(700.0))
        .otherwise(700.0 - (pl.col("unix_ms") - T0 - 1640) / 1000)
        .cast(pl.Float32)
        .alias("game_clock")
    )
    fi = build_frame_index(f)
    h = game_handler(f, fi)
    p = passes(h, ball_flights(h, f), f, fi)
    ib = inbounds(p, f, fi)
    r = ib.row(0, named=True)
    assert (r["inbounder_id"], r["receiver_id"], r["context"]) == (A, B, "stoppage")
    assert r["y_entry"] >= 0 and r["t_entry_ms"] <= r["t_catch_ms"]
