"""Tests for passes / handoffs (nbacore.events.passes) on synthetic frames."""

from __future__ import annotations

import polars as pl

from nbacore.clean.dedup import build_frame_index
from nbacore.events.flight import ball_flights
from nbacore.events.handler import game_handler
from nbacore.events.passes import passes
from test_events_handler import T0, A, B, D, _frames, _seq


def _passes(spec, freeze_from=None):
    f, fi = _frames(spec)
    if freeze_from is not None:  # stop the game clock from this time on
        gc_stop = f.filter(pl.col("unix_ms") == freeze_from)["game_clock"][0]
        f = f.with_columns(
            pl.when(pl.col("unix_ms") >= freeze_from)
            .then(pl.lit(gc_stop, dtype=pl.Float32))
            .otherwise(pl.col("game_clock"))
            .alias("game_clock")
        )
        fi = build_frame_index(f)
    h = game_handler(f, fi)
    return passes(h, ball_flights(h, f), f, fi)


def _hold(n, t0, ball_xy, pos, z=4.0):
    return _seq(n, t0, lambda i: (*ball_xy, z), lambda i: pos)


def test_pass_fields():
    spec = _hold(25, T0, (10.0, 10.0), {A: (11.0, 10.0), B: (30.0, 10.0)})
    spec += _seq(
        15,
        T0 + 1000,
        lambda i: (15.0 + 0.7 * i, 10.0, 7.0),
        lambda i: {A: (11.0, 10.0), B: (30.0, 10.0)},
    )
    spec += _hold(25, T0 + 1600, (30.0, 10.0), {A: (11.0, 10.0), B: (30.5, 10.0)})
    p = _passes(spec)
    assert p.height == 1
    r = p.row(0, named=True)
    assert (r["kind"], r["passer_id"], r["receiver_id"]) == ("pass", A, B)
    assert r["t_release_ms"] == T0 + 960 and r["t_catch_ms"] == T0 + 1600
    assert r["completed"] and not r["intercepted"] and not r["dead_ball"]
    assert r["max_free_run"] == 15 and r["event_uid"] == f"0021500001:pass:{T0 + 960}:{A}"


def _transfer(dist):
    """A holds, walks next to B, B takes the ball without any free frame; A-B distance = dist."""
    spec = _hold(25, T0, (10.0, 10.0), {A: (11.0, 10.0), B: (11.0 + dist, 10.0)})
    # ball moves from A to B while staying within 3 ft of one of them
    spec += _hold(
        25, T0 + 1000, (11.0 + dist - 0.2, 10.0), {A: (11.0, 10.0), B: (11.0 + dist, 10.0)}
    )
    return _passes(spec)


def test_handoff_and_ambiguous_by_distance():
    h = _transfer(2.0).row(0, named=True)
    assert h["kind"] == "handoff" and h["event_type"] == "handoff" and h["n_free_frames"] == 0
    a = _transfer(4.0).row(0, named=True)
    assert a["kind"] == "ambiguous" and a["gap_3_5ft"] and not a["flight_missing"]
    assert a["transfer_dist_ft"] == 4.0


def test_interception():
    spec = _hold(25, T0, (10.0, 10.0), {A: (11.0, 10.0), D: (30.0, 10.0)})
    spec += _seq(
        15,
        T0 + 1000,
        lambda i: (15.0 + 0.7 * i, 10.0, 6.0),
        lambda i: {A: (11.0, 10.0), D: (30.0, 10.0)},
    )
    spec += _hold(30, T0 + 1600, (30.0, 10.0), {A: (11.0, 10.0), D: (30.5, 10.0)})
    r = _passes(spec).row(0, named=True)
    assert r["intercepted"] and not r["completed"] and r["receiver_id"] == D


def test_dead_ball_toss_is_not_completed():
    # the clock stops before the toss and stays stopped: a dead-ball transfer
    spec = _hold(25, T0, (10.0, 10.0), {A: (11.0, 10.0), B: (30.0, 10.0)})
    spec += _seq(
        15,
        T0 + 1000,
        lambda i: (15.0 + 0.7 * i, 10.0, 7.0),
        lambda i: {A: (11.0, 10.0), B: (30.0, 10.0)},
    )
    spec += _hold(50, T0 + 1600, (30.0, 10.0), {A: (11.0, 10.0), B: (30.5, 10.0)})
    r = _passes(spec, freeze_from=T0 + 800).row(0, named=True)
    assert r["dead_ball"] and not r["completed"]
