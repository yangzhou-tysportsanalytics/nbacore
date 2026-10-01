"""Tests for the game-wide L2 handler (nbacore.events.handler) on synthetic frames."""

from __future__ import annotations

import polars as pl

from nbacore.clean.dedup import build_frame_index
from nbacore.events.handler import HandlerL2Config, game_handler
from nbacore.io.raw import BALL_ID, FRAME_SCHEMA

GID = "0021500001"
T0 = 1_447_287_043_000
OFF, DEF = 100, 200
A, B = 1000, 1001  # offence
D, E = 2000, 2001  # defence


def _frames(spec):
    """``spec``: list of (t_ms, ball_xyz, {player_id: (x, y)}); teams from the id."""
    rows = []
    for k, (t, ball, pos) in enumerate(spec):
        gc = 700.0 - (t - T0) / 1000
        rows.append((GID, 1, k, 1, t, gc, 20.0, BALL_ID, BALL_ID, *ball))
        for pid, (x, y) in pos.items():
            team = OFF if pid < 2000 else DEF
            rows.append((GID, 1, k, 1, t, gc, 20.0, team, pid, x, y, 0.0))
    f = pl.DataFrame(rows, schema=FRAME_SCHEMA, orient="row")
    return f, build_frame_index(f)


def _run(spec, **kw):
    f, fi = _frames(spec)
    return game_handler(f, fi, HandlerL2Config(**kw))


def _seq(n, t0, ball, pos):
    return [(t0 + 40 * i, ball(i), pos(i)) for i in range(n)]


def test_contest_does_not_switch_team():
    # A holds the ball for 1 s; D is closer to the ball for 0.6 s; A keeps holding for 1 s
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (13.0, 10.0)})
    spec += _seq(
        15, T0 + 1000, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (12.0, 10.0), D: (10.5, 10.0)}
    )
    spec += _seq(
        25, T0 + 1600, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (13.0, 10.0)}
    )
    hyst = _run(spec)
    assert set(hyst["handler_id"]) == {A}
    assert set(hyst["control_team_id"]) == {OFF}
    near = _run(spec, mode="nearest")
    assert (near["handler_id"] == D).sum() == 15  # the naive rule gives the ball to the defender


def test_steal_after_loose_ball_switches_team():
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (15.0, 10.0)})
    # loose ball for 0.6 s (nobody within 3 ft), then D controls for 0.6 s
    spec += _seq(
        15, T0 + 1000, lambda i: (20.0, 10.0, 1.0), lambda i: {A: (11.0, 10.0), D: (15.0, 10.0)}
    )
    spec += _seq(
        15, T0 + 1600, lambda i: (20.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (20.5, 10.0)}
    )
    h = _run(spec)
    assert h["handler_id"].to_list()[-15:] == [D] * 15
    assert h["control_team_id"].to_list()[-1] == DEF


def test_contest_right_before_flight_keeps_shooter():
    # A holds; D closest for 0.2 s at the release; then the ball flies (no candidate) for 1 s
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 6.0), lambda i: {A: (11.0, 10.0), D: (13.0, 10.0)})
    spec += _seq(
        5, T0 + 1000, lambda i: (10.0, 10.0, 8.0), lambda i: {A: (12.0, 10.0), D: (10.5, 10.0)}
    )
    spec += _seq(
        25,
        T0 + 1200,
        lambda i: (10.0 + i, 10.0, 12.0),
        lambda i: {A: (12.0, 10.0), D: (10.5, 10.0)},
    )
    h = _run(spec)
    last_handler = h.filter(pl.col("handler_id") >= 0)["handler_id"].to_list()[-1]
    assert last_handler == A


def test_catch_and_shoot_within_team_is_kept():
    # A holds, pass flight 0.6 s, B touches 0.2 s, then shot flight 1 s: B is a handler
    # (catch-and-shoot rule)
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 4.0), lambda i: {A: (11.0, 10.0), B: (30.0, 10.0)})
    spec += _seq(
        15, T0 + 1000, lambda i: (20.0, 10.0, 8.0), lambda i: {A: (11.0, 10.0), B: (30.0, 10.0)}
    )
    spec += _seq(
        5, T0 + 1600, lambda i: (30.0, 10.0, 6.0), lambda i: {A: (11.0, 10.0), B: (30.5, 10.0)}
    )
    spec += _seq(
        25, T0 + 1800, lambda i: (30.0, 10.0, 15.0), lambda i: {A: (11.0, 10.0), B: (30.5, 10.0)}
    )
    h = _run(spec)
    assert (h["handler_id"] == B).sum() == 5


def test_no_carry_across_a_hole_and_ambiguity_flag():
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (11.0, 10.0)})
    # 2 s tracking hole -> new segment; nobody near the ball afterwards
    spec += _seq(
        25, T0 + 3000, lambda i: (40.0, 10.0, 3.0), lambda i: {A: (11.0, 10.0), D: (11.0, 10.0)}
    )
    h = _run(spec)
    assert h["segment_id"].n_unique() == 2
    assert (h.filter(pl.col("segment_id") == 1)["handler_id"] == -1).all()
    # A and D share x, y (identity merge): candidate flagged ambiguous
    assert h.filter(pl.col("segment_id") == 0)["cand_ambiguous"].all()
