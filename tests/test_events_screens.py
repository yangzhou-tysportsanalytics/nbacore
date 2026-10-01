"""Tests for screen candidates (nbacore.events.screens) on a synthetic off-ball screen."""

from __future__ import annotations

import polars as pl

from nbacore.events.handler import game_handler
from nbacore.events.screens import screen_candidates
from test_events_handler import OFF, T0, A, B, D, E, _frames

C = 1002  # a third attacker: the screener


def _direction():
    return pl.DataFrame(
        {
            "game_id": ["0021500001"] * 2,
            "team_id": [OFF, 200],
            "period": [1, 1],
            "attacks_left": [True, False],
        },
        schema={
            "game_id": pl.Utf8,
            "team_id": pl.Int64,
            "period": pl.Int8,
            "attacks_left": pl.Boolean,
        },
    )


def test_off_ball_screen():
    # A holds the ball far away; screener C stands at (30, 20); user B runs past C along y = 22,
    # his defender D trails 2 ft behind him and runs into C; E guards A.
    spec = []
    for i in range(75):
        t = T0 + 40 * i
        bx = 20.0 + 0.4 * i  # B moves 10 ft/s in +x
        spec.append(
            (
                t,
                (60.0, 40.0, 4.0),
                {
                    A: (60.5, 40.0),
                    E: (63.0, 40.0),
                    C: (30.0, 20.0),
                    B: (bx, 22.0),
                    D: (bx - 2.0, 21.0),
                },
            )
        )
    f, fi = _frames(spec)
    s = screen_candidates(f, game_handler(f, fi), _direction())
    assert s.height >= 1
    r = s.sort("min_dist_ft").row(0, named=True)
    assert (r["screener_id"], r["user_id"], r["screened_def_id"]) == (C, B, D)
    assert not r["on_ball"] and r["handler_id"] == A
    # D is closest to C when D's x = 30 -> i = 30 (B at 32)
    assert abs(r["t_contact_ms"] - (T0 + 40 * 30)) <= 80
    assert r["min_dist_ft"] < 1.1 and r["screener_speed_fts"] < 1.0
    # B crosses x = 30 at i = 25, before the trailing defender reaches the screener
    assert r["t_pass_ms"] == T0 + 40 * 25 and r["t_set_start_ms"] <= r["t_contact_ms"]
    assert r["side_passed"] in ("left", "right")
    assert not r["xy_shared_at_contact"]


def test_screen_confidence_orders_the_obvious_cases():
    import polars as pl

    from nbacore.events.screens import screen_confidence

    df = pl.DataFrame(
        {
            "x": [70.0, 86.0],
            "y": [30.0, 25.0],
            "user_speed_fts": [15.0, 1.0],
            "def_mean_speed_m05_0": [14.0, 1.0],
            "min_dist_ft": [1.5, 5.0],
            "t_pass_ms": [1, None],
            "ball_in_flight_at_contact": [False, True],
            "on_ball": [False, False],
        }
    )
    s = screen_confidence(df).to_list()
    assert 0 < s[1] < 0.2 < 0.8 < s[0] < 1  # a moving user passing a close screen vs board crashing
