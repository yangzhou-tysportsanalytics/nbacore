"""Tests for ball flights (nbacore.events.flight) on synthetic frames."""

from __future__ import annotations

from nbacore.court import HOOP_LEFT
from nbacore.events.flight import ball_flights
from nbacore.events.handler import game_handler
from test_events_handler import T0, A, B, D, _frames, _seq


def _flights(spec):
    f, fi = _frames(spec)
    return ball_flights(game_handler(f, fi), f)


def test_pass_flight_between_teammates():
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 4.0), lambda i: {A: (11.0, 10.0), B: (30.0, 10.0)})
    spec += _seq(
        15,
        T0 + 1000,
        lambda i: (15.0 + 0.7 * i, 10.0, 7.0),
        lambda i: {A: (11.0, 10.0), B: (30.0, 10.0)},
    )
    spec += _seq(
        25, T0 + 1600, lambda i: (30.0, 10.0, 4.0), lambda i: {A: (11.0, 10.0), B: (30.5, 10.0)}
    )
    fl = _flights(spec)
    assert fl.height == 1
    r = fl.row(0, named=True)
    assert (r["from_id"], r["to_id"]) == (A, B)
    assert r["n_frames"] == 15 and r["t_start_ms"] == T0 + 1000 and r["t_to_ms"] == T0 + 1600
    assert not r["reaches_rim"] and not r["ends_segment"]


def test_shot_reaches_rim_and_single_free_frame_is_not_a_flight():
    hx, hy = float(HOOP_LEFT[0]), float(HOOP_LEFT[1])
    spec = _seq(25, T0, lambda i: (20.0, 25.0, 6.0), lambda i: {A: (20.5, 25.0), D: (40.0, 10.0)})
    # one free frame (jitter), back in hand
    spec += [(T0 + 1000, (24.0, 25.0, 6.0), {A: (20.5, 25.0), D: (40.0, 10.0)})]
    spec += _seq(
        10, T0 + 1040, lambda i: (20.0, 25.0, 6.0), lambda i: {A: (20.5, 25.0), D: (40.0, 10.0)}
    )
    # shot: ball travels to the hoop at rim height, ending at the rim
    spec += _seq(
        20,
        T0 + 1440,
        lambda i: (20.0 + (hx - 20.0) * (i + 1) / 20, hy, 9.5),
        lambda i: {A: (20.5, 25.0), D: (40.0, 10.0)},
    )
    fl = _flights(spec)
    assert fl.height == 1  # the one-frame blip is ignored
    r = fl.row(0, named=True)
    assert r["from_id"] == A and r["reaches_rim"] and r["to_id"] is None and r["ends_segment"]


def test_interception_goes_to_other_team():
    spec = _seq(25, T0, lambda i: (10.0, 10.0, 4.0), lambda i: {A: (11.0, 10.0), D: (30.0, 10.0)})
    spec += _seq(
        15, T0 + 1000, lambda i: (12.0 + i, 10.0, 6.0), lambda i: {A: (11.0, 10.0), D: (30.0, 10.0)}
    )
    spec += _seq(
        30, T0 + 1600, lambda i: (30.0, 10.0, 4.0), lambda i: {A: (11.0, 10.0), D: (30.5, 10.0)}
    )
    r = _flights(spec).row(0, named=True)
    assert (r["from_id"], r["to_id"]) == (A, D) and r["from_team_id"] != r["to_team_id"]
