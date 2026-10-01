"""Tests for clock stops (nbacore.events.clock_stop) on synthetic frames."""

from __future__ import annotations

import polars as pl

from nbacore.clean.dedup import build_frame_index
from nbacore.events.clock_stop import clock_stops
from test_events_handler import T0, A, D, _frames, _seq


def _with_clock(spec, clock):
    """Frames from ``spec`` with the game clock replaced by ``clock(t_ms)``."""
    f, _ = _frames(spec)
    gc = [clock(int(u)) for u in f["unix_ms"].to_list()]
    f = f.with_columns(pl.Series("game_clock", gc, dtype=pl.Float32))
    return f, build_frame_index(f)


def _play(n, t0):
    return _seq(n, t0, lambda i: (10.0, 10.0, 4.0), lambda i: {A: (11.0, 10.0), D: (15.0, 10.0)})


def test_exact_onset():
    # clock runs for 2 s, stops at 698.0 for 3 s, runs again
    spec = _play(150, T0)
    stop_t = T0 + 2000

    def clock(u):
        if u < stop_t:
            return 700.0 - (u - T0) / 1000
        if u < stop_t + 3000:
            return 698.0
        return 698.0 - (u - stop_t - 3000) / 1000

    s, short = clock_stops(*_with_clock(spec, clock))
    assert s.height == 1 and not short
    r = s.row(0, named=True)
    assert r["onset_exact"] and r["t_start_ms"] == stop_t and r["t_onset_est_ms"] == stop_t
    assert r["game_clock_at_stop"] == 698.0 and r["t_end_ms"] == stop_t + 3040
    assert abs(r["duration_s"] - 3.0) < 1e-3 and not r["micro_freeze"]
    # 2 s of the 3 s pre window are tracked (only 2 players here, so "tracked" = ball present)
    assert abs(r["pre3s_ball_missing_frac"] - 1 / 3) < 0.02 and r["pre3s_max_hole_ms"] == 1000


def test_onset_inside_a_hole_is_estimated():
    # running until 698.40 at T0+1600, a 5 s hole, tracking resumes with the clock stopped at 698.0
    spec = _play(41, T0) + _play(50, T0 + 6600)

    def clock(u):
        return 700.0 - (u - T0) / 1000 if u <= T0 + 1600 else 698.0

    r = clock_stops(*_with_clock(spec, clock))[0].row(0, named=True)
    assert not r["onset_exact"] and r["gap_before_ms"] == 5000
    assert abs(r["clock_left_s"] - 0.4) < 1e-3
    assert r["t_last_running_ms"] == T0 + 1600 and r["t_onset_est_ms"] == T0 + 2000
    assert r["t_start_ms"] == T0 + 6600
