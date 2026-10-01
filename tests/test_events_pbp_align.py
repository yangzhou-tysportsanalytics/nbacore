"""Tests for pbp alignment (nbacore.events.pbp_align) with synthetic frames, stops and pbp."""

from __future__ import annotations

import polars as pl

from nbacore.events.clock_stop import STOP_SCHEMA
from nbacore.events.flight import FLIGHT_SCHEMA
from nbacore.events.passes import PASS_SCHEMA
from nbacore.events.pbp_align import align_game
from nbacore.events.shots import SHOT_SCHEMA
from test_events_clock_stop import _play, _with_clock
from test_events_handler import T0, A, D

GID = "0021500001"


def _pbp(rows):
    cols = ["event_num", "period", "msg_type", "action_type", "pctime_sec", "p1_id", "p2_id", "p3_id",
            "p1_type", "home_desc", "visitor_desc"]  # fmt: skip
    schema = {
        "event_num": pl.Int32, "period": pl.Int8, "msg_type": pl.Int16, "action_type": pl.Int16,
        "pctime_sec": pl.Float32, "p1_id": pl.Int64, "p2_id": pl.Int64, "p3_id": pl.Int64,
        "p1_type": pl.Int8, "home_desc": pl.Utf8, "visitor_desc": pl.Utf8,
    }  # fmt: skip
    return pl.DataFrame(
        [dict(zip(cols, r, strict=True)) for r in rows], schema=schema
    ).with_columns(pl.lit(GID).alias("game_id"))


def _stop(t_on, gc, exact, uid):
    row = {c: None for c in STOP_SCHEMA}
    row.update(
        game_id=GID, period=1, t_onset_est_ms=t_on, t_start_ms=t_on, game_clock_at_stop=gc,
        onset_exact=exact, event_uid=uid, t_end_ms=None,
    )  # fmt: skip
    return row


def _align(pbp, stops):
    spec = _play(100, T0)
    _, fi = _with_clock(spec, lambda u: 700.0 - (u - T0) / 1000)
    empty = {
        "shots": pl.DataFrame(schema=SHOT_SCHEMA),
        "flights": pl.DataFrame(schema=FLIGHT_SCHEMA),
        "passes": pl.DataFrame(schema={**PASS_SCHEMA, "dead_ball": pl.Boolean}),
    }
    st = pl.DataFrame(stops, schema=STOP_SCHEMA)
    a, st2, _ = align_game(pbp, fi, empty["shots"], st, empty["flights"], empty["passes"])
    return a, st2


def test_anchors_and_quality():
    stops = [
        _stop(T0 + 400, 699.6, True, "s1"),  # exact onset at clock 699.6
        _stop(T0 + 1800, 698.2, False, "s2"),  # onset estimated inside a hole
        _stop(T0 + 3000, 697.0, True, "s3a"),  # two stops within the tolerance of 697
        _stop(T0 + 3200, 696.8, True, "s3b"),
    ]
    pbp = _pbp(
        [
            (1, 1, 6, 2, 699.0, A, D, None, 4, "Smith S.FOUL (P1.T1)", None),  # shooting foul
            (2, 1, 6, 1, 697.0, A, D, None, 4, None, None),  # personal foul, two candidate stops
            (3, 1, 7, 0, 698.0, A, None, None, 4, None, None),  # violation: only s2 in range
            (4, 1, 8, 0, 698.0, A, D, None, 4, None, None),  # substitution: clock match
            (5, 1, 3, 11, 699.0, A, None, None, 4, "Smith Free Throw 1 of 2 (1 PTS)", None),
            (6, 1, 6, 2, 100.0, A, D, None, 4, None, None),  # nothing at that clock
        ]
    )
    a, st = _align(pbp, stops)
    r = {x["event_num"]: x for x in a.iter_rows(named=True)}
    assert (r[1]["method"], r[1]["anchor_quality"], r[1]["unix_ms"]) == ("clock_stop", 4, T0 + 400)
    assert r[1]["foul_class"] == "shooting" and r[1]["match_level"] == "tracking_event"
    assert r[1]["stop_event_uid"] == "s1" and abs(r[1]["residual_s"] + 0.6) < 0.05
    assert (r[2]["method"], r[2]["anchor_quality"]) == ("clock_stop", 2)
    assert r[2]["foul_class"] == "personal_nonshooting"
    assert (r[3]["method"], r[3]["anchor_quality"], r[3]["stop_event_uid"]) == (
        "clock_stop",
        3,
        "s2",
    )
    assert (r[4]["method"], r[4]["anchor_quality"], r[4]["match_level"]) == (
        "clock_match",
        1,
        "clock_only",
    )
    assert (r[5]["ft_n"], r[5]["ft_m"]) == (1, 2)
    assert (r[6]["method"], r[6]["anchor_quality"], r[6]["match_level"]) == (
        "unlocated",
        0,
        "unmatched",
    )
    hints = dict(st.select("event_uid", "stop_cause_hint").iter_rows())
    assert hints["s1"] == "foul" and hints["s2"] == "violation"
