"""Tests for nbacore.clean.dedup and nbacore.possession.direction on synthetic data."""

from __future__ import annotations

import polars as pl
import pytest

from nbacore.clean.dedup import build_frame_index, dedup_entities, dedup_report
from nbacore.io import BALL_ID
from nbacore.io.raw import FRAME_SCHEMA
from nbacore.possession.direction import infer_attack_direction, shot_ball_positions

T0 = 1_447_287_043_000
HOME, AWAY = 100, 200


def _rows(event_id, moment_idx, period, unix, gc, sc, ball=True, players=True, ball_x=47.0):
    out = []
    if ball:
        out.append(
            (event_id, moment_idx, period, unix, gc, sc, BALL_ID, BALL_ID, ball_x, 25.0, 5.0)
        )
    if players:
        for i in range(10):
            tid = HOME if i < 5 else AWAY
            out.append(
                (event_id, moment_idx, period, unix, gc, sc, tid, 1000 + i, 10.0 + i, 20.0, 0.0)
            )
    return out


def _df(rows):
    cols = [c for c in FRAME_SCHEMA if c != "game_id"]
    data = {c: [r[i] for r in rows] for i, c in enumerate(cols)}
    data["game_id"] = ["0021500001"] * len(rows)
    return pl.DataFrame(data, schema=FRAME_SCHEMA)


@pytest.fixture
def raw():
    rows = []
    # event 1: frames 0..9
    for i in range(10):
        rows += _rows(1, i, 1, T0 + 40 * i, 720 - 0.04 * i, 24.0)
    # event 2: overlaps frames 5..9, then continues 10..14
    for i in range(5, 15):
        rows += _rows(2, i - 5, 1, T0 + 40 * i, 720 - 0.04 * i, 24.0)
    # split frame at i=15: ball-only in event 2, players-only in event 3 (different clocks)
    rows += _rows(2, 10, 1, T0 + 40 * 15, 719.4, 24.0, ball=True, players=False)
    rows += _rows(3, 0, 1, T0 + 40 * 15, 719.0, 24.0, ball=False, players=True)
    # gap of 2 s then 3 more frames (new segment), with a clock correction (+5 s) at the last
    for k, i in enumerate(range(66, 69)):
        gc = 717.0 - 0.04 * k if k < 2 else 722.5
        rows += _rows(3, 1 + k, 1, T0 + 40 * i, gc, None)
    # period 2, one frame
    rows += _rows(4, 0, 2, T0 + 40 * 100, 720.0, 24.0)
    return _df(rows)


def test_dedup_entities(raw):
    ent = dedup_entities(raw)
    assert raw.height == 10 * 11 + 10 * 11 + 1 + 10 + 3 * 11 + 11
    # 16 frames in period 1 (0..15), 3 after the gap, 1 in period 2 => 20 frames, all 11 entities
    assert ent.select(["period", "unix_ms"]).unique().height == 20
    assert ent.height == 20 * 11
    # first occurrence wins: frames 5..9 keep event_id 1
    f7 = ent.filter(pl.col("unix_ms") == T0 + 40 * 7)
    assert f7["event_id"].unique().to_list() == [1]
    # split frame merged: ball from event 2, players from event 3
    f15 = ent.filter(pl.col("unix_ms") == T0 + 40 * 15)
    assert f15.height == 11
    assert f15.filter(pl.col("team_id") == BALL_ID)["event_id"].item() == 2
    assert f15.filter(pl.col("team_id") != BALL_ID)["event_id"].unique().to_list() == [3]


def test_frame_index_segments(raw):
    ent = dedup_entities(raw)
    idx = build_frame_index(ent)
    assert idx.height == 20
    assert idx["n_entities"].unique().to_list() == [11]
    assert idx["has_ball"].all()
    # split frame takes the majority clock (players' 719.0)
    assert idx.filter(pl.col("unix_ms") == T0 + 40 * 15)["game_clock"].item() == pytest.approx(
        719.0
    )
    # segments: [0..15] | gap -> [66, 67] | clock jump -> [68] | new period -> [100]
    seg = idx["segment_id"].to_list()
    assert seg[:16] == [0] * 16
    assert seg[16:18] == [1, 1]
    assert seg[18] == 2
    assert seg[19] == 3
    assert idx["gap"].to_list()[16] is True and idx["clock_jump"].to_list()[18] is True
    assert idx["dt_ms"][0] is None and idx["dt_ms"][1] == 40
    rep = dedup_report(raw, ent, idx)
    assert rep["unique_frames"] == 20 and rep["n_segments"] == 4
    assert rep["n_gaps"] == 1 and rep["n_clock_jumps"] == 1
    assert rep["raw_moments"] == 10 + 11 + 1 + 3 + 1
    assert rep["shot_clock_null_frames"] == 3


# ------------------------------------------------------------------------------------------
# attack direction
# ------------------------------------------------------------------------------------------


def _pbp_row(event_num, period, pctime, msg_type, team_id, pid=5):
    return {
        "game_id": "0021500001",
        "event_num": event_num,
        "period": period,
        "pctime_sec": float(pctime),
        "msg_type": msg_type,
        "p1_team_id": team_id,
        "p1_id": pid,
    }


@pytest.fixture
def direction_case():
    """HOME attacks LEFT in periods 1-2 and RIGHT in 3-4; one noisy shot in period 3."""
    rows = []
    pbp = []
    n = 0
    # for each period, 4 shots per team at 1 s spacing; ball placed in the attacking half
    for period in (1, 2, 3, 4):
        home_left = period <= 2
        for k in range(4):
            for team in (HOME, AWAY):
                n += 1
                gc = 700.0 - 10 * n
                left = home_left if team == HOME else not home_left
                # noise: one HOME shot in period 3 recorded on the wrong side
                if period == 3 and team == HOME and k == 0:
                    left = not left
                bx = 10.0 if left else 84.0
                unix = T0 + 1000 * n
                rows += _rows(n, 0, period, unix, gc, 20.0, ball_x=bx)
                pbp.append(_pbp_row(n, period, gc, 1 if k % 2 else 2, team))
    ent = dedup_entities(_df(rows))
    return ent, pl.DataFrame(pbp)


def test_shot_ball_positions(direction_case):
    ent, pbp = direction_case
    s = shot_ball_positions(ent, pbp, tol_s=1.0)
    assert s.height == 32
    assert s["n_frames"].unique().to_list() == [1]


def test_infer_attack_direction(direction_case):
    ent, pbp = direction_case
    table, diag = infer_attack_direction(ent, pbp)
    assert table.height == 8
    home = table.filter(pl.col("team_id") == HOME).sort("period")
    away = table.filter(pl.col("team_id") == AWAY).sort("period")
    assert home["attacks_left"].to_list() == [True, True, False, False]
    assert away["attacks_left"].to_list() == [False, False, True, True]
    # the noisy shot lowers consistency for HOME in period 3 but does not flip the decision
    assert home.filter(pl.col("period") == 3)["consistency"].item() == pytest.approx(0.75)
    assert diag["n_shots_used"] == 32
    assert diag["overall_consistency"] == pytest.approx(31 / 32)
    assert diag["n_period_votes_overruled"] == 0
    assert diag["teams_agree"] is True
