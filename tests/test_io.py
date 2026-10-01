"""Tests for nbacore.io (raw loader, schema validation, pbp) and nbacore.court."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from nbacore import court, paths
from nbacore.io import (
    BALL_ID,
    SchemaError,
    game_to_events,
    game_to_frames,
    game_to_roster,
    load_pbp,
    validate_game,
)

# extracted game JSONs live in the data root's scratch area (see scripts/download_raw.py)
TINY_JSONS = sorted(paths.scratch_dir("extracted").glob("*/*.json"))


# ------------------------------------------------------------------------------------------
# synthetic game
# ------------------------------------------------------------------------------------------


def _moment(period: int, unix: int, gc: float, sc: float | None, with_ball: bool = True):
    rows = [[BALL_ID, BALL_ID, 47.0, 25.0, 5.0]] if with_ball else []
    for i in range(10):
        tid = 100 if i < 5 else 200
        rows.append([tid, 1000 + i, float(5 * i + 1), 10.0 + i, 0.0])
    return [period, unix, gc, sc, None, rows]


def _team(tid: int, abbr: str):
    return {
        "name": f"Team {abbr}",
        "teamid": tid,
        "abbreviation": abbr,
        "players": [
            {
                "lastname": f"L{i}",
                "firstname": f"F{i}",
                "playerid": pid,
                "jersey": str(i),
                "position": "G",
            }
            for i, pid in enumerate(
                range(1000 + (0 if tid == 100 else 5), 1005 + (0 if tid == 100 else 5))
            )
        ],
    }


@pytest.fixture
def toy_game() -> dict:
    home, visitor = _team(100, "HOM"), _team(200, "VIS")
    t0 = 1_447_287_043_000
    # event 1: frames 0..9 ; event 2 overlaps: frames 5..14 ; event 3: no moments ; event 4: one frame w/o ball
    ev1 = [_moment(1, t0 + 40 * i, 720 - 0.04 * i, 24 - 0.04 * i) for i in range(10)]
    ev2 = [_moment(1, t0 + 40 * i, 720 - 0.04 * i, 24 - 0.04 * i) for i in range(5, 15)]
    ev4 = [_moment(1, t0 + 40 * 20, 719.0, None, with_ball=False)]
    return {
        "gameid": "21500001",
        "gamedate": "2015-10-27",
        "events": [
            {"eventId": "1", "home": home, "visitor": visitor, "moments": ev1},
            {"eventId": "2", "home": home, "visitor": visitor, "moments": ev2},
            {"eventId": "3", "home": home, "visitor": visitor, "moments": []},
            {"eventId": "4", "home": home, "visitor": visitor, "moments": ev4},
        ],
    }


def test_frames_long_table(toy_game):
    f = game_to_frames(toy_game)
    assert f.height == 10 * 11 + 10 * 11 + 10
    assert f["game_id"].unique().to_list() == ["0021500001"]
    assert f.filter(pl.col("team_id") == BALL_ID).height == 20
    assert f["period"].dtype == pl.Int8 and f["unix_ms"].dtype == pl.Int64
    # shot clock null survives
    assert f.filter(pl.col("event_id") == 4)["shot_clock"].null_count() == 10
    # duplicates present (not deduplicated here)
    assert f.select(["period", "unix_ms"]).unique().height == 16


def test_events_and_roster(toy_game):
    e = game_to_events(toy_game)
    assert e["n_moments"].to_list() == [10, 10, 0, 1]
    assert e["game_date"].unique().to_list() == ["2015-10-27"]
    assert e.filter(pl.col("event_id") == 3)["period"].item() is None
    assert e["home_team_id"].unique().to_list() == [100]
    r = game_to_roster(toy_game)
    assert r.height == 10
    assert set(r["side"].unique().to_list()) == {"home", "visitor"}
    assert r.filter(pl.col("side") == "visitor")["team_id"].unique().to_list() == [200]


def test_validate_stats(toy_game):
    s = validate_game(toy_game)
    assert s.game_id == "0021500001"
    assert s.n_events == 4 and s.n_events_no_moments == 1
    assert s.n_moments_raw == 21 and s.n_frames_unique == 16
    assert s.dup_fraction == pytest.approx(1 - 16 / 21)
    assert s.n_duplicate_conflicts == 0
    assert s.entities_per_frame == {10: 1, 11: 15}
    assert s.n_frames_no_ball == 1
    assert s.n_frames_shotclock_null == 1
    assert s.unix_dt_ms_mode == 40
    assert s.n_roster_variants == 1 and s.roster_size == 10
    assert s.n_players_on_court_total == 10
    assert s.player_z_nonzero == 0 and s.ball_z_max == 5.0


def test_validate_hard_errors(toy_game):
    bad = dict(toy_game)
    bad["events"] = [dict(toy_game["events"][0])]
    bad["events"][0]["moments"] = [toy_game["events"][0]["moments"][0][:5]]  # moment too short
    with pytest.raises(SchemaError):
        validate_game(bad)
    bad2 = {"gameid": "1", "events": []}
    with pytest.raises(SchemaError):
        validate_game(bad2)


# ------------------------------------------------------------------------------------------
# pbp
# ------------------------------------------------------------------------------------------

PBP_HEADER = (
    ",EVENTMSGACTIONTYPE,EVENTMSGTYPE,EVENTNUM,GAME_ID,HOMEDESCRIPTION,NEUTRALDESCRIPTION,"
    "PCTIMESTRING,PERIOD,PERSON1TYPE,PERSON2TYPE,PERSON3TYPE,PLAYER1_ID,PLAYER1_NAME,"
    "PLAYER1_TEAM_ABBREVIATION,PLAYER1_TEAM_CITY,PLAYER1_TEAM_ID,PLAYER1_TEAM_NICKNAME,"
    "PLAYER2_ID,PLAYER2_NAME,PLAYER2_TEAM_ABBREVIATION,PLAYER2_TEAM_CITY,PLAYER2_TEAM_ID,"
    "PLAYER2_TEAM_NICKNAME,PLAYER3_ID,PLAYER3_NAME,PLAYER3_TEAM_ABBREVIATION,PLAYER3_TEAM_CITY,"
    "PLAYER3_TEAM_ID,PLAYER3_TEAM_NICKNAME,SCORE,SCOREMARGIN,VISITORDESCRIPTION,WCTIMESTRING"
)
PBP_ROWS = [
    "0,0,12,0,21500758,,,12:00,1,0.0,0,0,0,,,,,,0,,,,,,0,,,,,,,,,9:11 PM",
    "1,1,1,7,21500758,Gobert 2' Dunk (2 PTS),,11:39,1,4.0,0,0,203497,Rudy Gobert,UTA,Utah,1610612762.0,Jazz,0,,,,,,0,,,,,,0 - 2,2,,9:12 PM",
    "2,1,1,9,21500758,,,11:20,1,5.0,0,0,202328,Greg Monroe,MIL,Milwaukee,1610612749.0,Bucks,0,,,,,,0,,,,,,2 - 2,TIE,Monroe 3' Layup (2 PTS),9:12 PM",
]


def test_load_pbp(tmp_path):
    p = tmp_path / "pbp.csv"
    p.write_text("\n".join([PBP_HEADER, *PBP_ROWS]) + "\n")
    df = load_pbp(p)
    assert df.height == 3
    assert df["game_id"].unique().to_list() == ["0021500758"]
    assert df["msg_type_name"].to_list() == ["PERIOD_BEGIN", "FIELD_GOAL_MADE", "FIELD_GOAL_MADE"]
    assert df["pctime_sec"].to_list() == pytest.approx([720.0, 699.0, 680.0])
    assert df["score_visitor"].to_list() == [None, 0, 2]
    assert df["score_home"].to_list() == [None, 2, 2]
    assert df["score_margin"].to_list() == [None, 2, 0]
    assert df["p1_team_id"].to_list() == [None, 1610612762, 1610612749]
    assert df["p1_type"].to_list() == [0, 4, 5]


# ------------------------------------------------------------------------------------------
# court
# ------------------------------------------------------------------------------------------


def test_court_geometry():
    line = court.three_point_line(left=True)
    assert line.shape[1] == 2
    assert np.allclose(line[0], [0.0, court.HOOP_Y - 22.0])
    assert np.allclose(line[-1], [0.0, court.HOOP_Y + 22.0])
    # farthest point of the arc from the hoop is 23.75 ft
    d = np.hypot(line[:, 0] - court.HOOP_X_LEFT, line[:, 1] - court.HOOP_Y)
    assert d.max() == pytest.approx(23.75, abs=1e-6)
    assert court.is_three_point_location(np.array([3.0]), np.array([2.0]))[0]  # corner
    assert court.is_three_point_location(np.array([30.0]), np.array([25.0]))[0]  # top
    assert not court.is_three_point_location(np.array([10.0]), np.array([25.0]))[0]  # paint
    assert not court.is_three_point_location(np.array([20.0]), np.array([25.0]))[0]  # mid-range
    # right basket mirror
    assert court.is_three_point_location(np.array([91.0]), np.array([2.0]), left=False)[0]


def test_to_left_half():
    x = np.array([10.0, 80.0])
    y = np.array([5.0, 5.0])
    xo, yo = court.to_left_half(x, y, np.array([True, False]))
    assert np.allclose(xo, [10.0, 14.0])
    assert np.allclose(yo, [5.0, 45.0])


# ------------------------------------------------------------------------------------------
# integration on real tiny data (skipped when not downloaded)
# ------------------------------------------------------------------------------------------


@pytest.mark.skipif(not TINY_JSONS, reason="tiny raw data not present")
def test_real_game_invariants():
    from nbacore.io import load_game_json

    g = load_game_json(TINY_JSONS[0])
    s = validate_game(g)
    assert s.unix_dt_ms_mode == 40  # 25 Hz
    # split frames (ball-only + players-only moments) exist in some games, but are rare
    assert s.n_duplicate_conflicts <= 0.01 * s.n_frames_unique
    assert s.n_events_unix_nonmonotonic == 0
    assert s.player_z_nonzero == 0
    assert 0.4 < s.dup_fraction < 0.8
    assert s.n_roster_variants == 1
    # shot clock can be missing for long stretches in some games (see data_schema.md)
    assert (
        s.frac_shotclock_null_when_gc_le_24 is None
        or 0.0 <= s.frac_shotclock_null_when_gc_le_24 <= 1.0
    )
    f = game_to_frames(g)
    assert f.filter(pl.col("team_id") == BALL_ID)["player_id"].unique().to_list() == [BALL_ID]
