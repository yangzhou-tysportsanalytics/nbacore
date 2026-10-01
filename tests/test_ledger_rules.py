"""pbp_possessions edge cases found on the worst games of the L3 count check."""

import polars as pl

from nbacore.ledger import pbp_possessions

A, B = 1610612745, 1610612739
SCHEMA = {
    "game_id": pl.Utf8,
    "event_num": pl.Int32,
    "period": pl.Int8,
    "pctime_sec": pl.Float32,
    "msg_type": pl.Int16,
    "action_type": pl.Int16,
    "p1_type": pl.Int8,
    "p1_id": pl.Int64,
    "p1_team_id": pl.Int64,
    "p3_team_id": pl.Int64,
    "home_desc": pl.Utf8,
    "visitor_desc": pl.Utf8,
}


def _pbp(rows):
    rows = [*rows, (99, 1, 0.0, 13, 0, 0, None, None, None, None, None)]  # period end
    cols = list(SCHEMA)
    full = [dict(zip(cols, ["g", *r], strict=True)) for r in rows]
    return pl.DataFrame(full, schema=SCHEMA)


def _start():
    return [(1, 1, 720.0, 12, 0, 0, None, None, None, None, None),
            (2, 1, 719.0, 10, 0, 4, 11, A, A, "Jump Ball", None)]  # fmt: skip


def test_held_ball_turnover_is_not_a_second_change():
    rows = _start() + [
        (3, 1, 700.0, 10, 0, 4, 11, A, B, "Jump Ball: Tip to B", None),  # A loses the jump ball
        (4, 1, 700.0, 5, 40, 4, 11, A, None, "Poss Lost Ball Turnover", None),
        (5, 1, 690.0, 1, 1, 4, 21, B, None, None, "Jump Shot"),
    ]
    p = pbp_possessions(_pbp(rows))
    assert p["offense_team_id"].to_list() == [A, B, A]
    assert "unrecorded_change" not in p["end_type"].to_list()


def test_late_team_rebound_after_made_last_ft():
    rows = _start() + [
        (3, 1, 700.0, 3, 11, 4, 11, A, None, "MISS Free Throw 1 of 2", None),
        (4, 1, 700.0, 3, 12, 4, 11, A, None, "Free Throw 2 of 2", None),
        (5, 1, 700.0, 4, 0, 2, A, None, None, "Team Rebound", None),  # dead-ball rebound, late
        (6, 1, 690.0, 2, 1, 4, 21, B, None, None, "MISS Jump Shot"),
    ]
    p = pbp_possessions(_pbp(rows))
    assert p["end_type"].to_list() == ["made_ft", "period_end"]
    assert p["offense_team_id"].to_list() == [A, B]
    assert p.filter(pl.col("end_type") == "defensive_rebound").height == 0


def test_clear_path_free_throws_keep_the_ball():
    rows = _start() + [
        (3, 1, 700.0, 3, 25, 4, 11, A, None, "Free Throw Clear Path 1 of 2", None),
        (4, 1, 700.0, 3, 26, 4, 11, A, None, "Free Throw Clear Path 2 of 2", None),
        (5, 1, 690.0, 1, 1, 4, 11, A, None, "Jump Shot", None),
    ]
    p = pbp_possessions(_pbp(rows))
    assert p["offense_team_id"][0] == A and p["points"][0] == 4 and p["end_type"][0] == "made_fg"


def test_reversed_free_throw_numbering():
    rows = _start() + [
        (6, 1, 700.0, 3, 12, 4, 11, A, None, "Free Throw 2 of 2", None),  # numbered first
        (7, 1, 700.0, 3, 11, 4, 11, A, None, "Free Throw 1 of 2", None),
        (8, 1, 690.0, 1, 1, 4, 21, B, None, None, "Jump Shot"),
    ]
    p = pbp_possessions(_pbp(rows))
    assert p["end_type"].to_list()[:2] == ["made_ft", "made_fg"]
    assert "unrecorded_change" not in p["end_type"].to_list()


def test_pbp_possession_map_putback_and_held_ball():
    from nbacore.ledger import pbp_possession_map

    rows = _start() + [
        (3, 1, 700.0, 2, 1, 4, 11, A, None, "MISS Jump Shot", None),
        (4, 1, 699.0, 1, 1, 4, 12, A, None, "Putback Layup", None),
        (5, 1, 699.0, 4, 0, 4, 12, A, None, "REBOUND (Off:1)", None),  # listed after the make
        (6, 1, 680.0, 10, 0, 4, 21, B, A, None, "Jump Ball: Tip to A"),  # B loses a held ball
        (7, 1, 680.0, 5, 40, 4, 21, B, None, None, "Poss Lost Ball Turnover"),
    ]
    p = pbp_possessions(_pbp(rows))
    m = dict(pbp_possession_map(_pbp(rows)).select("event_num", "poss_seq").iter_rows())
    assert p["end_type"].to_list()[:2] == ["made_fg", "jump_ball_lost"]
    assert m[3] == m[4] == m[5] == 0  # the putback rebound belongs to the scoring possession
    assert m[6] == 1 and m[7] == 1  # held ball: jump ball and its turnover end B's possession
