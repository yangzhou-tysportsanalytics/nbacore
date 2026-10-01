import polars as pl

from nbacore.fouls import foul_states, fouls_to_give, free_throw_links

A, B = 1610612737, 1610612738


def _row(en, per, clk, msg, act, team, pid, desc):
    return {
        "game_id": "g",
        "event_num": en,
        "period": per,
        "pctime_sec": float(clk),
        "msg_type": msg,
        "action_type": act,
        "p1_team_id": team,
        "p1_id": pid,
        "home_desc": desc,
        "visitor_desc": None,
    }


def _pbp(rows):
    return pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Utf8,
            "event_num": pl.Int32,
            "period": pl.Int8,
            "pctime_sec": pl.Float32,
            "msg_type": pl.Int16,
            "action_type": pl.Int16,
            "p1_team_id": pl.Int64,
            "p1_id": pl.Int64,
            "home_desc": pl.Utf8,
            "visitor_desc": pl.Utf8,
        },
    )


def test_penalty_state_and_links():
    rows = [_row(i, 1, 700 - 10 * i, 6, 1, A, 11, "P.FOUL") for i in range(1, 5)]  # 4 fouls
    rows += [
        _row(5, 1, 600, 6, 4, A, 11, "OFFENSIVE.FOUL"),  # not a team foul
        _row(6, 1, 590, 6, 1, A, 12, "P.FOUL"),  # 5th team foul: in penalty
        _row(7, 1, 590, 3, 11, B, 21, "Free Throw 1 of 2"),
        _row(8, 1, 590, 3, 12, B, 21, "MISS Free Throw 2 of 2"),
        _row(9, 1, 500, 6, 11, B, 22, "T.FOUL"),
        _row(10, 1, 500, 3, 16, A, 11, "Free Throw Technical"),
    ]
    f = foul_states(_pbp(rows))
    last = f.filter(pl.col("event_num") == 6).row(0, named=True)
    assert (
        last["team_fouls_before"] == 4
        and last["in_bonus_for_fouled_team"]
        and last["fouls_to_give"] == 0
    )
    assert last["n_free_throws"] == 2
    assert not f.filter(pl.col("event_num") == 5)["counts_as_team_foul"][0]
    assert not f.filter(pl.col("event_num") == 9)["counts_as_team_foul"][0]
    assert last["team_fouls_in_period_after"] == 5 and last["opp_team_fouls_before"] == 0
    ft = free_throw_links(_pbp(rows))
    assert ft["linked_foul_event_num"].to_list() == [6, 6, 9]
    assert ft["is_last_ft"].to_list() == [False, True, False]
    assert ft["made"].to_list() == [True, False, True]
    assert ft["ft_kind"].to_list() == ["regular", "regular", "technical"]


def test_fouls_to_give_last_two_minutes():
    assert fouls_to_give(4, 300.0, 1, 0) == 3
    assert fouls_to_give(4, 100.0, 1, 0) == 1  # one free foul in the last two minutes
    assert fouls_to_give(4, 100.0, 2, 1) == 0
    assert fouls_to_give(5, 250.0, 2, 0) == 1  # overtime quota 3


def test_state_at():
    from nbacore.fouls import state_at

    rows = [_row(i, 4, 300 - 10 * i, 6, 1, A, 11, "P.FOUL") for i in range(1, 4)]
    f = foul_states(_pbp(rows)).with_columns(pl.Series("unix_ms", [100, 200, 300]))
    s = state_at(f, [A, B], 4, 250, 100.0)
    assert s[A]["team_fouls"] == 2 and s[A]["fouls_to_give"] == 1 and not s[A]["in_penalty"]
    assert s[B]["team_fouls"] == 0
