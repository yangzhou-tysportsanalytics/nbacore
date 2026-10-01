import polars as pl

from nbacore.ledger import attach_poss_uid, ledger_times

A, B = 1, 2
T0 = 1_000_000


def _fi():
    # one period, 25 Hz, game clock 720 -> 700 over 20 s, one segment
    n = 501
    t = [T0 + 40 * i for i in range(n)]
    gc = [720.0 - 0.04 * i for i in range(n)]
    return pl.DataFrame(
        {
            "period": [1] * n,
            "unix_ms": t,
            "game_clock": gc,
            "segment_id": [0] * n,
        },
        schema_overrides={"period": pl.Int8, "game_clock": pl.Float32},
    )


def _poss():
    return pl.DataFrame(
        {
            "game_id": ["g"] * 3,
            "poss_seq": [0, 1, 2],
            "period": [1, 1, 1],
            "offense_team_id": [A, B, A],
            "end_event_num": [10, 20, None],
            "end_type": ["made_fg", "defensive_rebound", "period_end"],
        }
    )


def _dir():
    return pl.DataFrame(
        {"team_id": [A, B], "period": [1, 1], "attacks_left": [True, False]},
    )


def test_ledger_times_anchor_and_period_end():
    align = pl.DataFrame(
        {
            "event_num": [10, 20],
            "unix_ms": [T0 + 5_000, T0 + 12_000],  # clocks 715 and 708
            "method": ["shot_release", "control"],
            "pctime_sec": [713.0, 707.0],
        }
    )
    led = ledger_times(_poss(), align, _fi(), _dir())
    assert led["t_start_ms"].to_list() == [T0, T0 + 5_000, T0 + 12_000]
    assert led["t_end_ms"].to_list() == [T0 + 5_000, T0 + 12_000, T0 + 20_000]
    assert led["end_time_method"].to_list() == ["shot_release", "control", "period_last_frame"]
    assert led["poss_uid"][1] == f"g:1:{T0 + 5_000}"
    assert led["defense_team_id"].to_list() == [B, A, B]
    assert led["attacks_left"].to_list() == [True, False, True]


def test_ledger_times_implausible_anchor_falls_back_to_clock():
    align = pl.DataFrame(
        {
            "event_num": [10, 20],
            # event 20 anchored at clock 701 while pbp says 707: 6 s *after* -> implausible
            "unix_ms": [T0 + 5_000, T0 + 19_000],
            "method": ["shot_release", "control"],
            "pctime_sec": [713.0, 709.5],
        }
    )
    led = ledger_times(_poss(), align, _fi(), _dir())
    assert led["end_time_method"][1] == "clock_fallback:clock"
    # frame nearest clock 710.0 (bin centre of 709.5 .. 710.5)
    assert abs(led["t_end_ms"][1] - (T0 + 10_000)) <= 40


def test_attach_poss_uid():
    align = pl.DataFrame(
        {
            "event_num": [10, 20],
            "unix_ms": [T0 + 5_000, T0 + 12_000],
            "method": ["shot_release", "control"],
            "pctime_sec": [713.0, 707.0],
        }
    )
    led = ledger_times(_poss(), align, _fi(), _dir())
    win = pl.DataFrame(
        {
            "period": [1, 1],
            "t_cross": [T0 + 7_000, T0 + 1_000],
            "t_terminal": [T0 + 12_600, T0 + 5_000],
            "offense_team_id": [B, B],
        },
        schema_overrides={"period": pl.Int8},
    )
    out = attach_poss_uid(win, led)
    assert out["poss_uid"].to_list() == [f"g:1:{T0 + 5_000}", f"g:1:{T0}"]
    assert abs(out["overlap_frac"][0] - 5_000 / 5_600) < 1e-6
    assert out["offense_match"].to_list() == [True, False]


def test_order_pbp_period_markers_first_and_last():
    from nbacore.ledger import order_pbp

    pbp = pl.DataFrame(
        {
            "event_num": [1, 2, 525, 112],
            "period": [1, 1, 1, 1],
            "pctime_sec": [720.0, 700.0, 720.0, 0.0],
            "msg_type": [10, 1, 12, 13],
        }
    )
    # the late-inserted period start (525) must precede the opening jump ball (0021500506)
    assert order_pbp(pbp)["event_num"].to_list() == [525, 1, 2, 112]
