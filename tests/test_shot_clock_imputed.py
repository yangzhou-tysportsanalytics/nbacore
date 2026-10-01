import polars as pl

from nbacore.shot_clock import RESET_SCHEMA, shot_clock_filled, shot_clock_imputed

T0 = 1_000_000


def _fi(seconds: float, shot_clock=None, gc0: float = 600.0):
    n = int(seconds * 25) + 1
    return pl.DataFrame(
        {
            "period": [1] * n,
            "unix_ms": [T0 + 40 * i for i in range(n)],
            "game_clock": [gc0 - 0.04 * i for i in range(n)],
            "shot_clock": shot_clock(n) if shot_clock else [None] * n,
            "clock_frozen": [False] * n,
        },
        schema_overrides={
            "period": pl.Int8,
            "game_clock": pl.Float32,
            "shot_clock": pl.Float32,
        },
    )


def _resets(rows):
    return pl.DataFrame(rows, schema=RESET_SCHEMA, orient="row")


def _at(df, sec):
    return df.filter(pl.col("unix_ms") == T0 + int(sec * 1000))["shot_clock_imputed"][0]


def test_full_reset_holds_until_run():
    fi = _fi(20)
    imp = shot_clock_imputed(fi, _resets([(1, T0 + 5000, T0 + 8000, "full", "x")]))
    assert abs(_at(imp, 4) - 20.0) < 1e-3  # 24 - 4 s since the period start
    assert abs(_at(imp, 6) - 24.0) < 1e-3  # held until the inbound catch
    assert abs(_at(imp, 12) - 20.0) < 1e-3  # running from 8 s


def test_min14_raises_low_clock_and_keeps_high():
    fi = _fi(20)
    low = shot_clock_imputed(fi, _resets([(1, T0 + 15000, None, "min14", "foul")]))
    assert abs(_at(low, 15) - 14.0) < 1e-3  # 24 - 15 = 9 -> 14
    high = shot_clock_imputed(fi, _resets([(1, T0 + 5000, None, "min14", "foul")]))
    assert abs(_at(high, 5) - 19.0) < 1e-3  # 19 >= 14: kept


def test_off_when_game_clock_is_lower():
    fi = _fi(10, gc0=20.0)
    imp = shot_clock_imputed(fi, _resets([(1, T0 + 1000, None, "full", "x")]))
    row = imp.filter(pl.col("unix_ms") == T0 + 2000).row(0, named=True)
    assert row["imputed_off"] and abs(row["shot_clock_imputed"] - 18.0) < 1e-3


def test_stuck_24_flagged():
    # 10 s at 24 while the game clock runs, then a normal countdown
    fi = _fi(
        20, shot_clock=lambda n: [24.0 if i < 251 else 24.0 - 0.04 * (i - 250) for i in range(n)]
    )
    m = shot_clock_filled(fi)["fill_method"]
    assert m[0] == "stuck_24" and m[-1] == "observed"
    fi_ok = _fi(
        20, shot_clock=lambda n: [24.0 if i < 76 else 24.0 - 0.04 * (i - 75) for i in range(n)]
    )
    assert shot_clock_filled(fi_ok)["fill_method"][0] == "observed"  # 3 s inbound hold is fine
