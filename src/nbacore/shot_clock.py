"""``shot_clock_filled`` per frame (L2).

``fill_method`` per frame of the frame index:

* ``observed``: the tracked shot clock, 0 ≤ value ≤ 24;
* ``suspect``: a tracked value outside [0, 24] (not overwritten, not used for filling);
* ``stuck_24``: the tracked value stays at 24 while the game clock runs more than
  ``STUCK_24_S`` = 8 s — impossible (the throw-in must be released within 5 s and the shot clock
  starts at the touch, 7-II-b, 8-III-a): a feed artefact, ~1.6 % of running frames; null;
* ``off_game_clock``: shot clock missing while the game clock is below 24 s, or below the last
  observed shot clock (the shot clock is switched off when the game clock is shorter): the game
  clock is used;
* ``carry_forward``: shot clock missing while the game clock is frozen (dead ball, free throws):
  the last observed value in the period;
* ``missing``: shot clock missing while the clock runs with more than 24 s left (not filled);
* ``unavailable``: games with no shot clock at all (the 12 HoopTransformer games); null until the
  rule-based rebuild of v1.2.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SHOT_CLOCK_MAX = 24.0
STUCK_24_S = 8.0


def _flag_stuck_24(fi: pl.DataFrame) -> pl.DataFrame:
    """``_stuck``: frames of a run of tracked values at 24 during which the game clock runs more
    than ``STUCK_24_S`` (frames sorted by period, time)."""
    at24 = pl.col("shot_clock").fill_null(-1) >= SHOT_CLOCK_MAX - 0.01
    fi = fi.with_columns(
        (at24 != at24.shift(1).over("period")).fill_null(True).cum_sum().alias("_run"),
        at24.alias("_at24"),
    )
    span = (pl.col("game_clock").max() - pl.col("game_clock").min()).over("_run")
    return fi.with_columns((pl.col("_at24") & (span > STUCK_24_S)).alias("_stuck")).drop(
        "_run", "_at24"
    )


def shot_clock_filled(frame_index: pl.DataFrame) -> pl.DataFrame:
    fi = frame_index.sort(["period", "unix_ms"])
    if fi["shot_clock"].null_count() == fi.height:
        return fi.select(
            "period",
            "unix_ms",
            pl.lit(None, dtype=pl.Float32).alias("shot_clock_filled"),
            pl.lit("unavailable").alias("fill_method"),
        )
    fi = _flag_stuck_24(fi)
    sc = pl.col("shot_clock")
    ok = sc.is_not_null() & (sc >= 0) & (sc <= SHOT_CLOCK_MAX) & ~pl.col("_stuck")
    fi = fi.with_columns(
        pl.when(ok).then(sc).otherwise(None).forward_fill().over("period").alias("_last"),
    )
    gc = pl.col("game_clock")
    off = sc.is_null() & ((gc < SHOT_CLOCK_MAX) | (gc < pl.col("_last")).fill_null(False))
    carry = sc.is_null() & pl.col("clock_frozen") & pl.col("_last").is_not_null()
    method = (
        pl.when(ok)
        .then(pl.lit("observed"))
        .when(pl.col("_stuck"))
        .then(pl.lit("stuck_24"))
        .when(sc.is_not_null())
        .then(pl.lit("suspect"))
        .when(off)
        .then(pl.lit("off_game_clock"))
        .when(carry)
        .then(pl.lit("carry_forward"))
        .otherwise(pl.lit("missing"))
    )
    value = (
        pl.when(ok)
        .then(sc)
        .when(sc.is_null() & off)
        .then(gc)
        .when(sc.is_null() & carry)
        .then(pl.col("_last"))
        .otherwise(None)
    )
    return fi.select(
        "period",
        "unix_ms",
        value.cast(pl.Float32).alias("shot_clock_filled"),
        method.alias("fill_method"),
    )


# ---------------------------------------------------------------------------------------------
# rule-based rebuild (v1.2): for the 12 games without a tracked shot clock
# ---------------------------------------------------------------------------------------------

RESET_SCHEMA: dict[str, pl.DataType] = {
    "period": pl.Int8,
    "unix_ms": pl.Int64,  # when the new value applies
    "t_run_ms": pl.Int64,  # when the shot clock starts running again (inbound catch / control)
    "rule": pl.Utf8,  # full (24) | min14 (max(remaining, 14))
    "cause": pl.Utf8,
}
NONSHOOTING_COMMON = ("personal", "personal_block", "personal_take", "loose_ball", "inbound")
HALF_COURT_X = 47.0
THROW_IN_CAUSES = (
    "poss:made_basket_inbound",
    "poss:made_ft_inbound",
    "poss:turnover",
    "foul_",
    "kicked_ball",
    "defensive_3_seconds",
    "technical",
    "delay_of_game",
    "flagrant",
)
INBOUND_TOUCH_MS = 300  # ball back on the court -> touched by a player


def _offense_at(ledger: pl.DataFrame):
    by = {
        int(p): (g["t_start_ms"].to_numpy(), g["offense_team_id"].to_list())
        for (p,), g in ledger.filter(pl.col("t_start_ms").is_not_null())
        .sort("t_start_ms")
        .group_by("period")
    }

    def f(period: int, t: int):
        if period not in by:
            return None
        s, off = by[period]
        i = int(np.searchsorted(s, t, side="right")) - 1
        return off[i] if i >= 0 else None

    return f


def shot_clock_resets(
    ledger: pl.DataFrame,
    shots: pl.DataFrame,
    inbounds: pl.DataFrame,
    foul_st: pl.DataFrame,
    pbp_game: pl.DataFrame,
    align_game: pl.DataFrame,
    frame_index: pl.DataFrame,
    ball: pl.DataFrame,
    direction: pl.DataFrame,
    rim_mode: str = "hold",
    live_lag_ms: int = 0,
    dedupe_rebound_s: float = 6.0,
) -> pl.DataFrame:
    """Shot-clock resets of one game from the 2015-16 rules (``nbacore.rules_2015_16``).

    * ``full`` at every ledger possession start (7-IV-c(1)), at the rim contact of a missed FG
      (7-IV-c(2)), at flagrant fouls (7-IV-c(6)) and at defensive non-shooting fouls without free
      throws with a backcourt throw-in (7-IV-c(3));
    * ``min14`` at such fouls with a frontcourt throw-in (7-IV-d(1)), at defensive three seconds /
      technicals / delay warnings on the defence (7-IV-d(2), (3)) and kicked balls (7-IV-d(4)).

    After a reset the clock waits for the next inbound catch (7-II-b) before the next reset, or,
    after a rim contact, for the offensive rebound control. Without an inbound event, for resets
    followed by a throw-in: the ball leaving the court and coming back (+ ``INBOUND_TOUCH_MS``).
    ``ball``: ball rows (``period, unix_ms, x, y``).

    Options (defaults chosen on 12 train games, ``scripts/l3_shot_clock_backtest.py``):
    ``rim_mode`` "hold" (24 from the rim contact, running from the first control of either team)
    or "control" (reset shown at the control); ``live_lag_ms`` operator delay added to live-ball
    resets; ``dedupe_rebound_s``: no second reset at a defensive-rebound possession start within
    that many seconds after a rim reset. All variants were within 0.1 s MAE of each other.
    """
    off_at = _offense_at(ledger)
    at = dict(zip(align_game["event_num"].to_list(), align_game["unix_ms"].to_list(), strict=True))
    left = {(r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)}
    fi = frame_index.sort("period", "unix_ms")
    rows: list[tuple] = []
    missed = shots.filter(~pl.col("made").fill_null(True) & pl.col("t_rim_ms").is_not_null())
    rim_rows = []
    for s in missed.iter_rows(named=True):
        per, t_rim, ctrl = int(s["period"]), int(s["t_rim_ms"]), s["t_control_ms"]
        if rim_mode == "hold":  # 24 from the rim contact, running from the control (either team)
            rim_rows.append((per, t_rim, ctrl, "full", "rim"))
        else:  # "control": the reset is shown at the control
            t = int(ctrl) + live_lag_ms if ctrl is not None else t_rim
            rim_rows.append((per, t, None, "full", "rim"))
    rows.extend(rim_rows)
    rim_t = {}
    for per, t, *_ in rim_rows:
        rim_t.setdefault(per, []).append(t)
    live = ("poss:defensive_rebound", "poss:turnover", "poss:unrecorded_change")
    for r in ledger.filter(pl.col("t_start_ms").is_not_null()).iter_rows(named=True):
        per, t = int(r["period"]), int(r["t_start_ms"])
        cause = f"poss:{r['start_type']}"
        if (
            cause == "poss:defensive_rebound"
            and dedupe_rebound_s > 0
            and any(t - dedupe_rebound_s * 1000 <= x <= t for x in rim_t.get(per, []))
        ):
            continue  # already reset at the rim / control of the missed shot
        rows.append((per, t + (live_lag_ms if cause in live else 0), None, "full", cause))

    def restart_half(period: int, t: int, offense: int | None) -> str | None:
        run = fi.filter(
            (pl.col("period") == period) & (pl.col("unix_ms") > t) & ~pl.col("clock_frozen")
        ).head(1)
        if run.height == 0 or offense is None or (offense, period) not in left:
            return None
        t_run = int(run["unix_ms"][0])
        b = ball.filter(
            (pl.col("period") == period) & pl.col("unix_ms").is_between(t_run - 200, t_run + 400)
        )
        if b.height == 0:
            return None
        x = float(b["x"].median())
        return "front" if (x < HALF_COURT_X) == left[(offense, period)] else "back"

    for f in foul_st.iter_rows(named=True):
        t = at.get(f["event_num"])
        if t is None or f["committed_by_team"] is None:
            continue
        per = int(f["period"])
        offense = off_at(per, int(t))
        defensive = offense is not None and f["committed_by_team"] != offense
        d = f["foul_detail"]
        if d in ("flagrant_1", "flagrant_2"):
            rows.append((per, int(t), None, "full", "flagrant"))
        elif d in NONSHOOTING_COMMON and defensive and f["n_free_throws"] == 0:
            half = restart_half(per, int(t), offense)
            if half is not None:
                rule = "min14" if half == "front" else "full"
                rows.append((per, int(t), None, rule, f"foul_{half}court"))
        elif defensive and d in ("defensive_3_seconds", "technical", "delay_of_game"):
            rows.append((per, int(t), None, "min14", d))
    kicked = pbp_game.filter((pl.col("msg_type") == 7) & (pl.col("action_type") == 5))
    for ev in kicked.iter_rows(named=True):
        t = at.get(ev["event_num"])
        if t is None:
            continue
        offense = off_at(int(ev["period"]), int(t))
        if offense is not None and ev["p1_team_id"] != offense:
            rows.append((int(ev["period"]), int(t), None, "min14", "kicked_ball"))

    res = pl.DataFrame(rows, schema=RESET_SCHEMA, orient="row")
    res = res.sort("period", "unix_ms").with_columns(
        pl.col("unix_ms").shift(-1).over("period").alias("_next")
    )
    # the clock starts at the next inbound catch (if it comes before the next reset)
    ib = (
        inbounds.filter(pl.col("t_catch_ms").is_not_null())
        .select(
            pl.col("period").cast(pl.Int8), pl.col("t_catch_ms").cast(pl.Int64).alias("t_catch")
        )
        .sort("period", "t_catch")
    )
    res = res.join_asof(ib, left_on="unix_ms", right_on="t_catch", by="period", strategy="forward")
    res = res.with_columns(
        pl.when(
            pl.col("t_run_ms").is_null()
            & pl.col("t_catch").is_not_null()
            & (pl.col("t_catch") < pl.col("_next").fill_null(2**62))
            & (pl.col("t_catch") - pl.col("unix_ms") <= 180_000)
        )
        .then(pl.col("t_catch"))
        .otherwise(pl.col("t_run_ms"))
        .alias("t_run_ms")
    )
    # no inbound event: the ball goes out of bounds and comes back in (release of the throw-in)
    bs = ball.sort("period", "unix_ms")
    by_p = {
        int(p_): (g["unix_ms"].to_numpy(), g["x"].to_numpy(), g["y"].to_numpy())
        for (p_,), g in bs.group_by("period")
    }
    t_run = res["t_run_ms"].to_list()
    for i, r in enumerate(res.iter_rows(named=True)):
        if t_run[i] is not None or not r["cause"].startswith(THROW_IN_CAUSES):
            continue
        if int(r["period"]) not in by_p:
            continue
        bt, bx, by = by_p[int(r["period"])]
        hi = min(r["_next"] if r["_next"] is not None else 2**62, r["unix_ms"] + 60_000)
        w = np.flatnonzero((bt > r["unix_ms"]) & (bt <= hi))
        if w.size == 0:
            continue
        x, y = bx[w], by[w]
        out = (x < -0.5) | (x > 94.5) | (y < -0.5) | (y > 50.5)
        if not out.any():
            continue
        first_out = int(np.argmax(out))
        inside = (x > 1) & (x < 93) & (y > 1) & (y < 49)
        back = np.flatnonzero(inside[first_out:])
        if back.size:
            t_run[i] = int(bt[w[first_out + back[0]]]) + INBOUND_TOUCH_MS
    res = res.with_columns(pl.Series("t_run_ms", t_run, dtype=pl.Int64))
    return res.select(list(RESET_SCHEMA)).cast(RESET_SCHEMA)


def shot_clock_imputed(frame_index: pl.DataFrame, resets: pl.DataFrame) -> pl.DataFrame:
    """Shot clock per frame from the resets: the value set at a reset holds until ``t_run_ms``
    and then falls with the game clock; ``min14`` keeps max(remaining, 14); shown as the game clock
    when that is lower (7-II-i, ``imputed_off``). Before the first reset of a period: 24."""
    out = []
    for (per,), g in frame_index.sort("period", "unix_ms").group_by("period", maintain_order=True):
        t = g["unix_ms"].to_numpy()
        gc = g["game_clock"].to_numpy().astype(np.float64)

        def gc_at(x: int, t=t, gc=gc) -> float:
            return float(gc[min(int(np.searchsorted(t, x, side="left")), len(t) - 1)])

        rs = resets.filter(pl.col("period") == per).sort("unix_ms")
        v, t_run, gc_run = SHOT_CLOCK_MAX, int(t[0]), float(gc[0])
        vs, runs, gcs = [], [], []
        for row in rs.iter_rows(named=True):
            cur = v - max(0.0, gc_run - gc_at(row["unix_ms"])) if row["unix_ms"] >= t_run else v
            cur = min(max(cur, 0.0), SHOT_CLOCK_MAX)
            v = SHOT_CLOCK_MAX if row["rule"] == "full" else max(cur, 14.0)
            t_run = int(row["t_run_ms"] if row["t_run_ms"] is not None else row["unix_ms"])
            gc_run = gc_at(t_run)
            vs.append(v)
            runs.append(t_run)
            gcs.append(gc_run)
        k = np.searchsorted(rs["unix_ms"].to_numpy(), t, side="right") - 1
        vs_, runs_, gcs_ = (
            np.array([SHOT_CLOCK_MAX] + vs),
            np.array([int(t[0])] + runs),
            np.array([float(gc[0])] + gcs),
        )
        j = k + 1  # index 0 = period start state
        val = np.where(t >= runs_[j], vs_[j] - np.clip(gcs_[j] - gc, 0, None), vs_[j])
        val = np.clip(val, 0.0, SHOT_CLOCK_MAX)
        off = val > gc
        out.append(
            pl.DataFrame(
                {
                    "period": g["period"],
                    "unix_ms": t,
                    "shot_clock_imputed": np.where(off, gc, val).astype(np.float32),
                    "imputed_off": off,
                }
            )
        )
    return pl.concat(out)


SHOT_CLOCK_TABLE_SCHEMA: dict[str, pl.DataType] = {
    "period": pl.Int8,
    "unix_ms": pl.Int64,
    "shot_clock_filled": pl.Float32,
    "fill_method": pl.Utf8,
    "shot_clock_imputed": pl.Boolean,
}


def shot_clock_table(frame_index: pl.DataFrame, imputed: pl.DataFrame) -> pl.DataFrame:
    """Per-frame shot clock of the L3 release: ``shot_clock_filled`` where
    it is observed / off the game clock / carried forward; elsewhere (``missing``, ``suspect``,
    ``stuck_24``, and every frame of the games without a shot clock) the rule-based value with
    ``fill_method = imputed:<original method>`` and ``shot_clock_imputed = True``."""
    f = shot_clock_filled(frame_index).join(
        imputed.select("period", "unix_ms", "shot_clock_imputed"),
        on=["period", "unix_ms"],
        how="left",
    )
    keep = pl.col("fill_method").is_in(["observed", "off_game_clock", "carry_forward"])
    return f.select(
        "period",
        "unix_ms",
        pl.when(keep)
        .then(pl.col("shot_clock_filled"))
        .otherwise(pl.col("shot_clock_imputed"))
        .cast(pl.Float32)
        .alias("shot_clock_filled"),
        pl.when(keep)
        .then(pl.col("fill_method"))
        .otherwise(pl.lit("imputed:") + pl.col("fill_method"))
        .alias("fill_method"),
        (~keep).alias("shot_clock_imputed"),
    ).cast(SHOT_CLOCK_TABLE_SCHEMA)
