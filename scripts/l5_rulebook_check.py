"""Check the 2015-16 rule constants against the data (L5; docs/external_sources.md).

1. Shot clock after rebounds of a missed FG (7-IV-c(2), 7-IV-b(5)): in the tracked shot clock
   between the release and the rebound control + 3 s, is there a reset, and to which value?
   Split by offensive / defensive rebound and by whether a rim contact was detected. Only games
   with a shot clock and rebounds with more than 24 s on the game clock.
2. Shot clock after defensive non-shooting fouls without free throws (7-IV-c(3), 7-IV-d(1)):
   the value shown while the clock is frozen vs the value when it stopped, by throw-in court half (ball x
   when the clock restarts, attack direction of the fouled team; |x − 47| < 3 ft = midcourt).
3. Penalty situation (12B-V): replay the team fouls of every period and compare "in penalty"
   with "free throws were awarded" for defensive non-shooting common fouls (personal, block,
   take, loose ball) in pbp.

Inputs: release (frame_index, frames, pbp, attack_direction) and the L2 build
``scratch/l2_full`` (events, pbp_align) — v1.1 content. Writes ``reports/l5_rulebook_check.md``
and ``.json``.

Usage:
    uv run python scripts/l5_rulebook_check.py [--games N] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore import rules_2015_16 as R
from nbacore.events.pbp_align import FOUL_DETAIL

REPO = Path(__file__).resolve().parents[1]
L2 = paths.scratch_dir("l2_full")
FT_RE = re.compile(r"Free Throw (Technical|Flagrant|Clear Path)?", re.I)
NONSHOOTING = {1: "personal", 27: "personal_block", 28: "personal_take", 3: "loose_ball"}


def _resets(fi: pl.DataFrame, period: int, t0: int, t1: int) -> list[float]:
    """Values the tracked shot clock jumps up to (> +1 s between observations) in [t0, t1]."""
    w = fi.filter(
        (pl.col("period") == period)
        & pl.col("unix_ms").is_between(t0, t1)
        & pl.col("shot_clock").is_not_null()
    )
    if w.height < 2:
        return []
    sc = w["shot_clock"].to_numpy()
    up = np.flatnonzero(np.diff(sc) > 1.0) + 1
    return [float(sc[i]) for i in up]


def _bucket(v: float) -> str:
    if v >= 23.0:
        return "24"
    if 13.0 <= v <= 15.0:
        return "14"
    return "other"


def rebound_check(g: str, fi: pl.DataFrame, shots: pl.DataFrame) -> list[dict]:
    out = []
    miss = shots.filter(
        ~pl.col("made").fill_null(True)
        & pl.col("t_release_ms").is_not_null()
        & pl.col("t_control_ms").is_not_null()
        & pl.col("control_team_id").is_not_null()
    )
    gc = fi.select("period", "unix_ms", "game_clock")
    for s in miss.iter_rows(named=True):
        t0, t1 = int(s["t_release_ms"]), int(s["t_control_ms"]) + 3000
        c = gc.filter((pl.col("period") == s["period"]) & (pl.col("unix_ms") <= t1)).tail(1)
        if c.height == 0 or c["game_clock"][0] is None or c["game_clock"][0] <= R.SHOT_CLOCK_S:
            continue
        obs = fi.filter(
            (pl.col("period") == s["period"])
            & pl.col("unix_ms").is_between(t0, t1)
            & pl.col("shot_clock").is_not_null()
        ).height
        rs = _resets(fi, s["period"], t0, t1)
        at_release = _sc_at(fi, s["period"], t0 - 500, t0, last=True)
        if rs:
            reset = _bucket(max(rs))
        elif at_release is not None and at_release >= 23.5:
            reset = "already_24"  # reset before the release (tip after an earlier rim contact)
        else:
            reset = "none"
        out.append(
            {
                "game_id": g,
                "event_uid": s["event_uid"],
                "offensive": bool(s["offensive_rebound"]),
                "rim": s["t_rim_ms"] is not None,
                "blocked": bool(s["blocked"]),
                "observed": obs >= 10,
                "reset": reset,
                "reset_value": max(rs) if rs else None,
            }
        )
    return out


def _sc_at(fi: pl.DataFrame, period: int, t_lo: int, t_hi: int, last: bool) -> float | None:
    w = fi.filter(
        (pl.col("period") == period)
        & pl.col("unix_ms").is_between(t_lo, t_hi)
        & pl.col("shot_clock").is_not_null()
    )
    if w.height == 0:
        return None
    return float(w["shot_clock"][-1] if last else w["shot_clock"][0])


def foul_shot_clock_check(
    g: str,
    fi: pl.DataFrame,
    stops: pl.DataFrame,
    align: pl.DataFrame,
    pbp: pl.DataFrame,
    ball: pl.DataFrame,
    direction: pl.DataFrame,
    no_ft: set[int],
) -> list[dict]:
    out = []
    fouls = align.filter(
        (pl.col("msg_type") == 6)
        & pl.col("action_type").is_in(list(NONSHOOTING))
        & pl.col("stop_event_uid").is_not_null()
        & pl.col("event_num").is_in(list(no_ft))
    )
    team = dict(zip(pbp["event_num"].to_list(), pbp["p1_team_id"].to_list(), strict=True))
    teams = [int(t) for t in pbp["p1_team_id"].drop_nulls().unique().to_list()]
    st = {r["event_uid"]: r for r in stops.iter_rows(named=True)}
    left = {(r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)}
    for f in fouls.iter_rows(named=True):
        s = st.get(f["stop_event_uid"])
        fouler = team.get(f["event_num"])
        if s is None or fouler is None or len(teams) != 2 or s["t_last_running_ms"] is None:
            continue
        if s["game_clock_at_stop"] is None or s["game_clock_at_stop"] <= R.SHOT_CLOCK_S:
            continue
        fouled = teams[0] if teams[1] == fouler else teams[1]
        p, t_run, t_end = f["period"], int(s["t_last_running_ms"]), int(s["t_end_ms"])
        last = fi.filter(
            (pl.col("period") == p)
            & pl.col("unix_ms").is_between(t_run - 1000, t_run)
            & pl.col("shot_clock").is_not_null()
        ).tail(1)
        # value when the clock stopped: last running value minus the game time still elapsed
        before = (
            float(last["shot_clock"][0] - (last["game_clock"][0] - s["game_clock_at_stop"]))
            if last.height
            else None
        )
        frozen = fi.filter(
            (pl.col("period") == p)
            & pl.col("unix_ms").is_between(t_run, t_end)
            & pl.col("clock_frozen")
            & pl.col("shot_clock").is_not_null()
        )
        during = float(frozen["shot_clock"][-1]) if frozen.height else None
        b = ball.filter((pl.col("period") == p) & pl.col("unix_ms").is_between(t_end, t_end + 400))
        half = "unknown"
        if b.height and (fouled, p) in left:
            x = float(b["x"][0])
            if abs(x - 47.0) < 3.0:
                half = "midcourt"
            else:
                half = "frontcourt" if (x < 47.0) == left[(fouled, p)] else "backcourt"
        rec = {
            "game_id": g,
            "event_num": f["event_num"],
            "foul": NONSHOOTING[int(f["action_type"])],
            "half": half,
            "before": before,
            "after": during,
            "outcome": None,
        }
        if before is not None and during is not None:
            if before >= 23.0:
                rec["outcome"] = "uninformative"  # 24 is both "kept" and "reset"
            elif abs(during - R.RESET_FULL_S) <= 0.5:
                rec["outcome"] = "reset_24"
            elif before < R.FRONTCOURT_RESET_MIN_S and abs(during - 14.0) <= 0.5:
                rec["outcome"] = "raised_to_14"
            elif abs(during - before) <= 1.0:
                rec["outcome"] = "kept"
            else:
                rec["outcome"] = "other"
        out.append(rec)
    return out


def penalty_check(g: str, pbp: pl.DataFrame) -> tuple[list[dict], set[int]]:
    """Replay team fouls; return per-foul records and the event numbers without free throws."""
    p = pbp.sort(["period", "pctime_sec", "event_num"], descending=[False, True, False])
    p = p.with_columns(pl.coalesce("home_desc", "visitor_desc").alias("_d"))
    fts = p.filter(pl.col("msg_type") == 3).with_columns(
        pl.col("_d").str.extract(FT_RE.pattern, 1).alias("_kind")
    )
    fts = fts.filter(
        pl.col("_kind").is_null() | (pl.col("_kind").str.to_lowercase() != "technical")
    )
    ft_keys = set(zip(fts["period"], fts["pctime_sec"], fts["p1_team_id"], strict=True))
    teams = [int(t) for t in p["p1_team_id"].drop_nulls().unique().to_list()]
    recs, no_ft = [], set()
    count: Counter = Counter()
    last2: Counter = Counter()
    for ev in p.iter_rows(named=True):
        if ev["msg_type"] != 6 or ev["p1_team_id"] is None or len(teams) != 2:
            continue
        detail = FOUL_DETAIL.get(int(ev["action_type"]), (None,))[0]
        team, per, clk = int(ev["p1_team_id"]), int(ev["period"]), float(ev["pctime_sec"])
        key = (team, per)
        if int(ev["action_type"]) in NONSHOOTING:
            fouled = teams[0] if teams[1] == team else teams[1]
            pen = R.in_penalty(per, clk, count[key], last2[key])
            got = (per, ev["pctime_sec"], fouled) in ft_keys
            if not got:
                no_ft.add(int(ev["event_num"]))
            recs.append(
                {
                    "game_id": g,
                    "event_num": int(ev["event_num"]),
                    "period": per,
                    "clock": clk,
                    "foul": NONSHOOTING[int(ev["action_type"])],
                    "fouls_before": count[key],
                    "last2_before": last2[key],
                    "penalty": pen,
                    "free_throws": got,
                }
            )
        if detail in R.TEAM_FOUL_DETAILS:
            count[key] += 1
            if clk <= R.TWO_MINUTE_S:
                last2[key] += 1
    return recs, no_ft


def run_game(g: str) -> dict:
    fi = L.frame_index(g).sort(["period", "unix_ms"])
    pbp = L.pbp(game_id=g)
    pen, no_ft = penalty_check(g, pbp)
    res = {"game_id": g, "penalty": pen, "rebounds": [], "fouls": []}
    if fi["shot_clock"].null_count() == fi.height:
        res["no_shot_clock"] = True
        return res
    ev = pl.read_parquet(L2 / "events" / f"{g}.parquet")
    shots = ev.filter(pl.col("event_type") == "shot_release")
    stops = ev.filter(pl.col("event_type") == "clock_stop")
    align = pl.read_parquet(L2 / "_pbp_align" / f"{g}.parquet")
    ball = (
        L.frames(g, columns=["period", "unix_ms", "player_id", "x"])
        .filter(pl.col("player_id") == -1)
        .sort("period", "unix_ms")
    )
    res["rebounds"] = rebound_check(g, fi, shots)
    res["fouls"] = foul_shot_clock_check(
        g, fi, stops, align, pbp, ball, L.attack_direction(game_id=g), no_ft
    )
    return res


def pct(k: int, n: int) -> str:
    return f"{k / n:.1%} ({k}/{n})" if n else "–"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    games = L.games().filter(pl.col("status") == "ok")["game_id"].to_list()
    games = games[: a.games] if a.games else games
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_game, games, chunksize=4))
    reb = pl.DataFrame([r for x in results for r in x["rebounds"]], infer_schema_length=None)
    fouls = pl.DataFrame([r for x in results for r in x["fouls"]], infer_schema_length=None)
    pen = pl.DataFrame([r for x in results for r in x["penalty"]], infer_schema_length=None)
    n_sc = sum(not x.get("no_shot_clock") for x in results)

    lines = [
        "# L5 rulebook check: 2015-16 rule constants vs data",
        "",
        f"`scripts/l5_rulebook_check.py`: {len(games)} games, {n_sc} with a shot clock. "
        "Constants: `nbacore.rules_2015_16` (docs/external_sources.md).",
        "",
        "## 1. Shot clock after the rebound of a missed FG (7-IV-c(2), 7-IV-b(5))",
        "",
        "Largest upward jump of the tracked shot clock between the release and control + 3 s "
        "(game clock > 24 s, ≥ 10 observed shot-clock frames in the window).",
        "",
        "| rebound | rim contact detected | n | reset to ~24 | already 24 at release | "
        "to 13–15 | other | none |",
        "|---|---|---|---|---|---|---|---|",
    ]
    stats: dict = {"games": len(games), "games_with_shot_clock": n_sc}
    rb = reb.filter(pl.col("observed") & ~pl.col("blocked"))
    for off in (True, False):
        for rim in (True, False):
            s = rb.filter((pl.col("offensive") == off) & (pl.col("rim") == rim))
            c = dict(s.group_by("reset").len().iter_rows())
            n = s.height
            row = [pct(c.get(k, 0), n) for k in ("24", "already_24", "14", "other", "none")]
            name = "offensive" if off else "defensive"
            lines.append(f"| {name} | {'yes' if rim else 'no'} | {n} | " + " | ".join(row) + " |")
            stats[f"rebound_{name}_rim{int(rim)}"] = {"n": n, **c}
    lines += [
        "",
        f"Blocked shots excluded ({reb.filter(pl.col('blocked')).height}); rebounds without "
        f"enough shot-clock frames: {reb.filter(~pl.col('observed')).height}.",
        "",
        "## 2. Shot clock after defensive non-shooting fouls without free throws "
        "(7-IV-c(3), 7-IV-d(1))",
        "",
        "Last value shown while the clock is frozen vs the value when it stopped (last running "
        "value minus the game time still elapsed; `kept` within 1 s); throw-in half from "
        "the ball when the clock restarts. Expected: backcourt → 24; frontcourt → "
        "max(remaining, 14), i.e. `kept` when ≥ 14 and `raised_to_14` when < 14.",
        "",
        "| throw-in | remaining before | n | reset_24 | raised_to_14 | kept | other |",
        "|---|---|---|---|---|---|---|",
    ]
    n_unin = fouls.filter(pl.col("outcome") == "uninformative").height
    fo = fouls.filter(pl.col("outcome").is_not_null() & (pl.col("outcome") != "uninformative"))
    for half in ("frontcourt", "backcourt", "midcourt"):
        for lo in (True, False):
            s = fo.filter(
                (pl.col("half") == half) & ((pl.col("before") < R.FRONTCOURT_RESET_MIN_S) == lo)
            )
            c = dict(s.group_by("outcome").len().iter_rows())
            n = s.height
            row = [pct(c.get(k, 0), n) for k in ("reset_24", "raised_to_14", "kept", "other")]
            lines.append(f"| {half} | {'< 14' if lo else '≥ 14'} | {n} | " + " | ".join(row) + " |")
            stats[f"foul_{half}_{'lt14' if lo else 'ge14'}"] = {"n": n, **c}
    ok_front = fo.filter(pl.col("half") == "frontcourt").with_columns(
        pl.when(pl.col("before") < R.FRONTCOURT_RESET_MIN_S)
        .then(pl.col("outcome") == "raised_to_14")
        .otherwise(pl.col("outcome") == "kept")
        .alias("ok")
    )
    ok_back = fo.filter(pl.col("half") == "backcourt").with_columns(
        (pl.col("outcome") == "reset_24").alias("ok")
    )
    lines += [
        "",
        f"Excluded: {n_unin} fouls with ≥ 23 s left (24 is both outcomes), "
        f"{fouls.filter(pl.col('outcome').is_null()).height} without a frozen shot-clock value. "
        f"Agreement with the rule: frontcourt {pct(int(ok_front['ok'].sum()), ok_front.height)}, "
        f"backcourt {pct(int(ok_back['ok'].sum()), ok_back.height)}.",
        "",
        "## 3. Penalty situation vs free throws (12B-V-a)",
        "",
        "Defensive non-shooting common fouls in pbp (personal, block, take, loose ball); team "
        "fouls replayed per period with `TEAM_FOUL_DETAILS`; `free throws` = a non-technical free "
        "throw of the fouled team at the same game clock.",
        "",
        "| predicted | n | with free throws | without |",
        "|---|---|---|---|",
    ]
    for pen_v in (True, False):
        s = pen.filter(pl.col("penalty") == pen_v)
        k = int(s["free_throws"].sum())
        lines.append(
            f"| {'in penalty' if pen_v else 'not in penalty'} | {s.height} | {pct(k, s.height)} | "
            f"{pct(s.height - k, s.height)} |"
        )
    agree = pen.filter(pl.col("penalty") == pl.col("free_throws")).height
    stats["penalty"] = {
        "n": pen.height,
        "agree": agree,
        "in_penalty": pen.filter(pl.col("penalty")).height,
        "in_penalty_with_ft": pen.filter(pl.col("penalty") & pl.col("free_throws")).height,
        "not_penalty_with_ft": pen.filter(~pl.col("penalty") & pl.col("free_throws")).height,
    }
    by = (
        pen.with_columns(
            pl.when(pl.col("period") > 4)
            .then(pl.lit("OT"))
            .when(pl.col("clock") <= R.TWO_MINUTE_S)
            .then(pl.lit("last 2:00"))
            .otherwise(pl.lit("before 2:00"))
            .alias("phase")
        )
        .group_by("phase")
        .agg(pl.len().alias("n"), (pl.col("penalty") == pl.col("free_throws")).sum().alias("agree"))
        .sort("phase")
    )
    lines += ["", f"Overall agreement {pct(agree, pen.height)}; by phase: "]
    lines += [f"- {r['phase']}: {pct(r['agree'], r['n'])}" for r in by.iter_rows(named=True)]
    dis = pen.filter(pl.col("penalty") != pl.col("free_throws")).head(15)
    lines += ["", "First disagreements (for manual reading):", "", "```"]
    lines += [str(r) for r in dis.iter_rows(named=True)]
    lines += ["```", ""]
    text = "\n".join(lines)
    print(text)
    (REPO / "reports" / "l5_rulebook_check.md").write_text(text, encoding="utf-8")
    (REPO / "reports" / "l5_rulebook_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
