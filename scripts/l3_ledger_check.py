"""L3 acceptance checks of the possession ledger (docs/ledger_schema.md).

Per game: ledger possessions (``pbp_possessions`` + ``ledger_times`` on the v1.1 ``pbp_align``)
vs the box-score estimate (acceptance: |diff| ≤ 3 %), and the ghost_v1 windows linked to the
ledger (``attach_poss_uid``): every window should lie inside one possession (overlap ≥ 0.9) with
the same offence. ghost_v1 windows are read from ghost-defense's output, which ghost_v1
reproduces exactly (``scripts/l3_ghost_v1_parity.py``).

Usage:
    uv run python scripts/l3_ledger_check.py [--set small|all] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.ledger import (
    attach_poss_uid,
    ledger_times,
    pbp_possession_estimate,
    pbp_possessions,
    possession_estimate_aligned,
)

GHOST = Path("../ghost-defense/data/processed/nbacore-v1.0")
REPO = Path(__file__).resolve().parents[1]
TOL_S = 1.5  # ghost ends at the pbp clock (+ 0.5 s pad); the ledger at the steal / release
L2 = paths.scratch_dir("l2_full")


def run_game(args: tuple[str, str]) -> dict:
    gid, game_set = args
    pbp = L.pbp(game_id=gid)
    poss = pbp_possessions(pbp)
    led = ledger_times(
        poss,
        pl.read_parquet(L2 / "_pbp_align" / f"{gid}.parquet"),
        L.frame_index(gid, columns=["period", "unix_ms", "game_clock", "segment_id"]),
        L.attack_direction(game_id=gid),
        handler=pl.read_parquet(
            L2 / "handler" / f"{gid}.parquet", columns=["period", "unix_ms", "control_team_id"]
        ),
        pbp_game=pbp,
    )
    est = pbp_possession_estimate(pbp)
    est_t = pbp_possession_estimate(pbp, team_oreb=True)
    est_a = possession_estimate_aligned(pbp, led)
    win = pl.read_parquet(GHOST / game_set / "possessions" / f"{gid}.parquet")
    lk = attach_poss_uid(win, led).join(
        led.select("poss_uid", "t_start_ms", "t_end_ms", "end_type", "end_time_method"),
        on="poss_uid",
        how="left",
    )
    lk = lk.with_columns(
        ((pl.col("t_start_ms") - pl.col("t_cross")).clip(0) / 1000).alias("out_before_s"),
        ((pl.col("t_terminal") - pl.col("t_end_ms")).clip(0) / 1000).alias("out_after_s"),
    )
    match = pl.col("offense_match").fill_null(False)
    bad = lk.filter(~match | (pl.col("overlap_frac") < 0.9))
    tol = match & (pl.col("out_before_s") <= TOL_S) & (pl.col("out_after_s") <= TOL_S)
    return {
        "game_id": gid,
        "n_poss": led.height,
        "estimate": est,
        "diff_frac": (led.height - est) / est,
        "estimate_team_oreb": est_t,
        "diff_frac_team_oreb": (led.height - est_t) / est_t,
        "estimate_aligned": est_a,
        "diff_frac_aligned": (led.height - est_a) / est_a,
        "end_methods": dict(led.group_by("end_time_method").len().iter_rows()),
        "unrecorded_change": led.filter(pl.col("end_type") == "unrecorded_change").height,
        "n_windows": win.height,
        "windows_ok": win.height - bad.height,
        "windows_ok_tol": lk.filter(tol).height,
        "offense_mismatch": lk.filter(~match).height,
        "bad": bad.select(
            "terminal_event_num",
            "terminal_type",
            "period",
            "offense_team_id",
            "poss_uid",
            "overlap_frac",
            "offense_match",
            "out_before_s",
            "out_after_s",
            "end_type",
            "end_time_method",
        ).to_dicts(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="small")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    games = sorted(p.stem for p in (GHOST / a.set / "possessions").glob("*.parquet"))
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_game, [(g, a.set) for g in games]))
    diffs = [r["diff_frac"] for r in res]
    n_win = sum(r["n_windows"] for r in res)
    ok = sum(r["windows_ok"] for r in res)
    methods: dict[str, int] = {}
    for r in res:
        for k, v in r["end_methods"].items():
            methods[str(k)] = methods.get(str(k), 0) + v
    summary = {
        "set": a.set,
        "games": len(games),
        "possessions": sum(r["n_poss"] for r in res),
        "estimate": round(sum(r["estimate"] for r in res), 1),
        "diff_mean": sum(diffs) / len(diffs),
        "diff_max_abs": max(abs(d) for d in diffs),
        "games_over_3pct": [r["game_id"] for r in res if abs(r["diff_frac"]) > 0.03],
        "diff_p95_abs": float(pl.Series([abs(x) for x in diffs]).quantile(0.95)),
        "estimate_team_oreb": round(sum(r["estimate_team_oreb"] for r in res), 1),
        "diff_team_oreb_season": sum(r["n_poss"] for r in res)
        / sum(r["estimate_team_oreb"] for r in res)
        - 1,
        "diff_team_oreb_p95_abs": float(
            pl.Series([abs(r["diff_frac_team_oreb"]) for r in res]).quantile(0.95)
        ),
        "games_over_3pct_team_oreb": sum(abs(r["diff_frac_team_oreb"]) > 0.03 for r in res),
        "estimate_aligned": round(sum(r["estimate_aligned"] for r in res), 1),
        "diff_aligned_season": sum(r["n_poss"] for r in res)
        / sum(r["estimate_aligned"] for r in res)
        - 1,
        "diff_aligned_p95_abs": float(
            pl.Series([abs(r["diff_frac_aligned"]) for r in res]).quantile(0.95)
        ),
        "games_over_3pct_aligned": sum(abs(r["diff_frac_aligned"]) > 0.03 for r in res),
        "unrecorded_change": sum(r["unrecorded_change"] for r in res),
        "end_time_methods": dict(sorted(methods.items(), key=lambda kv: -kv[1])),
        "ghost_windows": n_win,
        "ghost_windows_in_one_possession_same_offense": ok,
        "frac": ok / n_win if n_win else None,
        "ghost_windows_same_offense_outside_le_1_5s": sum(r["windows_ok_tol"] for r in res),
        "frac_tol": sum(r["windows_ok_tol"] for r in res) / n_win if n_win else None,
        "offense_mismatch": sum(r["offense_mismatch"] for r in res),
        "bad_by_terminal": {},
    }
    for r in res:
        for b in r["bad"]:
            k = b["terminal_type"]
            summary["bad_by_terminal"][k] = summary["bad_by_terminal"].get(k, 0) + 1
    print(json.dumps(summary, indent=1))
    out = REPO / "reports" / f"p3_ledger_check_{a.set}.json"
    out.write_text(
        json.dumps({**summary, "per_game": res}, indent=1, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
