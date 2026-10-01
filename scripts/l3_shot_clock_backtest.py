"""Backtest of the rule-based shot clock (``nbacore.shot_clock.shot_clock_imputed``).

On games with a tracked shot clock: imputed vs observed (``fill_method == observed``, so stuck-at-24
feed artefacts excluded) on running frames with a game clock above 24 s. Baseline: resets at possession starts only, no inbound
hold. Writes ``reports/l3_shot_clock_backtest.json``; ``--impute`` also writes the imputed shot
clock of the 12 games without one to ``<data root>/scratch/shot_clock_imputed/``.

Usage:
    uv run python scripts/l3_shot_clock_backtest.py [--games N] [--workers 3] [--impute]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.fouls import foul_states
from nbacore.ledger import ledger_times, pbp_possessions
from nbacore.shot_clock import shot_clock_filled, shot_clock_imputed, shot_clock_resets

REPO = Path(__file__).resolve().parents[1]
L2 = paths.scratch_dir("l2_full")


def impute(gid: str) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    pbp = L.pbp(game_id=gid)
    fi = L.frame_index(gid).sort("period", "unix_ms")
    align = pl.read_parquet(L2 / "_pbp_align" / f"{gid}.parquet")
    direction = L.attack_direction(game_id=gid)
    led = ledger_times(
        pbp_possessions(pbp),
        align,
        fi,
        direction,
        handler=pl.read_parquet(
            L2 / "handler" / f"{gid}.parquet", columns=["period", "unix_ms", "control_team_id"]
        ),
        pbp_game=pbp,
    )
    ev = pl.read_parquet(L2 / "events" / f"{gid}.parquet")
    ball = L.frames(gid, columns=["period", "unix_ms", "player_id", "x", "y"]).filter(
        pl.col("player_id") == -1
    )
    resets = shot_clock_resets(
        led,
        ev.filter(pl.col("event_type") == "shot_release"),
        ev.filter(pl.col("event_type") == "inbound"),
        foul_states(pbp),
        pbp,
        align,
        fi,
        ball,
        direction,
    )
    return fi, resets, shot_clock_imputed(fi, resets)


def run(gid: str) -> dict:
    fi, resets, imp = impute(gid)
    if fi["shot_clock"].null_count() == fi.height:
        return {"game_id": gid, "no_shot_clock": True}
    base = shot_clock_imputed(
        fi,
        resets.filter(pl.col("cause").str.starts_with("poss:")).with_columns(
            pl.lit(None).alias("t_run_ms")
        ),
    )
    d = (
        fi.select("period", "unix_ms", "game_clock", "shot_clock", "clock_frozen")
        .join(
            shot_clock_filled(fi).select("period", "unix_ms", "fill_method"),
            on=["period", "unix_ms"],
        )
        .filter(pl.col("fill_method") == "observed")
        .join(imp, on=["period", "unix_ms"])
        .join(
            base.select("period", "unix_ms", pl.col("shot_clock_imputed").alias("base")),
            on=["period", "unix_ms"],
        )
        .filter(
            ~pl.col("clock_frozen")
            & pl.col("shot_clock").is_between(0, 24)
            & (pl.col("game_clock") > 24)
        )
        .with_columns(
            (pl.col("shot_clock_imputed") - pl.col("shot_clock")).abs().alias("err"),
            (pl.col("base") - pl.col("shot_clock")).abs().alias("err_base"),
        )
    )
    return {
        "game_id": gid,
        "n": d.height,
        "sum_err": float(d["err"].sum()),
        "within_1": int((d["err"] <= 1).sum()),
        "within_2": int((d["err"] <= 2).sum()),
        "sum_err_base": float(d["err_base"].sum()),
        "within_1_base": int((d["err_base"] <= 1).sum()),
        "resets": dict(resets.group_by("cause").len().iter_rows()),
        "err_sample": d["err"].gather_every(10).to_list(),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--impute", action="store_true")
    a = ap.parse_args()
    games = L.games().filter(pl.col("status") == "ok")["game_id"].to_list()
    games = games[: a.games] if a.games else games
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, games, chunksize=2))
    ok = [r for r in res if not r.get("no_shot_clock")]
    n = sum(r["n"] for r in ok)
    causes: dict[str, int] = {}
    for r in ok:
        for k, v in r["resets"].items():
            causes[k] = causes.get(k, 0) + v
    stats = {
        "games": len(ok),
        "frames": n,
        "mae_s": sum(r["sum_err"] for r in ok) / n,
        "within_1s": sum(r["within_1"] for r in ok) / n,
        "within_2s": sum(r["within_2"] for r in ok) / n,
        "baseline_mae_s": sum(r["sum_err_base"] for r in ok) / n,
        "baseline_within_1s": sum(r["within_1_base"] for r in ok) / n,
        "resets_by_cause": causes,
        "err_quantiles_s": {
            str(q): float(pl.Series([x for r in ok for x in r["err_sample"]]).quantile(q))
            for q in (0.5, 0.75, 0.9, 0.95)
        },
        "per_game": [{k: v for k, v in r.items() if k != "err_sample"} for r in ok],
    }
    print(json.dumps({k: v for k, v in stats.items() if k != "per_game"}, indent=1))
    if not a.games:
        (REPO / "reports" / "l3_shot_clock_backtest.json").write_text(
            json.dumps(stats, indent=1), encoding="utf-8"
        )
    if a.impute:
        out = paths.scratch_dir("shot_clock_imputed")
        out.mkdir(parents=True, exist_ok=True)
        for r in res:
            if r.get("no_shot_clock"):
                _, _, imp = impute(r["game_id"])
                imp.write_parquet(out / f"{r['game_id']}.parquet")


if __name__ == "__main__":
    main()
