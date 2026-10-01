"""Independent check of the inferred attack direction (L1 acceptance).

``attack_direction`` votes with the ball position at pbp shot times (± 1 s), which is noisy
because pbp shot times are late (median 1.8 s). This check uses a different signal: for each
FG attempt, the **hoop the ball reaches** — first frame with the ball within ``RIM_XY`` ft (xy)
of either hoop centre at ``RIM_Z`` height, among frames whose game clock lies in
[pbp second - 2 s, pbp second + 5 s]. A shot "agrees" when that hoop is the one the shooter's
team attacks according to ``attack_direction``.

Writes ``reports/l1_direction_check.json`` (per game) and ``reports/l1_direction_check.md``.

Usage:
    uv run python scripts/l1_direction_check.py --build scratch/l1_full
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from nbacore import paths
from nbacore.court import HOOP_LEFT, HOOP_RIGHT
from nbacore.io.raw import BALL_ID

REPO = Path(__file__).resolve().parents[1]
RIM_XY = 2.0
RIM_Z = (8.0, 11.5)
WIN_BEFORE_S, WIN_AFTER_S = 5.0, 2.0  # clock counts down: earlier = larger clock


def check_game(build: Path, gid: str, pbp_game: pl.DataFrame, direction: pl.DataFrame) -> dict:
    ball = (
        pl.read_parquet(
            build / "frames" / f"{gid}.parquet",
            columns=["period", "unix_ms", "game_clock", "team_id", "x", "y", "z"],
        )
        .filter(pl.col("team_id") == BALL_ID)
        .sort(["period", "unix_ms"])
    )
    shots = pbp_game.filter(pl.col("msg_type").is_in([1, 2]) & pl.col("p1_team_id").is_not_null())
    dirs = {(r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)}
    n = agree = no_rim = 0
    bad = []
    for s in shots.iter_rows(named=True):
        b = ball.filter(
            (pl.col("period") == s["period"])
            & (pl.col("game_clock") >= s["pctime_sec"] - WIN_AFTER_S)
            & (pl.col("game_clock") <= s["pctime_sec"] + WIN_BEFORE_S)
        )
        if b.height == 0:
            continue
        x, y, z = (b[c].to_numpy().astype(float) for c in ("x", "y", "z"))
        hz = (z > RIM_Z[0]) & (z < RIM_Z[1])
        at_l = hz & (np.hypot(x - HOOP_LEFT[0], y - HOOP_LEFT[1]) < RIM_XY)
        at_r = hz & (np.hypot(x - HOOP_RIGHT[0], y - HOOP_RIGHT[1]) < RIM_XY)
        hit = np.flatnonzero(at_l | at_r)
        if hit.size == 0:
            no_rim += 1
            continue
        rim_left = bool(at_l[hit[0]])
        att = dirs.get((int(s["p1_team_id"]), int(s["period"])))
        if att is None:
            continue
        n += 1
        if rim_left == att:
            agree += 1
        else:
            bad.append(
                {
                    "event_num": int(s["event_num"]),
                    "period": int(s["period"]),
                    "pctime_sec": float(s["pctime_sec"]),
                    "team_id": int(s["p1_team_id"]),
                    "rim_left": rim_left,
                    "attacks_left": att,
                    "ball_x_first_rim": float(x[hit[0]]),
                }
            )
    by_period = {}
    for b_ in bad:
        k = f"{b_['team_id']}/{b_['period']}"
        by_period[k] = by_period.get(k, 0) + 1
    return {
        "game_id": gid,
        "n_shots_rim": n,
        "n_no_rim": no_rim,
        "rim_agreement": agree / n if n else None,
        "disagree_by_team_period": by_period,
        "disagreements": bad,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    build = paths.data_root() / args.build
    games = pl.read_parquet(build / "games.parquet").filter(pl.col("status") == "ok")
    pbp = pl.read_parquet(build / "pbp.parquet")
    direction = pl.read_parquet(build / "attack_direction.parquet")
    stats = {
        r["game_id"]: r
        for r in json.loads((build / "build_results.json").read_text())
        if r["status"] == "ok"
    }
    gids = games["game_id"].to_list()
    with ProcessPoolExecutor(args.workers) as ex:
        futs = [
            ex.submit(
                check_game,
                build,
                g,
                pbp.filter(pl.col("game_id") == g),
                direction.filter(pl.col("game_id") == g),
            )
            for g in gids
        ]
        res = [f.result() for f in futs]
    for r in res:
        r["vote_consistency"] = stats[r["game_id"]]["direction"]["overall_consistency"]
    (REPO / "reports" / "l1_direction_check.json").write_text(
        json.dumps(res, indent=1), encoding="utf-8"
    )

    R = pl.DataFrame(
        [
            {
                k: v
                for k, v in r.items()
                if k in ("game_id", "n_shots_rim", "n_no_rim", "rim_agreement", "vote_consistency")
            }
            for r in res
        ]
    )
    tot_n = R["n_shots_rim"].sum()
    tot_agree = sum(r["rim_agreement"] * r["n_shots_rim"] for r in res if r["n_shots_rim"])
    L = [
        "# Attack direction: independent rim check (L1)",
        "",
        "Generated by `scripts/l1_direction_check.py`. Signal: the hoop the ball first reaches",
        f"(xy < {RIM_XY} ft of the hoop centre, {RIM_Z[0]}–{RIM_Z[1]} ft high) within the game-clock",
        f"window [pbp s − {WIN_AFTER_S:g}, pbp s + {WIN_BEFORE_S:g}] of each FG attempt, compared with",
        "`attack_direction` for the shooter's team and period.",
        "",
        f"- games: {R.height}; shots with a rim contact: {tot_n}; overall rim agreement: {tot_agree / tot_n:.4f}",
        f"- rim agreement per game (min / 5 % / median): {R['rim_agreement'].min():.4f} / "
        f"{R['rim_agreement'].quantile(0.05):.4f} / {R['rim_agreement'].median():.4f}",
        f"- games with rim agreement < 0.95: {(R['rim_agreement'] < 0.95).sum()}",
        "",
        "## Games with vote consistency < 0.9 or rim agreement < 0.95",
        "",
        "| game_id | vote consistency | rim agreement | shots (rim) | disagreements by team/period |",
        "|---|---|---|---|---|",
    ]
    by_id = {r["game_id"]: r for r in res}
    for row in (
        R.filter((pl.col("vote_consistency") < 0.9) | (pl.col("rim_agreement") < 0.95))
        .sort("rim_agreement")
        .iter_rows(named=True)
    ):
        r = by_id[row["game_id"]]
        L.append(
            f"| {row['game_id']} | {row['vote_consistency']:.3f} | {row['rim_agreement']:.3f} | "
            f"{row['n_shots_rim']} | {r['disagree_by_team_period'] or '–'} |"
        )
    (REPO / "reports" / "l1_direction_check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
