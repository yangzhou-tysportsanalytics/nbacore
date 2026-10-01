"""Attack-direction inference: which basket does each team attack in each period?

Method: for every field-goal attempt in the play-by-play (made or missed),
take the ball's x position in the tracking frames whose game clock is within ``tol_s`` of the
pbp clock; the shot happened in the half that contains the ball. Vote per (team, period),
then impose the structure of a basketball game: the two teams attack opposite baskets, the
direction is constant within a half, it flips after period 2, and overtime keeps the
period-4 direction.
"""

from __future__ import annotations

import polars as pl

from nbacore.court import HALF
from nbacore.io.raw import BALL_ID

SHOT_MSG_TYPES = (1, 2)  # FIELD_GOAL_MADE, FIELD_GOAL_MISSED


def shot_ball_positions(
    entities: pl.DataFrame, pbp_game: pl.DataFrame, tol_s: float = 1.0
) -> pl.DataFrame:
    """Median ball (x, y) around each pbp shot. One row per shot event that has frames."""
    shots = pbp_game.filter(
        pl.col("msg_type").is_in(SHOT_MSG_TYPES) & pl.col("p1_team_id").is_not_null()
    ).select(["event_num", "period", "pctime_sec", "p1_team_id", "p1_id", "msg_type"])
    ball = entities.filter(pl.col("team_id") == BALL_ID).select(
        ["period", "unix_ms", "game_clock", "x", "y"]
    )
    joined = shots.join(ball, on="period", how="inner").filter(
        (pl.col("game_clock") - pl.col("pctime_sec")).abs() <= tol_s
    )
    return (
        joined.group_by(["event_num", "period", "pctime_sec", "p1_team_id", "p1_id", "msg_type"])
        .agg(
            pl.col("x").median().alias("ball_x"),
            pl.col("y").median().alias("ball_y"),
            pl.len().alias("n_frames"),
        )
        .with_columns((pl.col("ball_x") < HALF).alias("shot_left"))
        .sort("event_num")
    )


def _half_index(period: int) -> int:
    """0 for periods 1-2, 1 for periods 3, 4 and overtime."""
    return 0 if period <= 2 else 1


def infer_attack_direction(
    entities: pl.DataFrame, pbp_game: pl.DataFrame, tol_s: float = 1.0
) -> tuple[pl.DataFrame, dict]:
    """Return (direction table, diagnostics).

    Direction table columns: game_id, team_id, period, attacks_left (structured decision),
    attacks_left_raw (per-period majority vote; null if no shots), n_shots, n_left,
    consistency (fraction of shots agreeing with the structured decision).
    """
    game_id = entities["game_id"][0]
    shots = shot_ball_positions(entities, pbp_game, tol_s)
    teams = sorted(
        pbp_game.filter(pl.col("p1_team_id").is_not_null())["p1_team_id"].unique().to_list()
    )
    if len(teams) != 2:
        raise ValueError(f"expected 2 teams in pbp for game {game_id}, got {teams}")
    periods = sorted(entities["period"].unique().to_list())

    # per-team score: +1 for a first-half left shot / second-half right shot, -1 otherwise.
    # S > 0  => team attacks LEFT in periods 1-2 (and right afterwards).
    scores = {t: 0 for t in teams}
    for r in shots.iter_rows(named=True):
        s = 1 if r["shot_left"] else -1
        if _half_index(r["period"]) == 1:
            s = -s
        scores[r["p1_team_id"]] += s
    # the two teams must disagree; combine both votes into one decision for team A
    a, b = teams
    combined = scores[a] - scores[b]
    a_left_first_half = combined > 0
    diagnostics = {
        "game_id": game_id,
        "n_shots_used": shots.height,
        "n_shots_pbp": pbp_game.filter(pl.col("msg_type").is_in(SHOT_MSG_TYPES)).height,
        "score_team_a": scores[a],
        "score_team_b": scores[b],
        "combined_score": combined,
        "teams_agree": (scores[a] > 0) != (scores[b] > 0) if scores[a] and scores[b] else None,
    }

    rows = []
    for t in teams:
        left_first = a_left_first_half if t == a else not a_left_first_half
        for p in periods:
            attacks_left = left_first if _half_index(p) == 0 else not left_first
            sp = shots.filter((pl.col("p1_team_id") == t) & (pl.col("period") == p))
            n = sp.height
            n_left = int(sp["shot_left"].sum()) if n else 0
            raw = (n_left > n / 2) if n else None
            agree = n_left if attacks_left else n - n_left
            rows.append(
                {
                    "game_id": game_id,
                    "team_id": t,
                    "period": p,
                    "attacks_left": attacks_left,
                    "attacks_left_raw": raw,
                    "n_shots": n,
                    "n_left": n_left,
                    "consistency": agree / n if n else None,
                }
            )
    table = pl.DataFrame(
        rows,
        schema={
            "game_id": pl.Utf8,
            "team_id": pl.Int64,
            "period": pl.Int8,
            "attacks_left": pl.Boolean,
            "attacks_left_raw": pl.Boolean,
            "n_shots": pl.Int32,
            "n_left": pl.Int32,
            "consistency": pl.Float64,
        },
    )
    cons = table.filter(pl.col("n_shots") > 0)
    diagnostics["min_period_consistency"] = (
        float(cons["consistency"].min()) if cons.height else None
    )
    diagnostics["overall_consistency"] = (
        float((cons["consistency"] * cons["n_shots"]).sum() / cons["n_shots"].sum())
        if cons.height
        else None
    )
    diagnostics["n_period_votes_overruled"] = int(
        (cons["attacks_left_raw"] != cons["attacks_left"]).sum()
    )
    return table, diagnostics
