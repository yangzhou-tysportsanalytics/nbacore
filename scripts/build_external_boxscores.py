"""L5: 2015-16 regular-season box scores (docs/external_sources.md).

Source: Kaggle eoinamoore "NBA Dataset: Box Scores and Stats" v515 (``Games.csv``,
``PlayerStatistics.csv``; NBA.com data, provenance in
``<data root>/raw/external/boxscores_eoinamoore_v515/provenance.json``).

Outputs under ``<data root>/scratch/l5/`` (L5 build, released with a later version):
* ``box_games_2015_16.parquet``: one row per game: ``game_id`` (10-char NBA id), ``game_date``,
  home / away team ids and scores, ``arena_id`` (the source has no officials, attendance or
  arena name for 2015-16);
* ``box_players_2015_16.parquet``: one row per player-game: ``game_id, player_id, player_name,
  team_id, opponent_team_id, home, starter, starting_position, minutes, pts, fgm, fga, fg3m, fg3a,
  ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus, comment`` (``comment`` = DNP
  reason, etc.).

Checks against the pbp (``reports/l5_boxscores_check.json``): final score per game (last pbp
score), and per player-game FGM / FGA / 3PM / FTM / FTA / points counted from pbp rows.

Usage:
    uv run python scripts/build_external_boxscores.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths

REPO = Path(__file__).resolve().parents[1]
SRC = paths.raw_dir() / "external" / "boxscores_eoinamoore_v515"


def _gid(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64).cast(pl.Utf8).str.zfill(10).alias("game_id")


def games() -> pl.DataFrame:
    g = pl.read_csv(SRC / "Games.csv", infer_schema_length=0).filter(
        pl.col("gameId").str.starts_with("215")
    )
    return g.select(
        _gid("gameId"),
        pl.col("gameDate").str.slice(0, 10).str.to_date().alias("game_date"),
        pl.col("hometeamId").cast(pl.Int64).alias("home_team_id"),
        pl.col("awayteamId").cast(pl.Int64).alias("away_team_id"),
        pl.col("homeScore").cast(pl.Int16).alias("home_score"),
        pl.col("awayScore").cast(pl.Int16).alias("away_score"),
        pl.col("arenaId").cast(pl.Int64, strict=False).alias("arena_id"),
    ).sort("game_id")


def players() -> pl.DataFrame:
    b = pl.read_csv(SRC / "PlayerStatistics.csv", infer_schema_length=0).filter(
        pl.col("gameId").str.starts_with("215")
    )
    i16 = {
        "points": "pts",
        "fieldGoalsMade": "fgm",
        "fieldGoalsAttempted": "fga",
        "threePointersMade": "fg3m",
        "threePointersAttempted": "fg3a",
        "freeThrowsMade": "ftm",
        "freeThrowsAttempted": "fta",
        "reboundsOffensive": "oreb",
        "reboundsDefensive": "dreb",
        "reboundsTotal": "reb",
        "assists": "ast",
        "steals": "stl",
        "blocks": "blk",
        "turnovers": "tov",
        "foulsPersonal": "pf",
        "plusMinusPoints": "plus_minus",
    }
    return b.select(
        _gid("gameId"),
        pl.col("personId").cast(pl.Int64).alias("player_id"),
        (pl.col("firstName") + " " + pl.col("lastName")).alias("player_name"),
        pl.col("playerteamId").cast(pl.Int64).alias("team_id"),
        pl.col("opponentteamId").cast(pl.Int64).alias("opponent_team_id"),
        (pl.col("home") == "1").alias("home"),
        pl.col("startingPosition").fill_null("").ne("").alias("starter"),
        pl.col("startingPosition").alias("starting_position"),
        pl.col("numMinutes").cast(pl.Float32).alias("minutes"),
        *[pl.col(k).cast(pl.Float64).cast(pl.Int16).alias(v) for k, v in i16.items()],
        pl.col("comment"),
    ).sort("game_id", "team_id", "player_id")


def pbp_counts(pbp: pl.DataFrame) -> pl.DataFrame:
    """Per player-game shooting counts from pbp rows (FG made / missed, free throws)."""
    desc = pl.coalesce("home_desc", "visitor_desc", pl.lit(""))
    s = pbp.filter(pl.col("msg_type").is_in([1, 2, 3])).with_columns(
        desc.str.contains("3PT").alias("three"),
        (
            (pl.col("msg_type") == 1) | ((pl.col("msg_type") == 3) & ~desc.str.starts_with("MISS"))
        ).alias("made"),
    )
    fg, ft = pl.col("msg_type") < 3, pl.col("msg_type") == 3
    return s.group_by("game_id", pl.col("p1_id").cast(pl.Int64).alias("player_id")).agg(
        fg.sum().cast(pl.Int16).alias("fga_pbp"),
        (fg & pl.col("made")).sum().cast(pl.Int16).alias("fgm_pbp"),
        (fg & pl.col("three") & pl.col("made")).sum().cast(pl.Int16).alias("fg3m_pbp"),
        ft.sum().cast(pl.Int16).alias("fta_pbp"),
        (ft & pl.col("made")).sum().cast(pl.Int16).alias("ftm_pbp"),
    )


def main() -> None:
    g, b = games(), players()
    out = paths.scratch_dir("l5")
    out.mkdir(parents=True, exist_ok=True)
    g.write_parquet(out / "box_games_2015_16.parquet")
    b.write_parquet(out / "box_players_2015_16.parquet")

    pbp = L.pbp("v1.2")
    last = (
        pbp.filter(pl.col("score_home").is_not_null())
        .sort(
            "game_id", "period", "pctime_sec", "event_num", descending=[False, False, True, False]
        )
        .group_by("game_id")
        .agg(pl.col("score_home").last(), pl.col("score_visitor").last())
    )
    sc = g.join(last, on="game_id", how="left")
    score_ok = (sc["home_score"] == sc["score_home"]) & (sc["away_score"] == sc["score_visitor"])
    bad_scores = sc.filter(~score_ok.fill_null(False)).select(
        "game_id", "home_score", "away_score", "score_home", "score_visitor"
    )

    played = b.filter(pl.col("minutes").fill_null(0) > 0)
    c = played.join(pbp_counts(pbp), on=["game_id", "player_id"], how="full", coalesce=True)
    c = c.with_columns(pl.col(pl.Int16).fill_null(0))
    cols = ["fga", "fgm", "fg3m", "fta", "ftm"]
    agree = {k: float((c[k] == c[f"{k}_pbp"]).mean()) for k in cols}
    pts_pbp = 2 * c["fgm_pbp"] + c["fg3m_pbp"] + c["ftm_pbp"]
    agree["pts"] = float((c["pts"] == pts_pbp).mean())
    all_ok = pl.all_horizontal([pl.col(k) == pl.col(f"{k}_pbp") for k in cols])
    worst = (
        c.filter(~all_ok).group_by("game_id").len().sort("len", descending=True).head(10).to_dicts()
    )
    ids_rosters = set(L.rosters("v1.2")["player_id"].to_list())
    stats = {
        "games": g.height,
        "player_game_rows": b.height,
        "player_game_rows_with_minutes": played.height,
        "players": b["player_id"].n_unique(),
        "tracked_roster_players_in_box": len(ids_rosters & set(b["player_id"].to_list())),
        "tracked_roster_players": len(ids_rosters),
        "final_score_agrees_with_pbp": int(score_ok.fill_null(False).sum()),
        "final_score_disagrees": bad_scores.to_dicts(),
        "player_game_agreement_with_pbp": agree,
        "player_games_compared": c.height,
        "games_with_most_disagreeing_player_rows": worst,
        "outputs": [
            str(out / "box_games_2015_16.parquet"),
            str(out / "box_players_2015_16.parquet"),
        ],
    }
    (REPO / "reports" / "l5_boxscores_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
