"""Play-by-play loader (sumitrodatta/nba-alt-awards 2015-16_pbp.csv, NBA stats format)."""

from __future__ import annotations

from pathlib import Path

import polars as pl

# EVENTMSGTYPE codes of the NBA stats play-by-play feed.
EVENTMSGTYPE: dict[int, str] = {
    1: "FIELD_GOAL_MADE",
    2: "FIELD_GOAL_MISSED",
    3: "FREE_THROW",
    4: "REBOUND",
    5: "TURNOVER",
    6: "FOUL",
    7: "VIOLATION",
    8: "SUBSTITUTION",
    9: "TIMEOUT",
    10: "JUMP_BALL",
    11: "EJECTION",
    12: "PERIOD_BEGIN",
    13: "PERIOD_END",
    18: "INSTANT_REPLAY",
}

# PERSONxTYPE codes: 4 = home player, 5 = visitor player (others: 0 none, 1 official, 2/3 team)
PERSON_TYPE_HOME = 4
PERSON_TYPE_VISITOR = 5

PBP_COLUMNS = [
    "game_id",
    "event_num",
    "period",
    "pctime_str",
    "pctime_sec",
    "wctime_str",
    "msg_type",
    "msg_type_name",
    "action_type",
    "home_desc",
    "visitor_desc",
    "neutral_desc",
    "score_str",
    "score_visitor",
    "score_home",
    "score_margin",
    "p1_type",
    "p1_id",
    "p1_name",
    "p1_team_id",
    "p2_type",
    "p2_id",
    "p2_name",
    "p2_team_id",
    "p3_type",
    "p3_id",
    "p3_name",
    "p3_team_id",
]


def _clock_to_sec(col: str) -> pl.Expr:
    parts = pl.col(col).str.split(":")
    return (parts.list.get(0).cast(pl.Float32) * 60 + parts.list.get(1).cast(pl.Float32)).alias(
        "pctime_sec"
    )


def load_pbp(path: str | Path) -> pl.DataFrame:
    """Load the season play-by-play CSV into a typed, snake_case polars frame.

    Notes
    -----
    * ``GAME_ID`` loses its leading zeros in the CSV; it is restored to a 10-char string.
    * ``SCORE`` is formatted ``"<visitor> - <home>"`` and is only present on scoring rows;
      it is split into ``score_visitor`` / ``score_home`` (null elsewhere). Forward-filling
      is left to the caller because it must be done per game in event order.
    * ``pctime_sec`` is the game clock (seconds remaining in the period) at the event.
    """
    df = pl.read_csv(
        path,
        infer_schema_length=200_000,
        schema_overrides={
            "GAME_ID": pl.Int64,
            "EVENTNUM": pl.Int32,
            "PERIOD": pl.Int8,
            "EVENTMSGTYPE": pl.Int16,
            "EVENTMSGACTIONTYPE": pl.Int16,
            "PLAYER1_ID": pl.Int64,
            "PLAYER2_ID": pl.Int64,
            "PLAYER3_ID": pl.Int64,
            "PLAYER1_TEAM_ID": pl.Float64,
            "PLAYER2_TEAM_ID": pl.Float64,
            "PLAYER3_TEAM_ID": pl.Float64,
            "PERSON1TYPE": pl.Float64,
            "PERSON2TYPE": pl.Float64,
            "PERSON3TYPE": pl.Float64,
            "SCORE": pl.Utf8,
            "SCOREMARGIN": pl.Utf8,
        },
    )
    score = pl.col("SCORE").str.split(" - ")
    out = df.select(
        pl.col("GAME_ID").cast(pl.Utf8).str.zfill(10).alias("game_id"),
        pl.col("EVENTNUM").alias("event_num"),
        pl.col("PERIOD").alias("period"),
        pl.col("PCTIMESTRING").alias("pctime_str"),
        _clock_to_sec("PCTIMESTRING"),
        pl.col("WCTIMESTRING").alias("wctime_str"),
        pl.col("EVENTMSGTYPE").alias("msg_type"),
        pl.col("EVENTMSGTYPE")
        .replace_strict(EVENTMSGTYPE, default="UNKNOWN", return_dtype=pl.Utf8)
        .alias("msg_type_name"),
        pl.col("EVENTMSGACTIONTYPE").alias("action_type"),
        pl.col("HOMEDESCRIPTION").alias("home_desc"),
        pl.col("VISITORDESCRIPTION").alias("visitor_desc"),
        pl.col("NEUTRALDESCRIPTION").alias("neutral_desc"),
        pl.col("SCORE").alias("score_str"),
        score.list.get(0, null_on_oob=True).cast(pl.Int16, strict=False).alias("score_visitor"),
        score.list.get(1, null_on_oob=True).cast(pl.Int16, strict=False).alias("score_home"),
        pl.when(pl.col("SCOREMARGIN") == "TIE")
        .then(pl.lit(0, dtype=pl.Int16))
        .otherwise(pl.col("SCOREMARGIN").cast(pl.Int16, strict=False))
        .alias("score_margin"),
        pl.col("PERSON1TYPE").cast(pl.Int8).alias("p1_type"),
        pl.col("PLAYER1_ID").alias("p1_id"),
        pl.col("PLAYER1_NAME").alias("p1_name"),
        pl.col("PLAYER1_TEAM_ID").cast(pl.Int64).alias("p1_team_id"),
        pl.col("PERSON2TYPE").cast(pl.Int8).alias("p2_type"),
        pl.col("PLAYER2_ID").alias("p2_id"),
        pl.col("PLAYER2_NAME").alias("p2_name"),
        pl.col("PLAYER2_TEAM_ID").cast(pl.Int64).alias("p2_team_id"),
        pl.col("PERSON3TYPE").cast(pl.Int8).alias("p3_type"),
        pl.col("PLAYER3_ID").alias("p3_id"),
        pl.col("PLAYER3_NAME").alias("p3_name"),
        pl.col("PLAYER3_TEAM_ID").cast(pl.Int64).alias("p3_team_id"),
    )
    return out.select(PBP_COLUMNS).sort(["game_id", "event_num"])


def pbp_for_game(pbp: pl.DataFrame, game_id: str) -> pl.DataFrame:
    return pbp.filter(pl.col("game_id") == str(game_id).zfill(10))
