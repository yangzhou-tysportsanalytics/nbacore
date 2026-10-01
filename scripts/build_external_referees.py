"""L5: referees of every 2015-16 regular-season game (docs/external_sources.md).

Sources (``<data root>/raw/external/referees_kaggle/``, downloaded manually from Kaggle,
provenance.json there; both CC BY-SA 4.0):
* ``officials.csv`` (wyattowalsh "NBA Database"; from stats.nba.com box score summaries):
  ``game_id, official_id, first_name, last_name, jersey_num`` — primary source;
* ``2012-18_officialBoxScore.csv.zip`` (pablote "NBA Enhanced Box Score and Standings"): one row
  per team-game and official, names only, keyed by date + teams — fills the games missing in the
  primary source; names are mapped to NBA official ids through the primary source and the L2M
  reports (``L2M_stats_nba.csv`` of a local L2M collection, read only, ``OFFICIAL_n`` /
  ``OFFICIAL_ID_n``).

Output ``<data root>/scratch/l5/referees_2015_16.parquet``: one row per game and official —
``game_id, official_id, official_name, jersey_num, source`` (``wyattowalsh`` / ``pablote``).
Checks (``reports/l5_referees_check.json``): coverage, officials per game, agreement of the two
sources on the games both have, agreement with the L2M reports.

Usage:
    uv run python scripts/build_external_referees.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_external_player_bio import norm  # noqa: E402

SRC = paths.raw_dir() / "external" / "referees_kaggle"
L2M = Path("L2M_stats_nba.csv")
PABLOTE_TEAM = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "PHO": "PHX", "SA": "SAS"}
# pablote spellings of official names that differ from the NBA ones (normalized keys)
NAME_ALIAS = {"steven anderson": "steve anderson"}


def primary() -> pl.DataFrame:
    o = pl.read_csv(SRC / "officials.csv", infer_schema_length=0).filter(
        pl.col("game_id").str.starts_with("00215")
    )
    return o.select(
        "game_id",
        pl.col("official_id").cast(pl.Int64),
        (pl.col("first_name") + " " + pl.col("last_name")).alias("official_name"),
        pl.col("jersey_num").cast(pl.Int16, strict=False),
        pl.lit("wyattowalsh").alias("source"),
    ).unique(["game_id", "official_id"])


def l2m_officials() -> pl.DataFrame:
    """(game_id, official_id, name) of the 2015-16 L2M reports."""
    d = pl.read_csv(L2M, infer_schema_length=0)
    gid = pl.col("gid").str.zfill(10)
    parts = [
        d.select(
            gid.alias("game_id"),
            pl.col(f"OFFICIAL_{i}").alias("official_name"),
            pl.col(f"OFFICIAL_ID_{i}")
            .cast(pl.Float64, strict=False)
            .cast(pl.Int64)
            .alias("official_id"),
        )
        for i in (1, 2, 3, 4)
        if f"OFFICIAL_{i}" in d.columns
    ]
    return (
        pl.concat(parts)
        .filter(pl.col("game_id").str.starts_with("00215") & pl.col("official_id").is_not_null())
        .unique(["game_id", "official_id"])
    )


def secondary(games: pl.DataFrame) -> pl.DataFrame:
    z = zipfile.ZipFile(SRC / "2012-18_officialBoxScore.csv.zip")
    p = pl.read_csv(io.BytesIO(z.read("2012-18_officialBoxScore.csv")), infer_schema_length=0)
    h = p.filter(
        (pl.col("teamLoc") == "Home")
        & (pl.col("seasTyp") == "Regular")
        & pl.col("gmDate").is_between(pl.lit("2015-10-01"), pl.lit("2016-04-30"))
    ).select(
        pl.col("gmDate").str.to_date().alias("game_date"),
        pl.col("teamAbbr").replace(PABLOTE_TEAM).alias("home"),
        pl.col("opptAbbr").replace(PABLOTE_TEAM).alias("away"),
        (pl.col("offFNm") + " " + pl.col("offLNm")).alias("official_name"),
    )
    return h.join(games, on=["game_date", "home", "away"], how="left").unique(
        ["game_id", "official_name"]
    )


def main() -> None:
    tri = dict(L.rosters("v1.4").select("team_id", "abbreviation").unique().iter_rows())
    games = L.box_games("v1.4").select(
        "game_id",
        "game_date",
        pl.col("home_team_id").replace_strict(tri, default=None).alias("home"),
        pl.col("away_team_id").replace_strict(tri, default=None).alias("away"),
    )
    a = primary()
    lm = l2m_officials()
    b = secondary(games)
    # name -> official id: primary names first, then L2M names
    names = pl.concat(
        [a.select("official_name", "official_id"), lm.select("official_name", "official_id")]
    ).with_columns(pl.col("official_name").map_elements(norm, return_dtype=pl.Utf8).alias("key"))
    key_ids = names.group_by("key").agg(pl.col("official_id").unique())
    ambiguous = key_ids.filter(pl.col("official_id").list.len() > 1).to_dicts()
    key_id = key_ids.filter(pl.col("official_id").list.len() == 1).with_columns(
        pl.col("official_id").list.first()
    )
    b = b.with_columns(
        pl.col("official_name")
        .map_elements(norm, return_dtype=pl.Utf8)
        .replace(NAME_ALIAS)
        .alias("key")
    ).join(key_id, on="key", how="left")

    # agreement of the two sources where both have the game
    sa = a.group_by("game_id").agg(pl.col("official_id").sort().alias("ids_a"))
    sb = (
        b.filter(pl.col("game_id").is_not_null())
        .group_by("game_id")
        .agg(
            pl.col("official_id").sort().alias("ids_b"),
            pl.col("official_id").null_count().alias("nb_null"),
        )
    )
    both = sa.join(sb, on="game_id", how="inner")
    agree = both.filter(pl.col("ids_a") == pl.col("ids_b"))
    disagree = both.filter(pl.col("ids_a") != pl.col("ids_b")).head(10).to_dicts()

    missing = games.join(a.select("game_id").unique(), on="game_id", how="anti")
    fill = b.join(missing.select("game_id"), on="game_id", how="inner").select(
        "game_id",
        "official_id",
        "official_name",
        pl.lit(None, pl.Int16).alias("jersey_num"),
        pl.lit("pablote").alias("source"),
    )
    # use the primary spelling of the name and the jersey where the id is known
    canon = (
        a.sort("game_id")
        .unique("official_id", keep="last")
        .select(
            "official_id", pl.col("official_name").alias("_n"), pl.col("jersey_num").alias("_j")
        )
    )
    fill = (
        fill.join(canon, on="official_id", how="left")
        .with_columns(
            pl.coalesce("_n", "official_name").alias("official_name"),
            pl.coalesce("_j", "jersey_num").alias("jersey_num"),
        )
        .drop("_n", "_j")
    )
    out = pl.concat([a, fill]).sort("game_id", "official_id")
    dest = paths.scratch_dir("l5") / "referees_2015_16.parquet"
    out.write_parquet(dest)

    per = out.group_by("game_id").agg(
        pl.len().alias("n"), pl.col("official_id").null_count().alias("nul")
    )
    lg = lm.group_by("game_id").agg(pl.col("official_id").sort().alias("ids_l2m"))
    og = out.group_by("game_id").agg(pl.col("official_id").sort().alias("ids"))
    l2m_cmp = lg.join(og, on="game_id", how="left")
    tracked = set(L.games("v1.4").filter(pl.col("status") == "ok")["game_id"].to_list())
    stats = {
        "games": games.height,
        "games_with_referees": out["game_id"].n_unique(),
        "games_from_wyattowalsh": a["game_id"].n_unique(),
        "games_filled_from_pablote": fill["game_id"].n_unique(),
        "tracked_games_with_referees": len(tracked & set(out["game_id"].to_list())),
        "tracked_games": len(tracked),
        "officials_per_game": dict(per.group_by("n").len().iter_rows()),
        "rows_without_official_id": int(out["official_id"].null_count()),
        "unmapped_names_in_fill": fill.filter(pl.col("official_id").is_null())["official_name"]
        .unique()
        .to_list(),
        "ambiguous_name_keys": ambiguous,
        "distinct_officials": out["official_id"].n_unique(),
        "pablote_games_matched_to_game_id": int(b["game_id"].n_unique()),
        "sources_both_have_game": both.height,
        "sources_agree_exactly": agree.height,
        "sources_disagree_examples": disagree,
        "l2m_games": lg.height,
        "l2m_agree_exactly": int((l2m_cmp["ids_l2m"] == l2m_cmp["ids"]).fill_null(False).sum()),
        "l2m_subset_of_ours": int(
            l2m_cmp.filter(pl.col("ids").is_not_null())
            .select(pl.col("ids_l2m").list.set_difference(pl.col("ids")).list.len() == 0)
            .to_series()
            .sum()
        ),
        "output": str(dest),
    }
    (REPO / "reports" / "l5_referees_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
