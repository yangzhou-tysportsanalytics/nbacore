"""L5: 2015-16 advanced metrics, awards and RAPTOR (docs/external_sources.md).

Sources (``<data root>/raw/external/``, provenance.json in each):
* ``bbref_sumitrodatta/``: Basketball-Reference compilations — ``Advanced.csv`` (PER, TS%, USG%,
  WS, BPM, VORP, …), ``Player Award Shares.csv`` (MVP / ROY / DPOY / SMOY / MIP voting),
  ``End of Season Teams.csv`` and ``(Voting).csv`` (All-NBA / All-Defense / All-Rookie),
  ``All-Star Selections.csv``;
* ``raptor_538/modern_RAPTOR_by_player.csv``: FiveThirtyEight RAPTOR (CC BY 4.0).
Season 2015-16 = ``season == 2016`` in both sources. Both use Basketball-Reference player ids;
they are mapped to NBA person ids through the 2015-16 advanced table: normalized name + team
(tricode) against the 2015-16 box scores, then name alone when unique
(``build_external_player_bio.norm``).

Outputs under ``<data root>/scratch/l5/``:
* ``bbref_player_map_2015_16.parquet``: ``bbref_id, player_id, player_name, match``;
* ``advanced_2015_16.parquet``: one row per player and team stint, plus the season total of
  players with several teams (``team = "TOT"``, ``is_total``); ``player_id`` + metrics;
* ``awards_2015_16.parquet``: long table ``player_id, bbref_id, player_name, category, award,
  detail, pts_won, pts_max, share, first_place_votes, winner`` — categories ``award_voting``
  (mvp, roy, dpoy, smoy, mip), ``season_team`` (all_nba / all_defense / all_rookie, detail = 1st
  / 2nd / 3rd, with voting points), ``all_star`` (detail = East / West, ``winner`` false = named
  as a replacement);
* ``raptor_2015_16.parquet``: ``player_id, bbref_id, player_name, poss, mp, raptor_*``,
  ``war_*``, ``predator_*``, ``pace_impact``.
Checks in ``reports/l5_advanced_awards_check.json``.

Usage:
    uv run python scripts/build_external_bbref.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_external_player_bio import norm  # noqa: E402

RAW = paths.raw_dir() / "external"
BB = RAW / "bbref_sumitrodatta"
# Basketball-Reference team codes that differ from the NBA tricodes
BBREF_TEAM = {"BRK": "BKN", "CHO": "CHA", "PHO": "PHX"}
SEASON = "2016"


def read(name: str) -> pl.DataFrame:
    return pl.read_csv(BB / name, infer_schema_length=0, null_values=["NA", ""])


def box_players() -> pl.DataFrame:
    b = L.box_players("v1.4").select("player_id", "player_name", "team_id").unique()
    tri = dict(L.rosters("v1.4").select("team_id", "abbreviation").unique().iter_rows())
    return b.with_columns(
        pl.col("team_id").replace_strict(tri, default=None).alias("team"),
        pl.col("player_name").map_elements(norm, return_dtype=pl.Utf8).alias("key"),
    )


def player_map(adv: pl.DataFrame, box: pl.DataFrame) -> tuple[pl.DataFrame, list]:
    a = (
        adv.filter(~pl.col("team").is_in(["2TM", "3TM", "TOT"]))
        .select(
            pl.col("player_id").alias("bbref_id"),
            "player",
            pl.col("team").replace(BBREF_TEAM),
        )
        .with_columns(pl.col("player").map_elements(norm, return_dtype=pl.Utf8).alias("key"))
    )
    m1 = a.join(box.select("player_id", "key", "team").unique(), on=["key", "team"], how="left")
    by_team = (
        m1.filter(pl.col("player_id").is_not_null())
        .group_by("bbref_id")
        .agg(pl.col("player_id").unique())
        .filter(pl.col("player_id").list.len() == 1)
        .with_columns(pl.col("player_id").list.first(), pl.lit("name+team").alias("match"))
    )
    rest = a.join(by_team.select("bbref_id"), on="bbref_id", how="anti").unique("bbref_id")
    uniq = (
        box.group_by("key")
        .agg(pl.col("player_id").unique())
        .filter(pl.col("player_id").list.len() == 1)
        .with_columns(pl.col("player_id").list.first())
    )
    by_name = (
        rest.join(uniq, on="key", how="inner")
        .select("bbref_id", "player_id")
        .with_columns(pl.lit("name").alias("match"))
    )
    mp = pl.concat([by_team, by_name])
    names = box.select("player_id", "player_name").unique("player_id")
    mp = mp.join(names, on="player_id", how="left").select(
        "bbref_id", "player_id", "player_name", "match"
    )
    unmatched = (
        a.join(mp, on="bbref_id", how="anti")
        .select("bbref_id", "player", "team")
        .unique()
        .to_dicts()
    )
    return mp, unmatched


def main() -> None:
    adv = read("Advanced.csv").filter((pl.col("season") == SEASON) & (pl.col("lg") == "NBA"))
    box = box_players()
    mp, unmatched = player_map(adv, box)
    num = [
        c for c in adv.columns if c not in ("season", "lg", "player", "player_id", "team", "pos")
    ]
    advanced = (
        adv.with_columns(
            pl.col("team").replace({"2TM": "TOT", "3TM": "TOT"}).replace(BBREF_TEAM),
            *[pl.col(c).cast(pl.Float64) for c in num],
        )
        .with_columns((pl.col("team") == "TOT").alias("is_total"))
        .rename({"player_id": "bbref_id", "player": "player_name_bbref"})
        .join(mp.select("bbref_id", "player_id"), on="bbref_id", how="left")
        .drop("season", "lg")
    )
    advanced = advanced.select(
        "player_id", "bbref_id", "player_name_bbref", "team", "is_total", "pos", *num
    )

    ids = mp.select("bbref_id", "player_id")
    w = read("Player Award Shares.csv").filter(pl.col("season") == SEASON)
    voting = w.select(
        pl.lit("award_voting").alias("category"),
        pl.col("award").str.replace("nba ", "").alias("award"),
        pl.lit(None, pl.Utf8).alias("detail"),
        pl.col("player"),
        pl.col("player_id").alias("bbref_id"),
        pl.col("pts_won").cast(pl.Float64),
        pl.col("pts_max").cast(pl.Float64),
        pl.col("share").cast(pl.Float64),
        pl.col("first").cast(pl.Float64).alias("first_place_votes"),
        (pl.col("winner") == "TRUE").alias("winner"),
    )
    tv = read("End of Season Teams (Voting).csv").filter(pl.col("season") == SEASON)
    teams = tv.select(
        pl.lit("season_team").alias("category"),
        pl.col("type").alias("award"),
        pl.col("number_tm").alias("detail"),
        pl.col("player"),
        pl.col("player_id").alias("bbref_id"),
        pl.col("pts_won").cast(pl.Float64),
        pl.col("pts_max").cast(pl.Float64),
        pl.col("share").cast(pl.Float64),
        pl.col("x1st_tm").cast(pl.Float64).alias("first_place_votes"),
        pl.col("number_tm").is_in(["1st", "2nd", "3rd"]).alias("winner"),
    )
    st = read("All-Star Selections.csv").filter(
        (pl.col("season") == SEASON) & (pl.col("lg") == "NBA")
    )
    stars = st.select(
        pl.lit("all_star").alias("category"),
        pl.lit("all_star").alias("award"),
        pl.col("team").alias("detail"),
        pl.col("player"),
        pl.col("player_id").alias("bbref_id"),
        *[
            pl.lit(None, pl.Float64).alias(c)
            for c in ("pts_won", "pts_max", "share", "first_place_votes")
        ],
        (pl.col("replaced") != "TRUE").alias("winner"),
    )
    awards = (
        pl.concat([voting, teams, stars])
        .join(ids, on="bbref_id", how="left")
        .rename({"player": "player_name_bbref"})
    )
    awards = awards.select("player_id", *[c for c in awards.columns if c != "player_id"])

    r = pl.read_csv(
        RAW / "raptor_538" / "modern_RAPTOR_by_player.csv", infer_schema_length=0
    ).filter(pl.col("season") == SEASON)
    rnum = [c for c in r.columns if c not in ("player_name", "player_id", "season")]
    raptor = (
        r.with_columns(*[pl.col(c).cast(pl.Float64) for c in rnum])
        .rename({"player_id": "bbref_id", "player_name": "player_name_bbref"})
        .drop("season")
        .join(ids, on="bbref_id", how="left")
    )
    raptor = raptor.select("player_id", "bbref_id", "player_name_bbref", *rnum)

    out = paths.scratch_dir("l5")
    out.mkdir(parents=True, exist_ok=True)
    mp.write_parquet(out / "bbref_player_map_2015_16.parquet")
    advanced.write_parquet(out / "advanced_2015_16.parquet")
    awards.write_parquet(out / "awards_2015_16.parquet")
    raptor.write_parquet(out / "raptor_2015_16.parquet")

    tracked = set(L.rosters("v1.4")["player_id"].to_list())
    tot = advanced.filter(~pl.col("is_total") | pl.col("is_total"))
    # plausibility: games played in the advanced table vs box-score games with minutes
    gp = (
        L.box_players("v1.4")
        .filter(
            pl.col("comment").is_null()
        )  # played: no DNP comment (206 short appearances have 0 minutes)
        .group_by("player_id")
        .agg(pl.col("game_id").n_unique().alias("g_box"))
    )
    season_rows = advanced.filter(
        pl.col("is_total")
        | ~pl.col("bbref_id").is_in(advanced.filter(pl.col("is_total"))["bbref_id"].implode())
    )
    g = season_rows.join(gp, on="player_id", how="inner")
    stats = {
        "advanced_rows": advanced.height,
        "advanced_players": advanced["bbref_id"].n_unique(),
        "mapped_players": mp.height,
        "match": dict(mp.group_by("match").len().iter_rows()),
        "unmatched": unmatched,
        "map_player_id_unique": mp["player_id"].is_unique().all(),
        "tracked_roster_with_advanced": len(
            tracked & set(advanced["player_id"].drop_nulls().to_list())
        ),
        "tracked_roster_players": len(tracked),
        "games_played_equal_box": float((g["g"] == g["g_box"]).mean()),
        "awards_rows": dict(awards.group_by("category").len().iter_rows()),
        "awards_unmapped": awards.filter(pl.col("player_id").is_null())
        .select("player_name_bbref", "award")
        .to_dicts(),
        "award_winners": awards.filter((pl.col("category") == "award_voting") & pl.col("winner"))
        .select("award", "player_name_bbref")
        .to_dicts(),
        "raptor_rows": raptor.height,
        "raptor_unmapped": raptor.filter(pl.col("player_id").is_null())[
            "player_name_bbref"
        ].to_list(),
        "outputs": [
            str(out / f)
            for f in (
                "bbref_player_map_2015_16.parquet",
                "advanced_2015_16.parquet",
                "awards_2015_16.parquet",
                "raptor_2015_16.parquet",
            )
        ],
    }
    del tot
    (REPO / "reports" / "l5_advanced_awards_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
