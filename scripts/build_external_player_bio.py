"""L5: player height / weight / wingspan for 2015-16 (docs/external_sources.md).

Sources (``<data root>/raw/external/player_bio_kaggle/``, provenance.json there):
* ``all_seasons.csv`` (justinas): height (cm), weight (kg) per season, names + team only;
* ``NBA_Anthropometric.csv`` (CC BY 4.0): draft-combine wingspan / standing reach (cm).
NBA person ids come from the 2015-16 box scores (``raw/external/boxscores_eoinamoore_v515``):
names are normalized (accents, punctuation, suffixes, case) and matched on name + team, then name
alone when unique; the combine data is matched on name (+ draft year when ambiguous).

Output ``<data root>/scratch/l5/player_bio_2015_16.parquet``: ``player_id, player_name,
height_cm, weight_kg, wingspan_cm, standing_reach_cm, combine_year, bio_match, combine_match``;
checks in ``reports/l5_player_bio_check.json`` (coverage of the box-score players and of the
tracked roster, height by position).

Usage:
    uv run python scripts/build_external_player_bio.py
"""

from __future__ import annotations

import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths

REPO = Path(__file__).resolve().parents[1]
SRC = paths.raw_dir() / "external" / "player_bio_kaggle"
BOX = paths.raw_dir() / "external" / "boxscores_eoinamoore_v515" / "PlayerStatistics.csv"
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
# all_seasons team abbreviations that differ from the box-score / NBA ones
TEAM_FIX = {"NOH": "NOP", "NJN": "BKN", "PHO": "PHX", "CHH": "CHA", "GOS": "GSW", "SAN": "SAS"}


# letters NFKD does not decompose to ASCII
TRANSLIT = str.maketrans({"ß": "ss", "ı": "i", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l", "đ": "d"})


def norm(name: str | None) -> str | None:
    if name is None:
        return None
    s = name.translate(TRANSLIT)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"['`’]", "", s)  # "o'brien" -> "obrien", not an initial "o"
    s = re.sub(r"[.\-,]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(\w) (?=\w\b)", r"\1", s)  # initials: "c j wilcox" -> "cj wilcox"
    first, _, rest = s.partition(" ")
    rest = re.sub(r"\s+", " ", SUFFIX.sub(" ", rest)).strip()  # "JR Smith": jr is a first name
    s = f"{first} {rest}".strip()
    return ALIAS.get(s, s)


# names that changed or are written differently between the sources (the box score is the key)
ALIAS = {"enes kanter": "enes freedom", "nene": "nene hilario"}


def read_zip(name: str, member: str) -> pl.DataFrame:
    z = zipfile.ZipFile(SRC / name)
    return pl.read_csv(io.BytesIO(z.read(member)), infer_schema_length=20000)


def box_players() -> pl.DataFrame:
    """2015-16 regular-season players with NBA ids and their teams (tricode via games)."""
    b = pl.scan_csv(BOX, infer_schema_length=20000).filter(
        pl.col("gameId").cast(pl.Utf8).str.starts_with("215")
    )
    b = b.select(
        pl.col("personId").cast(pl.Int64).alias("player_id"),
        (pl.col("firstName") + " " + pl.col("lastName")).alias("player_name"),
        pl.col("playerteamCity"),
        pl.col("playerteamName"),
        pl.col("playerteamId").cast(pl.Int64).alias("team_id")
        if "playerteamId" in b.collect_schema().names()
        else pl.lit(None, pl.Int64).alias("team_id"),
    ).collect()
    return b.unique(["player_id", "team_id"])


def main() -> None:
    box = box_players()
    # team tricodes from the tracking rosters (team_id -> abbreviation)
    tri = dict(L.rosters("v1.2").select("team_id", "abbreviation").unique().iter_rows())
    box = box.with_columns(
        pl.col("team_id").replace_strict(tri, default=None).alias("team"),
        pl.col("player_name").map_elements(norm, return_dtype=pl.Utf8).alias("key"),
    )
    ids = box.select("player_id", "player_name", "key").unique("player_id")

    a = read_zip("nba-player-data.zip", "all_seasons.csv").filter(pl.col("season") == "2015-16")
    a = a.with_columns(
        pl.col("player_name").map_elements(norm, return_dtype=pl.Utf8).alias("key"),
        pl.col("team_abbreviation").replace(TEAM_FIX).alias("team"),
    )
    # 1) name + team
    m1 = a.join(box.select("player_id", "key", "team").unique(), on=["key", "team"], how="left")
    matched = m1.filter(pl.col("player_id").is_not_null()).with_columns(
        pl.lit("name+team").alias("bio_match")
    )
    # 2) name alone, when unique among 2015-16 players
    rest = m1.filter(pl.col("player_id").is_null()).drop("player_id")
    uniq = (
        ids.group_by("key")
        .agg(pl.col("player_id").first(), pl.len().alias("n"))
        .filter(pl.col("n") == 1)
    )
    m2 = rest.join(uniq.select("key", "player_id"), on="key", how="left").with_columns(
        pl.when(pl.col("player_id").is_not_null()).then(pl.lit("name")).alias("bio_match")
    )
    bio = pl.concat([matched, m2], how="diagonal_relaxed")
    unmatched_bio = (
        bio.filter(pl.col("player_id").is_null()).select("player_name", "team").to_dicts()
    )
    bio = (
        bio.filter(pl.col("player_id").is_not_null())
        .group_by("player_id")
        .agg(
            pl.col("player_height").mean().alias("height_cm"),
            pl.col("player_weight").mean().alias("weight_kg"),
            pl.col("draft_year").first(),
            pl.col("bio_match").first(),
        )
    )

    c = read_zip("nba-anthropometric.zip", "NBA_Anthropometric.csv").with_columns(
        pl.col("player_name").map_elements(norm, return_dtype=pl.Utf8).alias("key")
    )
    c = c.select("key", "wingspan", "standing_reach", pl.col("draft_year").alias("combine_year"))
    out = ids.join(bio, on="player_id", how="left")
    cm = out.join(c, on="key", how="left")
    # ambiguous combine names: keep the row whose year matches the draft year, else the latest
    cm = cm.with_columns(
        (pl.col("combine_year").cast(pl.Utf8) == pl.col("draft_year").cast(pl.Utf8)).alias("_yr")
    ).sort("player_id", "_yr", "combine_year", descending=[False, True, True])
    n_c = cm.group_by("player_id").agg(pl.col("wingspan").is_not_null().sum().alias("n"))
    cm = cm.unique("player_id", keep="first").join(n_c, on="player_id")
    out = cm.with_columns(
        pl.when(pl.col("wingspan").is_null())
        .then(None)
        .when(pl.col("n") == 1)
        .then(pl.lit("name"))
        .otherwise(pl.lit("name+year"))
        .alias("combine_match")
    ).select(
        "player_id",
        "player_name",
        pl.col("height_cm").cast(pl.Float32),
        pl.col("weight_kg").cast(pl.Float32),
        pl.col("wingspan").cast(pl.Float32).alias("wingspan_cm"),
        pl.col("standing_reach").cast(pl.Float32).alias("standing_reach_cm"),
        pl.col("combine_year").cast(pl.Int16),
        "bio_match",
        "combine_match",
    )
    dest = paths.scratch_dir("l5") / "player_bio_2015_16.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.sort("player_id").write_parquet(dest)

    tracked = L.rosters("v1.2").select("player_id", "position").unique("player_id")
    t = tracked.join(out, on="player_id", how="left")
    pos = (
        t.with_columns(pl.col("position").str.slice(0, 1).alias("p"))
        .group_by("p")
        .agg(pl.len(), pl.col("height_cm").median().alias("median_height_cm"))
        .sort("p")
    )
    stats = {
        "box_players_2015_16": ids.height,
        "all_seasons_rows_2015_16": a.height,
        "height_weight_coverage_box_players": float(out["height_cm"].is_not_null().mean()),
        "wingspan_coverage_box_players": float(out["wingspan_cm"].is_not_null().mean()),
        "tracked_roster_players": tracked.height,
        "height_weight_coverage_tracked": float(t["height_cm"].is_not_null().mean()),
        "wingspan_coverage_tracked": float(t["wingspan_cm"].is_not_null().mean()),
        "bio_match": dict(out.group_by("bio_match").len().iter_rows()),
        "combine_match": dict(out.group_by("combine_match").len().iter_rows()),
        "all_seasons_rows_unmatched": unmatched_bio,
        "median_height_by_position_tracked": pos.to_dicts(),
        "output": str(dest),
    }
    (REPO / "reports" / "l5_player_bio_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )
    print(json.dumps(stats, indent=1, default=str))


if __name__ == "__main__":
    main()
