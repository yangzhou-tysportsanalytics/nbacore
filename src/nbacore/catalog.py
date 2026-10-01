"""The games table (``games.parquet``): one row per game of the tracking period.

Rows come from three sources:

* every archive in the GitHub listing: ``status`` ``ok`` / ``empty_archive`` / ``failed:<kind>``;
  ``game_id`` from the JSON inside (``game_id_source = "json"``). For archives without a usable
  JSON the game is identified in the play-by-play by the teams in the archive name and the
  game-id order of the neighbouring dated games (``game_id_source = "pbp_match"``);
* pbp games inside the tracking period without any archive: ``status = "missing_archive"``,
  ``game_date`` null (the pbp file carries no date).

The tracking period is game ids up to the largest id of any archive (ids follow the schedule).
Split columns are added by ``nbacore.splits.assign_splits``.
"""

from __future__ import annotations

import re
from datetime import date, datetime

import polars as pl

ARCHIVE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})\.([A-Z]{3})\.at\.([A-Z]{3})\.7z$")

GAMES_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "game_date": pl.Utf8,  # YYYY-MM-DD
    "home_team_id": pl.Int64,
    "visitor_team_id": pl.Int64,
    "home_abbr": pl.Utf8,
    "visitor_abbr": pl.Utf8,
    "archive_name": pl.Utf8,
    "status": pl.Utf8,
    "game_id_source": pl.Utf8,  # json | pbp_match | pbp (missing archive) | null
    "n_unique_frames": pl.Int64,
    "dup_fraction": pl.Float64,
}


def parse_archive_name(name: str) -> tuple[date, str, str]:
    """``'12.05.2015.CHA.at.CHI.7z'`` (with or without the bogus prefix) -> (date, visitor, home)."""
    m = ARCHIVE_RE.search(name)
    if m is None:
        raise ValueError(f"unexpected archive name {name!r}")
    return datetime.strptime(m.group(1), "%m.%d.%Y").date(), m.group(2), m.group(3)


def pbp_game_teams(pbp: pl.DataFrame) -> pl.DataFrame:
    """game_id, home_team_id, visitor_team_id from typed pbp (PERSON1TYPE 4 = home, 5 = visitor)."""

    def side(t: int, name: str) -> pl.Expr:
        return (
            pl.col("p1_team_id")
            .filter((pl.col("p1_type") == t) & pl.col("p1_team_id").is_not_null())
            .mode()
            .first()
            .cast(pl.Int64)
            .alias(name)
        )

    return (
        pbp.group_by("game_id")
        .agg(side(4, "home_team_id"), side(5, "visitor_team_id"))
        .sort("game_id")
    )


def build_games(results: list[dict], pbp: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """games table from per-archive build results (``nbacore.build.process_archive``) + pbp.

    Returns (games, problems): ``problems`` lists every inconsistency found (archive name vs
    JSON, unmatched archives); an empty list means the table is fully consistent.
    """
    problems: list[str] = []
    teams = pbp_game_teams(pbp)
    ok = [r for r in results if r["status"] == "ok"]
    abbr = {}
    for r in ok:
        abbr[r["home_team_id"]] = r["home_abbr"]
        abbr[r["visitor_team_id"]] = r["visitor_abbr"]
    team_id_of = {v: k for k, v in abbr.items()}
    if len(team_id_of) != len(abbr):
        problems.append("team abbreviations are not one-to-one")

    rows = []
    for r in results:
        d, vis, home = parse_archive_name(r["archive_name"])
        row = {
            "archive_name": r["archive_name"],
            "status": r["status"],
            "game_date": d.isoformat(),
            "visitor_abbr": vis,
            "home_abbr": home,
            "home_team_id": team_id_of.get(home),
            "visitor_team_id": team_id_of.get(vis),
            "game_id": None,
            "game_id_source": None,
            "n_unique_frames": None,
            "dup_fraction": None,
        }
        if r["status"] == "ok":
            got = (r["game_date"], r["visitor_abbr"], r["home_abbr"])
            if got != (row["game_date"], vis, home):
                problems.append(f"{r['archive_name']}: JSON says {got}")
            row.update(
                game_id=r["game_id"],
                game_id_source="json",
                home_team_id=r["home_team_id"],
                visitor_team_id=r["visitor_team_id"],
                n_unique_frames=r["dedup"]["unique_frames"],
                dup_fraction=r["dedup"]["moment_dup_fraction"],
            )
        elif r.get("game_id"):  # JSON readable but unusable (e.g. no moments at all)
            row.update(game_id=r["game_id"], game_id_source="json")
        rows.append(row)

    # identify archives without a usable JSON through pbp teams + game-id order of dated games
    dated = sorted((row["game_date"], row["game_id"]) for row in rows if row["game_id"])
    taken = {gid for _, gid in dated}
    for row in rows:
        if row["game_id"] is not None:
            continue
        lo = max((g for d, g in dated if d < row["game_date"]), default="0000000000")
        hi = min((g for d, g in dated if d > row["game_date"]), default="9999999999")
        cand = teams.filter(
            (pl.col("home_team_id") == row["home_team_id"])
            & (pl.col("visitor_team_id") == row["visitor_team_id"])
            & (pl.col("game_id") > lo)
            & (pl.col("game_id") < hi)
            & ~pl.col("game_id").is_in(list(taken))
        )["game_id"].to_list()
        if len(cand) == 1:
            row.update(game_id=cand[0], game_id_source="pbp_match")
            taken.add(cand[0])
        else:
            problems.append(f"{row['archive_name']}: {len(cand)} pbp candidates {cand}")

    last_id = max(r["game_id"] for r in rows if r["game_id"])
    for t in teams.filter(
        (pl.col("game_id") <= last_id) & ~pl.col("game_id").is_in(list(taken))
    ).iter_rows(named=True):
        rows.append(
            {
                "archive_name": None,
                "status": "missing_archive",
                "game_date": None,
                "home_abbr": abbr.get(t["home_team_id"]),
                "visitor_abbr": abbr.get(t["visitor_team_id"]),
                "home_team_id": t["home_team_id"],
                "visitor_team_id": t["visitor_team_id"],
                "game_id": t["game_id"],
                "game_id_source": "pbp",
                "n_unique_frames": None,
                "dup_fraction": None,
            }
        )
    games = pl.DataFrame(rows, schema=GAMES_SCHEMA).sort(
        ["game_id", "archive_name"], nulls_last=True
    )
    dups = games.filter(pl.col("game_id").is_not_null() & pl.col("game_id").is_duplicated())
    if dups.height:
        problems.append(f"duplicate game ids: {dups['game_id'].unique().to_list()}")
    return games, problems
