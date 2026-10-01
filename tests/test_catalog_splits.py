"""Tests for the games table (nbacore.catalog) and the fixed L4 split (nbacore.splits)."""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from nbacore.catalog import build_games, parse_archive_name
from nbacore.possession.splits import make_game_splits
from nbacore.splits import N_FOLDS, assign_splits

A, B, C = 1, 2, 3  # team ids


def test_parse_archive_name():
    assert parse_archive_name("12.05.2015.CHA.at.CHI.7z") == (date(2015, 12, 5), "CHA", "CHI")
    assert parse_archive_name("2016.NBA.Raw.SportVU.Game.Logs12.05.2015.CHA.at.CHI.7z") == (
        date(2015, 12, 5),
        "CHA",
        "CHI",
    )
    with pytest.raises(ValueError):
        parse_archive_name("foo.7z")


def _ok(name, gid, d, vis, home, vid, hid):
    return {
        "archive_name": name,
        "status": "ok",
        "game_id": gid,
        "game_date": d,
        "visitor_abbr": vis,
        "home_abbr": home,
        "visitor_team_id": vid,
        "home_team_id": hid,
        "dedup": {"unique_frames": 100, "moment_dup_fraction": 0.5},
    }


def _pbp(games):  # games: [(gid, home_id, visitor_id)]
    rows = []
    for gid, h, v in games:
        rows += [(gid, 4, h), (gid, 5, v), (gid, 0, None)]
    return pl.DataFrame(
        rows, schema={"game_id": pl.Utf8, "p1_type": pl.Int64, "p1_team_id": pl.Int64}, orient="row"
    )


def test_build_games_matches_empty_archive_and_lists_missing():
    results = [
        _ok("11.01.2015.AAA.at.BBB.7z", "0021500001", "2015-11-01", "AAA", "BBB", A, B),
        _ok("11.03.2015.CCC.at.AAA.7z", "0021500004", "2015-11-03", "CCC", "AAA", C, A),
        # AAA at BBB again on 11-02: empty archive, must match game 3, not game 1 or 5
        {"archive_name": "11.02.2015.AAA.at.BBB.7z", "status": "empty_archive"},
        # JSON readable but unusable: keeps its JSON game id
        {
            "archive_name": "11.03.2015.BBB.at.CCC.7z",
            "status": "failed:SchemaError",
            "game_id": "0021500005",
        },
    ]
    pbp = _pbp(
        [
            ("0021500001", B, A),
            ("0021500002", C, B),  # no archive -> missing_archive
            ("0021500003", B, A),
            ("0021500004", A, C),
            ("0021500005", C, B),
            ("0021500006", B, A),  # after the last tracked id -> not listed
        ]
    )
    games, problems = build_games(results, pbp)
    assert problems == []
    g = {r["game_id"]: r for r in games.iter_rows(named=True)}
    assert sorted(g) == ["0021500001", "0021500002", "0021500003", "0021500004", "0021500005"]
    assert g["0021500005"]["game_id_source"] == "json"
    assert g["0021500005"]["status"] == "failed:SchemaError"
    assert g["0021500003"]["status"] == "empty_archive"
    assert g["0021500003"]["game_id_source"] == "pbp_match"
    assert g["0021500003"]["game_date"] == "2015-11-02"
    assert g["0021500002"]["status"] == "missing_archive" and g["0021500002"]["game_date"] is None
    assert g["0021500002"]["home_abbr"] == "CCC"
    assert g["0021500001"]["n_unique_frames"] == 100


def test_build_games_reports_name_mismatch():
    results = [_ok("11.01.2015.AAA.at.BBB.7z", "0021500001", "2015-11-02", "AAA", "BBB", A, B)]
    _, problems = build_games(results, _pbp([("0021500001", B, A)]))
    assert len(problems) == 1 and "JSON says" in problems[0]


def _games(n=100):
    return pl.DataFrame(
        {
            "game_id": [f"00215{i:05d}" for i in range(n)],
            "game_date": [f"2015-{10 + i // 40:02d}-{1 + (i % 40) // 2:02d}" for i in range(n)],
            "archive_name": [f"a{i}.7z" for i in range(n)],
            "status": ["ok"] * (n - 2) + ["empty_archive", "missing_archive"],
        }
    )


def test_assign_splits():
    g = assign_splits(_games())
    ok = g.filter(pl.col("status") == "ok")
    assert ok.height == 98
    counts = dict(ok.group_by("split").len().iter_rows())
    assert counts == {"train": 78 - 8, "val": 8, "test": 20}  # round(0.8*98)=78, round(7.8)=8
    # test = the latest games
    last_train = ok.filter(pl.col("split") != "test")["game_date"].max()
    assert ok.filter(pl.col("split") == "test")["game_date"].min() >= last_train
    # identical to ghost-defense's rule with the same seed
    ref = make_game_splits(ok.select(["game_id", "game_date"]), seed=9)
    assert ok.sort("game_id")["split"].to_list() == ref.sort("game_id")["split"].to_list()
    # balanced folds, alternating parity, nulls for unusable games
    assert sorted(ok.group_by("fold").len()["len"].to_list()) == [19, 19, 20, 20, 20]
    assert set(ok["fold"].to_list()) == set(range(N_FOLDS))
    par = ok.sort(["game_date", "game_id"])["parity"].to_list()
    assert par == [i % 2 for i in range(98)]
    bad = g.filter(pl.col("status") != "ok")
    assert bad["split"].null_count() == bad["fold"].null_count() == bad["parity"].null_count() == 2
    # deterministic
    assert assign_splits(_games()).equals(g)
