"""P0 acceptance: L1 built by nbacore on `tiny` equals ghost-defense's L1 row for row.

Reference: ghost-defense ``data/interim`` produced by ``scripts/phase1_clean.py``; the frame code
(``ghost.io.raw.game_to_frames``, ``ghost.clean.dedup``) is unchanged between the commit that wrote
those files and ed6d754. Override the location with ``GHOST_REF_DIR``.

Built through ``nbacore.build.process_archive`` (the code path of ``scripts/build_l1.py``).
Every table is compared exactly (values, dtypes, column order, row order):
frames, frame_index, rosters, events, attack_direction.

Skipped when the tiny raw data (``$NBA_DATA_ROOT/raw``) or the reference is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from nbacore import paths
from nbacore.build import process_archive
from nbacore.io import load_pbp

REF = Path(os.environ.get("GHOST_REF_DIR") or "../ghost-defense/data/interim")
MANIFEST = paths.raw_dir() / "manifest_tiny.json"
TINY_GAMES = ["0021500115", "0021500230", "0021500292", "0021500333", "0021500648"]

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (MANIFEST.exists() and paths.pbp_csv().exists() and (REF / "frames").is_dir()),
        reason="tiny raw data under NBA_DATA_ROOT or ghost-defense reference not present",
    ),
]


def _archives() -> list[Path]:
    names = json.loads(MANIFEST.read_text(encoding="utf-8"))["archives"]
    return [paths.sportvu_dir() / n for n in names]


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> dict[str, dict[str, pl.DataFrame]]:
    """L1 tables per tiny game, built through the production path (``process_archive``) and read
    back from the parquet files it writes."""
    out = tmp_path_factory.mktemp("l1")
    pbp_path = out / "pbp.parquet"
    load_pbp(paths.pbp_csv()).write_parquet(pbp_path)
    res = {}
    for arc in _archives():
        r = process_archive(arc, out, pbp_path)
        assert r["status"] == "ok", r.get("error")
        gid = r["game_id"]
        res[gid] = {
            "frames": pl.read_parquet(out / "frames" / f"{gid}.parquet"),
            "frame_index": pl.read_parquet(out / "frame_index" / f"{gid}.parquet"),
            "rosters": r["tables"]["rosters"],
            # ghost-defense calls the tracking-event table events.parquet
            "events": r["tables"]["tracking_events"],
            "attack_direction": r["tables"]["attack_direction"],
        }
    return res


def test_tiny_membership(built):
    assert sorted(built) == TINY_GAMES


@pytest.mark.parametrize("gid", TINY_GAMES)
@pytest.mark.parametrize("table", ["frames", "frame_index"])
def test_per_game_tables_identical(built, gid, table):
    ref = pl.read_parquet(REF / table / f"{gid}.parquet")
    new = built[gid][table]
    assert new.height == ref.height > 0
    assert_frame_equal(new, ref, check_exact=True)


@pytest.mark.parametrize("table", ["rosters", "events", "attack_direction"])
def test_game_level_tables_identical(built, table):
    ref = pl.read_parquet(REF / f"{table}.parquet").filter(pl.col("game_id").is_in(TINY_GAMES))
    for gid in TINY_GAMES:
        r = ref.filter(pl.col("game_id") == gid)
        assert r.height > 0, f"{table}: {gid} missing from reference"
        assert_frame_equal(built[gid][table], r, check_exact=True)
