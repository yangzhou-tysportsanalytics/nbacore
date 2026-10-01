"""ghost_v1 view = ghost-defense's possession windows on real tiny games (its segmentation
rules migrated into nbacore as a reference view).

Skipped without v1.0 or ghost-defense's processed possessions (nbacore-v1.0/tiny).
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import nbacore.load as L
from nbacore import paths
from nbacore.views.ghost_v1 import SegmentConfig, ghost_v1

GHOST = Path(os.environ.get("GHOST_PROCESSED_DIR") or "../ghost-defense/data/processed")
TINY_DIR = GHOST / "nbacore-v1.0" / "tiny" / "possessions"
TINY = ["0021500115", "0021500230", "0021500292", "0021500333", "0021500648"]


def _have_release() -> bool:
    try:
        return (paths.release_dir("v1.0") / "MANIFEST.json").exists()
    except (FileNotFoundError, ValueError):
        return False


pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (_have_release() and TINY_DIR.exists()),
        reason="release v1.0 or ghost-defense nbacore-v1.0/tiny possessions not present",
    ),
]


@pytest.mark.parametrize("gid", TINY)
def test_ghost_v1_identical(gid):
    home = int(L.games("v1.0").filter(pl.col("game_id") == gid)["home_team_id"][0])
    ours, _ = ghost_v1(
        L.frames(gid, "v1.0"),
        L.frame_index(gid, "v1.0"),
        L.pbp("v1.0", game_id=gid),
        L.attack_direction("v1.0", game_id=gid),
        home,
        SegmentConfig(),
    )
    ref = pl.read_parquet(TINY_DIR / f"{gid}.parquet")
    assert_frame_equal(ours, ref)
