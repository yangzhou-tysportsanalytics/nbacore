"""ghost_v1 with the corrected release ends FG windows no later than the original
rule, keeps the window keys, and equals the L2 (v1.3) release."""

from __future__ import annotations

import polars as pl
import pytest

import nbacore.load as L
from nbacore import paths
from nbacore.events.build import CONFIGS
from nbacore.events.handler import game_handler
from nbacore.events.shots import shot_events
from nbacore.possession.shots import ShotTimeConfig
from nbacore.views.ghost_v1 import SegmentConfig, segment_game

GID = "0021500333"


@pytest.mark.skipif(
    not (paths.release_dir("v1.2") / "MANIFEST.json").exists(), reason="release v1.2 not present"
)
def test_l2release_windows():
    fr, fi = L.frames(GID, "v1.2"), L.frame_index(GID, "v1.2")
    pbp, d = L.pbp("v1.2", game_id=GID), L.attack_direction("v1.2", game_id=GID)
    home = int(L.games("v1.2").filter(pl.col("game_id") == GID)["home_team_id"][0])
    old, _ = segment_game(fr, fi, pbp, d, home, SegmentConfig())
    new, _ = segment_game(
        fr, fi, pbp, d, home, SegmentConfig(shot_cfg=ShotTimeConfig(shooter_only=True))
    )
    j = old.join(new, on="terminal_event_num", suffix="_n")
    assert j.height >= 0.99 * old.height
    assert (j["t_end_n"] <= j["t_end"]).all()
    moved = (j["t_end_n"] < j["t_end"]).mean()
    assert 0.02 < moved < 0.3
    s = shot_events(fr, fi, pbp, d, game_handler(fr, fi), CONFIGS["shot"])
    fg = new.filter(pl.col("terminal_type").is_in(["fg_made", "fg_missed"])).join(
        s, left_on="terminal_event_num", right_on="pbp_event_num"
    )
    assert (fg["t_terminal"] == fg["t_release_ms"]).mean() > 0.99
