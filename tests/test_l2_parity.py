"""L2 parity with ghost-defense on real tiny games: shot release = ghost `t_terminal`
(taken from the ball, not from the pbp clock).

Skipped without v1.0 or ghost-defense's processed possessions.
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl
import pytest

import nbacore.load as L
from nbacore import paths
from nbacore.events.handler import game_handler
from nbacore.events.shots import shot_events

GHOST = Path(os.environ.get("GHOST_PROCESSED_DIR") or "../ghost-defense/data/processed")
TINY = ["0021500115", "0021500230", "0021500292", "0021500333", "0021500648"]


def _have_release() -> bool:
    try:
        paths.release_dir("v1.0")
    except (FileNotFoundError, ValueError):
        return False
    return (paths.release_dir("v1.0") / "MANIFEST.json").exists()


pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (_have_release() and (GHOST / "possessions.parquet").exists()),
        reason="release v1.0 or ghost-defense possessions not present",
    ),
]


@pytest.mark.parametrize("gid", TINY)
def test_shot_release_equals_ghost_terminal(gid):
    fr, fi = L.frames(gid, "v1.0"), L.frame_index(gid, "v1.0")
    s = shot_events(
        fr,
        fi,
        L.pbp("v1.0", game_id=gid),
        L.attack_direction("v1.0", game_id=gid),
        game_handler(fr, fi),
    )
    g = pl.read_parquet(GHOST / "possessions.parquet").filter(
        (pl.col("game_id") == gid) & pl.col("terminal_type").is_in(["fg_made", "fg_missed"])
    )
    j = g.join(s, left_on="terminal_event_num", right_on="pbp_event_num", how="left")
    assert j.height > 0
    eq = (j["t_terminal"] == j["t_release_ms"]).fill_null(False)
    # only allowed difference: pbp rows inserted late, refined by ghost onto an earlier release
    assert (eq | (j["method"] == "duplicate_release").fill_null(False)).all()
    assert s["event_uid"].drop_nulls().n_unique() == s["event_uid"].drop_nulls().len()


@pytest.mark.parametrize("gid", TINY[:2])
def test_grid_matches_ghost_and_aligns_across_rates(gid):
    import numpy as np

    from nbacore.ballhandler.infer import handler_on_grid, infer_handler_raw
    from nbacore.geometry import grid_distances
    from nbacore.grid import resample
    from nbacore.possession.resample import ResampleConfig, resample_possession

    fr, fi = L.frames(gid, "v1.0"), L.frame_index(gid, "v1.0")
    h = game_handler(fr, fi)
    poss = pl.read_parquet(GHOST / "possessions.parquet").filter(pl.col("game_id") == gid)
    for p in poss.head(15).iter_rows(named=True):
        g5 = resample(fr, fi, p, hz=5, handler=h, handler_mode="ghost")
        ref = resample_possession(fr, fi, p, ResampleConfig())
        np.testing.assert_array_equal(g5.off_xy, ref.off_xy)
        np.testing.assert_array_equal(g5.ball, ref.ball)
        t_raw, slot = infer_handler_raw(
            fr, p["period"], p["t_start"], p["t_end"], list(ref.off_ids)
        )
        np.testing.assert_array_equal(
            g5.handler_slot,
            np.where(ref.valid, handler_on_grid(t_raw, slot, ref.t.astype(np.int64)), -1),
        )
        for hz in (10, 25):
            g = resample(fr, fi, p, hz=hz, handler=h)
            k = hz // 5
            assert g.t.size == 24 * hz + 1
            np.testing.assert_array_equal(g.t[::k], g5.t)
            np.testing.assert_allclose(g.off_xy[::k][g5.valid], g5.off_xy[g5.valid], atol=1e-4)
        d = grid_distances(g5)
        assert d["off_def_dist"].shape == (121, 5, 5) and d["ball_dist"].shape == (121, 10)
        dd = np.linalg.norm(g5.off_xy[0, 0] - g5.def_xy[0, 0])
        assert abs(d["off_def_dist"][0, 0, 0] - dd) < 1e-5


def test_event_uids_unique_on_known_edge_cases():
    """0021500036: a flight split by a short touch produced two incomplete passes with the same
    release; 0021500504: late-inserted pbp rows shared a shot release."""
    from nbacore.events.build import build_game_l2

    for gid in ("0021500036", "0021500504"):
        r = build_game_l2(
            L.frames(gid, "v1.0"),
            L.frame_index(gid, "v1.0"),
            L.pbp("v1.0", game_id=gid),
            L.attack_direction("v1.0", game_id=gid),
        )
        assert r["stats"]["uid_duplicates"] == 0, gid


@pytest.mark.parametrize("gid", TINY)
def test_v13_release_never_later_than_ghost(gid):
    """v1.3 (``ShotTimeConfig(shooter_only=True)``): only the pbp shooter's hands count, so a
    release can only move earlier than ghost's (a ball in flight over a teammate was taken as
    still in hand); most releases are unchanged."""
    from nbacore.events.build import CONFIGS

    fr, fi = L.frames(gid, "v1.0"), L.frame_index(gid, "v1.0")
    s = shot_events(
        fr,
        fi,
        L.pbp("v1.0", game_id=gid),
        L.attack_direction("v1.0", game_id=gid),
        game_handler(fr, fi),
        CONFIGS["shot"],
    )
    g = pl.read_parquet(GHOST / "possessions.parquet").filter(
        (pl.col("game_id") == gid) & pl.col("terminal_type").is_in(["fg_made", "fg_missed"])
    )
    j = g.join(s, left_on="terminal_event_num", right_on="pbp_event_num", how="inner")
    j = j.filter(pl.col("method") != "duplicate_release")
    assert (j["t_release_ms"] <= j["t_terminal"]).all()
    assert (j["t_release_ms"] == j["t_terminal"]).mean() >= 0.75


def test_v16_tip_release_after_the_miss_left_the_rim():
    """Tip after a miss (0021500195 ev 455 miss, 456 tip-in): with ``after_miss_from_rim``
    the tip's release is searched after the miss left the rim zone, not 0.12 s after the miss's
    release."""
    from nbacore.events.build import CONFIGS

    gid = "0021500195"
    if not (paths.release_dir("v1.5") / "MANIFEST.json").exists():
        pytest.skip("release v1.5 not present")
    fr, fi = L.frames(gid, "v1.5"), L.frame_index(gid, "v1.5")
    s = shot_events(
        fr,
        fi,
        L.pbp("v1.5", game_id=gid),
        L.attack_direction("v1.5", game_id=gid),
        L.handler(gid, "v1.5"),
        CONFIGS["shot"],
    )
    t = dict(
        s.filter(pl.col("pbp_event_num").is_in([455, 456]))
        .select("pbp_event_num", "t_release_ms")
        .iter_rows()
    )
    rim = s.filter(pl.col("pbp_event_num") == 455)["t_rim_ms"][0]
    assert t[456] - t[455] > 400
    assert t[456] >= rim
