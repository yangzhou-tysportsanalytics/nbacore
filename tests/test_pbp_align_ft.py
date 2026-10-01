"""Free throws are never aligned before the foul / earlier rows of their trip (v1.3 fix: flights
assigned from the end of the trip; missing flights -> ``ft_stop_est`` inside the stop)."""

from __future__ import annotations

import polars as pl
import pytest

import nbacore.load as L
from nbacore import paths
from nbacore.events.build import build_game_l2

GID = "0021500572"  # event 93 / 94: only the flight of FT 2 of 2 is tracked


@pytest.mark.skipif(
    not (paths.release_dir("v1.2") / "MANIFEST.json").exists(), reason="release v1.2 not present"
)
def test_free_throws_after_their_foul():
    r = build_game_l2(
        L.frames(GID, "v1.2"),
        L.frame_index(GID, "v1.2"),
        L.pbp("v1.2", game_id=GID),
        L.attack_direction("v1.2", game_id=GID),
    )
    a = r["pbp_align"].sort("period", "event_num")
    prev = pl.col("unix_ms").cum_max().shift(1).over("period", "pctime_sec")
    ft = a.with_columns(prev.alias("prev")).filter(pl.col("msg_type") == 3)
    assert ft.filter(pl.col("unix_ms") < pl.col("prev")).height == 0
    t = dict(
        a.filter(pl.col("event_num").is_in([92, 93, 94])).select("event_num", "unix_ms").iter_rows()
    )
    assert t[92] < t[93] < t[94]
    m = dict(
        a.filter(pl.col("event_num").is_in([93, 94])).select("event_num", "method").iter_rows()
    )
    assert m == {93: "ft_stop_est", 94: "ft_flight"}
