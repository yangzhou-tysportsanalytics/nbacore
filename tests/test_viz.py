"""Smoke tests for nbacore.viz (not covered in ghost-defense): court drawing and mp4 export."""

from __future__ import annotations

import polars as pl

from nbacore.io.raw import BALL_ID, FRAME_SCHEMA
from nbacore.viz.animate import animate_frames, frames_to_arrays
from nbacore.viz.court import draw_court

T0 = 1_447_287_043_000


def _frames(n: int = 10) -> pl.DataFrame:
    rows = []
    for i in range(n):
        base = ("0021500001", 1, i, 1, T0 + 40 * i, 700.0 - 0.04 * i, 20.0)
        rows.append((*base, BALL_ID, BALL_ID, 30.0 + i, 25.0, 5.0))
        for k in range(10):
            team = 100 if k < 5 else 200
            rows.append((*base, team, 1000 + k, 10.0 + 5 * k, 10.0 + 2 * k + 0.1 * i, 0.0))
    return pl.DataFrame(rows, schema=FRAME_SCHEMA, orient="row")


def test_draw_court():
    ax = draw_court()
    assert ax.get_xlim()[1] >= 94 and ax.get_ylim()[1] >= 50


def test_frames_to_arrays_shapes():
    arr = frames_to_arrays(_frames())
    assert arr["xy"].shape == (10, 10, 2) and arr["ball"].shape == (10, 3)


def test_animate_mp4(tmp_path):
    out = animate_frames(_frames(), tmp_path / "clip.mp4", home_team_id=100)
    assert out.exists() and out.stat().st_size > 1000
