"""Tests for the private video index helpers (nbacore.video)."""

from __future__ import annotations

import polars as pl

from nbacore.video import ANCHOR_SCHEMA, to_video_time, validate_anchors


def _anchors(rows):
    return pl.DataFrame(rows, schema=ANCHOR_SCHEMA, orient="row")


def test_mapping_and_validity():
    a = _anchors(
        [
            ("g", 1, 1_000_000, "url", 100.0, 1_030_000, "yt", "x", "2026-09-26"),
            ("g", 1, 1_060_000, "url", 190.0, None, "yt", "x", "2026-09-26"),  # after an ad break
        ]
    )
    assert validate_anchors(a) == []
    assert to_video_time(a, "g", 1_010_000) == ("url", 110.0)
    assert to_video_time(a, "g", 1_045_000) is None  # between the stretches: not covered
    assert to_video_time(a, "g", 1_061_500) == ("url", 191.5)
    assert to_video_time(a, "other", 1_010_000) is None


def test_validation_problems():
    a = _anchors(
        [
            ("g", 1, 2_000, "url", -1.0, 1_000, "yt", "x", "d"),
            ("g", 1, 2_000, "url", 5.0, None, "yt", "x", "d"),
        ]
    )
    p = validate_anchors(a)
    assert any("negative" in s for s in p) and any("end before" in s for s in p)
    assert any("duplicated" in s for s in p)
