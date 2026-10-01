"""Tests for nbacore.events.uid.match_events."""

from __future__ import annotations

import polars as pl

from nbacore.events.uid import match_events


def test_match_events():
    old = pl.DataFrame(
        {
            "event_uid": [
                "g:pass:1000:7",  # moves 80 ms -> matched
                "g:pass:5000:7",  # moves 400 ms -> unmatched both sides
                "g:screen_candidate:9000:3-4",  # unchanged
                "g:pass:7000:8",  # disappears
            ]
        }
    )
    new = pl.DataFrame(
        {
            "event_uid": [
                "g:pass:1080:7",
                "g:pass:5400:7",
                "g:screen_candidate:9000:3-4",
                "g:pass:7000:9",
            ]
        }
    )
    m = match_events(old, new)
    pairs = {(r["old_uid"], r["new_uid"]) for r in m.iter_rows(named=True)}
    assert ("g:pass:1000:7", "g:pass:1080:7") in pairs
    assert ("g:screen_candidate:9000:3-4", "g:screen_candidate:9000:3-4") in pairs
    assert ("g:pass:5000:7", None) in pairs and (None, "g:pass:5400:7") in pairs
    assert ("g:pass:7000:8", None) in pairs and (None, "g:pass:7000:9") in pairs  # other actor
    assert m.height == 6
