"""Tests for drive / cut candidates (nbacore.events.motion) on synthetic frames."""

from __future__ import annotations

from nbacore.court import HOOP_LEFT
from nbacore.events.handler import game_handler
from nbacore.events.motion import motion_candidates
from test_events_handler import T0, A, B, D, E, _frames
from test_events_screens import _direction


def test_drive_and_cut():
    # OFF attacks left (hoop at x = 5.25). A dribbles from x = 30 to x = 12 at 10 ft/s along
    # y = 25 with D in front; B cuts from (30, 10) toward the hoop at 12 ft/s; E stands.
    spec = []
    hx, hy = HOOP_LEFT
    for i in range(70):
        t = T0 + 40 * i
        k = min(max(i - 10, 0), 45)
        ax = 30.0 - 0.4 * k
        bx, by = 30.0 - 0.48 * k * (30 - hx) / 30.0, 10.0 + 0.48 * k * (hy - 10.0) / 30.0
        spec.append(
            (
                t,
                (ax - 0.5, 25.0, 3.0),
                {A: (ax, 25.0), D: (ax - 3.0, 25.0), B: (bx, by), E: (40.0, 40.0)},
            )
        )
    f, fi = _frames(spec)
    drives, cuts = motion_candidates(f, game_handler(f, fi), _direction())
    d = drives.filter(drives["player_id"] == A).row(0, named=True)
    assert d["def_id"] == D and d["def_offset_start_ft"] > 0  # defender between A and the basket
    assert 9.0 < d["radial_peak_fts"] < 11.0 and d["dist_basket_start_ft"] > 20
    assert d["start_frontcourt"] and d["end_reason"] == "none"
    c = cuts.filter(cuts["player_id"] == B).row(0, named=True)
    assert c["radial_mean_fts"] > 8 and c["dist_basket_end_ft"] < c["dist_basket_start_ft"]
    assert c["ball_dist_start_ft"] > 10 and not c["caught"]
