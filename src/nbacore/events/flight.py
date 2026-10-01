"""Ball flight (L2): the ball free of every player.

A frame is *free* when the ball is tracked and no player is a handler candidate, i.e. no player
within ``max_dist_ft`` (xy) of the ball, or the ball above ``max_ball_z_ft`` (the same test as the
handler candidate, so flight and handler never contradict each other). A **flight** is a run of
≥ ``min_flight_frames`` free frames inside one continuity segment (default ≥ 2 frames).

Per flight: the last handler before it (``from_id``) and the first handler after it (``to_id``,
null when the segment ends first), ball kinematics, and whether the ball reached the rim zone
(``reaches_rim``: within ``rim_xy_ft`` of a hoop centre between ``rim_z_lo`` and ``rim_z_hi``) —
a flight that reaches the rim is a shot (or a tip), so a handler change across it is a rebound,
not a pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.court import HOOP_LEFT, HOOP_RIGHT

FLIGHT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "segment_id": pl.Int32,
    "t_start_ms": pl.Int64,  # first free frame
    "t_end_ms": pl.Int64,  # last free frame
    "n_frames": pl.Int32,
    "from_id": pl.Int64,  # handler on the frame before the flight (null: none)
    "from_team_id": pl.Int64,
    "to_id": pl.Int64,  # first handler after the flight within the segment (null: none)
    "to_team_id": pl.Int64,
    "t_to_ms": pl.Int64,  # first frame with that handler
    "x_start": pl.Float32,
    "y_start": pl.Float32,
    "x_end": pl.Float32,
    "y_end": pl.Float32,
    "max_z": pl.Float32,
    "path_ft": pl.Float32,  # ball path length in xy
    "reaches_rim": pl.Boolean,
    "t_rim_ms": pl.Int64,  # first frame in the rim zone (null: none)
    "ends_segment": pl.Boolean,  # the flight runs into the end of the continuity segment
}


@dataclass
class FlightConfig:
    min_flight_frames: int = 2
    rim_xy_ft: float = 2.0
    rim_z_lo: float = 8.0
    rim_z_hi: float = 11.5


def ball_flights(handler: pl.DataFrame, frames: pl.DataFrame, cfg: FlightConfig | None = None):
    """Flights of one game from the L2 handler table (``nbacore.events.handler``) + L1 frames."""
    cfg = cfg or FlightConfig()
    ball = frames.filter(pl.col("team_id") == -1).select("period", "unix_ms", "x", "y", "z")
    h = handler.join(ball, on=["period", "unix_ms"], how="left").sort(["period", "unix_ms"])
    free = (h["has_ball"] & (h["cand_id"] < 0)).to_numpy()
    seg = h["segment_id"].to_numpy()
    t = h["unix_ms"].to_numpy()
    hid = h["handler_id"].to_numpy()
    cand = h["cand_id"].to_numpy()
    team = h["handler_team_id"].to_numpy()
    x, y, z = (h[c].to_numpy().astype(float) for c in ("x", "y", "z"))
    per = h["period"].to_numpy()
    n = free.size
    rows = []
    i = 0
    while i < n:
        if not free[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and free[j + 1] and seg[j + 1] == seg[i]:
            j += 1
        if j - i + 1 >= cfg.min_flight_frames:
            rows.append(_flight_row(i, j, seg, t, hid, cand, team, x, y, z, per, cfg))
        i = j + 1
    gid = handler["game_id"][0]
    return pl.DataFrame(
        rows, schema={k: v for k, v in FLIGHT_SCHEMA.items() if k != "game_id"}, orient="row"
    ).select(pl.lit(gid).alias("game_id"), pl.all())


def _flight_row(i, j, seg, t, hid, cand, team, x, y, z, per, cfg):
    s = seg[i]
    before = i - 1
    from_id = from_team = None
    if before >= 0 and seg[before] == s and hid[before] >= 0:
        from_id, from_team = int(hid[before]), int(team[before])
    # first frame after the flight with a handler *candidate* in the same segment: the receiver
    # (the sticky handler may still carry the passer for a short run)
    to_id = to_team = t_to = None
    k = j + 1
    while k < t.size and seg[k] == s:
        if cand[k] >= 0 and hid[k] >= 0:
            to_id, to_team, t_to = int(hid[k]), int(team[k]), int(t[k])
            break
        k += 1
    ends_segment = j + 1 >= t.size or seg[j + 1] != s
    xs, ys, zs = x[i : j + 1], y[i : j + 1], z[i : j + 1]
    path = float(np.nansum(np.hypot(np.diff(xs), np.diff(ys)))) if xs.size > 1 else 0.0
    rim = np.zeros(xs.size, dtype=bool)
    for hoop in (HOOP_LEFT, HOOP_RIGHT):
        rim |= (
            (np.hypot(xs - hoop[0], ys - hoop[1]) < cfg.rim_xy_ft)
            & (zs > cfg.rim_z_lo)
            & (zs < cfg.rim_z_hi)
        )
    t_rim = int(t[i + int(np.flatnonzero(rim)[0])]) if rim.any() else None
    return (
        int(per[i]),
        int(s),
        int(t[i]),
        int(t[j]),
        j - i + 1,
        from_id,
        from_team,
        to_id,
        to_team,
        t_to,
        float(xs[0]),
        float(ys[0]),
        float(xs[-1]),
        float(ys[-1]),
        float(np.nanmax(zs)),
        path,
        bool(rim.any()),
        t_rim,
        bool(ends_segment),
    )
