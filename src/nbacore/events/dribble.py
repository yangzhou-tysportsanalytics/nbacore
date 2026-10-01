"""Dribbles and per-touch summaries (L2).

A **dribble** (``t_bounce_ms``) is a local minimum of the ball height during a possession touch
(a run of one handler, ``possession_touch``) with

(a) ``z_min`` ≤ ``z_floor_ft``;
(b) prominence ≥ ``min_prominence_ft``: the smaller of the highest ball z within
    ``prominence_s`` before and after the minimum, minus ``z_min``;
(c) vertical velocity (3-frame central difference) < 0 just before and > 0 just after;
(d) ≥ ``min_gap_s`` after the previous dribble of the same touch;
(e) ball within ``max_handler_ft`` (xy) of the handler at the minimum.

No dribble is inferred across frames without the ball; such touches get ``ball_gap``.
Per touch: ``n_dribbles``, ``live_duration_s`` (frames with a running game clock only),
``ball_gap`` and ``ends_with`` (pass / handoff / shot / turnover / none).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.io.raw import BALL_ID


@dataclass
class DribbleConfig:
    z_floor_ft: float = 1.0
    min_prominence_ft: float = 1.0
    prominence_s: float = 0.3
    min_gap_s: float = 0.2
    max_handler_ft: float = 4.0


DRIBBLE_SCHEMA = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,
    "handler_id": pl.Int64,
    "team_id": pl.Int64,
    "t_bounce_ms": pl.Int64,
    "z_min": pl.Float32,
    "prominence_ft": pl.Float32,
    "x": pl.Float32,
    "y": pl.Float32,
    "touch_uid": pl.Utf8,
    "event_uid": pl.Utf8,
}


def dribbles(
    frames: pl.DataFrame,
    touches: pl.DataFrame,
    frame_index: pl.DataFrame,
    passes_df: pl.DataFrame | None = None,
    shots: pl.DataFrame | None = None,
    cfg: DribbleConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(dribble events, touches with n_dribbles / live_duration_s / ball_gap / ends_with)."""
    cfg = cfg or DribbleConfig()
    gid = frame_index["game_id"][0]
    ball = frames.filter(pl.col("team_id") == BALL_ID).select("period", "unix_ms", "x", "y", "z")
    players = frames.filter(pl.col("team_id") != BALL_ID).select(
        "period", "unix_ms", "player_id", "x", "y"
    )
    fz = frame_index.select("period", "unix_ms", "clock_frozen", "has_ball")
    rows, summ = [], []
    half = max(1, int(round(cfg.prominence_s * 25)))
    gap = max(1, int(round(cfg.min_gap_s * 25)))
    for (period,), fi_p in fz.sort(["period", "unix_ms"]).group_by(["period"], maintain_order=True):
        t_all = fi_p["unix_ms"].to_numpy()
        b = fi_p.join(ball.filter(pl.col("period") == period), on=["period", "unix_ms"], how="left")
        z_all = b["z"].to_numpy().astype(float)
        bx, by = b["x"].to_numpy().astype(float), b["y"].to_numpy().astype(float)
        frozen = fi_p["clock_frozen"].to_numpy()
        pl_p = players.filter(pl.col("period") == period)
        for tch in touches.filter(pl.col("period") == period).iter_rows(named=True):
            i0 = int(np.searchsorted(t_all, tch["t_start_ms"]))
            i1 = int(np.searchsorted(t_all, tch["t_end_ms"]))
            z = z_all[i0 : i1 + 1]
            t = t_all[i0 : i1 + 1]
            gap_flag = bool(np.isnan(z).any())
            hp = pl_p.filter(
                (pl.col("player_id") == tch["handler_id"])
                & pl.col("unix_ms").is_between(t[0], t[-1])
            )
            hpos = {int(u): (x, y) for u, x, y in hp.select("unix_ms", "x", "y").iter_rows()}
            vz = np.full(z.size, np.nan)
            if z.size >= 3:
                vz[1:-1] = (z[2:] - z[:-2]) / ((t[2:] - t[:-2]) / 1000.0)
            n, last = 0, -(10**9)
            for k in range(1, z.size - 1):
                if (
                    np.isnan(z[k])
                    or z[k] > cfg.z_floor_ft
                    or not (z[k] <= z[k - 1] and z[k] <= z[k + 1])
                ):
                    continue
                if not (vz[k - 1] < 0 and vz[k + 1] > 0):
                    continue
                left, right = z[max(0, k - half) : k], z[k + 1 : k + 1 + half]
                if (
                    left.size == 0
                    or right.size == 0
                    or np.isnan(left).all()
                    or np.isnan(right).all()
                ):
                    continue
                prom = min(np.nanmax(left), np.nanmax(right)) - z[k]
                if prom < cfg.min_prominence_ft or k - last < gap:
                    continue
                hxy = hpos.get(int(t[k]))
                bxy = (bx[i0 + k], by[i0 + k])
                if hxy is None or np.hypot(hxy[0] - bxy[0], hxy[1] - bxy[1]) > cfg.max_handler_ft:
                    continue
                last = k
                n += 1
                rows.append(
                    {
                        "game_id": gid,
                        "period": period,
                        "event_type": "dribble",
                        "handler_id": tch["handler_id"],
                        "team_id": tch["handler_team_id"],
                        "t_bounce_ms": int(t[k]),
                        "z_min": float(z[k]),
                        "prominence_ft": float(prom),
                        "x": float(bxy[0]),
                        "y": float(bxy[1]),
                        "touch_uid": tch["event_uid"],
                        "event_uid": f"{gid}:dribble:{int(t[k])}:{tch['handler_id']}",
                    }
                )
            live = (~frozen[i0 : i1 + 1]).sum() / 25.0
            summ.append((tch["event_uid"], n, float(live), gap_flag))
    d = pl.DataFrame(rows, schema=DRIBBLE_SCHEMA)
    s = pl.DataFrame(
        summ,
        schema={
            "event_uid": pl.Utf8,
            "n_dribbles": pl.Int32,
            "live_duration_s": pl.Float32,
            "ball_gap": pl.Boolean,
        },
        orient="row",
    )
    out = touches.join(s, on="event_uid", how="left")
    return d, _ends_with(out, passes_df, shots)


def _ends_with(touches: pl.DataFrame, passes_df, shots) -> pl.DataFrame:
    """What ends a touch: the handler's pass / handoff / shot released within 0.2 s of the end."""
    ends = ["none"] * touches.height
    for i, r in enumerate(touches.iter_rows(named=True)):
        lo, hi = r["t_end_ms"] - 200, r["t_end_ms"] + 200
        if (
            shots is not None
            and shots.filter(
                (pl.col("period") == r["period"])
                & (pl.coalesce("shooter_inferred_id", "shooter_id") == r["handler_id"])
                & pl.col("t_release_ms").is_between(lo, hi)
            ).height
        ):
            ends[i] = "shot"
            continue
        if passes_df is not None:
            p = passes_df.filter(
                (pl.col("period") == r["period"])
                & (pl.col("passer_id") == r["handler_id"])
                & pl.col("t_release_ms").is_between(lo, hi)
            )
            if p.height:
                row = p.row(0, named=True)
                ends[i] = (
                    "turnover"
                    if row["intercepted"]
                    else ("handoff" if row["kind"] == "handoff" else "pass")
                )
    return touches.with_columns(pl.Series("ends_with", ends, dtype=pl.Utf8))
