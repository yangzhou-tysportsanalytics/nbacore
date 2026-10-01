"""Drive and cut candidates (L2). Loose.

Radial speed = rate at which a player's (smoothed) distance to the basket his team attacks
decreases (ft/s, > 0 toward the basket). With the controlling team T of the L2 handler table:

* **drive**: T's handler with radial speed > ``min_radial_fts`` for ≥ ``min_s``, starting ≥
  ``drive_min_start_ft`` from the basket **or** outside the paint (union of the conditions
  used for action segmentation and for foul-call analysis);
* **cut**: a T player who is not the handler, same speed rule, no start distance.

A candidate is a run of such frames inside one continuity segment; ``t_peak_ms`` = max radial
speed. Defender geometry uses the nearest opponent at the start (``def_id``); ``def_offset_*``
= driver's distance to the basket minus the defender's (> 0: defender between him and the
basket). Links to other events (optional inputs): drive ``end_reason`` (shot / pass / turnover /
foul / none, from shot releases, passes and clock stops by or around the driver up to 0.5 s
after the end); cut ``caught`` / ``t_catch_ms`` (a completed pass to the cutter caught in the cut
or ≤ 0.5 s after it) and ``screen_uid`` (a screen candidate whose screener came within 8 ft of
the cutter in the 1.5 s before the start).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.court import (
    HOOP_LEFT,
    HOOP_RIGHT,
    LANE_LENGTH,
    LANE_WIDTH,
    LENGTH,
    RESTRICTED_RADIUS,
    WIDTH,
)
from nbacore.events.screens import FPS, ScreenConfig, _Dense, _runs


@dataclass
class MotionConfig:
    min_radial_fts: float = 5.0
    min_s: float = 0.3
    drive_min_start_ft: float = 10.0
    link_after_s: float = 0.5
    screen_link_ft: float = 8.0
    screen_link_s: float = 1.5
    smooth_frames: int = 5


def _in_paint(xy: np.ndarray, left: bool) -> np.ndarray:
    x = xy[..., 0] if left else LENGTH - xy[..., 0]
    return (x >= 0) & (x <= LANE_LENGTH) & (np.abs(xy[..., 1] - WIDTH / 2) <= LANE_WIDTH / 2)


def motion_candidates(
    frames: pl.DataFrame,
    handler: pl.DataFrame,
    direction: pl.DataFrame,
    shots: pl.DataFrame | None = None,
    passes_df: pl.DataFrame | None = None,
    stops: pl.DataFrame | None = None,
    screens: pl.DataFrame | None = None,
    cfg: MotionConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(drives, cuts) of one game."""
    cfg = cfg or MotionConfig()
    gid = handler["game_id"][0]
    dir_map = {
        (r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)
    }
    drives, cuts = [], []
    for (period,), h in handler.sort(["period", "unix_ms"]).group_by(
        ["period"], maintain_order=True
    ):
        d, c = _period(gid, int(period), frames.filter(pl.col("period") == period), h, dir_map, cfg)
        drives += d
        cuts += c
    dr = pl.DataFrame(drives, schema=DRIVE_SCHEMA)
    ct = pl.DataFrame(cuts, schema=CUT_SCHEMA)
    return _link_drives(dr, shots, passes_df, stops, cfg), _link_cuts(ct, passes_df, screens, cfg)


_BASE = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,
    "player_id": pl.Int64,
    "team_id": pl.Int64,
    "def_id": pl.Int64,
    "t_start_ms": pl.Int64,
    "t_peak_ms": pl.Int64,
    "t_end_ms": pl.Int64,
    "duration_s": pl.Float32,
    "x": pl.Float32,  # player at start (raw court coordinates)
    "y": pl.Float32,
    "start_frontcourt": pl.Boolean,  # start in the half the team attacks
    "dist_basket_start_ft": pl.Float32,
    "dist_basket_end_ft": pl.Float32,
    "radial_peak_fts": pl.Float32,
    "radial_mean_fts": pl.Float32,
    "def_offset_start_ft": pl.Float32,
    "def_offset_end_ft": pl.Float32,
    "def_dist_start_ft": pl.Float32,
    "def_dist_end_ft": pl.Float32,
    "event_uid": pl.Utf8,
}
DRIVE_SCHEMA = {
    **_BASE,
    "start_in_paint": pl.Boolean,
    "speed_max_fts": pl.Float32,
    "second_def_dist_end_ft": pl.Float32,
    "t_paint_ms": pl.Int64,
    "t_restricted_ms": pl.Int64,
    "end_reason": pl.Utf8,
}
CUT_SCHEMA = {
    **_BASE,
    "heading_change_deg": pl.Float32,
    "ball_dist_start_ft": pl.Float32,
    "ball_minus_cutter_to_basket_ft": pl.Float32,  # > 0: ball farther from the basket
    "caught": pl.Boolean,
    "t_catch_ms": pl.Int64,
    "screen_uid": pl.Utf8,
}


def _period(gid, period, frames, h, dir_map, cfg):
    t = h["unix_ms"].to_numpy()
    seg = h["segment_id"].to_numpy()
    ctrl = h["control_team_id"].to_numpy()
    hid = h["handler_id"].to_numpy()
    D = _Dense(frames, t, ScreenConfig(smooth_frames=cfg.smooth_frames))
    ball = np.full((t.size, 2), np.nan)
    pos = {int(u): i for i, u in enumerate(t.tolist())}
    for u, x, y in frames.filter(pl.col("team_id") == -1).select("unix_ms", "x", "y").iter_rows():
        i = pos.get(int(u))
        if i is not None:
            ball[i] = (x, y)
    ids = np.array(D.ids)
    team = np.array([D.team[i] for i in D.ids])
    min_len = max(2, int(round(cfg.min_s * FPS)))
    dt = np.gradient(t.astype(float)) / 1000.0
    drives, cuts = [], []
    for T in np.unique(team):
        left = bool(dir_map.get((int(T), period), True))
        hoop = HOOP_LEFT if left else HOOP_RIGHT
        off = np.flatnonzero(team == T)
        opp = np.flatnonzero(team != T)
        for p in off:
            pid = int(ids[p])
            dist = np.linalg.norm(D.xy[:, p] - hoop, axis=-1)
            radial = -np.gradient(dist) / dt
            fast = (radial > cfg.min_radial_fts) & (ctrl == T) & ~np.isnan(radial)
            for kind, mask in (("drive", fast & (hid == pid)), ("cut", fast & (hid != pid))):
                for a, b in _runs(mask):
                    cuts_at = np.flatnonzero(np.diff(seg[a : b + 1]) != 0) + 1
                    for lo, hi in zip(np.r_[0, cuts_at], np.r_[cuts_at, b - a + 1], strict=True):
                        i0, i1 = a + lo, a + hi - 1
                        if i1 - i0 + 1 < min_len:
                            continue
                        xy0 = D.raw_xy[i0, p]
                        paint0 = bool(_in_paint(xy0, left))
                        if kind == "drive" and not (
                            dist[i0] >= cfg.drive_min_start_ft or not paint0
                        ):
                            continue
                        row = _row(
                            gid,
                            period,
                            kind,
                            D,
                            ids,
                            opp,
                            p,
                            pid,
                            int(T),
                            t,
                            i0,
                            i1,
                            dist,
                            radial,
                            hoop,
                            left,
                            ball,
                        )
                        if kind == "drive":
                            row["start_in_paint"] = paint0
                            drives.append(row)
                        else:
                            cuts.append(row)
    return drives, cuts


def _nearest(D, opp, k, p):
    d = np.linalg.norm(D.raw_xy[k, opp] - D.raw_xy[k, p], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(d)
    return order, d


def _row(gid, period, kind, D, ids, opp, p, pid, T, t, i0, i1, dist, radial, hoop, left, ball):
    order, d0 = _nearest(D, opp, i0, p)
    dj = opp[order[0]] if np.isfinite(d0[order[0]]) else None
    k_pk = i0 + int(np.nanargmax(radial[i0 : i1 + 1]))

    def f(v):
        return None if v is None or not np.isfinite(v) else float(v)

    def offset(i):
        if dj is None:
            return None
        return f(dist[i] - np.linalg.norm(D.xy[i, dj] - hoop))

    def ddist(i):
        return None if dj is None else f(np.linalg.norm(D.raw_xy[i, dj] - D.raw_xy[i, p]))

    row = {
        "game_id": gid,
        "period": period,
        "event_type": f"{kind}_candidate",
        "player_id": pid,
        "team_id": T,
        "def_id": int(ids[dj]) if dj is not None else None,
        "t_start_ms": int(t[i0]),
        "t_peak_ms": int(t[k_pk]),
        "t_end_ms": int(t[i1]),
        "duration_s": (t[i1] - t[i0]) / 1000,
        "x": f(D.raw_xy[i0, p, 0]),
        "y": f(D.raw_xy[i0, p, 1]),
        "start_frontcourt": bool(
            (D.raw_xy[i0, p, 0] < LENGTH / 2) if left else (D.raw_xy[i0, p, 0] > LENGTH / 2)
        ),
        "dist_basket_start_ft": f(dist[i0]),
        "dist_basket_end_ft": f(dist[i1]),
        "radial_peak_fts": f(radial[k_pk]),
        "radial_mean_fts": f(np.nanmean(radial[i0 : i1 + 1])),
        "def_offset_start_ft": offset(i0),
        "def_offset_end_ft": offset(i1),
        "def_dist_start_ft": ddist(i0),
        "def_dist_end_ft": ddist(i1),
        "event_uid": f"{gid}:{kind}_candidate:{int(t[i0])}:{pid}",
    }
    if kind == "drive":
        order1, d1 = _nearest(D, opp, i1, p)
        row["second_def_dist_end_ft"] = f(d1[order1[1]]) if len(order1) > 1 else None
        row["speed_max_fts"] = f(np.nanmax(D.speed[i0 : i1 + 1, p]))
        ext = min(t.size - 1, i1 + int(0.5 * FPS))
        pts = D.raw_xy[i0 : ext + 1, p]
        paint = np.flatnonzero(_in_paint(pts, left))
        ra = np.flatnonzero(np.linalg.norm(pts - hoop, axis=-1) <= RESTRICTED_RADIUS)
        row["t_paint_ms"] = int(t[i0 + paint[0]]) if paint.size else None
        row["t_restricted_ms"] = int(t[i0 + ra[0]]) if ra.size else None
    else:
        v0 = D.v[i0, p]
        v1 = D.v[i1, p]
        n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
        ok = n0 > 1e-6 and n1 > 1e-6 and np.isfinite(n0) and np.isfinite(n1)
        row["heading_change_deg"] = (
            float(np.degrees(np.arccos(np.clip(v0 @ v1 / (n0 * n1), -1, 1)))) if ok else None
        )
        b = ball[i0]
        row["ball_dist_start_ft"] = f(np.linalg.norm(b - D.raw_xy[i0, p]))
        row["ball_minus_cutter_to_basket_ft"] = f(np.linalg.norm(b - hoop) - dist[i0])
    return row


def _link_drives(dr, shots, passes_df, stops, cfg):
    reasons = []
    after = int(cfg.link_after_s * 1000)
    for r in dr.iter_rows(named=True):
        lo, hi = r["t_start_ms"], r["t_end_ms"] + after
        reason = "none"
        if (
            shots is not None
            and shots.filter(
                (pl.col("period") == r["period"])
                & (pl.coalesce("shooter_id", "shooter_inferred_id") == r["player_id"])
                & pl.col("t_release_ms").is_between(lo, hi)
            ).height
        ):
            reason = "shot"
        elif (
            passes_df is not None
            and (
                p := passes_df.filter(
                    (pl.col("period") == r["period"])
                    & (pl.col("passer_id") == r["player_id"])
                    & pl.col("t_release_ms").is_between(lo, hi)
                )
            ).height
        ):
            reason = "turnover" if p["intercepted"].any() else "pass"
        elif (
            stops is not None
            and stops.filter(
                (pl.col("period") == r["period"])
                & pl.col("t_onset_est_ms").is_between(lo, hi)
                & (pl.col("stop_cause_hint") == "foul")
            ).height
        ):
            reason = "foul"
        reasons.append(reason)
    return dr.with_columns(pl.Series("end_reason", reasons, dtype=pl.Utf8))


def _link_cuts(ct, passes_df, screens, cfg):
    if ct.height == 0:
        return ct
    after = int(cfg.link_after_s * 1000)
    caught, t_catch, s_uid = [], [], []
    for r in ct.iter_rows(named=True):
        lo, hi = r["t_start_ms"], r["t_end_ms"] + after
        c = None
        if passes_df is not None:
            c = passes_df.filter(
                (pl.col("period") == r["period"])
                & (pl.col("receiver_id") == r["player_id"])
                & pl.col("completed")
                & pl.col("t_catch_ms").is_between(lo, hi)
            )
        caught.append(bool(c is not None and c.height))
        t_catch.append(int(c["t_catch_ms"].min()) if c is not None and c.height else None)
        u = None
        if screens is not None:
            sc = screens.filter(
                (pl.col("period") == r["period"])
                & (pl.col("user_id") == r["player_id"])
                & pl.col("t_contact_ms").is_between(
                    r["t_start_ms"] - int(cfg.screen_link_s * 1000), r["t_start_ms"]
                )
                & (pl.col("min_dist_screener_user_ft") <= cfg.screen_link_ft)
            ).sort("t_contact_ms")
            u = sc["event_uid"][-1] if sc.height else None
        s_uid.append(u)
    out = ct.with_columns(
        pl.Series("caught", caught, dtype=pl.Boolean),
        pl.Series("t_catch_ms", t_catch, dtype=pl.Int64),
        pl.Series("screen_uid", s_uid, dtype=pl.Utf8),
    )
    return out


@dataclass
class PostUpConfig:
    max_basket_ft: float = 18.0
    max_speed_fts: float = 4.0
    max_def_ft: float = 4.0
    min_s: float = 1.0
    smooth_frames: int = 5


POSTUP_SCHEMA = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,
    "player_id": pl.Int64,
    "team_id": pl.Int64,
    "def_id": pl.Int64,  # opponent nearest over most of the post-up
    "t_start_ms": pl.Int64,
    "t_end_ms": pl.Int64,
    "duration_s": pl.Float32,
    "x": pl.Float32,
    "y": pl.Float32,
    "mean_dist_basket_ft": pl.Float32,
    "mean_speed_fts": pl.Float32,
    "mean_def_dist_ft": pl.Float32,
    "def_between_frac": pl.Float32,  # share of frames with the defender closer to the basket
    "n_dribbles": pl.Int32,
    "end_reason": pl.Utf8,
    "event_uid": pl.Utf8,
}


def postups(
    frames: pl.DataFrame,
    handler: pl.DataFrame,
    direction: pl.DataFrame,
    dribbles_df: pl.DataFrame | None = None,
    shots: pl.DataFrame | None = None,
    passes_df: pl.DataFrame | None = None,
    cfg: PostUpConfig | None = None,
) -> pl.DataFrame:
    """Post-up candidates: the handler within ``max_basket_ft`` of the basket,
    slower than ``max_speed_fts``, nearest opponent within ``max_def_ft``, for at least ``min_s``
    (no upper limit), inside one segment."""
    cfg = cfg or PostUpConfig()
    gid = handler["game_id"][0]
    dir_map = {
        (r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)
    }
    rows = []
    min_len = int(round(cfg.min_s * FPS))
    for (period,), h in handler.sort(["period", "unix_ms"]).group_by(
        ["period"], maintain_order=True
    ):
        period = int(period)
        t = h["unix_ms"].to_numpy()
        seg = h["segment_id"].to_numpy()
        hid = h["handler_id"].to_numpy()
        ctrl = h["control_team_id"].to_numpy()
        fp = frames.filter(pl.col("period") == period)
        D = _Dense(fp, t, ScreenConfig(smooth_frames=cfg.smooth_frames))
        ids = np.array(D.ids)
        team = np.array([D.team[i] for i in D.ids])
        for p, pid in enumerate(ids):
            T = team[p]
            left = bool(dir_map.get((int(T), period), True))
            hoop = HOOP_LEFT if left else HOOP_RIGHT
            opp = np.flatnonzero(team != T)
            dist = np.linalg.norm(D.xy[:, p] - hoop, axis=-1)
            dd = np.linalg.norm(D.raw_xy[:, opp] - D.raw_xy[:, [p]], axis=-1)
            dd = np.where(np.isnan(dd), np.inf, dd)
            near = np.argmin(dd, axis=1)
            dmin = dd[np.arange(t.size), near]
            m = (hid == pid) & (ctrl == T) & (dist <= cfg.max_basket_ft)
            m &= (D.speed[:, p] < cfg.max_speed_fts) & (dmin <= cfg.max_def_ft)
            for a, b in _runs(m):
                cuts_at = np.flatnonzero(np.diff(seg[a : b + 1]) != 0) + 1
                for lo, hi in zip(np.r_[0, cuts_at], np.r_[cuts_at, b - a + 1], strict=True):
                    i0, i1 = a + lo, a + hi - 1
                    if i1 - i0 + 1 < min_len:
                        continue
                    dj = opp[np.bincount(near[i0 : i1 + 1]).argmax()]
                    ddist = np.linalg.norm(D.xy[i0 : i1 + 1, dj] - hoop, axis=-1)
                    rows.append(
                        {
                            "game_id": gid,
                            "period": period,
                            "event_type": "postup_candidate",
                            "player_id": int(pid),
                            "team_id": int(T),
                            "def_id": int(ids[dj]),
                            "t_start_ms": int(t[i0]),
                            "t_end_ms": int(t[i1]),
                            "duration_s": (t[i1] - t[i0]) / 1000,
                            "x": float(D.raw_xy[i0, p, 0]),
                            "y": float(D.raw_xy[i0, p, 1]),
                            "mean_dist_basket_ft": float(np.nanmean(dist[i0 : i1 + 1])),
                            "mean_speed_fts": float(np.nanmean(D.speed[i0 : i1 + 1, p])),
                            "mean_def_dist_ft": float(np.mean(dmin[i0 : i1 + 1])),
                            "def_between_frac": float(np.nanmean(ddist < dist[i0 : i1 + 1])),
                            "event_uid": f"{gid}:postup_candidate:{int(t[i0])}:{int(pid)}",
                        }
                    )
    base = {k: v for k, v in POSTUP_SCHEMA.items() if k not in ("n_dribbles", "end_reason")}
    out = pl.DataFrame(rows, schema=base)
    nd, ends = [], []
    for r in out.iter_rows(named=True):
        lo, hi = r["t_start_ms"], r["t_end_ms"]
        if dribbles_df is not None:
            nd.append(
                dribbles_df.filter(
                    (pl.col("period") == r["period"])
                    & (pl.col("handler_id") == r["player_id"])
                    & pl.col("t_bounce_ms").is_between(lo, hi)
                ).height
            )
        else:
            nd.append(None)
        reason = "exit"
        shot_hit = shots is not None and (
            shots.filter(
                (pl.col("period") == r["period"])
                & (pl.coalesce("shooter_inferred_id", "shooter_id") == r["player_id"])
                & pl.col("t_release_ms").is_between(lo, hi + 1000)
            ).height
            > 0
        )
        if shot_hit:
            reason = "shot"
        elif passes_df is not None:
            p = passes_df.filter(
                (pl.col("period") == r["period"])
                & (pl.col("passer_id") == r["player_id"])
                & pl.col("t_release_ms").is_between(lo, hi + 1000)
            )
            if p.height:
                reason = "turnover" if p["intercepted"].any() else "pass"
        ends.append(reason)
    return out.with_columns(
        pl.Series("n_dribbles", nd, dtype=pl.Int32), pl.Series("end_reason", ends, dtype=pl.Utf8)
    ).select(list(POSTUP_SCHEMA))
