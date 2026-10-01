"""Shot release, rim contact, rebound (L2).

One row per pbp field-goal attempt (msg 1 / 2). Terminal pbp events are walked in order as in
ghost-defense ``segment_game``: each is placed on the tracking timeline with ``locate_clock``
(game clock only) after the previous terminal; FG attempts are then refined with
``refine_shot_release`` (ball-derived, window −5 s … +2 s, lower bound = previous terminal), so
``t_release_ms`` equals ghost-defense's ``t_terminal`` for shots.

After the release:
* ``t_rim_ms``: first frame within 3 s with the ball in the rim zone of the target hoop;
* ``t_control_ms`` / ``control_id`` / ``control_team_id``: first handler run ≥ ``control_s`` in
  the same continuity segment after the rim contact (or the release): the rebounder of a miss,
  usually the inbounder after a make;
* ``t_landing_ms``, ``x_landing``, ``y_landing``: first frame after the rim contact with the ball
  below ``landing_z_ft`` while falling (rebound landing, also used for box-outs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nbacore.court import HOOP_LEFT, HOOP_RIGHT
from nbacore.io.raw import BALL_ID
from nbacore.possession.segment import SHOT_TYPES, locate_clock, terminal_events
from nbacore.possession.shots import ShotTimeConfig, refine_shot_release

SHOT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "pbp_event_num": pl.Int32,
    "made": pl.Boolean,
    "blocked": pl.Boolean,  # pbp: missed FG with a PLAYER3 (block)
    "shooter_id": pl.Int64,  # pbp PLAYER1
    "shooter_team_id": pl.Int64,
    "shooter_inferred_id": pl.Int64,  # offensive player holding the ball at the release
    "attacks_left": pl.Boolean,
    "t_pbp_ms": pl.Int64,  # pbp clock reading placed on the timeline (null: not located)
    "t_release_ms": pl.Int64,
    "method": pl.Utf8,  # rim | apex | pbp (+ _nohand / _nooff) | unlocated
    "release_minus_pbp_s": pl.Float32,
    "x_release": pl.Float32,  # ball at release
    "y_release": pl.Float32,
    "t_rim_ms": pl.Int64,
    "t_control_ms": pl.Int64,
    "control_id": pl.Int64,
    "control_team_id": pl.Int64,
    "offensive_rebound": pl.Boolean,  # misses only: control by the shooting team
    "t_landing_ms": pl.Int64,
    "x_landing": pl.Float32,
    "y_landing": pl.Float32,
    "event_uid": pl.Utf8,
}


@dataclass
class ShotEventConfig:
    shot: ShotTimeConfig = field(default_factory=ShotTimeConfig)
    clock_bin_s: float = 1.0
    rim_search_s: float = 3.0
    control_s: float = 0.4
    landing_z_ft: float = 9.0
    # nbacore v1.6: after a missed shot, the next attempt (a tip / putback) is searched
    # only after the ball has left the rim zone of that miss. With the previous release as the lower
    # bound, the tip's "first rim contact" was the miss's own and its release landed 0.12–0.2 s
    # after the miss's release.
    after_miss_from_rim: bool = False


def shot_events(
    frames: pl.DataFrame,
    frame_index: pl.DataFrame,
    pbp_game: pl.DataFrame,
    direction: pl.DataFrame,
    handler: pl.DataFrame,
    cfg: ShotEventConfig | None = None,
) -> pl.DataFrame:
    cfg = cfg or ShotEventConfig()
    gid = frame_index["game_id"][0]
    dir_map = {
        (r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)
    }
    ent_by_period = {
        int(p): frames.filter(pl.col("period") == p) for p in frames["period"].unique()
    }
    ball = (
        frames.filter(pl.col("team_id") == BALL_ID)
        .select("period", "unix_ms", "x", "y", "z")
        .sort(["period", "unix_ms"])
    )
    hs = handler.sort(["period", "unix_ms"])
    pbp_full = pbp_game.sort("event_num")
    blocks = set(
        pbp_full.filter((pl.col("msg_type") == 2) & (pl.col("p3_id") > 0))["event_num"].to_list()
    )
    rows = []
    prev_unix: int | None = None
    prev_period: int | None = None
    for row in terminal_events(pbp_game).iter_rows(named=True):
        period = int(row["period"])
        if period != prev_period:
            prev_unix, prev_period = None, period
        msg = int(row["msg_type"])
        idx_p = frame_index.filter(pl.col("period") == period)
        t_term = (
            locate_clock(idx_p, float(row["pctime_sec"]), prev_unix, cfg.clock_bin_s)
            if idx_p.height
            else None
        )
        if msg not in SHOT_TYPES:
            if t_term is not None:
                prev_unix = t_term
            continue
        shooter = row["p1_id"]
        team = row["p1_team_id"]
        base = {
            "game_id": gid,
            "period": period,
            "pbp_event_num": int(row["event_num"]),
            "made": msg == 1,
            "blocked": int(row["event_num"]) in blocks,
            "shooter_id": int(shooter) if shooter else None,
            "shooter_team_id": int(team) if team is not None else None,
        }
        if t_term is None or team is None or (int(team), period) not in dir_map:
            rows.append({**base, "method": "unlocated"})
            continue
        attacks_left = bool(dir_map[(int(team), period)])
        hoop = HOOP_LEFT if attacks_left else HOOP_RIGHT
        t_rel, shooter_inf, method = refine_shot_release(
            ent_by_period[period],
            t_term,
            int(team),
            hoop,
            cfg.shot,
            t_min=prev_unix,
            shooter_id=int(shooter) if shooter else None,
        )
        after = _after_release(ball, hs, period, t_rel, hoop, int(team), msg == 1, cfg)
        rim_exit = after.pop("_t_rim_exit_ms", None)
        prev_unix = t_rel
        if cfg.after_miss_from_rim and msg == 2 and rim_exit is not None:
            prev_unix = rim_exit
        rows.append(
            {
                **base,
                "shooter_inferred_id": shooter_inf,
                "attacks_left": attacks_left,
                "t_pbp_ms": t_term,
                "t_release_ms": t_rel,
                "method": method,
                "release_minus_pbp_s": (t_rel - t_term) / 1000,
                **after,
            }
        )
    out = pl.DataFrame(rows, schema={k: v for k, v in SHOT_SCHEMA.items() if k != "event_uid"})
    # pbp rows inserted late (event numbers out of chronological order, e.g. Q4 corrections numbered
    # among OT events) can be refined onto the same release: keep the row whose pbp time is
    # closest to it, the others become method "duplicate_release" without a release
    dist = (pl.col("t_pbp_ms") - pl.col("t_release_ms")).abs()
    first = pl.col("pbp_event_num") == pl.col("pbp_event_num").sort_by(dist).first().over(
        "period", "t_release_ms"
    )
    dup = pl.col("t_release_ms").is_not_null() & ~first
    rel_cols = ["t_release_ms", "release_minus_pbp_s", "shooter_inferred_id", "x_release", "y_release",
                "t_rim_ms", "t_control_ms", "control_id", "control_team_id", "offensive_rebound",
                "t_landing_ms", "x_landing", "y_landing"]  # fmt: skip
    out = out.with_columns(
        [pl.when(dup).then(None).otherwise(pl.col(c)).alias(c) for c in rel_cols]
        + [
            pl.when(dup)
            .then(pl.lit("duplicate_release"))
            .otherwise(pl.col("method"))
            .alias("method")
        ]
    )
    # anything observed at or after the next attempt's release belongs to that attempt (an air
    # ball / block / rim bounce followed by a putback): clear it on this one, group by group
    nxt = pl.col("t_release_ms").shift(-1).over("period", order_by="t_release_ms")
    groups = {
        "t_rim_ms": ["t_rim_ms"],
        "t_landing_ms": ["t_landing_ms", "x_landing", "y_landing"],
        "t_control_ms": ["t_control_ms", "control_id", "control_team_id", "offensive_rebound"],
    }
    late = {k: (pl.col(k) >= nxt).fill_null(False) for k in groups}
    out = out.with_columns(
        [
            pl.when(late[k]).then(None).otherwise(pl.col(c)).alias(c)
            for k, cols in groups.items()
            for c in cols
        ]
    )
    return out.with_columns(
        pl.when(pl.col("t_release_ms").is_not_null())
        .then(
            pl.format(
                "{}:shot_release:{}:{}",
                "game_id",
                "t_release_ms",
                pl.coalesce("shooter_id", "shooter_inferred_id", pl.lit(-1)),
            )
        )
        .alias("event_uid")
    )


def _after_release(ball, hs, period, t_rel, hoop, team, made, cfg) -> dict:
    b = ball.filter(
        (pl.col("period") == period)
        & (pl.col("unix_ms") >= t_rel)
        & (pl.col("unix_ms") <= t_rel + int(cfg.rim_search_s * 1000))
    )
    out: dict = {}
    if b.height:
        rel = b.row(0, named=True)
        out.update(x_release=rel["x"], y_release=rel["y"])
    t, x, y, z = (b[c].to_numpy() for c in ("unix_ms", "x", "y", "z"))
    rim = (
        (np.hypot(x - hoop[0], y - hoop[1]) < cfg.shot.rim_xy_ft)
        & (z > cfg.shot.rim_z_lo)
        & (z < cfg.shot.rim_z_hi)
    )
    t_anchor = t_rel
    if rim.any():
        k = int(np.flatnonzero(rim)[0])
        out["t_rim_ms"] = int(t[k])
        t_anchor = int(t[k])
        # last frame of the first stay in the rim zone (internal: lower bound of the next attempt)
        e = k
        while e + 1 < rim.size and rim[e + 1]:
            e += 1
        out["_t_rim_exit_ms"] = int(t[e])
        # landing: first frame after the rim contact with the ball falling below landing_z
        dz = np.diff(z[k:], prepend=z[k])
        land = np.flatnonzero((z[k:] < cfg.landing_z_ft) & (dz < 0))
        if land.size:
            j = k + int(land[0])
            out.update(t_landing_ms=int(t[j]), x_landing=float(x[j]), y_landing=float(y[j]))
    # first real control after the anchor, in the same continuity segment
    after = hs.filter((pl.col("period") == period) & (pl.col("unix_ms") >= t_anchor))
    if after.height:
        seg0 = after["segment_id"][0]
        after = after.filter(pl.col("segment_id") == seg0)
        hid = after["handler_id"].to_numpy()
        min_len = max(1, int(round(cfg.control_s * 25)))
        i = 0
        while i < hid.size:
            if hid[i] < 0:
                i += 1
                continue
            j = i
            while j + 1 < hid.size and hid[j + 1] == hid[i]:
                j += 1
            if j - i + 1 >= min_len:
                ctrl_team = int(after["handler_team_id"][i])
                out.update(
                    t_control_ms=int(after["unix_ms"][i]),
                    control_id=int(hid[i]),
                    control_team_id=ctrl_team,
                    offensive_rebound=None if made else ctrl_team == team,
                )
                break
            i = j + 1
    return out
