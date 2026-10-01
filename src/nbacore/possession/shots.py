"""Refine the time of a field-goal attempt from the ball trajectory.

Play-by-play clock readings for shots are late by a median 0.6 s (up to several seconds) and
occasionally early, so a window ending at the pbp time either includes the rebound scramble or
cuts the shot off. We therefore locate the shot from the ball:

1. *Rim moment*: the **first** frame after the previous terminal event (``t_min``) and within
   the search window around the pbp time where the ball is within ``rim_xy_ft`` of the hoop
   centre at rim height (``rim_z_lo`` .. ``rim_z_hi``). Taking the first contact (not the one
   closest to the pbp time) avoids latching onto tip-ins and rim bounces. If none (blocked
   shot, air ball), fall back to the ball's highest point above ``apex_z_ft`` closest to the
   pbp time.
2. *Release*: the last frame at or before the rim/apex moment where the ball is *in hand*:
   within ``hand_ft`` (xy) of some offensive player and not sitting at the rim (a ball at rim
   height inside ``rim_zone_ft`` of the hoop is never "in hand": it is above the player standing
   under the basket). That player is the inferred shooter. With ``hand_z_ft`` set, a ball higher
   than that is never in hand either. With ``shooter_only`` (nbacore v1.3) and a known pbp
   shooter, only that player's hands count; if the shooter never has the ball in hand in the
   window, any offensive player counts again and the method gets the suffix ``_anyhand``.

If neither anchor exists the pbp time is kept (``method == "pbp"``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from nbacore.io.raw import BALL_ID


@dataclass
class ShotTimeConfig:
    before_s: float = 5.0  # search window before pbp time
    after_s: float = 2.0  # ... and after
    rim_xy_ft: float = 2.0
    rim_z_lo: float = 8.0
    rim_z_hi: float = 11.5
    apex_z_ft: float = 9.0
    hand_ft: float = 3.0
    rim_zone_ft: float = 4.0  # ball within this of the hoop and above hand_z_max is not in hand
    hand_z_max: float = 8.5
    # the ball is never "in hand" above this height (None = the original ghost rule, xy distance
    # only). Tried for v1.3 and rejected: jump shots are released at 10-12 ft, so any cap that
    # removes the mid-air releases also moves most correct releases 80-120 ms earlier.
    hand_z_ft: float | None = None
    # nbacore v1.3: only the pbp shooter's hands count (when known). Under the original rule a
    # ball in flight passing over a teammate counted as in hand and the release was placed
    # mid-air (14.8 % of shots on v1.2; in 71 % of those the "holder" was not the pbp shooter).
    shooter_only: bool = False
    # nbacore v1.6: fallbacks when neither a rim contact nor an apex is found (method "pbp").
    # ``wide_before_s``: search again with this window before the pbp time (the pbp clock lags
    # the shot by > 5 s for some attempts); the method gets the suffix "_wide". ``hand_fallback``:
    # then take the last frame with the ball within ``hand_ft`` of the pbp shooter before the pbp
    # time (low blocked shots never reach the rim or 9 ft); method "hand".
    wide_before_s: float | None = None
    hand_fallback: bool = False


def refine_shot_release(
    entities_period: pl.DataFrame,
    t_pbp: int,
    offense_team_id: int,
    hoop: np.ndarray,
    cfg: ShotTimeConfig | None = None,
    t_min: int | None = None,
    shooter_id: int | None = None,
) -> tuple[int, int | None, str]:
    """Return (t_release_unix_ms, shooter_player_id or None, method).

    ``entities_period``: deduped entity rows of the period. ``hoop``: (2,) hoop centre in the
    raw (unflipped) frame. ``t_min``: unix_ms of the previous terminal event (exclusive).
    ``shooter_id``: the pbp shooter, used with ``cfg.shooter_only`` and ``cfg.hand_fallback``.
    """
    cfg = cfg or ShotTimeConfig()
    if cfg.wide_before_s is None and not cfg.hand_fallback:
        return _refine(entities_period, t_pbp, offense_team_id, hoop, cfg, t_min, shooter_id)
    base = replace(cfg, wide_before_s=None, hand_fallback=False)
    res = _refine(entities_period, t_pbp, offense_team_id, hoop, base, t_min, shooter_id)
    if res[2] != "pbp":
        return res
    if cfg.wide_before_s is not None:
        wide = replace(base, before_s=cfg.wide_before_s)
        r2 = _refine(entities_period, t_pbp, offense_team_id, hoop, wide, t_min, shooter_id)
        if r2[2] != "pbp":
            return r2[0], r2[1], r2[2] + "_wide"
    if cfg.hand_fallback and shooter_id is not None:
        before = cfg.wide_before_s if cfg.wide_before_s is not None else cfg.before_s
        lo = t_pbp - int(before * 1000)
        if t_min is not None:
            lo = max(lo, t_min + 1)
        w = entities_period.filter(pl.col("unix_ms").is_between(lo, t_pbp))
        b = w.filter(pl.col("team_id") == BALL_ID).select("unix_ms", "x", "y")
        s = w.filter(pl.col("player_id") == shooter_id).select(
            "unix_ms", pl.col("x").alias("_sx"), pl.col("y").alias("_sy")
        )
        j = b.join(s, on="unix_ms").filter(
            ((pl.col("x") - pl.col("_sx")) ** 2 + (pl.col("y") - pl.col("_sy")) ** 2).sqrt()
            <= cfg.hand_ft
        )
        if j.height:
            return int(j["unix_ms"].max()), shooter_id, "hand"
    return res


def _refine(
    entities_period: pl.DataFrame,
    t_pbp: int,
    offense_team_id: int,
    hoop: np.ndarray,
    cfg: ShotTimeConfig | None = None,
    t_min: int | None = None,
    shooter_id: int | None = None,
) -> tuple[int, int | None, str]:
    """Return (t_release_unix_ms, shooter_player_id or None, method).

    ``entities_period``: deduped entity rows of the period. ``hoop``: (2,) hoop centre in the
    raw (unflipped) frame. ``t_min``: unix_ms of the previous terminal event (exclusive).
    ``shooter_id``: the pbp shooter, used with ``cfg.shooter_only``.
    """
    cfg = cfg or ShotTimeConfig()
    lo, hi = t_pbp - int(cfg.before_s * 1000), t_pbp + int(cfg.after_s * 1000)
    if t_min is not None:
        lo = max(lo, t_min + 1)
    sub = entities_period.filter((pl.col("unix_ms") >= lo) & (pl.col("unix_ms") <= hi))
    ball = sub.filter(pl.col("team_id") == BALL_ID).sort("unix_ms")
    if ball.height < 5:
        return t_pbp, None, "pbp"
    t = ball["unix_ms"].to_numpy()
    bx = ball["x"].to_numpy().astype(np.float64)
    by = ball["y"].to_numpy().astype(np.float64)
    bz = ball["z"].to_numpy().astype(np.float64)
    dh = np.hypot(bx - hoop[0], by - hoop[1])
    rim = np.flatnonzero((dh < cfg.rim_xy_ft) & (bz > cfg.rim_z_lo) & (bz < cfg.rim_z_hi))
    method = "rim"
    if rim.size:
        k = int(rim[0])  # first rim contact after the previous terminal event
    else:
        high = np.flatnonzero(bz > cfg.apex_z_ft)
        if high.size == 0:
            return t_pbp, None, "pbp"
        # local maxima of z among high frames, closest to pbp time
        method = "apex"
        k = int(high[np.argmin(np.abs(t[high] - t_pbp))])
        # walk to the local maximum around k
        while k + 1 < bz.size and bz[k + 1] > bz[k]:
            k += 1
        while k - 1 >= 0 and bz[k - 1] > bz[k]:
            k -= 1

    off = sub.filter(pl.col("team_id") == offense_team_id).select(
        ["unix_ms", "player_id", "x", "y"]
    )
    if off.height == 0:
        return int(t[k]), None, method + "_nooff"
    # distance of every offensive row to the ball at the same timestamp
    bt = pl.DataFrame({"unix_ms": t, "_bx": bx, "_by": by})
    j = off.join(bt, on="unix_ms", how="inner").with_columns(
        ((pl.col("x") - pl.col("_bx")) ** 2 + (pl.col("y") - pl.col("_by")) ** 2).sqrt().alias("_d")
    )
    at_rim = pl.DataFrame({"unix_ms": t, "_dh": dh, "_bz": bz})
    j = j.join(at_rim, on="unix_ms", how="inner")
    in_hand = (pl.col("_d") <= cfg.hand_ft) & ~(
        (pl.col("_dh") <= cfg.rim_zone_ft) & (pl.col("_bz") > cfg.hand_z_max)
    )
    if cfg.hand_z_ft is not None:
        in_hand = in_hand & (pl.col("_bz") <= cfg.hand_z_ft)
    near = j.filter(in_hand & (pl.col("unix_ms") <= int(t[k])))
    if near.height == 0:
        return int(t[k]), None, method + "_nohand"
    if cfg.shooter_only and shooter_id is not None:
        own = near.filter(pl.col("player_id") == shooter_id)
        if own.height:
            near = own
        else:
            method += "_anyhand"
    t_rel = int(near["unix_ms"].max())
    last = near.filter(pl.col("unix_ms") == t_rel).sort("_d")
    return t_rel, int(last["player_id"][0]), method
