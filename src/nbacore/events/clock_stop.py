"""Clock stops (L2).

In the frame index, ``clock_frozen`` is True on a frame whose game clock equals the previous
frame's (|Δ| < 0.005 s, within the period, also across a tracking hole). For a run of frozen
frames a..b the frame a−1 already shows the stopped value: it is the **first stopped frame**
(``t_start_ms``). Frame a−2 is the **last running frame** (``t_last_running_ms``, clock
``clock_left_s`` above the stop value).

* ``onset_exact``: frames a−2, a−1 are contiguous (no hole) and the clock was running at a−2 (its
  value differs from a−3's). Then the whistle lies between the two frames.
* Otherwise the stop happened in a tracking hole (``gap_before_ms``). The public data keep only
  windows around pbp events, so the last running frame is often a fraction of a second of game
  clock before the stop, then tracking resumes with the clock already stopped. Because a running
  game clock advances in real time, the onset is estimated as
  ``t_onset_est_ms = t_last_running_ms + clock_left_s · 1000`` (capped at ``t_start_ms``).
* ``t_end_ms``: first frame with a running clock again (restart; null if the period ends first);
  ``ends_at_gap``: the frozen run ends at a hole, so the restart time is only bounded.
* ``duration_s`` = last stopped frame − ``t_onset_est_ms``; kept if ≥ ``min_stop_s``;
  ``micro_freeze`` = ``duration_s`` < ``micro_s`` (possession-value models count stoppages ≥ 1 s).
* ``clock_correction``: an official clock correction (``clock_jump``) within ``correction_s``.
* Window columns for foul-call analysis over [onset − 3 s, onset) and [onset, onset + 1 s) with
  onset = ``t_onset_est_ms``: tracked share of the 25 Hz slots with 10 players and the ball, largest hole,
  ball-missing share, players absent for more than 0.2 s of the pre window.

``stop_cause_hint`` is filled by ``nbacore.events.pbp_align`` from the pbp event anchored on the
stop. Shorter freezes are returned as counts by number of frames.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.clean.dedup import FRAME_GAP_MS
from nbacore.io.raw import BALL_ID

STOP_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "t_last_running_ms": pl.Int64,
    "clock_left_s": pl.Float32,  # game clock at the last running frame minus the stop value
    "t_onset_est_ms": pl.Int64,
    "onset_exact": pl.Boolean,
    "t_start_ms": pl.Int64,  # first frame showing the stopped clock
    "t_end_ms": pl.Int64,
    "game_clock_at_stop": pl.Float32,
    "duration_s": pl.Float32,
    "n_frozen_frames": pl.Int32,
    "micro_freeze": pl.Boolean,
    "gap_before_ms": pl.Int64,
    "ends_at_gap": pl.Boolean,
    "clock_correction": pl.Boolean,
    "pre3s_tracked_frac": pl.Float32,
    "post1s_tracked_frac": pl.Float32,
    "pre3s_max_hole_ms": pl.Int64,
    "pre3s_ball_missing_frac": pl.Float32,
    "pre3s_missing_player_ids": pl.List(pl.Int64),
    "event_uid": pl.Utf8,
}


@dataclass
class StopConfig:
    min_stop_s: float = 0.2
    micro_s: float = 1.0
    correction_s: float = 1.0
    pre_s: float = 3.0
    post_s: float = 1.0
    missing_player_s: float = 0.2
    hz: float = 25.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(int(a[0]), int(a[-1])) for a in np.split(idx, brk)]


def clock_stops(frames: pl.DataFrame, frame_index: pl.DataFrame, cfg: StopConfig | None = None):
    """Stops of one game (``STOP_SCHEMA``), plus the count of shorter freezes by frame count."""
    cfg = cfg or StopConfig()
    gid = frame_index["game_id"][0]
    fi = frame_index.sort(["period", "unix_ms"])
    rows, short = [], {}
    players = frames.filter(pl.col("team_id") != BALL_ID).select("period", "unix_ms", "player_id")
    for (period,), f in fi.group_by(["period"], maintain_order=True):
        t = f["unix_ms"].to_numpy()
        gc = f["game_clock"].to_numpy()
        frozen = f["clock_frozen"].to_numpy()
        dt = f["dt_ms"].fill_null(0).to_numpy()
        gap = f["gap"].to_numpy()
        jump_t = t[f["clock_jump"].to_numpy()]
        full = (f["has_ball"] & (f["n_players"] == 10)).to_numpy()
        ball = f["has_ball"].to_numpy()
        pl_p = players.filter(pl.col("period") == period)
        for a, b in _runs(frozen):
            k = a - 1  # first frame showing the stopped value
            r = k - 1  # last running frame
            if r < 0:
                continue  # stopped from the start of the period
            gc_stop = float(gc[a])
            clock_left = max(0.0, float(gc[r]) - gc_stop)
            exact = bool(not gap[k] and not gap[r] and not frozen[r])
            t_on = int(t[k]) if exact else min(int(t[k]), int(t[r] + clock_left * 1000))
            dur = (int(t[b]) - t_on) / 1000
            n = b - k + 1
            if dur < cfg.min_stop_s:
                short[n] = short.get(n, 0) + 1
                continue
            t_end = int(t[b + 1]) if b + 1 < t.size else None
            pre = (t >= t_on - cfg.pre_s * 1000) & (t < t_on)
            post = (t >= t_on) & (t < t_on + cfg.post_s * 1000)
            n_pre_slots, n_post_slots = cfg.pre_s * cfg.hz, cfg.post_s * cfg.hz
            tp = np.r_[t_on - cfg.pre_s * 1000, t[pre], t_on]
            max_hole = int(np.max(np.diff(tp)))
            win = pl_p.filter(
                (pl.col("unix_ms") >= t_on - cfg.pre_s * 1000) & (pl.col("unix_ms") < t_on)
            )
            n_fr = int(pre.sum())
            miss = []
            if n_fr:
                cnt = win.group_by("player_id").len()
                lim = n_fr - cfg.missing_player_s * cfg.hz
                miss = sorted(int(p) for p, c in cnt.iter_rows() if c < lim)
            rows.append(
                {
                    "game_id": gid,
                    "period": period,
                    "t_last_running_ms": int(t[r]),
                    "clock_left_s": clock_left,
                    "t_onset_est_ms": t_on,
                    "onset_exact": exact,
                    "t_start_ms": int(t[k]),
                    "t_end_ms": t_end,
                    "game_clock_at_stop": gc_stop,
                    "duration_s": dur,
                    "n_frozen_frames": n,
                    "micro_freeze": dur < cfg.micro_s,
                    "gap_before_ms": 0 if exact else int(max(dt[k], dt[r])),
                    "ends_at_gap": bool(b + 1 < t.size and dt[b + 1] > FRAME_GAP_MS),
                    "clock_correction": bool(
                        np.any(np.abs(jump_t - t_on) <= cfg.correction_s * 1000)
                    ),
                    "pre3s_tracked_frac": min(1.0, float(full[pre].sum()) / n_pre_slots),
                    "post1s_tracked_frac": min(1.0, float(full[post].sum()) / n_post_slots),
                    "pre3s_max_hole_ms": max_hole,
                    "pre3s_ball_missing_frac": max(0.0, 1.0 - float(ball[pre].sum()) / n_pre_slots),
                    "pre3s_missing_player_ids": miss,
                    "event_uid": f"{gid}:clock_stop:{t_on}:-1",
                }
            )
    return pl.DataFrame(rows, schema=STOP_SCHEMA), short
