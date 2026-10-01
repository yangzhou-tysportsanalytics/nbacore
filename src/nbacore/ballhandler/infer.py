"""Ball-handler inference on raw 25 Hz frames (Phase 1, task 4).

Rule: at each frame the *candidate* handler is the nearest offensive
player if they are within ``max_dist_ft`` of the ball and the ball is lower than ``max_ball_z_ft``;
otherwise there is no candidate (pass / shot in flight, loose ball). The label is made *sticky*:
a run of identical candidates shorter than ``min_hold_s`` does not change the handler and
inherits the previous accepted handler (so a bounce that briefly pushes the ball out of range
does not create a gap, and a 0.2 s touch does not create a new handler) - unless the short
touch is immediately followed by ball flight (catch-and-shoot, tip pass), which is a real touch.

Validation against play-by-play: the last accepted handler before a shot should be the pbp
shooter (``handler_vs_shooter``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.io.raw import BALL_ID


@dataclass
class HandlerConfig:
    max_dist_ft: float = 3.0
    max_ball_z_ft: float = 10.0
    min_hold_s: float = 0.4
    hz: float = 25.0


def candidate_handlers(
    t: np.ndarray,
    ball: np.ndarray,
    off_xy: np.ndarray,
    cfg: HandlerConfig,
) -> np.ndarray:
    """Per-frame candidate offence slot (0-4) or -1.

    ``ball`` (n, 3), ``off_xy`` (n, 5, 2) aligned on the same frames ``t``.
    """
    d = np.linalg.norm(off_xy - ball[:, None, :2], axis=-1)  # (n, 5)
    d = np.where(np.isnan(d), np.inf, d)
    nearest = np.argmin(d, axis=1)
    dmin = d[np.arange(d.shape[0]), nearest]
    ok = (dmin <= cfg.max_dist_ft) & (ball[:, 2] <= cfg.max_ball_z_ft)
    return np.where(ok, nearest, -1).astype(np.int8)


def sticky_handler(cand: np.ndarray, cfg: HandlerConfig) -> np.ndarray:
    """Apply the minimum-hold rule to a per-frame candidate sequence."""
    n = cand.size
    out = np.full(n, -1, dtype=np.int8)
    if n == 0:
        return out
    min_len = max(1, int(round(cfg.min_hold_s * cfg.hz)))
    # run-length encode
    change = np.flatnonzero(np.diff(cand)) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [n]])
    current = -1
    n_runs = starts.size
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        accept = e - s >= min_len
        # a short touch that is immediately followed by ball flight (a long no-candidate run)
        # is a real touch: catch-and-shoot, tip pass
        if not accept and cand[s] >= 0 and i + 1 < n_runs:
            ns, ne = starts[i + 1], ends[i + 1]
            accept = cand[ns] < 0 and (ne - ns) >= min_len
        if accept:
            current = int(cand[s])
        out[s:e] = current
    return out


def infer_handler_raw(
    entities: pl.DataFrame,
    period: int,
    t_start: int,
    t_end: int,
    off_ids: list[int],
    cfg: HandlerConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Handler slot per raw frame in [t_start, t_end]. Returns (t_raw, handler_slot)."""
    cfg = cfg or HandlerConfig()
    ent = entities.filter(
        (pl.col("period") == period) & (pl.col("unix_ms") >= t_start) & (pl.col("unix_ms") <= t_end)
    )
    ball = ent.filter(pl.col("team_id") == BALL_ID).sort("unix_ms")
    t_raw = ball["unix_ms"].to_numpy()
    if t_raw.size == 0:
        return t_raw, np.zeros(0, dtype=np.int8)
    bxyz = ball.select(["x", "y", "z"]).to_numpy().astype(np.float64)
    off_ids = sorted(int(p) for p in off_ids)
    off_xy = np.full((t_raw.size, 5, 2), np.nan)
    pos = {int(u): i for i, u in enumerate(t_raw.tolist())}
    for j, pid in enumerate(off_ids):
        e = ent.filter(pl.col("player_id") == pid)
        for u, x, y in zip(e["unix_ms"].to_list(), e["x"].to_list(), e["y"].to_list(), strict=True):
            i = pos.get(int(u))
            if i is not None:
                off_xy[i, j] = (x, y)
    cand = candidate_handlers(t_raw, bxyz, off_xy, cfg)
    return t_raw, sticky_handler(cand, cfg)


def handler_on_grid(t_raw: np.ndarray, handler: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """Nearest-frame lookup of the raw handler label on the 5 Hz grid (-1 when no raw frame)."""
    if t_raw.size == 0:
        return np.full(t_grid.size, -1, dtype=np.int8)
    idx = np.searchsorted(t_raw, t_grid).clip(1, t_raw.size - 1)
    left = idx - 1
    choose_left = np.abs(t_grid - t_raw[left]) <= np.abs(t_raw[idx] - t_grid)
    nearest = np.where(choose_left, left, idx)
    return handler[nearest]


def last_handler_before(t_raw: np.ndarray, handler: np.ndarray, t: int) -> int:
    """Last non-negative handler slot at or before time ``t`` (-1 if none)."""
    m = (t_raw <= t) & (handler >= 0)
    if not m.any():
        return -1
    return int(handler[np.flatnonzero(m)[-1]])
