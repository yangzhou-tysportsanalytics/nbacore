"""Aligned resampling grids.

``resample`` wraps ghost-defense's ``resample_possession`` (unchanged) and adds what the
consumers asked for:

* any rate ``hz``: steps t_k = t_start + k · 1000 / hz, so for hz ∈ {5, 10, 25} the steps with
  k divisible by hz / 5 coincide with the 5 Hz grid (all grids start at ``t_start``);
* ``n_steps`` default (24 s + ``post_release_s``) · hz + 1; ``post_release_s`` extends the window
  past ``t_end`` (box-outs);
* ``flip_to_left`` (default True, ghost's rotation; False keeps raw coordinates);
* per step: ``handler_slot`` (offence slot 0–4, −1 none / defence), from the L2 handler table
  (game-wide, team hysteresis; default) or ghost's per-possession handler
  (``handler_mode="ghost"``, for parity);
  ``ball_in_flight`` (no handler candidate at the nearest frame); ``shot_clock_filled`` and
  ``fill_method`` (``nbacore.shot_clock``).

With hz = 5, ``flip_to_left`` and ``handler_mode="ghost"`` the arrays are identical to
ghost-defense ``resample_possession`` + ``handler_on_grid`` (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.ballhandler.infer import handler_on_grid, infer_handler_raw
from nbacore.possession.resample import PossessionArrays, ResampleConfig, resample_possession
from nbacore.shot_clock import shot_clock_filled


@dataclass
class GridArrays(PossessionArrays):
    hz: int = 5
    handler_slot: np.ndarray | None = None  # (n_steps,) int8
    ball_in_flight: np.ndarray | None = None  # (n_steps,) bool
    shot_clock_filled: np.ndarray | None = None  # (n_steps,) float32, NaN when not filled
    fill_method: np.ndarray | None = None  # (n_steps,) str


def _nearest(t_raw: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(t_raw, t_grid).clip(1, t_raw.size - 1)
    left = idx - 1
    return np.where(np.abs(t_grid - t_raw[left]) <= np.abs(t_raw[idx] - t_grid), left, idx)


def resample(
    frames: pl.DataFrame,
    frame_index: pl.DataFrame,
    poss: dict,
    hz: int = 5,
    n_steps: int | None = None,
    flip_to_left: bool = True,
    post_release_s: float = 0.0,
    handler: pl.DataFrame | None = None,
    handler_mode: str = "l2",
    shot_clock: pl.DataFrame | bool = True,
) -> GridArrays:
    """``poss``: a possession row with ``game_id, possession_id, period, t_start, t_end,
    attacks_left, offense_player_ids, defense_player_ids``. ``handler``: the L2 handler table
    of the game (required for ``handler_mode="l2"``). ``shot_clock``: True computes
    ``shot_clock_filled`` from the frame index; a per-frame table with ``period, unix_ms,
    shot_clock_filled, fill_method`` (e.g. the L3 ``nbacore.load.shot_clock(game_id)``) is used
    as given; False skips it (NaN / None) when the computation would dominate the run time."""
    if n_steps is None:
        n_steps = int(round((24.0 + post_release_s) * hz)) + 1
    p = dict(poss)
    p["t_end"] = int(poss["t_end"] + post_release_s * 1000)
    if not flip_to_left:
        p["attacks_left"] = True  # no rotation
    pa = resample_possession(frames, frame_index, p, ResampleConfig(hz=hz, n_steps=n_steps))
    t_grid = pa.t.astype(np.int64)
    off_ids = [int(i) for i in pa.off_ids]
    if handler_mode == "ghost":
        t_raw, slot = infer_handler_raw(frames, p["period"], p["t_start"], p["t_end"], off_ids)
        hslot = handler_on_grid(t_raw, slot, t_grid)
        in_flight = np.zeros(n_steps, dtype=bool)
        if handler is not None:
            in_flight = _flight_on_grid(handler, p, t_grid)
    elif handler_mode == "l2":
        if handler is None:
            raise ValueError("handler_mode='l2' needs the L2 handler table")
        h = handler.filter(
            (pl.col("period") == p["period"])
            & (pl.col("unix_ms") >= p["t_start"])
            & (pl.col("unix_ms") <= p["t_end"])
            & pl.col("has_ball")
        ).sort("unix_ms")
        slot_of = {pid: k for k, pid in enumerate(off_ids)}
        hs = np.array([slot_of.get(int(i), -1) for i in h["handler_id"].to_list()], dtype=np.int8)
        hslot = handler_on_grid(h["unix_ms"].to_numpy(), hs, t_grid)
        in_flight = _flight_on_grid(handler, p, t_grid)
    else:
        raise ValueError(f"unknown handler_mode {handler_mode!r}")
    hslot = np.where(pa.valid, hslot, -1).astype(np.int8)

    if shot_clock is False:
        scf = np.full(n_steps, np.nan, dtype=np.float32)
        fm = np.full(n_steps, None, dtype=object)
    else:
        if isinstance(shot_clock, pl.DataFrame):
            sc = shot_clock.filter(pl.col("period") == p["period"]).sort("unix_ms")
        else:
            sc = shot_clock_filled(frame_index.filter(pl.col("period") == p["period"]))
        tf = sc["unix_ms"].to_numpy()
        k = _nearest(tf, t_grid)
        scf = sc["shot_clock_filled"].cast(pl.Float64).to_numpy()[k].astype(np.float32)
        fm = sc["fill_method"].to_numpy()[k]
    return GridArrays(
        **{f: getattr(pa, f) for f in PossessionArrays.__dataclass_fields__},
        hz=hz,
        handler_slot=hslot,
        ball_in_flight=in_flight & pa.valid,
        shot_clock_filled=np.where(pa.valid, scf, np.nan).astype(np.float32),
        fill_method=np.where(pa.valid, fm, None),
    )


def _flight_on_grid(handler: pl.DataFrame, p: dict, t_grid: np.ndarray) -> np.ndarray:
    h = handler.filter(
        (pl.col("period") == p["period"])
        & (pl.col("unix_ms") >= p["t_start"] - 1000)
        & (pl.col("unix_ms") <= p["t_end"] + 1000)
    ).sort("unix_ms")
    if h.height < 2:
        return np.zeros(t_grid.size, dtype=bool)
    free = (h["has_ball"] & (h["cand_id"] < 0)).to_numpy()
    return free[_nearest(h["unix_ms"].to_numpy(), t_grid)]
