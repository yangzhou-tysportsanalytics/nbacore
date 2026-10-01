"""Turn one possession window into fixed-rate, left-half, dense arrays.

* Uniform grid at ``hz`` (default 5 Hz) starting at ``t_start``: t_k = t_start + k * 1000/hz ms,
  k = 0 .. n_steps-1, valid while t_k <= t_end. Missing tail steps are masked.
* Every entity's (x, y[, z]) is linearly interpolated on ``unix_ms`` from the raw ~25 Hz rows
  (jitter of 39-41 ms is absorbed by the interpolation).
* If the offence attacks the right basket, the court is rotated 180 degrees about its centre
  (x -> 94 - x, y -> 50 - y) so that every possession attacks the LEFT basket.
* Velocities are central differences on the grid (ft/s).

Slots: offence players are ordered by player_id (slot 0-4), defence likewise (slot 0-4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.court import LENGTH, WIDTH
from nbacore.io.raw import BALL_ID


@dataclass
class ResampleConfig:
    hz: int = 5
    n_steps: int = 121  # 24 s at 5 Hz, inclusive of both ends


@dataclass
class PossessionArrays:
    game_id: str
    possession_id: int
    t: np.ndarray  # (n_steps,) unix_ms of grid points
    valid: np.ndarray  # (n_steps,) bool
    game_clock: np.ndarray  # (n_steps,) float32
    shot_clock: np.ndarray  # (n_steps,) float32 with NaN
    ball: np.ndarray  # (n_steps, 3) x, y, z
    off_ids: np.ndarray  # (5,) int64
    def_ids: np.ndarray  # (5,) int64
    off_xy: np.ndarray  # (n_steps, 5, 2)
    def_xy: np.ndarray  # (n_steps, 5, 2)
    off_v: np.ndarray  # (n_steps, 5, 2)
    def_v: np.ndarray  # (n_steps, 5, 2)
    ball_v: np.ndarray  # (n_steps, 2)
    flipped: bool


def _interp_entity(t_grid: np.ndarray, t_raw: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """Linear interpolation per column; extrapolates by holding end values."""
    order = np.argsort(t_raw, kind="stable")
    t_raw = t_raw[order]
    vals = vals[order]
    out = np.empty((t_grid.size, vals.shape[1]), dtype=np.float32)
    for j in range(vals.shape[1]):
        out[:, j] = np.interp(t_grid, t_raw, vals[:, j])
    return out


def _velocity(xy: np.ndarray, dt: float, valid: np.ndarray) -> np.ndarray:
    """Central differences along axis 0; one-sided at the valid ends; 0 where invalid."""
    v = np.zeros_like(xy)
    n = int(valid.sum())
    if n >= 2:
        v[1 : n - 1] = (xy[2:n] - xy[: n - 2]) / (2 * dt)
        v[0] = (xy[1] - xy[0]) / dt
        v[n - 1] = (xy[n - 1] - xy[n - 2]) / dt
    return v


def resample_possession(
    entities: pl.DataFrame,
    frame_index: pl.DataFrame,
    poss: dict,
    cfg: ResampleConfig | None = None,
) -> PossessionArrays:
    """``poss`` is one row of the possessions table as a dict (``row(named=True)``)."""
    cfg = cfg or ResampleConfig()
    step_ms = 1000.0 / cfg.hz
    period = int(poss["period"])
    t0, t1 = int(poss["t_start"]), int(poss["t_end"])
    # raw rows with a little slack on both sides for interpolation
    slack = 200
    ent = entities.filter(
        (pl.col("period") == period)
        & (pl.col("unix_ms") >= t0 - slack)
        & (pl.col("unix_ms") <= t1 + slack)
    )
    fi = frame_index.filter(
        (pl.col("period") == period)
        & (pl.col("unix_ms") >= t0 - slack)
        & (pl.col("unix_ms") <= t1 + slack)
    ).sort("unix_ms")

    t_grid = (t0 + np.arange(cfg.n_steps) * step_ms).astype(np.float64)
    valid = t_grid <= t1 + 1e-6
    t_eval = np.where(valid, t_grid, t1)  # hold the last valid time for masked steps

    off_ids = np.asarray(sorted(poss["offense_player_ids"]), dtype=np.int64)
    def_ids = np.asarray(sorted(poss["defense_player_ids"]), dtype=np.int64)

    def ent_xy(pid: int) -> np.ndarray:
        e = ent.filter(pl.col("player_id") == pid)
        if e.height == 0:
            raise ValueError(f"player {pid} has no rows in possession window")
        return _interp_entity(
            t_eval, e["unix_ms"].to_numpy().astype(np.float64), e.select(["x", "y"]).to_numpy()
        )

    off_xy = np.stack([ent_xy(int(p)) for p in off_ids], axis=1)
    def_xy = np.stack([ent_xy(int(p)) for p in def_ids], axis=1)
    b = ent.filter(pl.col("team_id") == BALL_ID)
    if b.height == 0:
        raise ValueError("no ball rows in possession window")
    ball = _interp_entity(
        t_eval, b["unix_ms"].to_numpy().astype(np.float64), b.select(["x", "y", "z"]).to_numpy()
    )
    # clocks: game clock interpolates; shot clock interpolates only where defined
    tf = fi["unix_ms"].to_numpy().astype(np.float64)
    gc = np.interp(t_eval, tf, fi["game_clock"].to_numpy().astype(np.float64)).astype(np.float32)
    sc_raw = fi["shot_clock"].cast(pl.Float64).to_numpy()  # nulls become NaN
    sc = np.full(cfg.n_steps, np.nan, dtype=np.float32)
    ok = ~np.isnan(sc_raw)
    if ok.sum() >= 2:
        sc_i = np.interp(t_eval, tf[ok], sc_raw[ok].astype(np.float64))
        # only trust interpolated values where the nearest raw frame had a shot clock
        nearest = np.searchsorted(tf, t_eval).clip(0, tf.size - 1)
        sc = np.where(ok[nearest], sc_i, np.nan).astype(np.float32)

    flipped = not bool(poss["attacks_left"])
    if flipped:
        off_xy[..., 0] = LENGTH - off_xy[..., 0]
        off_xy[..., 1] = WIDTH - off_xy[..., 1]
        def_xy[..., 0] = LENGTH - def_xy[..., 0]
        def_xy[..., 1] = WIDTH - def_xy[..., 1]
        ball[:, 0] = LENGTH - ball[:, 0]
        ball[:, 1] = WIDTH - ball[:, 1]

    dt = step_ms / 1000.0
    off_v = np.stack([_velocity(off_xy[:, j], dt, valid) for j in range(5)], axis=1)
    def_v = np.stack([_velocity(def_xy[:, j], dt, valid) for j in range(5)], axis=1)
    ball_v = _velocity(ball[:, :2], dt, valid)

    return PossessionArrays(
        game_id=str(poss["game_id"]),
        possession_id=int(poss["possession_id"]),
        t=t_grid.astype(np.int64),
        valid=valid,
        game_clock=gc,
        shot_clock=sc,
        ball=ball,
        off_ids=off_ids,
        def_ids=def_ids,
        off_xy=off_xy,
        def_xy=def_xy,
        off_v=off_v,
        def_v=def_v,
        ball_v=ball_v,
        flipped=flipped,
    )


def arrays_to_long(pa: PossessionArrays, handler_slot: np.ndarray | None = None) -> pl.DataFrame:
    """Long table: 11 rows per step (ball + 5 offence + 5 defence), all steps (masked ones too).

    ``handler_slot``: (n_steps,) int offence slot holding the ball, -1 when none/unknown.
    """
    n = pa.t.size
    rows = 11 * n
    role = np.array(["ball"] + ["off"] * 5 + ["def"] * 5)
    slot = np.array([0] + list(range(5)) + list(range(5)), dtype=np.int8)
    pid = np.concatenate([[BALL_ID], pa.off_ids, pa.def_ids])
    xy = np.concatenate([pa.ball[:, None, :2], pa.off_xy, pa.def_xy], axis=1)  # (n, 11, 2)
    v = np.concatenate([pa.ball_v[:, None, :], pa.off_v, pa.def_v], axis=1)
    z = np.concatenate([pa.ball[:, None, 2], np.zeros((n, 10), dtype=np.float32)], axis=1)
    hs = handler_slot if handler_slot is not None else np.full(n, -1, dtype=np.int8)
    is_handler = np.zeros((n, 11), dtype=bool)
    for k in range(n):
        if hs[k] >= 0:
            is_handler[k, 1 + int(hs[k])] = True
    return pl.DataFrame(
        {
            "game_id": pl.Series([pa.game_id] * rows, dtype=pl.Utf8),
            "possession_id": pl.Series(np.full(rows, pa.possession_id, dtype=np.int32)),
            "step": pl.Series(np.repeat(np.arange(n, dtype=np.int16), 11)),
            "valid": pl.Series(np.repeat(pa.valid, 11)),
            "t_unix": pl.Series(np.repeat(pa.t, 11)),
            "game_clock": pl.Series(np.repeat(pa.game_clock, 11)),
            "shot_clock": pl.Series(np.repeat(pa.shot_clock, 11)),
            "role": pl.Series(np.tile(role, n), dtype=pl.Utf8),
            "slot": pl.Series(np.tile(slot, n)),
            "player_id": pl.Series(np.tile(pid, n)),
            "x": pl.Series(xy[..., 0].reshape(-1).astype(np.float32)),
            "y": pl.Series(xy[..., 1].reshape(-1).astype(np.float32)),
            "z": pl.Series(z.reshape(-1).astype(np.float32)),
            "vx": pl.Series(v[..., 0].reshape(-1).astype(np.float32)),
            "vy": pl.Series(v[..., 1].reshape(-1).astype(np.float32)),
            "is_handler": pl.Series(is_handler.reshape(-1)),
        }
    ).with_columns(pl.col("shot_clock").fill_nan(None))
