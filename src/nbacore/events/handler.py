"""Game-wide 25 Hz ball handler (L2).

``nbacore.ballhandler.infer.infer_handler_raw`` (migrated from ghost-defense) runs inside one
possession window and only considers the five offensive players. Over a whole game the offence is
not known in advance, so two rules are implemented (``HandlerL2Config.mode``):

``"nearest"``
    Per-frame candidate = nearest player of **either team** within ``max_dist_ft`` (xy) of the
    ball while the ball is below ``max_ball_z_ft``; then the sticky rule (runs shorter than
    ``min_hold_s`` inherit the previous handler unless followed by ball flight).

``"team_hysteresis"``
    Two stages. (1) Controlling team: run-length encode the team of the per-frame candidate; a
    run of the *other* team replaces the current team only if it lasts ≥ ``team_switch_s`` or is
    preceded by ≥ ``min_hold_s`` without any candidate (loose ball: rebound, interception,
    strip). Shorter runs of the other team keep the current team: a defender closest to the ball
    while contesting, including right before a shot's flight (so "touch then flight", which the
    sticky rule accepts for catch-and-shoot within a team, is *not* a reason to switch teams unless
    ``switch_on_flight_after``). (2) Handler candidate = nearest player **of the controlling team** within
    ``max_dist_ft``; then the sticky rule. With the controlling team known this is the same as
    ``infer_handler_raw``, which only looks at the offence.

Both are applied separately inside each continuity segment of the frame index (a new segment at a
tracking hole, an upward clock jump or a period change), so nothing carries across a hole.
Frames without the ball have no candidate (``has_ball = False``, ``handler_id = -1``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.ballhandler.infer import HandlerConfig, sticky_handler
from nbacore.io.raw import BALL_ID

NO_HANDLER = -1


@dataclass
class HandlerL2Config(HandlerConfig):
    mode: str = "team_hysteresis"  # or "nearest"
    team_switch_s: float = 1.0
    switch_on_flight_after: bool = False


HANDLER_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "unix_ms": pl.Int64,
    "segment_id": pl.Int32,
    "has_ball": pl.Boolean,
    "ball_z": pl.Float32,
    "cand_id": pl.Int64,  # nearest player of either team within range (-1 none)
    "cand_dist_ft": pl.Float32,  # xy distance ball - nearest player (null without ball)
    # nearest and second-nearest player at the identical distance: typically two players with
    # identical x, y (a source identity merge); the candidate id is then arbitrary
    "cand_ambiguous": pl.Boolean,
    "control_team_id": pl.Int64,  # team_hysteresis stage 1 (-1 none); = handler team in "nearest"
    "handler_id": pl.Int64,  # after the sticky rule (-1 none)
    "handler_team_id": pl.Int64,  # team of handler_id (-1 none)
}


def _per_team_nearest(frames: pl.DataFrame) -> tuple[pl.DataFrame, list[int]]:
    """Ball frames with ball z and, per team, the nearest player and distance (xy)."""
    ball = frames.filter(pl.col("team_id") == BALL_ID).select(
        "period",
        "unix_ms",
        pl.col("x").alias("bx"),
        pl.col("y").alias("by"),
        pl.col("z").alias("bz"),
    )
    players = frames.filter(pl.col("team_id") != BALL_ID).select(
        "period", "unix_ms", "team_id", "player_id", "x", "y"
    )
    teams = sorted(int(t) for t in players["team_id"].unique())
    d = players.join(ball, on=["period", "unix_ms"], how="inner").with_columns(
        ((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt().alias("d")
    )
    d = d.sort(["period", "unix_ms", "d", "player_id"])
    overall = d.group_by(["period", "unix_ms"], maintain_order=True).agg(
        pl.col("player_id").first().alias("near_id"),
        pl.col("team_id").first().alias("near_team"),
        pl.col("d").first().alias("near_d"),
        pl.col("d").get(1, null_on_oob=True).alias("near_d2"),
    )
    out = ball.select("period", "unix_ms", "bz").join(overall, on=["period", "unix_ms"], how="left")
    by_team = d.group_by(["period", "unix_ms", "team_id"], maintain_order=True).agg(
        pl.col("player_id").first().alias("id"), pl.col("d").first().alias("d")
    )
    for k, t in enumerate(teams):
        bt = by_team.filter(pl.col("team_id") == t).select(
            "period", "unix_ms", pl.col("id").alias(f"id{k}"), pl.col("d").alias(f"d{k}")
        )
        out = out.join(bt, on=["period", "unix_ms"], how="left")
    return out, teams


def _runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    change = np.flatnonzero(np.diff(x)) + 1
    return np.r_[0, change], np.r_[change, x.size]


def control_team(team_cand: np.ndarray, cfg: HandlerL2Config) -> np.ndarray:
    """Stage 1 of ``team_hysteresis`` on one segment: controlling team per frame (-1 none)."""
    n = team_cand.size
    out = np.full(n, NO_HANDLER, dtype=np.int64)
    if n == 0:
        return out
    min_len = max(1, int(round(cfg.min_hold_s * cfg.hz)))
    switch_len = max(1, int(round(cfg.team_switch_s * cfg.hz)))
    starts, ends = _runs(team_cand)
    current = NO_HANDLER
    for i, (s, e) in enumerate(zip(starts, ends, strict=True)):
        t = int(team_cand[s])
        if t < 0 or t == current:
            out[s:e] = current
            continue
        length = e - s
        loose_before = (
            i > 0 and team_cand[starts[i - 1]] < 0 and ends[i - 1] - starts[i - 1] >= min_len
        )
        loose_after = cfg.switch_on_flight_after and (
            i + 1 < starts.size
            and team_cand[starts[i + 1]] < 0
            and ends[i + 1] - starts[i + 1] >= min_len
        )
        # start of a segment: the first team holding the ball for >= min_hold_s
        first_control = current < 0 and length >= min_len
        if length >= switch_len or loose_before or loose_after or first_control:
            current = t
        out[s:e] = current
    return out


def game_handler(
    frames: pl.DataFrame, frame_index: pl.DataFrame, cfg: HandlerL2Config | None = None
) -> pl.DataFrame:
    """One row per frame of ``frame_index`` (see module doc for the two modes)."""
    cfg = cfg or HandlerL2Config()
    gid = frame_index["game_id"][0]
    near, teams = _per_team_nearest(frames)
    fi = (
        frame_index.select("period", "unix_ms", "segment_id", "has_ball")
        .join(near, on=["period", "unix_ms"], how="left")
        .sort(["period", "unix_ms"])
    )
    in_z = (fi["has_ball"] & (fi["bz"] <= cfg.max_ball_z_ft)).fill_null(False).to_numpy()
    near_ok = in_z & (fi["near_d"] <= cfg.max_dist_ft).fill_null(False).to_numpy()
    cand = np.where(near_ok, fi["near_id"].fill_null(NO_HANDLER).to_numpy(), NO_HANDLER)
    cand_team = np.where(near_ok, fi["near_team"].fill_null(NO_HANDLER).to_numpy(), NO_HANDLER)
    seg = fi["segment_id"].to_numpy()
    bounds = np.flatnonzero(np.diff(seg)) + 1
    spans = list(zip(np.r_[0, bounds], np.r_[bounds, cand.size], strict=True))

    if cfg.mode == "nearest":
        handler_cand = cand
        ctrl = None
    elif cfg.mode == "team_hysteresis":
        ctrl = np.full(cand.size, NO_HANDLER, dtype=np.int64)
        for s, e in spans:
            ctrl[s:e] = control_team(cand_team[s:e], cfg)
        handler_cand = np.full(cand.size, NO_HANDLER, dtype=np.int64)
        for k, t in enumerate(teams):
            ids = fi[f"id{k}"].fill_null(NO_HANDLER).to_numpy()
            dk = fi[f"d{k}"].fill_null(np.inf).to_numpy()
            m = (ctrl == t) & in_z & (dk <= cfg.max_dist_ft)
            handler_cand[m] = ids[m]
    else:
        raise ValueError(f"unknown mode {cfg.mode!r}")

    handler = np.full(cand.size, NO_HANDLER, dtype=np.int64)
    for s, e in spans:
        handler[s:e] = _sticky_ids(handler_cand[s:e], cfg)
    team_of = dict(
        frames.filter(pl.col("team_id") != BALL_ID)
        .select("player_id", "team_id")
        .unique()
        .iter_rows()
    )
    hteam = np.array([team_of.get(int(h), NO_HANDLER) if h >= 0 else NO_HANDLER for h in handler])
    ambiguous = (cand >= 0) & (fi["near_d2"] == fi["near_d"]).fill_null(False).to_numpy()
    return pl.DataFrame(
        {
            "game_id": gid,
            "period": fi["period"],
            "unix_ms": fi["unix_ms"],
            "segment_id": fi["segment_id"],
            "has_ball": fi["has_ball"],
            "ball_z": fi["bz"],
            "cand_id": cand,
            "cand_dist_ft": fi["near_d"],
            "cand_ambiguous": ambiguous,
            "control_team_id": ctrl if ctrl is not None else hteam,
            "handler_id": handler,
            "handler_team_id": hteam,
        },
        schema=HANDLER_SCHEMA,
    )


def _sticky_ids(cand_ids: np.ndarray, cfg: HandlerConfig) -> np.ndarray:
    """``sticky_handler`` on player ids: encode ids as small codes, apply, decode."""
    uniq, codes = np.unique(cand_ids, return_inverse=True)
    codes = codes.astype(np.int16)
    none_code = np.flatnonzero(uniq == NO_HANDLER)
    if none_code.size:
        codes = np.where(codes == none_code[0], -1, codes)
    out = sticky_handler(codes, cfg)  # int8 codes: fine while < 128 distinct ids per segment
    return np.where(out >= 0, uniq[np.clip(out, 0, None)], NO_HANDLER)
