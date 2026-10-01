"""Assemble the L2 tables of one game.

``build_game_l2`` runs every detector on the L1 tables of one game and returns

* ``handler``: the 25 Hz handler table (``handler/<game_id>.parquet``);
* ``events``: one long table (``events/<game_id>.parquet``) with the common columns
  ``game_id, period, event_type, t_start_ms, t_end_ms, team_id, actor_id, target_id, x, y,
  confidence, method, params_hash, event_uid`` followed by the type-specific columns of every
  type (null for other types; ``nbacore.load.events`` drops all-null columns after filtering);
* ``pbp_align``: rows of ``pbp_align.parquet`` for the game;
* ``stats``: counts and diagnostics for the build report.

Types: ``possession_touch``, ``ball_flight``, ``pass``, ``handoff``, ``shot_release``,
``rim_contact``, ``rebound_landing``, ``clock_stop``, ``screen_candidate``, ``drive_candidate``,
``cut_candidate``, ``dribble``, ``postup_candidate``, ``inbound``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json

import polars as pl

from nbacore.events.clock_stop import StopConfig, clock_stops
from nbacore.events.dribble import DribbleConfig, dribbles
from nbacore.events.flight import FlightConfig, ball_flights
from nbacore.events.handler import HandlerL2Config, game_handler
from nbacore.events.motion import MotionConfig, PostUpConfig, motion_candidates, postups
from nbacore.events.passes import PassConfig, inbounds, mark_after_made_fg, mark_shots, passes
from nbacore.events.pbp_align import align_game
from nbacore.events.screens import ScreenConfig, screen_candidates
from nbacore.events.shots import ShotEventConfig, shot_events
from nbacore.io.raw import BALL_ID
from nbacore.possession.shots import ShotTimeConfig


@dataclasses.dataclass
class CandidateConfig:
    # v1.3: screen / drive / cut / post-up candidates only where the L2 controlling team is the
    # ledger offence (the handler gave the defence control for > 1 s at times: 16.4 % of the v1.1
    # screen candidates had screener and user on the defence)
    ledger_offense: bool = True


COMMON: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,
    "t_start_ms": pl.Int64,
    "t_end_ms": pl.Int64,
    "team_id": pl.Int64,
    "actor_id": pl.Int64,
    "target_id": pl.Int64,
    "x": pl.Float32,
    "y": pl.Float32,
    "confidence": pl.Float32,
    "method": pl.Utf8,
    "params_hash": pl.Utf8,
    "event_uid": pl.Utf8,
}

CONFIGS = {
    "handler": HandlerL2Config(),
    "flight": FlightConfig(),
    "pass": PassConfig(),
    # v1.3: only the pbp shooter's hands count (late releases 14.8 % -> 5.1 %); v1.6: a tip is
    # searched after the miss left the rim, wider window and last-hand fallback when neither rim
    # contact nor apex is found
    "shot": ShotEventConfig(
        shot=ShotTimeConfig(shooter_only=True, wide_before_s=10.0, hand_fallback=True),
        after_miss_from_rim=True,
    ),
    "stop": StopConfig(),
    "screen": ScreenConfig(),
    "motion": MotionConfig(),
    "dribble": DribbleConfig(),
    "postup": PostUpConfig(),
    "candidates": CandidateConfig(),
}


def params_hash(cfg) -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode()).hexdigest()[
        :12
    ]


def all_params() -> dict:
    return {k: dataclasses.asdict(v) for k, v in CONFIGS.items()}


def _common(df: pl.DataFrame, event_type: str | pl.Expr, cfg, **cols) -> pl.DataFrame:
    """Common columns first (from ``cols`` expressions), type-specific columns after."""
    et = event_type if isinstance(event_type, pl.Expr) else pl.lit(event_type)
    exprs = {
        "game_id": pl.col("game_id"),
        "period": pl.col("period"),
        "event_type": et,
        "confidence": pl.lit(None),
        "method": pl.lit(None),
        "params_hash": pl.lit(params_hash(cfg)),
        "event_uid": pl.col("event_uid") if "event_uid" in df.columns else pl.lit(None),
    }
    exprs.update(cols)
    common = df.select([exprs.get(c, pl.lit(None)).cast(t).alias(c) for c, t in COMMON.items()])
    extra = df.drop([c for c in df.columns if c in COMMON])
    return pl.concat([common, extra], how="horizontal")


def possession_touches(handler: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Runs of a constant handler (>= 0) inside a segment."""
    h = handler.sort(["period", "unix_ms"]).with_columns(
        (
            (pl.col("handler_id") != pl.col("handler_id").shift(1))
            | (pl.col("segment_id") != pl.col("segment_id").shift(1))
        )
        .fill_null(True)
        .cum_sum()
        .alias("_run")
    )
    runs = (
        h.filter(pl.col("handler_id") >= 0)
        .group_by("_run", maintain_order=True)
        .agg(
            pl.col("game_id").first(),
            pl.col("period").first(),
            pl.col("handler_id").first(),
            pl.col("handler_team_id").first(),
            pl.col("unix_ms").min().alias("t_start_ms"),
            pl.col("unix_ms").max().alias("t_end_ms"),
            pl.len().alias("n_frames"),
            pl.col("cand_ambiguous").any().alias("cand_ambiguous"),
        )
        .drop("_run")
    )
    ball = frames.filter(pl.col("team_id") == BALL_ID).select(
        "period", pl.col("unix_ms").alias("t_start_ms"), "x", "y"
    )
    return runs.join(ball, on=["period", "t_start_ms"], how="left").with_columns(
        pl.format("{}:possession_touch:{}:{}", "game_id", "t_start_ms", "handler_id").alias(
            "event_uid"
        )
    )


def _ledger_constrained(h, pbp_game, align, frame_index, direction) -> pl.DataFrame:
    """The handler table with ``control_team_id`` = -1 wherever it is not the ledger offence
    (``CandidateConfig.ledger_offense``); used only by the candidate detectors."""
    if not CONFIGS["candidates"].ledger_offense:
        return h
    from nbacore.ledger import ledger_times, offense_by_frame, pbp_possessions

    led = ledger_times(
        pbp_possessions(pbp_game),
        align,
        frame_index,
        direction,
        handler=h.select("period", "unix_ms", "control_team_id"),
        pbp_game=pbp_game,
    )
    off = offense_by_frame(led, h)
    return h.with_columns(
        pl.when(pl.col("control_team_id") == off)
        .then(pl.col("control_team_id"))
        .otherwise(-1)
        .alias("control_team_id")
    )


def _no_handler_frames(ps: pl.DataFrame, h: pl.DataFrame) -> pl.DataFrame:
    """Pass / handoff rows: frames without a handler between release and catch."""
    import numpy as np

    by = {
        int(p): (g["unix_ms"].to_numpy(), (g["handler_id"].fill_null(-1) < 0).to_numpy())
        for (p,), g in h.sort("period", "unix_ms").group_by("period")
    }
    out = []
    for per, a, b in ps.select("period", "t_release_ms", "t_catch_ms").iter_rows():
        if per not in by or a is None or b is None:
            out.append(None)
            continue
        t, free = by[per]
        i0, i1 = np.searchsorted(t, a, side="right"), np.searchsorted(t, b, side="left")
        out.append(int(free[i0:i1].sum()) if i1 >= i0 else 0)
    return ps.with_columns(pl.Series("n_no_handler_frames", out, dtype=pl.Int32))


def build_game_l2(frames, frame_index, pbp_game, direction) -> dict:
    h = game_handler(frames, frame_index, CONFIGS["handler"])
    fl = ball_flights(h, frames, CONFIGS["flight"])
    sh = shot_events(frames, frame_index, pbp_game, direction, h, CONFIGS["shot"])
    ps = mark_shots(
        mark_after_made_fg(passes(h, fl, frames, frame_index, CONFIGS["pass"]), sh, h), sh
    )
    stops, short = clock_stops(frames, frame_index, CONFIGS["stop"])
    align, stops, ps = align_game(pbp_game, frame_index, sh, stops, fl, ps)
    hc = _ledger_constrained(h, pbp_game, align, frame_index, direction)
    sc = screen_candidates(frames, hc, direction, CONFIGS["screen"])
    touches = possession_touches(h, frames)
    drives, cuts = motion_candidates(frames, hc, direction, sh, ps, stops, sc, CONFIGS["motion"])
    drb, touches = dribbles(frames, touches, frame_index, ps, sh, CONFIGS["dribble"])
    pus = postups(frames, hc, direction, drb, sh, ps, CONFIGS["postup"])
    ps = _no_handler_frames(ps, h)
    inb = inbounds(ps, frames, frame_index, sh)
    fl = fl.with_columns(
        pl.format(
            "{}:ball_flight:{}:{}", "game_id", "t_start_ms", pl.col("from_id").fill_null(-1)
        ).alias("event_uid")
    )

    parts = [
        _common(
            touches,
            "possession_touch",
            CONFIGS["handler"],
            t_start_ms=pl.col("t_start_ms"),
            t_end_ms=pl.col("t_end_ms"),
            team_id=pl.col("handler_team_id"),
            actor_id=pl.col("handler_id"),
            x=pl.col("x"),
            y=pl.col("y"),
        ),
        _common(
            fl,
            "ball_flight",
            CONFIGS["flight"],
            t_start_ms=pl.col("t_start_ms"),
            t_end_ms=pl.col("t_end_ms"),
            team_id=pl.col("from_team_id"),
            actor_id=pl.col("from_id"),
            target_id=pl.col("to_id"),
            x=pl.col("x_start"),
            y=pl.col("y_start"),
        ),
        _common(
            ps,
            pl.col("event_type"),
            CONFIGS["pass"],
            t_start_ms=pl.col("t_release_ms"),
            t_end_ms=pl.col("t_catch_ms"),
            team_id=pl.col("passer_team_id"),
            actor_id=pl.col("passer_id"),
            target_id=pl.col("receiver_id"),
            x=pl.col("x_release"),
            y=pl.col("y_release"),
            method=pl.col("kind"),
        ).drop("event_type_right", strict=False),
        _common(
            sh.filter(pl.col("t_release_ms").is_not_null()),
            "shot_release",
            CONFIGS["shot"],
            t_start_ms=pl.col("t_release_ms"),
            t_end_ms=pl.col("t_rim_ms"),
            team_id=pl.col("shooter_team_id"),
            actor_id=pl.coalesce("shooter_id", "shooter_inferred_id"),
            x=pl.col("x_release"),
            y=pl.col("y_release"),
            method=pl.col("method"),
        ),
        _common(
            sh.filter(pl.col("t_rim_ms").is_not_null())
            .with_columns(
                pl.format(
                    "{}:rim_contact:{}:{}",
                    "game_id",
                    "t_rim_ms",
                    pl.coalesce("shooter_id", "shooter_inferred_id", pl.lit(-1)),
                ).alias("event_uid")
            )
            .select(
                "game_id",
                "period",
                "t_rim_ms",
                "shooter_id",
                "shooter_team_id",
                "pbp_event_num",
                "event_uid",
            ),
            "rim_contact",
            CONFIGS["shot"],
            t_start_ms=pl.col("t_rim_ms"),
            t_end_ms=pl.col("t_rim_ms"),
            team_id=pl.col("shooter_team_id"),
            actor_id=pl.col("shooter_id"),
        ),
        _common(
            sh.filter(pl.col("t_landing_ms").is_not_null())
            .with_columns(
                pl.format(
                    "{}:rebound_landing:{}:{}",
                    "game_id",
                    "t_landing_ms",
                    pl.col("control_id").fill_null(-1),
                ).alias("event_uid")
            )
            .select(
                "game_id",
                "period",
                "t_landing_ms",
                "x_landing",
                "y_landing",
                "t_control_ms",
                "control_id",
                "control_team_id",
                "offensive_rebound",
                "pbp_event_num",
                "event_uid",
            ),
            "rebound_landing",
            CONFIGS["shot"],
            t_start_ms=pl.col("t_landing_ms"),
            t_end_ms=pl.max_horizontal("t_landing_ms", "t_control_ms"),
            team_id=pl.col("control_team_id"),
            actor_id=pl.col("control_id"),
            x=pl.col("x_landing"),
            y=pl.col("y_landing"),
        ),
        _common(
            stops,
            "clock_stop",
            CONFIGS["stop"],
            t_start_ms=pl.col("t_onset_est_ms"),
            t_end_ms=pl.col("t_end_ms"),
            method=pl.when(pl.col("onset_exact"))
            .then(pl.lit("exact"))
            .otherwise(pl.lit("estimated")),
        ),
        _common(
            sc,
            "screen_candidate",
            CONFIGS["screen"],
            t_start_ms=pl.col("t_set_start_ms"),
            t_end_ms=pl.col("t_end_ms"),
            team_id=pl.col("offense_team_id"),
            actor_id=pl.col("screener_id"),
            target_id=pl.col("user_id"),
            x=pl.col("x"),
            y=pl.col("y"),
        ),
    ]
    for mv in (drives, cuts):
        parts.append(
            _common(
                mv,
                pl.col("event_type"),
                CONFIGS["motion"],
                t_start_ms=pl.col("t_start_ms"),
                t_end_ms=pl.col("t_end_ms"),
                team_id=pl.col("team_id"),
                actor_id=pl.col("player_id"),
                target_id=pl.col("def_id"),
                x=pl.col("x"),
                y=pl.col("y"),
            )
        )
    parts += [
        _common(
            drb,
            "dribble",
            CONFIGS["dribble"],
            t_start_ms=pl.col("t_bounce_ms"),
            t_end_ms=pl.col("t_bounce_ms"),
            team_id=pl.col("team_id"),
            actor_id=pl.col("handler_id"),
            x=pl.col("x"),
            y=pl.col("y"),
        ),
        _common(
            pus,
            "postup_candidate",
            CONFIGS["postup"],
            t_start_ms=pl.col("t_start_ms"),
            t_end_ms=pl.col("t_end_ms"),
            team_id=pl.col("team_id"),
            actor_id=pl.col("player_id"),
            target_id=pl.col("def_id"),
            x=pl.col("x"),
            y=pl.col("y"),
        ),
        _common(
            inb,
            "inbound",
            CONFIGS["pass"],
            t_start_ms=pl.col("t_release_ms"),
            t_end_ms=pl.col("t_catch_ms"),
            team_id=pl.col("team_id"),
            actor_id=pl.col("inbounder_id"),
            target_id=pl.col("receiver_id"),
            x=pl.col("x_entry"),
            y=pl.col("y_entry"),
            method=pl.col("context"),
        ),
    ]
    events = pl.concat(parts, how="diagonal_relaxed").sort(["period", "t_start_ms", "event_type"])
    stats = {
        "n_events": events.group_by("event_type").len().sort("event_type").rows(),
        "short_freezes": short,
        "n_pbp": align.height,
        "pbp_tracking_event_share": float((align["anchor_quality"] >= 2).mean())
        if align.height
        else None,
        "uid_duplicates": events.height - events["event_uid"].n_unique(),
    }
    return {"handler": h, "events": events, "pbp_align": align, "stats": stats}
