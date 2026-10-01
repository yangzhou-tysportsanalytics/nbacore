"""De-duplication of raw tracking frames and construction of a per-frame index.

Raw events overlap heavily (54-62 % duplicate moments) and a few timestamps are *split* into a
ball-only moment and a players-only moment. We therefore de-duplicate at the **entity**
level: one row per (game_id, period, unix_ms, team_id, player_id), keeping the first
occurrence in (event_id, moment_idx) order.

The frame index marks continuity: a new ``segment_id`` starts whenever
* the period changes,
* the gap to the previous frame exceeds ``FRAME_GAP_MS`` (a missing frame), or
* the game clock jumps *up* by more than ``CLOCK_JUMP_S`` (an official clock correction).

``unix_ms`` is the time axis; ``game_clock`` is treated as a noisy label.
"""

from __future__ import annotations

import polars as pl

from nbacore.io.raw import BALL_ID

ENTITY_KEY = ["game_id", "period", "unix_ms", "team_id", "player_id"]
FRAME_KEY = ["game_id", "period", "unix_ms"]
FRAME_GAP_MS = 60  # nominal 40 ms with 38-42 ms jitter; >= 2 frames apart means a hole
CLOCK_JUMP_S = 1.0  # game clock moving *up* by more than this = official correction


def dedup_entities(frames: pl.DataFrame) -> pl.DataFrame:
    """Entity-level de-duplication of the long raw table.

    Returns the same columns as the input, sorted by time then entity, with exactly one row per
    entity per timestamp. ``event_id`` / ``moment_idx`` are those of the first occurrence.
    """
    return (
        frames.sort(["game_id", "period", "unix_ms", "event_id", "moment_idx"])
        .unique(subset=ENTITY_KEY, keep="first", maintain_order=True)
        .sort(ENTITY_KEY)
    )


def build_frame_index(entities: pl.DataFrame) -> pl.DataFrame:
    """One row per unique frame with continuity flags.

    Columns: game_id, period, unix_ms, game_clock, shot_clock, n_entities, n_players, has_ball,
    dt_ms (to previous frame within period; null on first), gap, clock_jump, clock_frozen,
    segment_id (int, per game, increasing).
    """
    idx = (
        entities.group_by(FRAME_KEY, maintain_order=True)
        .agg(
            # when a timestamp was split across two moments they may carry different clocks;
            # the most common value (the 10-player moment) wins
            pl.col("game_clock").mode().first().alias("game_clock"),
            pl.col("shot_clock").mode().first().alias("shot_clock"),
            pl.len().cast(pl.Int8).alias("n_entities"),
            (pl.col("team_id") != BALL_ID).sum().cast(pl.Int8).alias("n_players"),
            (pl.col("team_id") == BALL_ID).any().alias("has_ball"),
        )
        .sort(FRAME_KEY)
    )
    dt = (pl.col("unix_ms") - pl.col("unix_ms").shift(1)).over(["game_id", "period"])
    dgc = (pl.col("game_clock") - pl.col("game_clock").shift(1)).over(["game_id", "period"])
    idx = idx.with_columns(dt.alias("dt_ms"), dgc.alias("_dgc"))
    idx = idx.with_columns(
        (pl.col("dt_ms").is_null() | (pl.col("dt_ms") > FRAME_GAP_MS)).alias("gap"),
        (pl.col("_dgc") > CLOCK_JUMP_S).fill_null(False).alias("clock_jump"),
        (pl.col("_dgc").abs() < 0.005).fill_null(False).alias("clock_frozen"),
    )
    new_seg = pl.col("gap") | pl.col("clock_jump")
    idx = idx.with_columns(
        (new_seg.cast(pl.Int32).cum_sum().over("game_id") - 1).alias("segment_id")
    ).drop("_dgc")
    return idx


def dedup_report(raw: pl.DataFrame, entities: pl.DataFrame, index: pl.DataFrame) -> dict:
    """Numbers to log: raw vs unique counts, gaps, clock corrections."""
    n_raw_frames = raw.select(["event_id", "moment_idx"]).unique().height
    n_raw_rows = raw.height
    n_rows = entities.height
    n_frames = index.height
    seg_len = index.group_by("segment_id").len()["len"]
    return {
        "raw_moments": n_raw_frames,
        "raw_entity_rows": n_raw_rows,
        "unique_frames": n_frames,
        "unique_entity_rows": n_rows,
        "moment_dup_fraction": 1 - n_frames / n_raw_frames if n_raw_frames else None,
        "entity_dup_fraction": 1 - n_rows / n_raw_rows if n_raw_rows else None,
        "n_gaps": int((index["gap"] & index["dt_ms"].is_not_null()).sum()),
        "n_clock_jumps": int(index["clock_jump"].sum()),
        "n_segments": int(index["segment_id"].n_unique()),
        "segment_len_median_frames": float(seg_len.median()),
        "segment_len_max_frames": int(seg_len.max()),
        "frames_no_ball": int((~index["has_ball"]).sum()),
        "frames_not_10_players": int((index["n_players"] != 10).sum()),
        "frames_clock_frozen": int(index["clock_frozen"].sum()),
        "shot_clock_null_frames": int(index["shot_clock"].null_count()),
    }
