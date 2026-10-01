"""``with_post_release`` view (L3 v1.2): half-court windows extended past the
shot release, for box-outs.

Applies to ghost_v1 windows (half-court segments): for windows ending in a
FG attempt, ``t_end_post_ms = t_terminal + seconds`` (the release plus up to 3 s); for the other
terminal types it equals ``t_end``. ``post_release_s`` records the parameter.
"""

from __future__ import annotations

import polars as pl

MAX_POST_RELEASE_S = 3.0
SHOT_TYPES = (1, 2)


def with_post_release(windows: pl.DataFrame, seconds: float) -> pl.DataFrame:
    if not 0 <= seconds <= MAX_POST_RELEASE_S:
        raise ValueError(f"seconds must be in [0, {MAX_POST_RELEASE_S}], got {seconds}")
    is_shot = pl.col("terminal_msg_type").is_in(SHOT_TYPES)
    return windows.with_columns(
        pl.when(is_shot)
        .then(pl.col("t_terminal") + int(round(seconds * 1000)))
        .otherwise(pl.col("t_end"))
        .alias("t_end_post_ms"),
        pl.lit(seconds, dtype=pl.Float32).alias("post_release_s"),
    )
