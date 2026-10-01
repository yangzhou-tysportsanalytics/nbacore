"""Private broadcast video index.

Metadata only — no video is downloaded or stored, and the index is never part of a release.
Location: ``<data root>/private/video_index/anchors.parquet`` (``nbacore.paths.private_dir``).

One row per anchor: a tracking frame (``unix_ms``) shown at ``video_time_s`` in ``video_url``.
Between anchors the broadcast runs in real time, so for a frame u the nearest anchor a at or
before u (same game, same URL, u ≤ ``valid_until_unix_ms`` when set) gives
``video_time_s(u) = video_time_s(a) + (u − unix_ms(a)) / 1000``. Add anchors after replays / ad
breaks and set ``valid_until_unix_ms`` where a stretch of live play ends.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nbacore import paths

ANCHOR_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "unix_ms": pl.Int64,
    "video_url": pl.Utf8,
    "video_time_s": pl.Float64,
    "valid_until_unix_ms": pl.Int64,  # optional
    "source": pl.Utf8,
    "annotator": pl.Utf8,
    "date": pl.Utf8,
}


def index_path() -> Path:
    return paths.private_dir("video_index") / "anchors.parquet"


def validate_anchors(anchors: pl.DataFrame) -> list[str]:
    """Problems in an anchor table (empty list = valid)."""
    problems = [f"missing column {c}" for c in ANCHOR_SCHEMA if c not in anchors.columns]
    if problems:
        return problems
    if anchors.select(["game_id", "unix_ms", "video_url"]).null_count().sum_horizontal()[0]:
        problems.append("null game_id / unix_ms / video_url")
    if (anchors["video_time_s"] < 0).any():
        problems.append("negative video_time_s")
    bad = anchors.filter(
        pl.col("valid_until_unix_ms").is_not_null()
        & (pl.col("valid_until_unix_ms") < pl.col("unix_ms"))
    )
    if bad.height:
        problems.append(f"{bad.height} anchors end before they start")
    dup = anchors.filter(pl.struct("game_id", "video_url", "unix_ms").is_duplicated())
    if dup.height:
        problems.append(f"{dup.height} duplicated anchors (game, url, unix_ms)")
    return problems


def to_video_time(anchors: pl.DataFrame, game_id: str, unix_ms: int) -> tuple[str, float] | None:
    """(video_url, video_time_s) for a tracking frame, or None when no anchor covers it."""
    a = (
        anchors.filter((pl.col("game_id") == game_id) & (pl.col("unix_ms") <= unix_ms))
        .filter(
            pl.col("valid_until_unix_ms").is_null() | (pl.col("valid_until_unix_ms") >= unix_ms)
        )
        .sort("unix_ms")
    )
    if a.height == 0:
        return None
    r = a.row(-1, named=True)
    return r["video_url"], r["video_time_s"] + (unix_ms - r["unix_ms"]) / 1000.0
