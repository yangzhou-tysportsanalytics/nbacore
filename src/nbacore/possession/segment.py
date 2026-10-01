"""Terminal play-by-play events and pbp-clock -> tracking-time lookup.

Migrated from ghost-defense ``ghost/possession/segment.py`` (commit ed6d754). Only the two
project-neutral helpers are kept here: ``terminal_events`` and ``locate_clock``. The half-court
possession segmentation (``segment_game``, ghost-defense's v1 possession rules) stays in
ghost-defense; the shared possession ledger (L3) will replace it in this project.

Behaviour is unchanged from the source commit.
"""

from __future__ import annotations

import polars as pl

TERMINAL_TYPES: dict[int, str] = {
    1: "fg_made",
    2: "fg_missed",
    5: "turnover",
    6: "foul",
    7: "violation",
    9: "timeout",
    13: "period_end",
}
SHOT_TYPES = (1, 2)
# terminal types whose pbp PLAYER1 team is the offence
OFFENSE_IS_P1 = (1, 2, 5)


def terminal_events(pbp_game: pl.DataFrame) -> pl.DataFrame:
    """Terminal pbp rows in event order with a forward-filled score margin (home - visitor)."""
    # margin *before* the event: forward-fill the scoring rows, then look at the previous row
    ff = pbp_game.sort("event_num").with_columns(
        pl.col("score_margin").forward_fill().shift(1).fill_null(0).alias("margin_home"),
    )
    return ff.filter(pl.col("msg_type").is_in(list(TERMINAL_TYPES))).select(
        [
            "event_num",
            "period",
            "pctime_sec",
            "msg_type",
            "action_type",
            "p1_team_id",
            "p1_id",
            "home_desc",
            "visitor_desc",
            "margin_home",
        ]
    )


def locate_clock(
    idx_period: pl.DataFrame, clock_s: float, after_unix: int | None, bin_s: float = 1.0
) -> int | None:
    """unix_ms of the frame best matching a floored pbp clock reading, after ``after_unix``.

    Candidates are frames with game_clock in [clock, clock + bin). Because official clock
    corrections can make the same reading occur twice, candidates are clustered by
    ``segment_id`` and the first cluster after ``after_unix`` is used; inside it the frame whose
    clock is closest to the bin centre is returned.
    """
    cand = idx_period.filter(
        (pl.col("game_clock") >= clock_s) & (pl.col("game_clock") < clock_s + bin_s)
    )
    if after_unix is not None:
        cand = cand.filter(pl.col("unix_ms") > after_unix)
    if cand.height == 0:
        # clock frozen exactly on the reading, or untracked: fall back to nearest reading
        cand = idx_period.filter((pl.col("game_clock") - clock_s).abs() <= bin_s)
        if after_unix is not None:
            cand = cand.filter(pl.col("unix_ms") > after_unix)
        if cand.height == 0:
            return None
    first_seg = cand["segment_id"].min()
    cand = cand.filter(pl.col("segment_id") == first_seg)
    centre = clock_s + bin_s / 2
    best = cand.with_columns((pl.col("game_clock") - centre).abs().alias("_d")).sort(
        ["_d", "unix_ms"]
    )
    return int(best["unix_ms"][0])
