"""``epv_full`` view of the ledger (L3 v1.2), for expected-possession-value models.

One row per team possession, from gaining the ball to losing it (ledger rules), with
all sub-segments (the backcourt ones kept; users may drop them) and flags instead of drops:
``is_transition`` (4 s), ``has_jump_ball``, ``has_technical``, ``has_flagrant``,
``has_clear_path``, ``data_gap``; ``ghost_v1_window_uids`` maps to the ghost_v1 windows.
"""

from __future__ import annotations

import polars as pl

EPV_COLUMNS = [
    "game_id",
    "poss_uid",
    "poss_seq",
    "period",
    "offense_team_id",
    "defense_team_id",
    "attacks_left",
    "t_start_ms",
    "t_end_ms",
    "start_type",
    "end_type",
    "start_event_num",
    "end_event_num",
    "end_time_method",
    "points",
    "n_fga",
    "n_fta",
    "n_oreb",
    "tech_points_offense",
    "tech_points_defense",
    "score_margin_offense_start",
    "offense_player_ids",
    "defense_player_ids",
    "lineup_stable",
    "subsegments",
    "t_first_frontcourt_ms",
    "t_first_shot_ms",
    "is_transition",
    "has_jump_ball",
    "has_technical",
    "has_flagrant",
    "has_clear_path",
    "data_gap",
    "ghost_v1_window_uids",
]


def epv_full(ledger: pl.DataFrame, drop_backcourt_subsegments: bool = False) -> pl.DataFrame:
    """The EPV view of the (detailed) ledger. ``drop_backcourt_subsegments`` removes the
    ``backcourt`` entries from ``subsegments`` (they are only flagged by default)."""
    out = ledger.select(EPV_COLUMNS)
    if drop_backcourt_subsegments:
        out = out.with_columns(
            pl.col("subsegments").list.eval(
                pl.element().filter(pl.element().struct.field("kind") != "backcourt")
            )
        )
    return out
