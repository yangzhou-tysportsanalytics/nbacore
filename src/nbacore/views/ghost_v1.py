"""``ghost_v1`` reference view: ghost-defense's half-court possession windows.

Migrated verbatim from ghost-defense ``ghost/possession/segment.py`` (commit ed6d754, unchanged at
d7bd23a; ghost-defense's v1 possession rules): ``SegmentConfig``, ``segment_game`` and helpers. Only
the imports differ — the project-neutral helpers ``terminal_events`` / ``locate_clock`` and the
shot release refinement already live in ``nbacore.possession``. Inputs are nbacore L1 tables,
identical to ghost's (v1.0 parity).

A window is delimited by consecutive terminal pbp events; inside (previous terminal, this terminal
+ ``end_pad_s``] the possession starts at the **last** back→front crossing of the ball. Windows are
rejected (and counted) when the terminal cannot be placed on the tracking timeline, the offence is
ambiguous, the ball never crosses, the frontcourt stretch is shorter than ``min_frontcourt_s``, the
frames are not continuous, or the lineup changes / a player is missing. Windows longer than
``max_window_s`` are cropped (``cropped``); shorter than ``transition_s`` → ``is_transition``.

``ghost_v1`` = ``segment_game``; ``ghost_v1_windows`` adds the stable keys ``window_uid`` and
``poss_uid`` and the ``include_second_chance`` option (off by default: then the output
is identical to ghost-defense's).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nbacore.court import HALF, HOOP_LEFT, HOOP_RIGHT
from nbacore.io.raw import BALL_ID
from nbacore.possession.segment import (
    OFFENSE_IS_P1,
    SHOT_TYPES,
    TERMINAL_TYPES,
    locate_clock,
    terminal_events,
)
from nbacore.possession.shots import ShotTimeConfig, refine_shot_release

__all__ = ["SHOT_TYPES", "SegmentConfig", "ghost_v1", "ghost_v1_windows", "segment_game"]


@dataclass
class SegmentConfig:
    max_window_s: float = 24.0
    transition_s: float = 4.0
    end_pad_s: float = 0.5
    min_frontcourt_s: float = 1.0
    midcourt_margin_ft: float = 3.0  # ball within this of midcourt => offence ambiguous
    max_missing_frac: float = 0.05
    clock_bin_s: float = 1.0  # pbp clock is floored to whole seconds
    refine_shots: bool = True  # replace pbp time of FG attempts by the ball-derived release
    # nbacore addition (optional second-chance windows): a window whose ball
    # never left the frontcourt starts at the last offensive rebound control inside it
    include_second_chance: bool = False
    # nbacore v1.4: ``ShotTimeConfig(shooter_only=True)`` ends FG windows at the
    # corrected (v1.3 L2) release; the default keeps ghost-defense's original rule (last
    # frame in the hand of any offensive player)
    shot_cfg: ShotTimeConfig = field(default_factory=ShotTimeConfig)


@dataclass
class RejectCounts:
    counts: dict[str, int] = field(default_factory=dict)

    last: str | None = None

    def add(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1
        self.last = reason


POSSESSION_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "possession_id": pl.Int32,
    "period": pl.Int8,
    "offense_team_id": pl.Int64,
    "defense_team_id": pl.Int64,
    "attacks_left": pl.Boolean,
    "t_start": pl.Int64,  # unix_ms of the (possibly cropped) crossing
    "t_cross": pl.Int64,  # unix_ms of the actual crossing
    "t_end": pl.Int64,  # unix_ms of terminal event + end_pad
    "t_terminal": pl.Int64,  # unix_ms of the terminal event (release time for shots)
    "t_pbp_terminal": pl.Int64,  # unix_ms matched to the pbp clock reading
    "shot_time_method": pl.Utf8,  # rim | apex | pbp (+ _nohand/_nooff) | null for non-shots
    "shooter_inferred_id": pl.Int64,  # offensive player holding the ball at release
    "duration_s": pl.Float32,
    "terminal_event_num": pl.Int32,
    "terminal_type": pl.Utf8,
    "terminal_msg_type": pl.Int16,
    "terminal_action_type": pl.Int16,
    "terminal_desc": pl.Utf8,
    "terminal_player_id": pl.Int64,
    "prev_terminal_event_num": pl.Int32,
    "game_clock_start": pl.Float32,
    "game_clock_end": pl.Float32,
    "shot_clock_start": pl.Float32,
    "score_margin_offense": pl.Int16,
    "is_transition": pl.Boolean,
    "cropped": pl.Boolean,
    "offense_player_ids": pl.List(pl.Int64),
    "defense_player_ids": pl.List(pl.Int64),
    "n_raw_frames": pl.Int32,
}


# ------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------


def _ball_x_at(entities_ball: pl.DataFrame, unix_ms: int, tol_ms: int = 500) -> float | None:
    near = entities_ball.filter((pl.col("unix_ms") - unix_ms).abs() <= tol_ms)
    if near.height == 0:
        return None
    return float(near["x"].median())


def _crossing_index(front: np.ndarray) -> int | None:
    """Index of the last back->front transition after which the ball stays in front."""
    if front.size == 0 or not front[-1]:
        return None
    # last index where front is False
    back_idx = np.flatnonzero(~front)
    if back_idx.size == 0:
        return None  # never in the backcourt inside the window
    return int(back_idx[-1] + 1)


# ------------------------------------------------------------------------------------------
# main
# ------------------------------------------------------------------------------------------


def segment_game(
    entities: pl.DataFrame,
    frame_index: pl.DataFrame,
    pbp_game: pl.DataFrame,
    direction: pl.DataFrame,
    home_team_id: int,
    cfg: SegmentConfig | None = None,
    oreb_times: dict[int, np.ndarray] | None = None,
    trace: list[dict] | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Segment one game into half-court possessions.

    Returns (possessions table with POSSESSION_SCHEMA, reject counts). With ``trace`` (a list,
    nbacore diagnostic for the shot-release comparison), one dict per terminal event is appended:
    ``event_num``, ``outcome`` (``accepted`` or the reject reason), ``prev_event_num``,
    ``w_start`` (= previous terminal time), ``t_terminal``, ``t_cross``, ``t_start``, ``t_end``,
    ``cropped``. Results are identical with and without ``trace``.
    """
    cfg = cfg or SegmentConfig()
    game_id = str(entities["game_id"][0])
    rejects = RejectCounts()
    ball = entities.filter(pl.col("team_id") == BALL_ID).select(["period", "unix_ms", "x"])
    players = entities.filter(pl.col("team_id") != BALL_ID).select(
        ["period", "unix_ms", "team_id", "player_id"]
    )
    dir_map = {
        (r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)
    }
    teams = sorted({t for t, _ in dir_map})
    if len(teams) != 2:
        raise ValueError(f"{game_id}: expected two teams in direction table, got {teams}")
    other = {teams[0]: teams[1], teams[1]: teams[0]}

    ent_by_period = {
        int(p): entities.filter(pl.col("period") == p)
        for p in entities["period"].unique().to_list()
    }
    terms = terminal_events(pbp_game)
    records: list[dict] = []
    prev_unix: int | None = None
    prev_event: int | None = None
    prev_period: int | None = None
    pid = 0

    for row in terms.iter_rows(named=True):
        t_term = t_cross = t_start = t_end = w_start = None
        cropped = None
        prev_before = prev_event
        n_rec, rejects.last = len(records), None
        try:
            period = int(row["period"])
            if period != prev_period:
                prev_unix, prev_event, prev_period = None, None, period
            idx_p = frame_index.filter(pl.col("period") == period)
            if idx_p.height == 0:
                rejects.add("period_untracked")
                continue
            t_term = locate_clock(idx_p, float(row["pctime_sec"]), prev_unix, cfg.clock_bin_s)
            if t_term is None:
                rejects.add(f"terminal_not_located:{TERMINAL_TYPES[int(row['msg_type'])]}")
                continue
            t_end = t_term + int(cfg.end_pad_s * 1000)
            msg = int(row["msg_type"])

            # ---- offence
            ball_p = ball.filter(pl.col("period") == period)
            if msg in OFFENSE_IS_P1 and row["p1_team_id"] is not None:
                offense = int(row["p1_team_id"])
            else:
                bx = _ball_x_at(ball_p, t_term)
                if bx is None or abs(bx - HALF) < cfg.midcourt_margin_ft:
                    rejects.add("offense_ambiguous")
                    prev_unix, prev_event = t_term, int(row["event_num"])
                    continue
                left_team = [t for t in teams if dir_map.get((t, period)) is True]
                if len(left_team) != 1:
                    rejects.add("direction_missing")
                    prev_unix, prev_event = t_term, int(row["event_num"])
                    continue
                offense = left_team[0] if bx < HALF else other[left_team[0]]
            if offense not in other:
                rejects.add("offense_unknown_team")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            attacks_left = dir_map.get((offense, period))
            if attacks_left is None:
                rejects.add("direction_missing")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue

            # ---- shots: replace the pbp time by the ball-derived release time
            t_pbp = t_term
            method: str | None = None
            shooter_inf: int | None = None
            if msg in SHOT_TYPES:
                method = "pbp"
                if cfg.refine_shots:
                    hoop = HOOP_LEFT if attacks_left else HOOP_RIGHT
                    t_term, shooter_inf, method = refine_shot_release(
                        ent_by_period[period],
                        t_pbp,
                        offense,
                        hoop,
                        cfg.shot_cfg,
                        t_min=prev_unix,
                        # used only with shot_cfg.shooter_only (nbacore v1.4)
                        shooter_id=int(row["p1_id"]) if row["p1_id"] else None,
                    )
                    t_end = t_term + int(cfg.end_pad_s * 1000)

            # ---- window frames and crossing
            w_start = prev_unix if prev_unix is not None else int(idx_p["unix_ms"].min())
            win = ball_p.filter((pl.col("unix_ms") > w_start) & (pl.col("unix_ms") <= t_end)).sort(
                "unix_ms"
            )
            if win.height == 0:
                rejects.add("no_ball_frames")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            x = win["x"].to_numpy()
            front = (x < HALF) if attacks_left else (x > HALF)
            ci = _crossing_index(front)
            t_second = None
            if ci is None and cfg.include_second_chance and front.all() and oreb_times:
                ob = oreb_times.get(period)
                if ob is not None:
                    c = ob[(ob > w_start) & (ob <= t_end - int(cfg.min_frontcourt_s * 1000))]
                    if c.size:
                        t_second = int(c[-1])
            if ci is None and t_second is None:
                # finer accounting: paired events with an (almost) empty window, second chances
                # that never left the frontcourt, or windows ending in the backcourt
                if (t_end - w_start) < 1500:
                    why = "no_crossing:short_window"
                elif front.all():
                    why = "no_crossing:always_frontcourt"
                else:
                    why = "no_crossing:ends_backcourt"
                rejects.add(f"{why}:{TERMINAL_TYPES[msg]}")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            t_cross = t_second if t_second is not None else int(win["unix_ms"][ci])
            if (t_end - t_cross) < cfg.min_frontcourt_s * 1000:
                rejects.add("frontcourt_too_short")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            cropped = False
            t_start = t_cross
            if (t_end - t_cross) > cfg.max_window_s * 1000:
                t_start = t_end - int(cfg.max_window_s * 1000)
                cropped = True

            # ---- continuity and lineup
            fi = idx_p.filter((pl.col("unix_ms") >= t_start) & (pl.col("unix_ms") <= t_end))
            if fi.height < 2 or fi["segment_id"].n_unique() != 1:
                rejects.add("gap_in_window")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue
            pw = players.filter(
                (pl.col("period") == period)
                & (pl.col("unix_ms") >= t_start)
                & (pl.col("unix_ms") <= t_end)
            )
            n_frames = fi.height
            presence = pw.group_by(["team_id", "player_id"]).len().sort(["team_id", "player_id"])
            good = presence.filter(pl.col("len") >= (1 - cfg.max_missing_frac) * n_frames)
            off_ids = good.filter(pl.col("team_id") == offense)["player_id"].to_list()
            def_ids = good.filter(pl.col("team_id") == other[offense])["player_id"].to_list()
            if presence.height != 10 or len(off_ids) != 5 or len(def_ids) != 5:
                rejects.add("lineup_change" if presence.height > 10 else "missing_players")
                prev_unix, prev_event = t_term, int(row["event_num"])
                continue

            # ---- context
            gc_start = float(fi["game_clock"][0])
            gc_end = float(fi["game_clock"][-1])
            sc_start = fi["shot_clock"][0]
            margin_home = int(row["margin_home"])  # before the terminal event (see terminal_events)
            margin_off = margin_home if offense == home_team_id else -margin_home
            duration = (t_end - t_start) / 1000.0
            pid += 1
            records.append(
                {
                    "game_id": game_id,
                    "possession_id": pid,
                    "period": period,
                    "offense_team_id": offense,
                    "defense_team_id": other[offense],
                    "attacks_left": bool(attacks_left),
                    "t_start": t_start,
                    "t_cross": t_cross,
                    "t_end": t_end,
                    "t_terminal": t_term,
                    "t_pbp_terminal": t_pbp,
                    "shot_time_method": method,
                    "shooter_inferred_id": shooter_inf,
                    "duration_s": duration,
                    "terminal_event_num": int(row["event_num"]),
                    "terminal_type": TERMINAL_TYPES[msg],
                    "terminal_msg_type": msg,
                    "terminal_action_type": row["action_type"],
                    "terminal_desc": row["home_desc"] or row["visitor_desc"],
                    "terminal_player_id": row["p1_id"] if msg in OFFENSE_IS_P1 else None,
                    "prev_terminal_event_num": prev_event,
                    "game_clock_start": gc_start,
                    "game_clock_end": gc_end,
                    "shot_clock_start": None if sc_start is None else float(sc_start),
                    "score_margin_offense": margin_off,
                    "is_transition": (t_end - t_cross) < cfg.transition_s * 1000,
                    "cropped": cropped,
                    "offense_player_ids": sorted(off_ids),
                    "defense_player_ids": sorted(def_ids),
                    "n_raw_frames": n_frames,
                }
            )
            prev_unix, prev_event = t_term, int(row["event_num"])

        finally:
            if trace is not None:
                trace.append(
                    {
                        "event_num": int(row["event_num"]),
                        "terminal_type": TERMINAL_TYPES.get(int(row["msg_type"])),
                        "outcome": "accepted" if len(records) > n_rec else rejects.last,
                        "prev_event_num": prev_before,
                        "w_start": w_start,
                        "t_terminal": t_term,
                        "t_cross": t_cross,
                        "t_start": t_start,
                        "t_end": t_end,
                        "cropped": cropped,
                    }
                )

    table = pl.DataFrame(records, schema=POSSESSION_SCHEMA)
    return table, rejects.counts


ghost_v1 = segment_game


def ghost_v1_windows(
    entities: pl.DataFrame,
    frame_index: pl.DataFrame,
    pbp_game: pl.DataFrame,
    direction: pl.DataFrame,
    home_team_id: int,
    ledger: pl.DataFrame,
    shots: pl.DataFrame | None = None,
    cfg: SegmentConfig | None = None,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """``segment_game`` plus the stable keys: ``window_uid =
    <game_id>:<terminal_event_num>`` and the ``poss_uid`` of the ledger possession the window
    overlaps most (``overlap_frac``, ``offense_match``). ``shots`` (L2 shot events) gives the
    offensive rebound times for ``include_second_chance``."""
    from nbacore.ledger import attach_poss_uid

    cfg = cfg or SegmentConfig()
    oreb = None
    if cfg.include_second_chance and shots is not None:
        o = shots.filter(pl.col("offensive_rebound") & pl.col("t_control_ms").is_not_null())
        oreb = {int(p): np.sort(g["t_control_ms"].to_numpy()) for (p,), g in o.group_by("period")}
    win, rej = segment_game(
        entities, frame_index, pbp_game, direction, home_team_id, cfg, oreb_times=oreb
    )
    win = attach_poss_uid(win, ledger).with_columns(
        (pl.col("game_id") + ":" + pl.col("terminal_event_num").cast(pl.Utf8)).alias("window_uid")
    )
    return win, rej
