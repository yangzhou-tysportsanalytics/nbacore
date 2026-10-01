"""Per-possession detail of the L3 ledger: sub-segments, lineups, score, flags (L3, v1.2).

``possession_details(ledger, ...)`` adds to the timed ledger (``ledger.ledger_times``):

* ``subsegments``: list of ``{kind, t_start_ms, t_end_ms, ref_event_uid}`` sorted by time —
  ``backcourt`` / ``frontcourt`` (ball half for the offence; runs shorter than ``MIN_RUN_MS``
  merged into the previous run), ``shot`` (release → rim contact / landing), ``offensive_rebound``
  (control instant), ``second_chance`` (offensive rebound → next shot release or possession end),
  ``free_throws`` (one per trip: first → last attempt; ref = ``<game_id>:pbp:<foul event_num>``),
  ``inbound`` (release → catch), ``dead_ball`` (clock stops, clipped to the possession);
* ``offense_player_ids`` / ``defense_player_ids`` (on court at the first frame of the
  possession), ``lineup_stable`` (the same ten players at its last frame);
* ``score_margin_offense_start`` (offence − defence after the start event);
* flags: ``is_transition`` (first frontcourt entry → first shot release, or the
  possession end, shorter than ``transition_s``), ``has_jump_ball``, ``has_technical``,
  ``has_flagrant``, ``has_clear_path`` (pbp events of the possession), ``data_gap`` (a tracking
  hole during live play: the game clock ran more than ``LIVE_GAP_CLOCK_S`` across it), ``ghost_v1_window_uids`` (windows
  of the ghost_v1 view linked to it).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nbacore.ledger import order_pbp

MIN_RUN_MS = 500
LIVE_GAP_CLOCK_S = 0.5
TRANSITION_S = 4.0  # ghost-defense's transition threshold
HALF_X = 47.0

SUBSEG = pl.Struct(
    {"kind": pl.Utf8, "t_start_ms": pl.Int64, "t_end_ms": pl.Int64, "ref_event_uid": pl.Utf8}
)
DETAIL_SCHEMA: dict[str, pl.DataType] = {
    "subsegments": pl.List(SUBSEG),
    "offense_player_ids": pl.List(pl.Int64),
    "defense_player_ids": pl.List(pl.Int64),
    "lineup_stable": pl.Boolean,
    "score_margin_offense_start": pl.Int16,
    "t_first_frontcourt_ms": pl.Int64,
    "t_first_shot_ms": pl.Int64,
    "is_transition": pl.Boolean,
    "has_jump_ball": pl.Boolean,
    "has_technical": pl.Boolean,
    "has_flagrant": pl.Boolean,
    "has_clear_path": pl.Boolean,
    "data_gap": pl.Boolean,
    "ghost_v1_window_uids": pl.List(pl.Utf8),
}
TECH_ACTIONS = (11, 12, 13, 16, 17, 18, 19, 25, 30)  # pbp foul action types (technicals)


def event_possession(pbp_game: pl.DataFrame, poss: pl.DataFrame) -> dict[int, int]:
    """pbp event_num → poss_seq of the ledger possession it belongs to (ledger order; the end
    event of a possession belongs to it)."""
    ends = poss.select("poss_seq", "period", "end_event_num").rows()
    owner: dict[int, int] = {}
    k = 0
    for ev in order_pbp(pbp_game).select("event_num", "period").iter_rows():
        en, per = int(ev[0]), int(ev[1])
        while k < len(ends) - 1 and ends[k][1] < per:
            k += 1
        if not ends:
            break
        owner[en] = ends[k][0]
        if ends[k][2] is not None and en == ends[k][2] and k < len(ends) - 1:
            k += 1
    return owner


def _live_gap(fi_period: pl.DataFrame) -> np.ndarray:
    """Per frame: the frame follows a tracking hole across which the game clock ran more than
    ``LIVE_GAP_CLOCK_S`` (live play without data; holes in dead balls do not count)."""
    gc = fi_period["game_clock"].to_numpy().astype(np.float64)
    ran = np.r_[0.0, gc[:-1] - gc[1:]] > LIVE_GAP_CLOCK_S
    return fi_period["gap"].to_numpy() & ran


def _half_runs(t: np.ndarray, front: np.ndarray) -> list[tuple[bool, int, int]]:
    """(is_front, t_start, t_end) runs; runs shorter than MIN_RUN_MS merged into the previous."""
    if t.size == 0:
        return []
    cut = np.flatnonzero(front[1:] != front[:-1]) + 1
    starts, ends = np.r_[0, cut], np.r_[cut, t.size]
    runs: list[list] = []
    for s, e in zip(starts, ends, strict=True):
        t0, t1 = int(t[s]), int(t[e - 1])
        if runs and (t1 - t0 < MIN_RUN_MS or runs[-1][0] == bool(front[s])):
            runs[-1][2] = t1
            continue
        runs.append([bool(front[s]), t0, t1])
    return [tuple(r) for r in runs]


def possession_details(
    ledger: pl.DataFrame,
    pbp_game: pl.DataFrame,
    align_game: pl.DataFrame,
    frame_index: pl.DataFrame,
    frames: pl.DataFrame,
    events: pl.DataFrame,
    fts: pl.DataFrame,
    home_team_id: int,
    ghost_windows: pl.DataFrame | None = None,
    transition_s: float = TRANSITION_S,
) -> pl.DataFrame:
    gid = ledger["game_id"][0]
    fi = frame_index.sort("period", "unix_ms")
    ball = frames.filter(pl.col("player_id") == -1).sort("period", "unix_ms")
    players = frames.filter(pl.col("player_id") != -1).select(
        "period", "unix_ms", "team_id", "player_id"
    )
    by_p_ball = {
        int(p): (g["unix_ms"].to_numpy(), g["x"].to_numpy()) for (p,), g in ball.group_by("period")
    }
    by_p_fi = {
        int(p): (g["unix_ms"].to_numpy(), _live_gap(g))
        for (p,), g in fi.group_by("period", maintain_order=True)
    }
    shots = events.filter(pl.col("event_type") == "shot_release")
    inb = events.filter(pl.col("event_type") == "inbound")
    stops = events.filter(pl.col("event_type") == "clock_stop")
    at = dict(zip(align_game["event_num"].to_list(), align_game["unix_ms"].to_list(), strict=True))

    # pbp flags per possession
    owner = event_possession(pbp_game, ledger)
    flags: dict[int, set[str]] = {}
    for ev in pbp_game.iter_rows(named=True):
        seq = owner.get(int(ev["event_num"]))
        if seq is None:
            continue
        msg, act = int(ev["msg_type"]), ev["action_type"]
        f = flags.setdefault(seq, set())
        if msg == 10:
            f.add("jump_ball")
        if msg == 6 and act in TECH_ACTIONS:
            f.add("technical")
        if msg == 6 and act in (14, 15):
            f.add("flagrant")
        if msg == 6 and act == 9:
            f.add("clear_path")

    # score margin (home − visitor) after each event, in ledger order
    ordered = order_pbp(pbp_game).with_columns(
        (pl.col("score_home").cast(pl.Int32) - pl.col("score_visitor").cast(pl.Int32))
        .forward_fill()
        .fill_null(0)
        .alias("_m")
    )
    margin_after = dict(zip(ordered["event_num"].to_list(), ordered["_m"].to_list(), strict=True))
    period_first = {
        int(p): int(g["_m"][0]) for (p,), g in ordered.group_by("period", maintain_order=True)
    }

    # lineups at the first / last frame of each possession
    frame_times = players.select("period", "unix_ms").unique().sort("period", "unix_ms")
    by_p_ft = {int(p): g["unix_ms"].to_numpy() for (p,), g in frame_times.group_by("period")}

    def lineup(period: int, t: int, first: bool) -> dict[int, list[int]] | None:
        ts = by_p_ft.get(period)
        if ts is None or ts.size == 0:
            return None
        i = int(np.searchsorted(ts, t, side="left" if first else "right")) - (0 if first else 1)
        if i < 0 or i >= ts.size:
            return None
        rows = players.filter((pl.col("period") == period) & (pl.col("unix_ms") == int(ts[i])))
        out: dict[int, list[int]] = {}
        for team, pid in rows.select("team_id", "player_id").iter_rows():
            out.setdefault(int(team), []).append(int(pid))
        return {k: sorted(v) for k, v in out.items()}

    win_by_poss: dict[str, list[str]] = {}
    if ghost_windows is not None and ghost_windows.height:
        for u, w in ghost_windows.select("poss_uid", "window_uid").iter_rows():
            if u is not None:
                win_by_poss.setdefault(u, []).append(w)

    rows = []
    for r in ledger.iter_rows(named=True):
        per, off = int(r["period"]), r["offense_team_id"]
        t0, t1 = r["t_start_ms"], r["t_end_ms"]
        seq = r["poss_seq"]
        fl = flags.get(seq, set())
        base = {
            "subsegments": [],
            "offense_player_ids": None,
            "defense_player_ids": None,
            "lineup_stable": None,
            "score_margin_offense_start": None,
            "t_first_frontcourt_ms": None,
            "t_first_shot_ms": None,
            "is_transition": None,
            "has_jump_ball": "jump_ball" in fl,
            "has_technical": "technical" in fl,
            "has_flagrant": "flagrant" in fl,
            "has_clear_path": "clear_path" in fl,
            "data_gap": None,
            "ghost_v1_window_uids": win_by_poss.get(r["poss_uid"], []),
        }
        se = r["start_event_num"]
        m_home = margin_after.get(int(se)) if se is not None else period_first.get(per)
        if m_home is not None and off is not None:
            base["score_margin_offense_start"] = m_home if off == home_team_id else -m_home
        if t0 is None or t1 is None or off is None:
            rows.append(base)
            continue
        sub: list[dict] = []
        # court halves
        att = r["attacks_left"]
        if per in by_p_ball and att is not None:
            bt, bx = by_p_ball[per]
            sel = (bt >= t0) & (bt <= t1)
            for is_front, a, b in _half_runs(bt[sel], (bx[sel] < HALF_X) == att):
                sub.append(
                    {
                        "kind": "frontcourt" if is_front else "backcourt",
                        "t_start_ms": a,
                        "t_end_ms": b,
                        "ref_event_uid": None,
                    }  # fmt: skip
                )
        # shots, offensive rebounds, second chances
        sh = shots.filter(
            (pl.col("period") == per)
            & pl.col("t_release_ms").is_between(t0, t1)
            & (pl.col("shooter_team_id") == off)
        ).sort("t_release_ms")
        releases = sh["t_release_ms"].to_list()
        for s in sh.iter_rows(named=True):
            end = s["t_rim_ms"] or s["t_landing_ms"] or s["t_release_ms"]
            sub.append({"kind": "shot", "t_start_ms": s["t_release_ms"],
                        "t_end_ms": max(end, s["t_release_ms"]), "ref_event_uid": s["event_uid"]})  # fmt: skip
            c = s["t_control_ms"]
            if s["offensive_rebound"] and c is not None and t0 <= c <= t1:
                nxt = [x for x in releases if x > c]
                sub.append({"kind": "offensive_rebound", "t_start_ms": c, "t_end_ms": c,
                            "ref_event_uid": s["event_uid"]})  # fmt: skip
                sub.append({"kind": "second_chance", "t_start_ms": c,
                            "t_end_ms": nxt[0] if nxt else t1, "ref_event_uid": s["event_uid"]})  # fmt: skip
        # free-throw trips of the offence
        trip: dict[int | None, list[int]] = {}
        for f in fts.filter(
            (pl.col("period") == per)
            & (pl.col("team_id") == off)
            & (pl.col("ft_kind") != "technical")
        ).iter_rows(named=True):
            t = at.get(f["event_num"])
            if t is not None and t0 <= t <= t1:
                trip.setdefault(f["linked_foul_event_num"], []).append(int(t))
        for foul, ts in trip.items():
            sub.append({"kind": "free_throws", "t_start_ms": min(ts), "t_end_ms": max(ts),
                        "ref_event_uid": f"{gid}:pbp:{foul}" if foul is not None else None})  # fmt: skip
        for i in inb.filter(
            (pl.col("period") == per) & pl.col("t_start_ms").is_between(t0, t1)
        ).iter_rows(named=True):
            sub.append({"kind": "inbound", "t_start_ms": i["t_start_ms"],
                        "t_end_ms": i["t_catch_ms"] or i["t_end_ms"], "ref_event_uid": i["event_uid"]})  # fmt: skip
        for s in stops.filter(
            (pl.col("period") == per) & (pl.col("t_start_ms") < t1) & (pl.col("t_end_ms") > t0)
        ).iter_rows(named=True):
            sub.append({"kind": "dead_ball", "t_start_ms": max(s["t_start_ms"], t0),
                        "t_end_ms": min(s["t_end_ms"], t1), "ref_event_uid": s["event_uid"]})  # fmt: skip
        sub.sort(key=lambda d: (d["t_start_ms"], d["t_end_ms"]))
        base["subsegments"] = sub
        # transition flag
        fronts = [d["t_start_ms"] for d in sub if d["kind"] == "frontcourt"]
        t_front = fronts[0] if fronts else None
        t_shot = releases[0] if releases else None
        base["t_first_frontcourt_ms"], base["t_first_shot_ms"] = t_front, t_shot
        if t_front is not None:
            base["is_transition"] = ((t_shot or t1) - t_front) < transition_s * 1000
        # lineups
        a, b = lineup(per, t0, True), lineup(per, t1, False)
        if a:
            base["offense_player_ids"] = a.get(off)
            base["defense_player_ids"] = a.get(r["defense_team_id"])
            base["lineup_stable"] = b is not None and a == b
        # tracking hole during live play
        if per in by_p_fi:
            ft, live_gap = by_p_fi[per]
            sel = (ft > t0) & (ft <= t1)
            base["data_gap"] = bool(live_gap[sel].any()) if sel.any() else t1 > t0
        rows.append(base)
    return ledger.hstack(pl.DataFrame(rows, schema=DETAIL_SCHEMA))
