"""pbp → tracking time alignment (L2).

One row per pbp event. Only the game clock is used, never the tracking ``event_id``, which is
not a reliable time key.

| pbp | anchor (``unix_ms``) | ``method`` |
|---|---|---|
| FG attempt (1, 2) | shot release (``shot_events``) | ``shot_release`` |
| free throw (3) | the n-th rim-reaching flight of the shooter inside the stop at that clock | ``ft_flight`` |
| free throw (3) without its flight (v1.3) | inside the stop: the last FT at the stop end (clock restart), earlier ones just after the onset, in order | ``ft_stop_est`` |
| rebound (4) after a FG miss | first control after the shot (``shot_events.t_control_ms``) if the controlling player / team matches | ``control`` |
| rebound (4) of a player after a FG miss, first control not by him | the pbp rebounder catching the ball at the end of a flight ≤ 6 s after the release (quality 3) | ``control`` |
| rebound (4) after a missed last free throw | first handler after the FT flight, if player / team match (quality 3) | ``control`` |
| rebound (4) after a missed free throw that is not the last | the free throw itself: a dead ball, pbp books a team rebound (quality 3) | ``ft_dead_ball`` |
| turnover (5) | catch of a live intercepted pass by the other team, passer = the turnover player, catch clock within [pbp s − 1, pbp s + 2] | ``pass`` |
| foul, violation, turnover, timeout, jump ball, replay, ejection (6, 7, 5, 9, 10, 18, 11) | onset of a clock stop at that clock | ``clock_stop`` |
| everything else, or no anchor found | ``locate_clock`` (frame whose clock is nearest the centre of the 1 s pbp bin) | ``clock_match`` |

Free-throw flights are assigned from the end of the trip (v1.3): when the stop holds fewer
flights than free throws left (a tracking hole early in the stop), the missing flights are the
early ones. Before, the first flight went to FT 1 and the last FT fell back to ``clock_match`` —
the first frame at that clock, i.e. live play *before* the foul (24 % of last free throws on the
v1.3 L2 build; the ledger then ended those possessions before the foul).

Shot-release, free-throw and rebound anchors must be plausible: game clock at the anchor − pbp
second within ``ANCHOR_CLOCK_WINDOW_S`` = [−3, +8] s; otherwise the next rule applies. (The walk is
in event-number order, so a late-inserted rebound row would otherwise be paired with a shot of
another moment.)

Stop matching: stops with ``pbp_s − 0.5 ≤ game_clock_at_stop < pbp_s + 1.5`` in the period; one
→ used; several → the one nearest ``pbp_s + 0.5``. Quality:

* ``anchor_quality`` (0–4): 4 = anchored on a tracking event with an exact time (unique
  stop with exact onset; shot release by rim / apex; interception; FT flight; rebound control);
  3 = unique stop whose onset lies in a tracking hole (estimated); 2 = several candidate stops;
  1 = ``clock_match``; 0 = not located.
* ``match_level``: ``tracking_event`` (quality 2–4), ``clock_only`` (1), ``unmatched`` (0).

Also ``residual_s`` = pbp second − game clock at the anchor, ``n_stops_within_2s``, the linked
event uid, and ``foul_class`` / ``foul_detail`` for fouls. ``align_game`` also returns the stop
table with ``stop_cause_hint`` and the pass table with ``pbp_turnover_event_num``.
"""

from __future__ import annotations

import re

import polars as pl

from nbacore.possession.segment import locate_clock

# EVENTMSGACTIONTYPE of fouls (msg 6), checked against the descriptions of the 2015-16 file;
# 10 and 16 have no description: every row names two players of opposite teams (double fouls).
FOUL_DETAIL: dict[int, tuple[str, str]] = {
    1: ("personal", "personal_nonshooting"),
    2: ("shooting", "shooting"),
    3: ("loose_ball", "loose_ball"),
    4: ("offensive", "offensive"),
    5: ("inbound", "personal_nonshooting"),
    6: ("away_from_play", "away_from_play"),
    9: ("clear_path", "clear_path"),
    10: ("double_personal", "double_personal"),
    11: ("technical", "technical"),
    12: ("non_unsportsmanlike_technical", "technical"),
    13: ("hanging_technical", "technical"),
    14: ("flagrant_1", "flagrant_1"),
    15: ("flagrant_2", "flagrant_2"),
    16: ("double_technical", "double_technical"),
    17: ("defensive_3_seconds", "technical"),
    18: ("delay_of_game", "technical"),
    19: ("taunting", "technical"),
    25: ("excess_timeout", "technical"),
    26: ("offensive_charge", "offensive"),
    27: ("personal_block", "personal_nonshooting"),
    28: ("personal_take", "personal_nonshooting"),
    29: ("shooting_block", "shooting"),
    30: ("too_many_players", "technical"),
}
STOP_TYPES = (5, 6, 7, 9, 10, 11, 18)
# plausible game clock at an anchor minus the (floored) pbp clock: releases and rebounds happen up
# to ~5 s before the scorer's clock reading, stops within a second of it. Anchors outside are
# dropped (late-inserted pbp rows paired with a shot of another moment by event-number order).
ANCHOR_CLOCK_WINDOW_S = (-3.0, 8.0)
CAUSE = {
    6: "foul",
    7: "violation",
    5: "turnover",
    9: "timeout",
    10: "jump_ball",
    18: "replay",
    11: "ejection",
}
FT_RE = re.compile(r"Free Throw (?:Technical|Flagrant|Clear Path)?\s*(\d) of (\d)", re.I)

ALIGN_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "event_num": pl.Int32,
    "period": pl.Int8,
    "msg_type": pl.Int16,
    "action_type": pl.Int16,
    "p1_id": pl.Int64,
    "p2_id": pl.Int64,
    "p3_id": pl.Int64,
    "pctime_sec": pl.Float32,
    "unix_ms": pl.Int64,
    "method": pl.Utf8,
    "match_level": pl.Utf8,
    "anchor_quality": pl.Int8,
    "residual_s": pl.Float32,
    "n_stops_within_2s": pl.Int16,
    "stop_event_uid": pl.Utf8,
    "linked_event_uid": pl.Utf8,
    "foul_class": pl.Utf8,
    "foul_detail": pl.Utf8,
    "ft_n": pl.Int8,
    "ft_m": pl.Int8,
}


def _level(q: int) -> str:
    return "tracking_event" if q >= 2 else ("clock_only" if q == 1 else "unmatched")


def align_game(
    pbp_game: pl.DataFrame,
    frame_index: pl.DataFrame,
    shots: pl.DataFrame,
    stops: pl.DataFrame,
    flights: pl.DataFrame,
    passes_df: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    gid = frame_index["game_id"][0]
    fi = frame_index.sort(["period", "unix_ms"])
    idx = {int(p): g for (p,), g in fi.group_by(["period"], maintain_order=True)}
    shot_by_ev = {r["pbp_event_num"]: r for r in shots.iter_rows(named=True)}
    rim = flights.filter(pl.col("reaches_rim")).sort("t_start_ms")
    ic = passes_df.filter(pl.col("intercepted")).join(
        fi.select("period", "unix_ms", "game_clock"),
        left_on=["period", "t_catch_ms"],
        right_on=["period", "unix_ms"],
        how="left",
    )
    rows = []
    last_shot = None  # previous FG attempt row (for rebounds)
    last_ft = None  # previous free throw: {t, flight, miss, n, m} (for rebounds)
    ft_used: set[int] = set()  # rim flights already assigned to a free throw
    pbp = pbp_game.sort("event_num").with_columns(
        pl.coalesce("home_desc", "visitor_desc").alias("_d")
    )
    for ev in pbp.iter_rows(named=True):
        period, msg, pct = int(ev["period"]), int(ev["msg_type"]), float(ev["pctime_sec"])
        rec = {
            "game_id": gid,
            "event_num": int(ev["event_num"]),
            "period": period,
            "msg_type": msg,
            "action_type": ev["action_type"],
            "p1_id": ev["p1_id"],
            "p2_id": ev["p2_id"],
            "p3_id": ev["p3_id"],
            "pctime_sec": pct,
        }
        if msg == 6 and ev["action_type"] in FOUL_DETAIL:
            rec["foul_detail"], rec["foul_class"] = FOUL_DETAIL[int(ev["action_type"])]
        idx_p = idx.get(period)
        cand = stops.filter(
            (pl.col("period") == period)
            & (pl.col("game_clock_at_stop") >= pct - 0.5)
            & (pl.col("game_clock_at_stop") < pct + 1.5)
        )
        rec["n_stops_within_2s"] = stops.filter(
            (pl.col("period") == period) & ((pl.col("game_clock_at_stop") - pct).abs() <= 2)
        ).height
        t, method, q, link, stop_uid = None, None, 0, None, None

        if msg in (1, 2):
            s = shot_by_ev.get(int(ev["event_num"]))
            last_shot, last_ft = s, None
            if s is not None and s["t_release_ms"] is not None:
                t, method, link = s["t_release_ms"], "shot_release", s["event_uid"]
                q = (
                    4
                    if s["method"] in ("rim", "apex")
                    else (2 if s["method"].startswith(("rim", "apex")) else 1)
                )
        elif msg == 3:
            m = FT_RE.search(ev["_d"] or "")
            if m:
                rec["ft_n"], rec["ft_m"] = int(m.group(1)), int(m.group(2))
            last_shot = None
            last_ft = {
                "t": None,
                "flight": None,
                "miss": "MISS" in (ev["_d"] or ""),
                "n": rec.get("ft_n"),
                "m": rec.get("ft_m"),
                "uid": None,
            }
            if cand.height and ev["p1_id"]:
                st = cand.sort((pl.col("game_clock_at_stop") - (pct + 0.5)).abs()).row(
                    0, named=True
                )
                t_hi = st["t_end_ms"] or st["t_start_ms"] + 120_000
                ff = rim.filter(
                    (pl.col("period") == period)
                    & (pl.col("from_id") == ev["p1_id"])
                    & (pl.col("t_start_ms") >= st["t_onset_est_ms"])
                    & (pl.col("t_start_ms") <= t_hi + 3000)
                )
                ff = ff.filter(~pl.col("t_start_ms").is_in(list(ft_used)))
                n_, m_ = rec.get("ft_n"), rec.get("ft_m")
                left = (m_ - n_ + 1) if n_ is not None and m_ is not None and m_ >= n_ else 1
                if ff.height >= left:
                    t = int(ff["t_start_ms"][0])
                    ft_used.add(t)
                    method, q, stop_uid = "ft_flight", 4, st["event_uid"]
                    last_ft.update(t=t, flight=ff.row(0, named=True), uid=st["event_uid"])
                elif st["t_end_ms"] is not None and n_ is not None and m_ is not None:
                    # this flight is missing (tracking hole early in the stop): the last FT
                    # just before the clock restarts, earlier ones just after the onset
                    t = (
                        int(st["t_end_ms"]) - 40
                        if n_ == m_
                        else int(st["t_onset_est_ms"]) + 40 * n_
                    )
                    method, q, stop_uid = "ft_stop_est", 2, st["event_uid"]
                    last_ft.update(t=t, uid=st["event_uid"])
        elif msg == 4 and last_shot is not None and last_shot.get("t_control_ms") is not None:
            same_player = ev["p1_id"] == last_shot["control_id"]
            team_reb = ev["p1_type"] not in (4, 5) and ev["p1_id"] == last_shot["control_team_id"]
            if same_player or team_reb:
                t, method, q = last_shot["t_control_ms"], "control", 4
                link = last_shot["event_uid"]
        elif msg == 4 and last_ft is not None and last_ft["miss"] and last_ft["t"] is not None:
            n, m_ = last_ft["n"], last_ft["m"]
            if n is not None and m_ is not None and n < m_:
                # a missed free throw that is not the last one: the ball is dead, pbp books a
                # team rebound; anchor it on the free throw itself
                t, method, q, link = last_ft["t"], "ft_dead_ball", 3, last_ft["uid"]
            elif last_ft["flight"] is not None:
                f = last_ft["flight"]
                same_player = f["to_id"] is not None and ev["p1_id"] == f["to_id"]
                team_reb = ev["p1_type"] not in (4, 5) and ev["p1_id"] == f["to_team_id"]
                if f["t_to_ms"] is not None and (same_player or team_reb):
                    t, method, q, link = f["t_to_ms"], "control", 3, last_ft["uid"]
        if t is not None and not _plausible(idx_p, t, pct):
            t, method, q, link, stop_uid = None, None, 0, None, None
        if (
            t is None
            and msg == 4
            and ev["p1_type"] in (4, 5)
            and last_shot is not None
            and last_shot["t_release_ms"] is not None
        ):
            # the pbp rebounder catches the ball at the end of a flight within 6 s of the release
            # (a tip or a tracking hole can make someone else the first control)
            r0 = int(last_shot["t_release_ms"])
            hit = flights.filter(
                (pl.col("period") == period)
                & (pl.col("to_id") == ev["p1_id"])
                & pl.col("t_to_ms").is_between(r0, r0 + 6000)
            ).sort("t_to_ms")
            if hit.height:
                t, method, q, link = int(hit["t_to_ms"][0]), "control", 3, last_shot["event_uid"]
                if not _plausible(idx_p, t, pct):
                    t, method, q, link = None, None, 0, None
        if t is None and msg == 5 and ev["p1_id"]:
            hit = ic.filter(
                (pl.col("period") == period)
                & (pl.col("passer_id") == ev["p1_id"])
                & (pl.col("game_clock") >= pct - 1)
                & (pl.col("game_clock") <= pct + 2)
            )
            if hit.height:
                r = hit.row(0, named=True)
                t, method, q, link = r["t_catch_ms"], "pass", 4, r["event_uid"]
        if t is None and msg in STOP_TYPES and cand.height:
            st = cand.sort((pl.col("game_clock_at_stop") - (pct + 0.5)).abs()).row(0, named=True)
            t, method, stop_uid = st["t_onset_est_ms"], "clock_stop", st["event_uid"]
            q = 2 if cand.height > 1 else (4 if st["onset_exact"] else 3)
        if t is None and idx_p is not None:
            t = locate_clock(idx_p, pct, None)
            if t is not None:
                method, q = "clock_match", 1
        if t is None:
            method, q = "unlocated", 0
        rec.update(unix_ms=t, method=method, anchor_quality=q, match_level=_level(q))
        rec["stop_event_uid"] = stop_uid
        rec["linked_event_uid"] = link or stop_uid
        if t is not None and idx_p is not None:
            k = idx_p.filter(pl.col("unix_ms") <= t)
            gc = float(k["game_clock"][-1]) if k.height else None
            rec["residual_s"] = pct - gc if gc is not None else None
        rows.append(rec)
    align = pl.DataFrame(rows, schema=ALIGN_SCHEMA)
    return align, _stop_causes(stops, align, shots), _link_turnovers(passes_df, align)


def _plausible(idx_p: pl.DataFrame | None, t: int, pct: float) -> bool:
    """Game clock at ``t`` minus the pbp clock inside ``ANCHOR_CLOCK_WINDOW_S``."""
    if idx_p is None:
        return True
    k = idx_p.filter(pl.col("unix_ms") >= t).head(1)
    if k.height == 0 or k["game_clock"][0] is None:
        return True
    lo, hi = ANCHOR_CLOCK_WINDOW_S
    return lo <= float(k["game_clock"][0]) - pct <= hi


def _stop_causes(stops: pl.DataFrame, align: pl.DataFrame, shots: pl.DataFrame) -> pl.DataFrame:
    """``stop_cause_hint`` from the pbp events anchored on each stop (priority order), else a
    made FG in the last 2:00 of the 4th period / OT released within 5 s before the onset."""
    order = ["foul", "violation", "turnover", "jump_ball", "timeout", "replay", "ejection"]
    anchored = align.filter(pl.col("method") == "clock_stop").with_columns(
        pl.col("msg_type").replace_strict(CAUSE, default=None).alias("cause")
    )
    by_stop: dict[str, list[str]] = {}
    for u, c in anchored.select("stop_event_uid", "cause").iter_rows():
        if c:
            by_stop.setdefault(u, []).append(c)
    made = shots.filter(pl.col("made") & pl.col("t_release_ms").is_not_null())
    hints = []
    for st in stops.iter_rows(named=True):
        cs = by_stop.get(st["event_uid"], [])
        hint = next((c for c in order if c in cs), None)
        if hint is None and (
            st["period"] >= 5 or (st["period"] == 4 and st["game_clock_at_stop"] <= 120)
        ):
            m = made.filter(
                (pl.col("period") == st["period"])
                & (pl.col("t_release_ms") <= st["t_onset_est_ms"])
                & (pl.col("t_release_ms") >= st["t_onset_est_ms"] - 5000)
            )
            if m.height:
                hint = "made_fg_last2min"
        hints.append(hint or "unknown")
    return stops.with_columns(pl.Series("stop_cause_hint", hints, dtype=pl.Utf8))


def _link_turnovers(passes_df: pl.DataFrame, align: pl.DataFrame) -> pl.DataFrame:
    link = align.filter(pl.col("method") == "pass").select(
        pl.col("linked_event_uid").alias("event_uid"),
        pl.col("event_num").alias("pbp_turnover_event_num"),
    )
    return passes_df.join(link, on="event_uid", how="left")
