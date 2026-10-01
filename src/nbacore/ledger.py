"""L3 possession ledger (L3; design in docs/ledger_schema.md) — work in
progress for v1.2.

``pbp_possessions`` walks the play-by-play of one game and splits it into **possession
segments**: one team controls the ball until the other gains it. Ownership rules:

* a FG attempt (1, 2) or a free throw (3) is taken by the team in possession (the shooter's);
* the possession ends with a made FG — unless free throws of the same team follow at the same
  clock (and-one): then it ends after the last of them —, a made last free throw, a defensive
  rebound (a rebound by the other team; team rebounds count), a turnover (5), a jump ball won by
  the other team, or the end of the period;
* a FG attempt, free throw or turnover of the team *not* in possession means the ball changed hands
  without a recorded event: the possession ends there (``unrecorded_change``);
* offensive rebounds, technical free throws, flagrant / clear-path free throws (the fouled team
  keeps the ball), dead-ball team rebounds after a missed free throw that is not the last (also
  when listed after the made last one), and a turnover recorded for a change already made at the
  same clock (held ball → jump ball + "Poss Lost Ball") do not change the possession.

Events are ordered by period, then game clock (descending), then event number, so that rows
inserted late by the scorer (event numbers out of order) are placed where they happened.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl

FT_RE = re.compile(r"Free Throw (Technical|Flagrant|Clear Path)?\s*(\d)? ?(?:of (\d))?", re.I)

POSSESSION_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "poss_seq": pl.Int32,
    "period": pl.Int8,
    "offense_team_id": pl.Int64,
    "start_event_num": pl.Int32,  # event that gave the ball to the offence (null: period start)
    "end_event_num": pl.Int32,  # event that ended the possession
    "start_type": pl.Utf8,
    "end_type": pl.Utf8,
    "n_fga": pl.Int16,
    "n_fta": pl.Int16,
    "n_oreb": pl.Int16,
    "points": pl.Int16,  # points of the offence (technical free throws excluded)
    "tech_points_offense": pl.Int16,  # technical free throws made during the possession
    "tech_points_defense": pl.Int16,
}


def _team_of(ev: dict) -> int | None:
    """Team of PLAYER1 (for team events, PLAYER1 is the team id itself)."""
    if ev["p1_team_id"] is not None:
        return int(ev["p1_team_id"])
    if ev["p1_type"] in (2, 3) and ev["p1_id"]:
        return int(ev["p1_id"])
    return None


def _ft_info(desc: str) -> tuple[str, int | None, int | None, bool]:
    """(kind: "" | technical | flagrant | clear path, n, m, made) from a free-throw description."""
    m = FT_RE.search(desc or "")
    kind = (m.group(1) or "").lower() if m else ""
    n = int(m.group(2)) if m and m.group(2) else None
    tot = int(m.group(3)) if m and m.group(3) else None
    return kind, n, tot, not (desc or "").upper().startswith("MISS")


def order_pbp(pbp_game: pl.DataFrame) -> pl.DataFrame:
    """Period, then period start first / period end last (late-inserted start rows exist, e.g.
    0021500506), then game clock descending, then event number — except that the free throws of
    one shooter at one clock are ordered by attempt number ("1 of 2" before "2 of 2"; some games,
    mainly 0021500506, number stretches of events in reverse)."""
    rank = (
        pl.when(pl.col("msg_type") == 12)
        .then(0)
        .when(pl.col("msg_type") == 13)
        .then(2)
        .otherwise(1)
    )
    desc = pl.coalesce("home_desc", "visitor_desc") if "home_desc" in pbp_game.columns else None
    key = pl.col("event_num").cast(pl.Float64)
    if desc is not None:
        n = desc.str.extract(r"Free Throw .*?(\d) of \d", 1).cast(pl.Float64)
        is_ft = (pl.col("msg_type") == 3) & n.is_not_null()
        grp = ["period", "pctime_sec", "p1_id", "msg_type"]
        key = pl.when(is_ft).then(pl.col("event_num").min().over(grp) + n / 10).otherwise(key)
    return (
        pbp_game.with_columns(rank.alias("_rank"), key.alias("_key"))
        .sort(["period", "_rank", "pctime_sec", "_key"], descending=[False, False, True, False])
        .drop("_rank", "_key")
    )


def pbp_possessions(pbp_game: pl.DataFrame, _event_map: dict | None = None) -> pl.DataFrame:
    """Ledger possessions of one game from its pbp. ``_event_map`` (internal, see
    ``pbp_possession_map``) is filled with event_num → poss_seq as the rows are processed."""
    gid = pbp_game["game_id"][0]
    p = order_pbp(pbp_game).with_columns(pl.coalesce("home_desc", "visitor_desc").alias("_d"))
    teams = sorted({int(t) for t in p["p1_team_id"].drop_nulls().unique()})
    other = {teams[0]: teams[1], teams[1]: teams[0]} if len(teams) == 2 else {}
    rows: list[dict] = []
    cur: dict | None = None
    pending_make: dict | None = None  # made FG waiting for a possible and-one free throw
    now: tuple = (None, None)  # (period, clock) of the current event
    last_change: dict = {}  # the latest change of possession: where / why / to whom
    pid = [0]  # provisional possession ids (possessions without an offence are dropped)
    pid_seq: dict[int, int] = {}
    emap: dict[int, int | None] = {}

    def open_(period, team, ev_num, start_type):
        nonlocal cur
        pid[0] += 1
        cur = {
            "game_id": gid, "period": period, "offense_team_id": team, "start_event_num": ev_num,
            "start_type": start_type, "n_fga": 0, "n_fta": 0, "n_oreb": 0, "points": 0,
            "tech_points_offense": 0, "tech_points_defense": 0, "_tech": {}, "_pid": pid[0],
        }  # fmt: skip

    def close(ev_num, end_type):
        nonlocal cur
        if cur is not None and cur["offense_team_id"] is not None:
            for t, pts in cur.pop("_tech").items():  # split once the offence is known
                k = "tech_points_offense" if t == cur["offense_team_id"] else "tech_points_defense"
                cur[k] += pts
            cur.update(end_event_num=ev_num, end_type=end_type)
            pid_seq[cur.pop("_pid")] = len(rows)
            rows.append(cur)
        cur = None

    def change(period, new_team, ev_num, end_type, start_type):
        nonlocal last_change
        before = cur["_pid"] if cur is not None else None
        close(ev_num, end_type)
        open_(period, new_team, ev_num, start_type)
        last_change = {"at": now, "end_type": end_type, "to": new_team, "from_pid": before}

    def here(en):  # the row belongs to the current possession
        emap[en] = cur["_pid"] if cur is not None else None

    for ev in p.iter_rows(named=True):
        period, msg, en = int(ev["period"]), int(ev["msg_type"]), int(ev["event_num"])
        team = _team_of(ev)
        now = (period, ev["pctime_sec"])
        same_clock_change = last_change.get("at") == now
        if pending_make is not None:
            # an and-one: a free throw of the scoring team at the same clock keeps the possession
            is_ft_same = (
                msg == 3
                and team == pending_make["team"]
                and ev["pctime_sec"] == pending_make["clock"]
            )
            # the scorer often lists the offensive rebound of a putback after the made basket (same
            # second): it happened before the make; fouls / substitutions / timeouts / replays at
            # that clock may precede an and-one free throw
            waiting = ev["pctime_sec"] == pending_make["clock"] and (
                msg in (6, 8, 9, 18) or (msg == 4 and team == pending_make["team"])
            )
            if not (is_ft_same or waiting):
                change(
                    period,
                    other.get(pending_make["team"]),
                    pending_make["ev"],
                    "made_fg",
                    "made_basket_inbound",
                )
                pending_make = None
        if msg == 12:  # period start
            if cur is not None:
                close(en, "period_end")
            open_(period, None, None, "period_start")
            here(en)
            continue
        if msg == 13:  # period end
            if pending_make is not None:
                change(period, None, pending_make["ev"], "made_fg", "made_basket_inbound")
                pending_make = None
            here(en)
            close(en, "period_end")
            continue
        if cur is None:
            open_(period, None, None, "period_start")
        here(en)  # before any change this row causes: it ends the current possession
        if msg == 10:  # jump ball: the tip goes to PLAYER3's team
            won = int(ev["p3_team_id"]) if ev["p3_team_id"] is not None else None
            if won is None:
                continue
            if cur["offense_team_id"] is None:
                cur["offense_team_id"] = won
                cur["start_type"] = (
                    "jump_ball" if cur["start_type"] == "period_start" else cur["start_type"]
                )
            elif won != cur["offense_team_id"]:
                change(period, won, en, "jump_ball_lost", "jump_ball")
            continue
        if msg in (1, 2):
            if team is None:
                continue
            if cur["offense_team_id"] is None:
                cur["offense_team_id"] = team
            elif team != cur["offense_team_id"]:
                # the ball changed hands without a recorded event: start a new possession here
                change(period, team, en, "unrecorded_change", "unrecorded_change")
                here(en)
            cur["n_fga"] += 1
            if msg == 1:
                pts = 3 if "3PT" in (ev["_d"] or "") else 2
                cur["points"] += pts
                pending_make = {"team": team, "clock": ev["pctime_sec"], "ev": en}
            continue
        if msg == 3:
            kind, n, m, made = _ft_info(ev["_d"])
            if team is None:
                continue
            if kind == "technical":  # no effect on the possession; the points are kept apart
                if made:
                    cur["_tech"][team] = cur["_tech"].get(team, 0) + 1
                continue
            if cur["offense_team_id"] is None:
                cur["offense_team_id"] = team
            elif team != cur["offense_team_id"]:
                change(period, team, en, "unrecorded_change", "free_throws")
                here(en)
            cur["n_fta"] += 1
            cur["points"] += int(made)
            last = n is not None and m is not None and n == m
            # flagrant / clear-path free throws: the fouled team keeps the ball (12B-IV, 12B-I (6))
            if last and made and kind not in ("flagrant", "clear path"):
                change(period, other.get(team), en, "made_ft", "made_ft_inbound")
            if pending_make is not None and last:
                pending_make = None  # the and-one sequence is over
            continue
        if msg == 4:
            if team is None or cur["offense_team_id"] is None:
                if team is not None and cur["offense_team_id"] is None:
                    cur["offense_team_id"] = team
                continue
            if team == cur["offense_team_id"]:
                cur["n_oreb"] += 1
            elif (
                same_clock_change
                and last_change["end_type"] == "made_ft"
                and ev["p1_type"] not in (4, 5)
            ):
                # dead-ball team rebound of an earlier missed free throw of the trip, listed after
                # the made last one (late-inserted row): no change
                emap[en] = last_change["from_pid"]
                continue
            else:
                change(period, team, en, "defensive_rebound", "defensive_rebound")
            continue
        if msg == 5:
            if team is None:
                continue
            if cur["offense_team_id"] is None:
                cur["offense_team_id"] = team
            elif team != cur["offense_team_id"]:
                if same_clock_change and last_change["to"] == other.get(team):
                    # the scorer's turnover for a change already made at this clock (e.g. a held
                    # ball: jump ball won by the other side + "Poss Lost Ball" turnover)
                    emap[en] = last_change["from_pid"]
                    continue
                # the turnover team had the ball without a recorded change (missing rebound, ...)
                change(period, team, en, "unrecorded_change", "unrecorded_change")
                here(en)
            change(period, other.get(team), en, "turnover", "turnover")
            continue
    if cur is not None:
        close(None, "game_end")
    if _event_map is not None:
        _event_map.update({e: pid_seq.get(q) for e, q in emap.items()})
    out = pl.DataFrame(rows, schema={k: v for k, v in POSSESSION_SCHEMA.items() if k != "poss_seq"})
    return (
        out.with_row_index("poss_seq")
        .with_columns(pl.col("poss_seq").cast(pl.Int32))
        .select(list(POSSESSION_SCHEMA))
    )


def pbp_possession_map(pbp_game: pl.DataFrame, ledger: pl.DataFrame | None = None) -> pl.DataFrame:
    """Every pbp row of one game → the ledger possession it was assigned to:
    ``event_num, poss_seq`` (+ ``poss_uid`` with a timed ``ledger``). A row that ends a possession
    (made last FT, defensive rebound, turnover, lost jump ball) belongs to it; rows the ledger
    treats as part of the scoring possession (a putback rebound listed after the make, and-one
    free throws) belong to it too; ``poss_seq`` is null for rows before any offence is known.
    ``role``: ``end_event`` (the row that ends the possession), ``start_event`` (the row that
    starts it, when it is not also the end of the previous one), ``inside``,
    ``between_possessions`` (no offence known: e.g. a period start before the tip)."""
    emap: dict = {}
    poss = pbp_possessions(pbp_game, _event_map=emap)
    out = pl.DataFrame(
        {"event_num": list(emap), "poss_seq": list(emap.values())},
        schema={"event_num": pl.Int32, "poss_seq": pl.Int32},
    ).with_columns(pl.lit(pbp_game["game_id"][0]).alias("game_id"))
    ends = poss.select("poss_seq", "start_event_num", "end_event_num")
    out = out.join(ends, on="poss_seq", how="left").with_columns(
        pl.when(pl.col("poss_seq").is_null())
        .then(pl.lit("between_possessions"))
        .when(pl.col("event_num") == pl.col("end_event_num"))
        .then(pl.lit("end_event"))
        .when(pl.col("event_num") == pl.col("start_event_num"))
        .then(pl.lit("start_event"))
        .otherwise(pl.lit("inside"))
        .alias("role")
    )
    out = out.select("game_id", "event_num", "poss_seq", "role")
    if ledger is not None:
        out = out.join(ledger.select("poss_seq", "poss_uid"), on="poss_seq", how="left")
    return out


LEDGER_TIME_SCHEMA: dict[str, pl.DataType] = {
    "poss_uid": pl.Utf8,  # <game_id>:<period>:<t_start_ms> (stable key)
    "defense_team_id": pl.Int64,
    "attacks_left": pl.Boolean,
    "t_start_ms": pl.Int64,
    "t_end_ms": pl.Int64,
    "end_time_method": pl.Utf8,  # pbp_align method of the end event | clock | period_last_frame
}


HOLE_MS = 1000  # an anchor farther than this from any frame lies in a tracking hole
SWITCH_FRAMES = 10  # sustained control by the new offence (0.4 s at 25 Hz)


def _clock_ok(fi: pl.DataFrame, period: int, unix_ms: int, pctime: float | None) -> bool:
    """Does the game clock at an anchored frame agree with the (floored) pbp clock?

    Same window as ``pbp_align.ANCHOR_CLOCK_WINDOW_S``; anchors inside a tracking hole are not
    trusted either."""
    from nbacore.events.pbp_align import ANCHOR_CLOCK_WINDOW_S

    fp = fi.filter(pl.col("period") == period)
    after = fp.filter(pl.col("unix_ms") >= unix_ms).head(1)
    before = fp.filter(pl.col("unix_ms") <= unix_ms).tail(1)
    gaps = [abs(int(x["unix_ms"][0]) - unix_ms) for x in (after, before) if x.height]
    if not gaps or min(gaps) > HOLE_MS:
        return False  # inside a tracking hole (e.g. a stop onset estimated across a timeout)
    if pctime is None:
        return True
    near = after if after.height else before
    if near["game_clock"][0] is None:
        return True
    lo, hi = ANCHOR_CLOCK_WINDOW_S
    return lo <= float(near["game_clock"][0]) - float(pctime) <= hi


def ledger_times(
    poss: pl.DataFrame,
    align_game: pl.DataFrame,
    frame_index: pl.DataFrame,
    direction: pl.DataFrame,
    handler: pl.DataFrame | None = None,
    pbp_game: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Put the pbp possessions of one game on the tracking time axis.

    The end of a possession is the tracking time of its end event (``pbp_align.unix_ms``:
    release of the made FG / last FT, rebound control, steal catch or clock stop, ...); when the
    event is not located, the frame matching its game clock (``locate_clock``); at the end of a
    period, the last frame of the period. The start is the end of the previous possession of the
    period, or the first frame of the period. Ends are forced monotone within a period.

    With the L2 ``handler`` table and the pbp, boundaries after a defensive rebound without a
    tracked control and after an unrecorded change are moved to the moment the new offence takes
    sustained control (``control_team_id``, ≥ ``SWITCH_FRAMES`` frames) between the neighbouring
    anchors (``handler_switch``): the scorer's clock of such rebounds lags by 1–5 s.
    """
    from nbacore.possession.segment import locate_clock

    at = dict(zip(align_game["event_num"].to_list(), align_game["unix_ms"].to_list(), strict=True))
    meth = dict(zip(align_game["event_num"].to_list(), align_game["method"].to_list(), strict=True))
    clk = dict(
        zip(align_game["event_num"].to_list(), align_game["pctime_sec"].to_list(), strict=True)
    )
    fi = frame_index.sort(["period", "unix_ms"])
    bounds = {
        int(r["period"]): (int(r["t0"]), int(r["t1"]))
        for r in fi.group_by("period")
        .agg(pl.col("unix_ms").min().alias("t0"), pl.col("unix_ms").max().alias("t1"))
        .iter_rows(named=True)
    }
    left = {(r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)}
    teams = sorted({t for t, _ in left})
    other = {teams[0]: teams[1], teams[1]: teams[0]} if len(teams) == 2 else {}
    gid = poss["game_id"][0] if poss.height else None

    rows = []
    prev_end: dict[int, int] = {}
    for r in poss.iter_rows(named=True):
        per = int(r["period"])
        if per not in bounds:
            rows.append(
                {**dict.fromkeys(LEDGER_TIME_SCHEMA), "end_time_method": "period_untracked"}
            )
            continue
        t0, t1 = bounds[per]
        start = prev_end.get(per, t0)
        en = r["end_event_num"]
        if r["end_type"] in ("period_end", "game_end") or en is None:
            end, how = t1, "period_last_frame"
        elif at.get(en) is not None and _clock_ok(fi, per, int(at[en]), clk.get(en)):
            end, how = int(at[en]), meth.get(en)
        else:
            if at.get(en) is not None:
                how_prefix = "clock_fallback"  # anchor disagrees with the pbp clock
            else:
                how_prefix = None
            idx_p = fi.filter(pl.col("period") == per)
            c = clk.get(en)
            loc = locate_clock(idx_p, float(c), start) if c is not None else None
            end, how = (loc, "clock") if loc is not None else (None, "unlocated")
            if end is None and c is not None:
                # the event lies in a tracking hole: end at the first frame past its clock
                past = idx_p.filter((pl.col("unix_ms") > start) & (pl.col("game_clock") < c))
                if past.height:
                    end, how = int(past["unix_ms"][0]), "after_gap"
            if how_prefix:
                how = f"{how_prefix}:{how}"
        if end is None or end < start:
            how = f"{how}:clamped" if end is not None else how
            end = start
        prev_end[per] = end
        off = r["offense_team_id"]
        rows.append(
            {
                "poss_uid": None,
                "defense_team_id": other.get(off),
                "attacks_left": left.get((off, per)),
                "t_start_ms": start,
                "t_end_ms": end,
                "end_time_method": how,
            }
        )
    if handler is not None and pbp_game is not None:
        _refine_with_handler(poss.to_dicts(), rows, pbp_game, at, handler)
    for pr, r in zip(poss.iter_rows(named=True), rows, strict=True):
        if r["t_start_ms"] is not None:
            r["poss_uid"] = f"{gid}:{int(pr['period'])}:{r['t_start_ms']}"
    times = pl.DataFrame(rows, schema=LEDGER_TIME_SCHEMA)
    # zero-length possessions share their start with the next one: keep the uid unique
    times = times.with_columns(
        pl.when(pl.col("poss_uid").is_duplicated())
        .then(pl.col("poss_uid") + ":" + pl.int_range(pl.len()).over("poss_uid").cast(pl.Utf8))
        .otherwise(pl.col("poss_uid"))
        .alias("poss_uid")
    )
    return poss.hstack(times)


def _first_run(mask: np.ndarray, n: int) -> int | None:
    """First index starting ``n`` consecutive True values."""
    if mask.size < n:
        return None
    full = np.flatnonzero(np.convolve(mask.astype(np.int16), np.ones(n, np.int16), "valid") == n)
    return int(full[0]) if full.size else None


def _refine_with_handler(
    poss: list[dict], rows: list[dict], pbp_game: pl.DataFrame, at: dict, handler: pl.DataFrame
) -> None:
    evs = order_pbp(pbp_game).select("event_num", "period", "msg_type").rows()
    pos = {int(e): k for k, (e, _, _) in enumerate(evs)}
    hs = handler.sort("period", "unix_ms")
    arr = {
        int(p): (g["unix_ms"].to_numpy(), g["control_team_id"].fill_null(-1).to_numpy())
        for (p,), g in hs.group_by("period")
    }
    for i in range(len(poss) - 1):
        r, nx = poss[i], poss[i + 1]
        kind, how = r["end_type"], rows[i]["end_time_method"] or ""
        if r["period"] != nx["period"] or rows[i]["t_end_ms"] is None:
            continue
        if kind not in ("defensive_rebound", "unrecorded_change") or how == "control":
            continue
        a_team, b_team, en = r["offense_team_id"], nx["offense_team_id"], r["end_event_num"]
        per = int(r["period"])
        if a_team is None or b_team is None or a_team == b_team or en not in pos or per not in arr:
            continue
        k = pos[en]
        lo = rows[i]["t_start_ms"]
        ok = kind == "unrecorded_change"
        for e, p_, m in reversed(evs[:k]):
            if p_ != per:
                break
            if kind == "unrecorded_change" and at.get(e) is not None:
                lo = max(lo, int(at[e]))  # the last anchored event before the change
                break
            if kind == "defensive_rebound" and m in (2, 3):
                # the miss this rebound follows; without its anchor the window is unknown
                if at.get(e) is not None and int(at[e]) >= lo:
                    lo, ok = int(at[e]), True
                break
        if not ok:
            continue
        if kind == "unrecorded_change":
            hi = int(at[en]) if at.get(en) is not None else rows[i]["t_end_ms"]
        else:
            hi = rows[i + 1]["t_end_ms"]
            for e, p_, _ in evs[k + 1 :]:
                if p_ != per:
                    break
                if at.get(e) is not None and int(at[e]) > lo:
                    hi = min(hi, int(at[e]))
                    break
        hi = min(hi, rows[i + 1]["t_end_ms"])
        t, c = arr[per]
        sel = np.flatnonzero((t > lo) & (t <= hi))
        if sel.size == 0:
            continue
        cs = c[sel]
        if kind == "unrecorded_change":
            last_a = np.flatnonzero(cs == a_team)
            start = int(last_a[-1]) + 1 if last_a.size else 0
        else:
            start = 0
        j = _first_run(cs[start:] == b_team, SWITCH_FRAMES)
        if j is None:
            continue
        t_new = int(t[sel[start + j]])
        if rows[i]["t_start_ms"] <= t_new <= rows[i + 1]["t_end_ms"]:
            rows[i]["t_end_ms"] = t_new
            rows[i + 1]["t_start_ms"] = t_new
            rows[i]["end_time_method"] = "handler_switch"


def attach_poss_uid(windows: pl.DataFrame, ledger: pl.DataFrame) -> pl.DataFrame:
    """Link view windows (``period``, ``t_cross``, ``t_terminal``, ``offense_team_id``) to the
    ledger possession they overlap most; adds ``poss_uid``, ``overlap_frac`` (share of
    [t_cross, t_terminal] inside it) and ``offense_match``."""
    out = []
    led = ledger.filter(pl.col("t_start_ms").is_not_null())
    by_period = {int(p): g for (p,), g in led.group_by("period")}
    for w in windows.iter_rows(named=True):
        g = by_period.get(int(w["period"]))
        a, b = int(w["t_cross"]), int(w["t_terminal"])
        best, frac = None, 0.0
        if g is not None:
            s, e = g["t_start_ms"].to_numpy(), g["t_end_ms"].to_numpy()
            ov = (np.minimum(e, b) - np.maximum(s, a)).clip(min=0)
            if b > a:
                i = int(ov.argmax())
                if ov[i] > 0:
                    best, frac = i, float(ov[i]) / (b - a)
            else:  # zero-length window: the possession containing the instant
                hit = np.flatnonzero((s <= a) & (a <= e))
                if hit.size:
                    best, frac = int(hit[-1]), 1.0
        if best is not None:
            row = g.row(best, named=True)
            out.append((row["poss_uid"], frac, row["offense_team_id"] == w["offense_team_id"]))
        else:
            out.append((None, 0.0, None))
    return windows.with_columns(
        pl.Series("poss_uid", [o[0] for o in out], dtype=pl.Utf8),
        pl.Series("overlap_frac", [o[1] for o in out], dtype=pl.Float32),
        pl.Series("offense_match", [o[2] for o in out], dtype=pl.Boolean),
    )


def pbp_possession_estimate(pbp_game: pl.DataFrame, team_oreb: bool = False) -> float:
    """Box-score estimate: FGA − OREB + TOV + 0.44 · FTA (both teams).

    OREB = player rebounds of the shooting team after a missed FG / FT (box score). With
    ``team_oreb`` also team rebounds of the shooting team after a missed FG (e.g. the ball knocked
    out of bounds by the defence): they do not end the possession either, but the box score does
    not count them, which biases the classic estimate upwards."""
    p = pbp_game.with_columns(pl.coalesce("home_desc", "visitor_desc").alias("_d"))
    fga = p.filter(pl.col("msg_type").is_in([1, 2])).height
    tov = p.filter(pl.col("msg_type") == 5).height
    fta = p.filter((pl.col("msg_type") == 3) & ~pl.col("_d").str.contains("Technical")).height
    shots = order_pbp(p)
    oreb = 0
    last_team, last_msg = None, None
    for ev in shots.iter_rows(named=True):
        if ev["msg_type"] in (2, 3):
            last_team, last_msg = _team_of(ev), ev["msg_type"]
        elif ev["msg_type"] == 4 and last_team is not None:
            same = _team_of(ev) == last_team
            player = ev["p1_type"] in (4, 5)
            oreb += int(same and (player or (team_oreb and last_msg == 2)))
            last_team = None
    return fga - oreb + tov + 0.44 * fta


def possession_estimate_aligned(pbp_game: pl.DataFrame, poss: pl.DataFrame) -> float:
    """Box-score estimate aligned with the ledger's definition (L3 acceptance yardstick):
    ``pbp_possession_estimate(team_oreb=True)`` plus the possessions that end with the period
    without a FG attempt, free throw or turnover (the box-score formula has no term for them)."""
    empty = poss.filter(
        pl.col("end_type").is_in(["period_end", "game_end"])
        & (pl.col("n_fga") == 0)
        & (pl.col("n_fta") == 0)
    ).height
    return pbp_possession_estimate(pbp_game, team_oreb=True) + empty


def offense_by_frame(ledger: pl.DataFrame, times: pl.DataFrame) -> pl.Series:
    """Ledger offence at each (period, unix_ms) of ``times`` (null outside possessions)."""
    led = (
        ledger.filter(pl.col("t_start_ms").is_not_null())
        .select(
            pl.col("period").cast(pl.Int8),
            pl.col("t_start_ms").alias("unix_ms"),
            "t_end_ms",
            "offense_team_id",
        )
        .sort("period", "unix_ms")
    )
    q = times.select(pl.col("period").cast(pl.Int8), "unix_ms").with_row_index("_i")
    j = q.sort("period", "unix_ms").join_asof(
        led, on="unix_ms", by="period", strategy="backward", check_sortedness=False
    )
    j = j.with_columns(
        pl.when(pl.col("unix_ms") <= pl.col("t_end_ms")).then(pl.col("offense_team_id")).alias("o")
    )
    return j.sort("_i")["o"].alias("ledger_offense_team_id")
