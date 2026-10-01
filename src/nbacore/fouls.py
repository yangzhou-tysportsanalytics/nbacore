"""Team-foul / penalty state and free throw ↔ foul links (L3, L3).

Rules from ``nbacore.rules_2015_16`` (12B-V): team fouls count per period (``TEAM_FOUL_DETAILS``);
a team is in the penalty from its 5th team foul of a quarter (4th in overtime), or from its 2nd
team foul of the last two minutes when it was below the quota at 2:00.

``foul_states(pbp_game)``: one row per foul (msg 6), in ledger order. For the fouling team
(``committed_by_team``): team fouls of the period before / after this foul, the same for the last
two minutes, ``in_bonus_for_fouled_team`` (the fouling team is in the penalty before this foul),
``fouls_to_give``; for the other team the state before the foul (``opp_*``);
``n_free_throws`` linked to the foul.

``free_throw_links(pbp_game)``: one row per free throw (msg 3) with ``linked_foul_event_num`` (the
latest foul at the same game clock by the other team not yet used — for technical free throws a
technical foul of the other team; the 2nd / 3rd attempt of a trip uses the foul of the 1st),
``ft_n`` / ``ft_m``, ``is_last_ft``, ``ft_kind`` (regular | technical | flagrant | clear_path),
``made``.

``state_at(fouls, period, unix_ms, game_clock_s)``: both teams' state at a tracking time (the
foul table needs ``unix_ms``, added from ``pbp_align`` by the L3 build).

Consistency with the data: ``reports/l3_fouls_check.json`` (99.88 % of free throws linked; in
penalty → free throws 99.9 %) and ``reports/l5_rulebook_check.md`` §3.
"""

from __future__ import annotations

import re

import polars as pl

from nbacore import rules_2015_16 as R
from nbacore.events.pbp_align import FOUL_DETAIL
from nbacore.ledger import order_pbp

FT_RE = re.compile(r"Free Throw (Technical|Flagrant|Clear Path)?\s*(\d)? ?(?:of (\d))?", re.I)
TECHNICAL_DETAILS = frozenset(
    d
    for d in R.NON_TEAM_FOUL_DETAILS
    if d not in ("offensive", "offensive_charge", "double_personal")
)

FOUL_STATE_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "event_num": pl.Int32,
    "period": pl.Int8,
    "pctime_sec": pl.Float32,
    "committed_by_team": pl.Int64,
    "player_id": pl.Int64,
    "foul_class": pl.Utf8,  # pbp_align FOUL_DETAIL class
    "foul_detail": pl.Utf8,
    "counts_as_team_foul": pl.Boolean,
    "team_fouls_before": pl.Int16,
    "team_fouls_in_period_after": pl.Int16,
    "fouls_in_last2min_before": pl.Int16,
    "fouls_in_last2min_after": pl.Int16,
    "in_bonus_for_fouled_team": pl.Boolean,
    "fouls_to_give": pl.Int16,
    "opp_team_fouls_before": pl.Int16,
    "opp_in_penalty": pl.Boolean,
    "opp_fouls_to_give": pl.Int16,
    "n_free_throws": pl.Int16,
}

FT_LINK_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "event_num": pl.Int32,
    "period": pl.Int8,
    "pctime_sec": pl.Float32,
    "team_id": pl.Int64,  # shooting team
    "player_id": pl.Int64,
    "ft_n": pl.Int8,
    "ft_m": pl.Int8,
    "is_last_ft": pl.Boolean,
    "ft_kind": pl.Utf8,
    "made": pl.Boolean,
    "linked_foul_event_num": pl.Int32,
}


def fouls_to_give(period: int, game_clock_s: float, fouls_before: int, last2_before: int) -> int:
    """Team fouls a team may still commit without penalty (0 when in the penalty)."""
    if R.in_penalty(period, game_clock_s, fouls_before, last2_before):
        return 0
    left = R.team_foul_quota(period) - fouls_before
    if game_clock_s <= R.TWO_MINUTE_S:
        left = min(left, R.LAST_TWO_MINUTES_FREE_FOULS - last2_before)
    return max(left, 0)


def _analyse(pbp_game: pl.DataFrame) -> tuple[list[dict], list[dict]]:
    gid = pbp_game["game_id"][0]
    p = order_pbp(pbp_game).with_columns(pl.coalesce("home_desc", "visitor_desc").alias("_d"))
    teams = sorted({int(t) for t in p["p1_team_id"].drop_nulls().unique()})
    other = {teams[0]: teams[1], teams[1]: teams[0]} if len(teams) == 2 else {}
    count: dict[tuple[int, int], int] = {}
    last2: dict[tuple[int, int], int] = {}
    fouls: list[dict] = []
    fts: list[dict] = []
    used: dict[int, int] = {}  # foul event_num -> free throws linked
    trip: dict[tuple[int, bool], int] = {}  # (shooter, technical) -> foul of the current trip
    for ev in p.iter_rows(named=True):
        msg, per = int(ev["msg_type"]), int(ev["period"])
        clk = float(ev["pctime_sec"])
        team = int(ev["p1_team_id"]) if ev["p1_team_id"] is not None else None
        if msg == 6:
            detail, klass = FOUL_DETAIL.get(int(ev["action_type"]), (None, None))
            key = (team, per)
            before, l2 = count.get(key, 0), last2.get(key, 0)
            is_team = detail in R.TEAM_FOUL_DETAILS and team is not None
            after, l2_after = before + int(is_team), l2 + int(is_team and clk <= R.TWO_MINUTE_S)
            opp = other.get(team)
            ob, ol2 = count.get((opp, per), 0), last2.get((opp, per), 0)
            known = team is not None
            fouls.append(
                {
                    "game_id": gid,
                    "event_num": int(ev["event_num"]),
                    "period": per,
                    "pctime_sec": clk,
                    "committed_by_team": team,
                    "player_id": ev["p1_id"],
                    "foul_class": klass,
                    "foul_detail": detail,
                    "counts_as_team_foul": is_team,
                    "team_fouls_before": before,
                    "team_fouls_in_period_after": after,
                    "fouls_in_last2min_before": l2,
                    "fouls_in_last2min_after": l2_after,
                    "in_bonus_for_fouled_team": R.in_penalty(per, clk, before, l2)
                    if known
                    else None,
                    "fouls_to_give": fouls_to_give(per, clk, before, l2) if known else None,
                    "opp_team_fouls_before": ob if opp is not None else None,
                    "opp_in_penalty": R.in_penalty(per, clk, ob, ol2) if opp is not None else None,
                    "opp_fouls_to_give": fouls_to_give(per, clk, ob, ol2)
                    if opp is not None
                    else None,
                    "n_free_throws": 0,
                }
            )
            if is_team:
                count[key], last2[key] = after, l2_after
        elif msg == 3 and team is not None:
            m = FT_RE.search(ev["_d"] or "")
            kind = (m.group(1) or "").lower().replace(" ", "_") if m else ""
            tech = kind == "technical"
            n = int(m.group(2)) if m and m.group(2) else None
            tot = int(m.group(3)) if m and m.group(3) else None
            shooter = ev["p1_id"]
            link = trip.get((shooter, tech)) if n is not None and n > 1 else None
            if link is None:
                for f in reversed(fouls):  # latest first; the clock grows going back
                    if f["period"] != per or f["pctime_sec"] > clk + 1:
                        break
                    if f["pctime_sec"] != clk or f["committed_by_team"] == team:
                        continue
                    if tech != (f["foul_detail"] in TECHNICAL_DETAILS):
                        continue
                    if not tech and (f["committed_by_team"] is None or used.get(f["event_num"], 0)):
                        continue  # a personal foul awards one trip
                    link = f["event_num"]
                    break
            if link is not None:
                used[link] = used.get(link, 0) + 1
                trip[(shooter, tech)] = link
            fts.append(
                {
                    "game_id": gid,
                    "event_num": int(ev["event_num"]),
                    "period": per,
                    "pctime_sec": clk,
                    "team_id": team,
                    "player_id": shooter,
                    "ft_n": n,
                    "ft_m": tot,
                    "is_last_ft": n is not None and tot is not None and n == tot,
                    "ft_kind": kind or "regular",
                    "made": not (ev["_d"] or "").upper().startswith("MISS"),
                    "linked_foul_event_num": link,
                }
            )
    for f in fouls:
        f["n_free_throws"] = used.get(f["event_num"], 0)
    return fouls, fts


def foul_states(pbp_game: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(_analyse(pbp_game)[0], schema=FOUL_STATE_SCHEMA)


def free_throw_links(pbp_game: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame(_analyse(pbp_game)[1], schema=FT_LINK_SCHEMA)


def state_at(
    fouls: pl.DataFrame, teams: list[int], period: int, unix_ms: int, game_clock_s: float
) -> dict[int, dict]:
    """Per team: ``team_fouls``, ``fouls_last2min``, ``in_penalty``, ``fouls_to_give`` at a
    tracking time (team fouls of the period whose ``unix_ms`` ≤ the time)."""
    f = fouls.filter(
        (pl.col("period") == period)
        & (pl.col("unix_ms") <= unix_ms)
        & pl.col("counts_as_team_foul")
    )
    out = {}
    for t in teams:
        ft = f.filter(pl.col("committed_by_team") == t)
        n = ft.height
        l2 = ft.filter(pl.col("pctime_sec") <= R.TWO_MINUTE_S).height
        out[t] = {
            "team_fouls": n,
            "fouls_last2min": l2,
            "in_penalty": R.in_penalty(period, game_clock_s, n, l2),
            "fouls_to_give": fouls_to_give(period, game_clock_s, n, l2),
        }
    return out
