"""Rule constants of the 2015-16 NBA season (L5; docs/external_sources.md).

Source: "Official Rules of the National Basketball Association 2015-2016" (edition of
10-02-2015, 67 pp.; provenance in ``<data root>/raw/external/rulebook/provenance.json``). The PDF
itself stays on this machine and is never distributed; only the values below are, each with its
rule / section / article and page. Values marked *inferred* are not stated literally.

Differences to later seasons that matter here: in 2015-16 an offensive rebound after the ball
touched the ring resets the shot clock to **24** (the 14-second reset dates from 2018-19), and the
bonus is reached with the 5th team foul of a quarter (4th in overtime).
"""

from __future__ import annotations

# --- timing (Rule 5 Section II, p. 21) --------------------------------------------------------
PERIOD_S = 720.0  # 5-II-a: regulation periods of twelve minutes
OT_PERIOD_S = 300.0  # 5-II-b: overtime periods of five minutes
TWO_MINUTE_S = 120.0  # 5-II-f: "two-minute part" when the game clock shows 2:00 or less
BACKCOURT_S = 8.0  # 4-V-f (p. 18): ball across the midcourt line within 8 seconds
FREE_THROW_LIMIT_S = 10.0  # 4-IV (p. 18): free throw attempted within 10 seconds

# --- 24-second clock (Rule 7, pp. 27-29) ------------------------------------------------------
SHOT_CLOCK_S = 24.0  # 7-II-d: field goal attempt within 24 seconds of gaining possession
SHOT_CLOCK_TENTHS_BELOW_S = 5.0  # 7-I: tenths displayed once the clock reaches 4.9
# 7-II-i: with 24 s or less left in the period the shot clock does not function after a change of
# possession
SHOT_CLOCK_OFF_GAME_CLOCK_S = 24.0
# 7-IV-c: reset to 24 on (1) change of possession, (2) ball contacting the ring of the team in
# possession (so an offensive rebound after rim contact -> 24), (3) personal foul / (4) violation
# with the throw-in in the backcourt, (5) jump balls not caused by a defensive held ball, (6) all
# flagrant and punching fouls
RESET_FULL_S = 24.0
OREB_RESET_S = 24.0  # 7-IV-c(2)
# 7-IV-b: never reset on (5) a field goal attempt that fails to touch the rim (air ball), (1) a
# defensive player causing the ball out of bounds, (2) technical / delay warnings on the offence
# 7-IV-d: stays the same or is reset to 14, whichever is greater, on (1) a defensive personal foul
# with the throw-in in the frontcourt, (2) defensive three seconds, (3) technical / delay
# warnings on the defence, (4) kicked or punched ball by the defence (frontcourt throw-in)
FRONTCOURT_RESET_MIN_S = 14.0
# 12B-VI-c (p. 47): double foul -> 24 if the throw-in is in the backcourt, else max(remaining, 14)

# --- team fouls and penalty (Rule 12B Section V, pp. 46-47) ----------------------------------
TEAM_FOUL_QUOTA = 4  # 12B-V-a / a(1): four team fouls per regulation period without penalty
TEAM_FOUL_QUOTA_OT = 3  # 12B-V-a(2), a(4): three per overtime period
# 12B-V-a(3): a team that has not reached its quota before the last 2:00 may commit one team foul
# in the last two minutes without penalty (the 2nd team foul of the last two minutes is penalised)
LAST_TWO_MINUTES_FREE_FOULS = 1
# penalty for a common foul in the penalty situation: one free throw plus a penalty free throw
# (12B-V-a; 12B-I PENALTIES (5); loose ball 12B-VIII-a(4))
PENALTY_FREE_THROWS = 2

# Which pbp foul classes (``nbacore.events.pbp_align.FOUL_DETAIL``) are charged as team fouls:
#  * personal, shooting, loose ball (12B-VIII-a(1)), flagrant (12B-IV-a/b), away-from-play,
#    clear path, punching (12B-IX-a) -> team foul (12B-I PENALTIES; 12B-V-a(5));
#  * offensive fouls -> no team foul (12B-VII-a(3); 12B-I PENALTIES "no team foul ... against an
#    offensive player"), except punching / flagrant (12B-VII-b(3));
#  * double personal fouls -> no team foul (12B-VI-b);
#  * technical fouls (Rule 12A) -> no team foul (*inferred*: 12B-V counts "common fouls charged as
#    team fouls"; technicals are penalised under 12A).
TEAM_FOUL_DETAILS = frozenset(
    {
        "personal",
        "shooting",
        "loose_ball",
        "inbound",
        "away_from_play",
        "clear_path",
        "flagrant_1",
        "flagrant_2",
        "personal_block",
        "personal_take",
        "shooting_block",
    }
)
NON_TEAM_FOUL_DETAILS = frozenset(
    {
        "offensive",
        "offensive_charge",
        "double_personal",
        "technical",
        "non_unsportsmanlike_technical",
        "hanging_technical",
        "double_technical",
        "defensive_3_seconds",
        "delay_of_game",
        "taunting",
        "excess_timeout",
        "too_many_players",
    }
)


def team_foul_quota(period: int) -> int:
    return TEAM_FOUL_QUOTA if period <= 4 else TEAM_FOUL_QUOTA_OT


def in_penalty(
    period: int, game_clock_s: float, fouls_before: int, last2_fouls_before: int
) -> bool:
    """Is a team in the penalty situation for its next common foul (12B-V-a)?

    ``fouls_before``: team fouls of the team earlier in the period; ``last2_fouls_before``: of
    those, the ones committed with the game clock at 2:00 or less.
    """
    if fouls_before >= team_foul_quota(period):
        return True
    return game_clock_s <= TWO_MINUTE_S and last2_fouls_before >= LAST_TWO_MINUTES_FREE_FOULS


def shot_clock_after_defensive_foul(remaining_s: float, frontcourt_throw_in: bool) -> float:
    """Shot clock after a defensive non-shooting foul without free throws (7-IV-c(3), 7-IV-d(1))."""
    if not frontcourt_throw_in:
        return RESET_FULL_S
    return max(remaining_s, FRONTCOURT_RESET_MIN_S)
