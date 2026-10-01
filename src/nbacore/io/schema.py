"""Schema validation and descriptive statistics for raw game JSON.

``validate_game`` raises ``SchemaError`` on *hard* violations (structure we rely on) and
returns a ``GameStats`` record of *soft* irregularities (things we handle downstream).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from nbacore.io.raw import BALL_ID, COURT_LENGTH_FT, COURT_WIDTH_FT

MOMENT_LEN = 6
ROW_LEN = 5
ENTITIES_PER_FRAME = 11


class SchemaError(ValueError):
    pass


@dataclass
class GameStats:
    game_id: str
    game_date: str
    n_events: int
    n_events_no_moments: int
    n_moments_raw: int
    n_frames_unique: int  # unique (period, unix_ms)
    dup_fraction: float
    n_duplicate_conflicts: int  # same key, different content
    entities_per_frame: dict[int, int]
    n_frames_no_ball: int
    n_frames_shotclock_null: int
    frac_shotclock_null_when_gc_le_24: float | None
    periods: dict[int, int]
    unix_dt_ms_mode: int | None
    unix_dt_ms_counts: dict[int, int]
    n_events_unix_nonmonotonic: int
    n_events_gc_increase: int  # game clock increases > 0.05 s inside an event
    max_gc_increase_in_event: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    ball_z_max: float
    player_z_nonzero: int
    n_roster_variants: int
    roster_size: int
    n_players_on_court_total: int
    frac_frames_any_oob: float  # any entity outside [0,94]x[0,50]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise SchemaError(msg)


def validate_game(game: dict) -> GameStats:  # noqa: C901 - one long, explicit pass
    _check(set(game) >= {"gameid", "gamedate", "events"}, f"top-level keys: {list(game)}")
    gid = str(game["gameid"]).zfill(10)
    events = game["events"]
    _check(isinstance(events, list) and len(events) > 0, "events must be a non-empty list")

    warnings: list[str] = []
    n_no_moments = 0
    n_raw = 0
    seen: dict[tuple[int, int], tuple] = {}
    conflicts = 0
    ent_counter: Counter[int] = Counter()
    no_ball = 0
    sc_null = 0
    sc_null_gc_le_24 = 0
    periods: Counter[int] = Counter()
    dt_counter: Counter[int] = Counter()
    unix_nonmono = 0
    gc_inc_events = 0
    max_gc_inc = 0.0
    xs_min = ys_min = math.inf
    xs_max = ys_max = -math.inf
    ball_z_max = -math.inf
    player_z_nonzero = 0
    rosters: set = set()
    roster_size = 0
    on_court: set[int] = set()
    frames_oob = 0

    for ev in events:
        _check(set(ev) >= {"eventId", "visitor", "home", "moments"}, f"event keys: {list(ev)}")
        for side in ("home", "visitor"):
            t = ev[side]
            _check(
                set(t) >= {"name", "teamid", "abbreviation", "players"}, f"{side} keys {list(t)}"
            )
        rosters.add(
            (
                tuple(sorted(p["playerid"] for p in ev["home"]["players"])),
                tuple(sorted(p["playerid"] for p in ev["visitor"]["players"])),
            )
        )
        roster_size = len(ev["home"]["players"]) + len(ev["visitor"]["players"])
        ms = ev["moments"]
        if not ms:
            n_no_moments += 1
            continue
        ux = np.fromiter((m[1] for m in ms), dtype=np.int64, count=len(ms))
        gc = np.fromiter((m[2] for m in ms), dtype=np.float64, count=len(ms))
        if len(ms) > 1:
            d = np.diff(ux)
            if (d < 0).any():
                unix_nonmono += 1
            dt_counter.update(d.tolist())
            inc = np.diff(gc)
            if (inc > 0.05).any():
                gc_inc_events += 1
                max_gc_inc = max(max_gc_inc, float(inc.max()))
        for m in ms:
            _check(
                len(m) == MOMENT_LEN,
                f"moment length {len(m)} != {MOMENT_LEN} (event {ev['eventId']})",
            )
            period, unix_ms, game_clock, shot_clock, _, rows = m
            _check(isinstance(period, int) and 1 <= period <= 10, f"bad period {period}")
            _check(isinstance(unix_ms, int), f"unix_ms not int: {unix_ms!r}")
            _check(isinstance(game_clock, (int, float)), f"game_clock not numeric: {game_clock!r}")
            _check(shot_clock is None or isinstance(shot_clock, (int, float)), "shot_clock type")
            n_raw += 1
            periods[period] += 1
            key = (period, unix_ms)
            content = (game_clock, shot_clock, tuple(tuple(r) for r in rows))
            if key in seen:
                if seen[key] != content:
                    conflicts += 1
                continue  # stats below are on unique frames only
            seen[key] = content
            ent_counter[len(rows)] += 1
            has_ball = False
            oob = False
            for r in rows:
                _check(len(r) == ROW_LEN, f"row length {len(r)} != {ROW_LEN}")
                tid, pid, x, y, z = r
                _check(all(math.isfinite(v) for v in (x, y, z)), "non-finite coordinate")
                xs_min = min(xs_min, x)
                xs_max = max(xs_max, x)
                ys_min = min(ys_min, y)
                ys_max = max(ys_max, y)
                if not (0 <= x <= COURT_LENGTH_FT and 0 <= y <= COURT_WIDTH_FT):
                    oob = True
                if tid == BALL_ID:
                    has_ball = True
                    ball_z_max = max(ball_z_max, z)
                else:
                    on_court.add(int(pid))
                    if z != 0:
                        player_z_nonzero += 1
            if not has_ball:
                no_ball += 1
            if oob:
                frames_oob += 1
            if shot_clock is None:
                sc_null += 1
                if game_clock <= 24.0:
                    sc_null_gc_le_24 += 1

    n_unique = len(seen)
    if n_unique == 0:
        raise SchemaError(f"game {gid} has no moments at all")
    if len(rosters) != 1:
        warnings.append(f"roster varies across events ({len(rosters)} variants)")
    if unix_nonmono:
        warnings.append(f"{unix_nonmono} events with non-monotonic unix time")
    if conflicts:
        warnings.append(f"{conflicts} duplicate frames with conflicting content")

    return GameStats(
        game_id=gid,
        game_date=str(game["gamedate"]),
        n_events=len(events),
        n_events_no_moments=n_no_moments,
        n_moments_raw=n_raw,
        n_frames_unique=n_unique,
        dup_fraction=1.0 - n_unique / n_raw,
        n_duplicate_conflicts=conflicts,
        entities_per_frame=dict(sorted(ent_counter.items())),
        n_frames_no_ball=no_ball,
        n_frames_shotclock_null=sc_null,
        frac_shotclock_null_when_gc_le_24=(sc_null_gc_le_24 / sc_null) if sc_null else None,
        periods=dict(sorted(periods.items())),
        unix_dt_ms_mode=dt_counter.most_common(1)[0][0] if dt_counter else None,
        unix_dt_ms_counts=dict(dt_counter.most_common(8)),
        n_events_unix_nonmonotonic=unix_nonmono,
        n_events_gc_increase=gc_inc_events,
        max_gc_increase_in_event=max_gc_inc,
        x_min=xs_min,
        x_max=xs_max,
        y_min=ys_min,
        y_max=ys_max,
        ball_z_max=ball_z_max,
        player_z_nonzero=player_z_nonzero,
        n_roster_variants=len(rosters),
        roster_size=roster_size,
        n_players_on_court_total=len(on_court),
        frac_frames_any_oob=frames_oob / n_unique,
        warnings=warnings,
    )
