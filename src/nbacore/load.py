"""Read a published data release (``<data root>/releases/<version>/``).

Research projects should pin ``version`` (e.g. ``"v1.0"``) in their own config; ``"latest"``
follows ``releases/latest.txt``.

    import nbacore.load as L
    games = L.games("v1.0").filter(pl.col("status") == "ok")
    f = L.frames("0021500115", "v1.0")
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from nbacore import paths


def _gid(game_id: str | int) -> str:
    return str(game_id).zfill(10)


def _file(version: str, rel: str) -> Path:
    p = paths.release_dir(version) / rel
    if not p.exists():
        raise FileNotFoundError(f"{rel} not in release {paths.resolve_version(version)} ({p})")
    return p


def _table(version: str, name: str, game_id: str | int | None) -> pl.DataFrame:
    lf = pl.scan_parquet(_file(version, f"{name}.parquet"))
    if game_id is not None:
        lf = lf.filter(pl.col("game_id") == _gid(game_id))
    return lf.collect()


def _per_game(version: str, name: str, game_id: str | int, columns: list[str] | None):
    gid = _gid(game_id)
    p = paths.release_dir(version) / name / f"{gid}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"no {name} for game {gid} in release {paths.resolve_version(version)}; "
            "see games(...)['status'] for games without usable tracking"
        )
    return pl.read_parquet(p, columns=columns)


def manifest(version: str = "latest") -> dict:
    return json.loads(_file(version, "MANIFEST.json").read_text(encoding="utf-8"))


def games(version: str = "latest") -> pl.DataFrame:
    """One row per game of the tracking period: status, date, teams, split / fold / parity."""
    return _table(version, "games", None)


def frames(game_id: str | int, version: str = "latest", columns: list[str] | None = None):
    """L1: entity-level deduped 25 Hz rows of one game (raw court coordinates, feet)."""
    return _per_game(version, "frames", game_id, columns)


def frame_index(game_id: str | int, version: str = "latest", columns: list[str] | None = None):
    """L1: one row per frame with continuity flags and ``segment_id``."""
    return _per_game(version, "frame_index", game_id, columns)


def rosters(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    return _table(version, "rosters", game_id)


def pbp(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """Typed play-by-play of the whole 2015-16 regular season (1,230 games)."""
    return _table(version, "pbp", game_id)


def attack_direction(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    return _table(version, "attack_direction", game_id)


def tracking_events(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """Raw tracking events (moment windows per pbp EVENTNUM); ghost-defense's ``events.parquet``."""
    return _table(version, "tracking_events", game_id)


def events(
    game_id: str | int, types: list[str] | str | None = None, version: str = "latest"
) -> pl.DataFrame:
    """L2 events of one game (long table). With ``types``, only those event types, and columns
    that are null for all of them are dropped (the common columns are always kept)."""
    df = _per_game(version, "events", game_id, None)
    if types is None:
        return df
    types = [types] if isinstance(types, str) else list(types)
    df = df.filter(pl.col("event_type").is_in(types))
    common = df.columns[: df.columns.index("event_uid") + 1]
    keep = [c for c in df.columns if c in common or df[c].null_count() < df.height]
    return df.select(keep)


def handler(game_id: str | int, version: str = "latest", columns: list[str] | None = None):
    """L2: 25 Hz ball handler per frame (game-wide, team hysteresis)."""
    return _per_game(version, "handler", game_id, columns)


def pbp_align(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L2: every pbp event placed on the tracking time axis."""
    return _table(version, "pbp_align", game_id)


def event_frames(game_id: str | int, version: str = "latest") -> pl.DataFrame:
    """L2: offence x defence and ball distances at shot releases, pass releases and catches."""
    return _per_game(version, "event_frames", game_id, None)


# ---------------------------------------------------------------------------------------------
# L3 (v1.2): possession ledger, fouls, free throws, ghost_v1, shot clock
# ---------------------------------------------------------------------------------------------


def ledger(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L3: one row per team possession on the tracking axis, with sub-segments, lineups and
    flags (``docs/data_schema.md``)."""
    return _table(version, "ledger", game_id)


def epv_full(
    version: str = "latest",
    game_id: str | int | None = None,
    drop_backcourt_subsegments: bool = False,
) -> pl.DataFrame:
    """L3 view for expected-possession-value models: ``nbacore.views.epv.epv_full`` of the
    ledger."""
    from nbacore.views.epv import epv_full as _epv

    return _epv(ledger(version, game_id), drop_backcourt_subsegments)


def fouls(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L3: per foul, team-foul / penalty state of both teams and linked free throws."""
    return _table(version, "fouls", game_id)


def free_throws(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L3: per free throw, the foul that awarded it, ``ft_n / ft_m``, ``ft_kind``."""
    return _table(version, "free_throws", game_id)


def foul_state_at(game_id: str | int, unix_ms: int, version: str = "latest") -> dict:
    """Team-foul state of both teams at a tracking time."""
    from nbacore.fouls import state_at

    fi = frame_index(game_id, version, columns=["period", "unix_ms", "game_clock"])
    k = fi.filter(pl.col("unix_ms") <= unix_ms).tail(1)
    if k.height == 0:
        raise ValueError(f"{unix_ms} is before the first frame of {game_id}")
    per, gc = int(k["period"][0]), float(k["game_clock"][0])
    g = games(version).filter(pl.col("game_id") == _gid(game_id))
    teams = [int(g["home_team_id"][0]), int(g["visitor_team_id"][0])]
    return state_at(fouls(version, game_id), teams, per, unix_ms, gc)


def ghost_v1(
    version: str = "latest", game_id: str | int | None = None, release: str = "ghost"
) -> pl.DataFrame:
    """L3: ghost-defense's half-court windows with ``window_uid`` and ``poss_uid``.

    ``release="l2"`` (v1.4+): FG windows end at the corrected L2 shot release
    (``ShotTimeConfig(shooter_only=True)``) instead of ghost's original rule (last frame in the
    hand of any offensive player); same ``window_uid`` keys (``ghost_v1_l2release.parquet``)."""
    if release not in ("ghost", "l2"):
        raise ValueError("release must be 'ghost' or 'l2'")
    return _table(version, "ghost_v1" if release == "ghost" else "ghost_v1_l2release", game_id)


def with_post_release(
    seconds: float, version: str = "latest", game_id: str | int | None = None
) -> pl.DataFrame:
    """L3 view for action-sequence models: ghost_v1 windows with ``t_end_post_ms`` = release +
    ``seconds`` (≤ 3) for shot windows."""
    from nbacore.views.post_release import with_post_release as _wpr

    return _wpr(ghost_v1(version, game_id), seconds)


def shot_clock(game_id: str | int, version: str = "latest") -> pl.DataFrame:
    """L3: per-frame shot clock, observed or rule-imputed (``shot_clock_imputed``)."""
    return _per_game(version, "shot_clock", game_id, None)


def pbp_possession(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L3 (v1.3): every pbp row → the ledger possession it belongs to: ``game_id,
    event_num, poss_seq, role, poss_uid``; ``role`` ∈ start_event / inside / end_event /
    between_possessions (``nbacore.ledger.pbp_possession_map``). Use it instead of slicing pbp by
    ``start_event_num < event_num <= end_event_num`` (pbp numbering is not always time order)."""
    return _table(version, "pbp_possession", game_id)


def event_possession(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L3: every offensive-action L2 event → its ledger possession (``poss_uid``), the ledger
    offence and ``offense_match``. Filter screen / cut / drive / post-up candidates on
    ``offense_match`` (candidates of the defence come from moments when the L2 handler's
    controlling team is wrong)."""
    return _table(version, "event_possession", game_id)


def external(name: str, version: str = "latest") -> pl.DataFrame:
    """L5 (v1.3+): an external table, ``external/<name>.parquet`` — ``shots_2015_16``,
    ``player_bio_2015_16``, ``box_games_2015_16``, ``box_players_2015_16``; from v1.5 also
    ``advanced_2015_16``, ``awards_2015_16``, ``raptor_2015_16``, ``bbref_player_map_2015_16``. Sources, licences and
    terms: ``external/SOURCES.json`` of the release and ``docs/external_sources.md``. Local research use
    only; credit NBA.com."""
    return pl.read_parquet(_file(version, f"external/{name}.parquet"))


def shots(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L5: official 2015-16 shot locations of all 1,230 games, linked to pbp
    (``docs/external_sources.md``)."""
    return _table(version, "external/shots_2015_16", game_id)


def player_bio(version: str = "latest") -> pl.DataFrame:
    """L5: 2015-16 height / weight / wingspan per ``player_id`` (``docs/external_sources.md``)."""
    return external("player_bio_2015_16", version)


def box_players(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L5: 2015-16 player box scores (``docs/external_sources.md``)."""
    return _table(version, "external/box_players_2015_16", game_id)


def box_games(version: str = "latest") -> pl.DataFrame:
    """L5: 2015-16 games with final scores (``docs/external_sources.md``)."""
    return external("box_games_2015_16", version)


def advanced(version: str = "latest") -> pl.DataFrame:
    """L5 (v1.5+): 2015-16 Basketball-Reference advanced metrics per player and team stint
    (``team == "TOT"`` = season total of traded players); ``docs/external_sources.md``."""
    return external("advanced_2015_16", version)


def awards(version: str = "latest") -> pl.DataFrame:
    """L5 (v1.5+): 2015-16 award voting, All-NBA / All-Defense / All-Rookie, All-Star."""
    return external("awards_2015_16", version)


def raptor(version: str = "latest") -> pl.DataFrame:
    """L5 (v1.5+): 2015-16 FiveThirtyEight RAPTOR per player (CC BY 4.0: credit FiveThirtyEight)."""
    return external("raptor_2015_16", version)


def referees(version: str = "latest", game_id: str | int | None = None) -> pl.DataFrame:
    """L5 (v1.5+): the officials of every 2015-16 regular-season game — ``game_id, official_id``
    (NBA person id), ``official_name, jersey_num, source`` (CC BY-SA 4.0 sources:
    ``docs/external_sources.md``)."""
    return _table(version, "external/referees_2015_16", game_id)
