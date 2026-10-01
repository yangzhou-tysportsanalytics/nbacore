"""Loaders for the raw SportVU game logs (linouk23/NBA-Player-Movements .7z -> JSON).

Raw JSON layout (verified on the 2015-16 public dump, see docs/data_schema.md):

    game   = {"gameid": str, "gamedate": "YYYY-MM-DD", "events": [event, ...]}
    event  = {"eventId": str, "visitor": team, "home": team, "moments": [moment, ...]}
    team   = {"name", "teamid", "abbreviation",
              "players": [{"lastname","firstname","playerid","jersey","position"}]}
    moment = [period, unix_ms, game_clock_s, shot_clock_s | None, None, rows]
    rows   = [[team_id, player_id, x, y, z], ...]   # ball row has team_id == player_id == -1

Nothing here deduplicates: the same (period, unix_ms) frame appears in several events.
Deduplication lives in ``nbacore.clean``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import py7zr

BALL_ID = -1
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
NOMINAL_HZ = 25

FRAME_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "event_id": pl.Int32,
    "moment_idx": pl.Int32,  # index of the moment inside its event (pre-dedup)
    "period": pl.Int8,
    "unix_ms": pl.Int64,
    "game_clock": pl.Float32,
    "shot_clock": pl.Float32,  # null when the shot clock is off
    "team_id": pl.Int64,  # -1 for the ball
    "player_id": pl.Int64,  # -1 for the ball
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,  # ball height; always 0 for players
}

EVENT_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "game_date": pl.Utf8,
    "event_id": pl.Int32,
    "n_moments": pl.Int32,
    "period": pl.Int8,  # period of the first moment (null if no moments)
    "unix_start": pl.Int64,
    "unix_end": pl.Int64,
    "gc_start": pl.Float32,
    "gc_end": pl.Float32,
    "home_team_id": pl.Int64,
    "visitor_team_id": pl.Int64,
}

ROSTER_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "team_id": pl.Int64,
    "player_id": pl.Int64,
    "firstname": pl.Utf8,
    "lastname": pl.Utf8,
    "jersey": pl.Utf8,
    "position": pl.Utf8,
    "side": pl.Utf8,  # "home" | "visitor"
    "abbreviation": pl.Utf8,
    "team_name": pl.Utf8,
}


def find_archives(raw_dir: str | Path) -> list[Path]:
    """All .7z game archives under ``raw_dir`` (sorted by name)."""
    return sorted(Path(raw_dir).glob("*.7z"))


def extract_archive(archive: str | Path, out_dir: str | Path | None = None) -> Path:
    """Extract a .7z game archive; return the path of the JSON inside.

    Idempotent: if the target directory already holds a .json, nothing is extracted.
    """
    archive = Path(archive)
    out_dir = Path(out_dir) if out_dir is not None else archive.with_suffix("")
    existing = sorted(out_dir.glob("*.json"))
    if existing:
        return existing[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive, "r") as z:
        z.extractall(out_dir)
    jsons = sorted(out_dir.glob("*.json"))
    if len(jsons) != 1:
        raise FileNotFoundError(f"expected exactly one .json in {out_dir}, found {jsons}")
    return jsons[0]


def find_game_jsons(raw_dir: str | Path) -> list[Path]:
    """All extracted game JSON files under ``raw_dir`` (one level deep)."""
    return sorted(Path(raw_dir).glob("*/*.json"))


def load_game_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def game_id_of(game: dict) -> str:
    """10-char zero-padded game id, e.g. '0021500115'."""
    return str(game["gameid"]).zfill(10)


# ----------------------------------------------------------------------------------------
# JSON -> tables
# ----------------------------------------------------------------------------------------


def game_to_frames(game: dict) -> pl.DataFrame:
    """Long table: one row per (event, moment, entity). Not deduplicated."""
    gid = game_id_of(game)
    ev_ids: list[int] = []
    mom_idx: list[int] = []
    periods: list[int] = []
    unix: list[int] = []
    gcs: list[float] = []
    scs: list[float | None] = []
    rows: list[list] = []
    for ev in game["events"]:
        eid = int(ev["eventId"])
        for mi, m in enumerate(ev["moments"]):
            ents = m[5]
            n = len(ents)
            if n == 0:
                continue
            ev_ids.extend([eid] * n)
            mom_idx.extend([mi] * n)
            periods.extend([m[0]] * n)
            unix.extend([m[1]] * n)
            gcs.extend([m[2]] * n)
            scs.extend([m[3]] * n)
            rows.extend(ents)
    arr = np.asarray(rows, dtype=np.float64) if rows else np.zeros((0, 5))
    df = pl.DataFrame(
        {
            "game_id": pl.Series([gid] * len(ev_ids), dtype=pl.Utf8),
            "event_id": pl.Series(ev_ids, dtype=pl.Int32),
            "moment_idx": pl.Series(mom_idx, dtype=pl.Int32),
            "period": pl.Series(periods, dtype=pl.Int8),
            "unix_ms": pl.Series(unix, dtype=pl.Int64),
            "game_clock": pl.Series(gcs, dtype=pl.Float32),
            "shot_clock": pl.Series(scs, dtype=pl.Float32),
            "team_id": pl.Series(arr[:, 0].astype(np.int64), dtype=pl.Int64),
            "player_id": pl.Series(arr[:, 1].astype(np.int64), dtype=pl.Int64),
            "x": pl.Series(arr[:, 2], dtype=pl.Float32),
            "y": pl.Series(arr[:, 3], dtype=pl.Float32),
            "z": pl.Series(arr[:, 4], dtype=pl.Float32),
        }
    )
    return df.select(list(FRAME_SCHEMA))


def game_to_events(game: dict) -> pl.DataFrame:
    gid = game_id_of(game)
    recs = []
    for ev in game["events"]:
        ms = ev["moments"]
        recs.append(
            {
                "game_id": gid,
                "game_date": str(game["gamedate"]),
                "event_id": int(ev["eventId"]),
                "n_moments": len(ms),
                "period": ms[0][0] if ms else None,
                "unix_start": ms[0][1] if ms else None,
                "unix_end": ms[-1][1] if ms else None,
                "gc_start": ms[0][2] if ms else None,
                "gc_end": ms[-1][2] if ms else None,
                "home_team_id": ev["home"]["teamid"],
                "visitor_team_id": ev["visitor"]["teamid"],
            }
        )
    return pl.DataFrame(recs, schema=EVENT_SCHEMA)


def game_to_roster(game: dict) -> pl.DataFrame:
    """Roster from the first event that lists players (rosters are constant within a game)."""
    gid = game_id_of(game)
    recs = []
    for ev in game["events"]:
        for side in ("home", "visitor"):
            t = ev[side]
            for p in t["players"]:
                recs.append(
                    {
                        "game_id": gid,
                        "team_id": int(t["teamid"]),
                        "player_id": int(p["playerid"]),
                        "firstname": p["firstname"],
                        "lastname": p["lastname"],
                        "jersey": str(p["jersey"]),
                        "position": p["position"],
                        "side": side,
                        "abbreviation": t["abbreviation"],
                        "team_name": t["name"],
                    }
                )
        if recs:
            break
    return pl.DataFrame(recs, schema=ROSTER_SCHEMA)


def load_game_tables(path: str | Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(frames, events, roster) for one extracted game JSON."""
    game = load_game_json(path)
    return game_to_frames(game), game_to_events(game), game_to_roster(game)
