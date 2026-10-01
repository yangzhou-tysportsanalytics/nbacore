"""Per-game L1 build: raw archive -> deduped frames, frame index, rosters, tracking events,
attack direction, plus raw-level diagnostics.

``process_archive`` is the single code path used by ``scripts/build_l1.py`` and by the parity
test against ghost-defense (``tests/test_l1_parity.py``). It never raises for a bad archive:
the outcome is reported in ``result["status"]``:

* ``ok``                     tables written
* ``empty_archive``          the .7z holds no file
* ``failed:<kind>``          anything else (``kind`` = exception class name), with the message in
                             ``result["error"]``
"""

from __future__ import annotations

import time
import traceback
from pathlib import Path

import polars as pl
import py7zr

from nbacore import paths
from nbacore.clean.dedup import build_frame_index, dedup_entities, dedup_report
from nbacore.io import (
    extract_archive,
    game_id_of,
    game_to_events,
    game_to_frames,
    game_to_roster,
    load_game_json,
    validate_game,
)
from nbacore.possession.direction import infer_attack_direction


def archive_is_empty(archive: str | Path) -> bool:
    with py7zr.SevenZipFile(archive, "r") as z:
        return len(z.getnames()) == 0


def build_game_tables(game: dict, pbp_game: pl.DataFrame) -> tuple[dict, dict]:
    """L1 tables for one parsed game JSON, and the raw-level diagnostics."""
    raw = game_to_frames(game)
    ent = dedup_entities(raw)
    idx = build_frame_index(ent)
    direction, diag = infer_attack_direction(ent, pbp_game)
    tables = {
        "frames": ent,
        "frame_index": idx,
        "rosters": game_to_roster(game),
        "tracking_events": game_to_events(game),
        "attack_direction": direction,
    }
    return tables, {"dedup": dedup_report(raw, ent, idx), "direction": diag}


def process_archive(
    archive: str | Path,
    out_dir: str | Path,
    pbp_parquet: str | Path,
    force: bool = False,
) -> dict:
    """Build L1 for one archive; write per-game parquet files under ``out_dir``.

    ``out_dir/frames/<gid>.parquet`` and ``out_dir/frame_index/<gid>.parquet`` are written;
    the small per-game tables (rosters, tracking_events, attack_direction) are returned in
    ``result["tables"]`` for the caller to concatenate. ``pbp_parquet`` is the typed pbp of the
    whole season (only this game's rows are read).
    """
    archive, out_dir = Path(archive), Path(out_dir)
    t0 = time.time()
    res: dict = {"archive_name": archive.name, "status": None, "game_id": None}
    try:
        if archive_is_empty(archive):
            res["status"] = "empty_archive"
            return res
        game = load_game_json(
            extract_archive(archive, paths.scratch_dir("extracted", archive.stem))
        )
        gid = game_id_of(game)
        res.update(
            game_id=gid,
            game_date=str(game["gamedate"]),
            home_team_id=int(game["events"][0]["home"]["teamid"]),
            visitor_team_id=int(game["events"][0]["visitor"]["teamid"]),
            home_abbr=game["events"][0]["home"]["abbreviation"],
            visitor_abbr=game["events"][0]["visitor"]["abbreviation"],
        )
        res["raw_stats"] = validate_game(game).to_dict()
        pbp_game = pl.scan_parquet(pbp_parquet).filter(pl.col("game_id") == gid).collect()
        out_frames = out_dir / "frames" / f"{gid}.parquet"
        out_index = out_dir / "frame_index" / f"{gid}.parquet"
        tables, diag = build_game_tables(game, pbp_game)
        del game
        if force or not (out_frames.exists() and out_index.exists()):
            out_frames.parent.mkdir(parents=True, exist_ok=True)
            out_index.parent.mkdir(parents=True, exist_ok=True)
            tables["frames"].write_parquet(out_frames, compression="zstd")
            tables["frame_index"].write_parquet(out_index, compression="zstd")
        res.update(diag)
        res["tables"] = {k: tables[k] for k in ("rosters", "tracking_events", "attack_direction")}
        res["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - every failure is recorded per game
        res["status"] = f"failed:{type(e).__name__}"
        res["error"] = f"{e}\n{traceback.format_exc(limit=3)}"
    finally:
        res["seconds"] = round(time.time() - t0, 1)
    return res
