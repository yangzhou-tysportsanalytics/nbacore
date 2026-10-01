"""Export a public data package of nbacore derived tables (no frame-level coordinates).

Scope: the tables the ghost-defense study reads, from a published
release — games / rosters / attack direction / typed pbp, ``ghost_v1`` (both release rules),
the L2 event types it uses (``screen_candidate``, ``pass``, ``handoff``, ``shot_release``,
``possession_touch``; one file per type, all tracked games), ``event_possession``, ``ledger``,
the per-frame shot clock (readings only) and FiveThirtyEight RAPTOR (CC BY 4.0).
Never included: ``frames``, ``frame_index``, ``handler``, ``event_frames`` (per-frame or
distance snapshots derived from player positions) and Basketball-Reference / NBA.com /
Kaggle-sourced L5 tables whose terms do not allow redistribution.

Event rows carry the event location (``x``, ``y``, ``x_release`` …), not tracks. ``--coords``
: ``all`` keeps them, ``shot`` keeps only the shot release
location (``shot_release.x_release / y_release``), ``none`` drops every point coordinate.
Kept and dropped coordinate columns are listed per file in ``MANIFEST.json``.

Output: ``<out>/nbacore-derived-<version>-<coords>-p<N>/`` (``N`` = ``PACKAGE_REVISION``, bumped
when the package content changes for the same data version) with ``tables/``, ``events/``,
``external/``, ``MANIFEST.json`` (package id, rows, columns, sha256) and ``README_DATA.md``.
Refuses an existing output.

Usage:
    uv run python scripts/export_public_data.py --coords shot [--version v1.5]
        [--out public_data]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

import nbacore.load as L

EVENT_TYPES = ["screen_candidate", "pass", "handoff", "shot_release", "possession_touch"]
COORD = ("x", "y", "z")
PACKAGE_REVISION = 1
SHOT_LOCATION = {"events/shot_release.parquet": ["x_release", "y_release"]}


def coord_cols(df: pl.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c in COORD or c.startswith(("x_", "y_", "z_")) or c.endswith(("_x", "_y"))
    ]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1.5")
    ap.add_argument("--out", default="public_data")
    ap.add_argument("--coords", required=True, choices=["all", "shot", "none"])
    a = ap.parse_args()
    v = a.version
    package_id = f"nbacore-derived-{v}-{a.coords}-p{PACKAGE_REVISION}"
    out = Path(a.out) / package_id
    if out.exists():
        raise SystemExit(f"{out} exists; bump PACKAGE_REVISION or choose another --out")
    for d in ("tables", "events", "external"):
        (out / d).mkdir(parents=True)
    gids = L.games(v).filter(pl.col("status") == "ok")["game_id"].sort().to_list()

    written: dict[str, pl.DataFrame] = {}
    dropped: dict[str, list[str]] = {}

    def put(rel: str, df: pl.DataFrame) -> None:
        cc = coord_cols(df)
        keep = cc if a.coords == "all" else SHOT_LOCATION.get(rel, []) if a.coords == "shot" else []
        drop = [c for c in cc if c not in keep]
        df = df.drop(drop)
        df.write_parquet(out / rel, compression="zstd")
        written[rel], dropped[rel] = df, drop

    put("tables/games.parquet", L.games(v))
    put("tables/rosters.parquet", L.rosters(v))
    put("tables/attack_direction.parquet", L.attack_direction(v))
    put("tables/pbp.parquet", L.pbp(v))
    put("tables/ghost_v1.parquet", L.ghost_v1(v))
    put("tables/ghost_v1_l2release.parquet", L.ghost_v1(v, release="l2"))
    put("tables/event_possession.parquet", L.event_possession(v))
    put("tables/ledger.parquet", L.ledger(v))
    for t in EVENT_TYPES:
        put(
            f"events/{t}.parquet",
            pl.concat([L.events(g, t, v) for g in gids], how="diagonal_relaxed"),
        )
    sc = pl.concat(
        [L.shot_clock(g, v).with_columns(pl.lit(g).alias("game_id")) for g in gids],
        how="vertical_relaxed",
    )
    put(
        "tables/shot_clock.parquet",
        sc.select("game_id", *[c for c in sc.columns if c != "game_id"]),
    )
    put("external/raptor_2015_16.parquet", L.raptor(v))

    manifest = {
        "package_id": package_id,
        "package_revision": PACKAGE_REVISION,
        "coords_mode": a.coords,
        "nbacore_data_version": v,
        "nbacore_code_tag": f"data-{v}",
        "tracked_games": len(gids),
        "files": {
            rel: {
                "rows": df.height,
                "bytes": (out / rel).stat().st_size,
                "sha256": sha256(out / rel),
                "columns": df.columns,
                "coordinate_columns": coord_cols(df),
                "dropped_coordinate_columns": dropped[rel],
            }
            for rel, df in sorted(written.items())
        },
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    mode = {
        "all": "Event rows keep their point locations (release, catch, landing, screen contact).",
        "shot": "Only the shot release location is kept (shot_release.x_release / y_release); "
        "all other point coordinates are dropped.",
        "none": "No point coordinates at all.",
    }[a.coords]
    (out / "README_DATA.md").write_text(
        README.format(v=v, n=len(gids), pid=package_id, mode=mode), encoding="utf-8"
    )
    total = sum(f["bytes"] for f in manifest["files"].values())
    print(f"{len(written)} files, {total / 1e6:.0f} MB in {out}")
    for rel, f in manifest["files"].items():
        print(f"  {rel}: {f['rows']:,} rows, coords {f['coordinate_columns']}")


README = """# {pid} (nbacore derived tables, public data package)

Derived tables of the public 2015-16 NBA SportVU release, built by nbacore (data version {v},
code tag `data-{v}`) for the {n} usable tracked games; no player or ball tracks. {mode} Column definitions: `docs/DATA_GUIDE.md`,
`docs/event_definitions.md`, `docs/ledger_schema.md` in the nbacore repository; row counts,
columns and sha256 of every file: `MANIFEST.json`.

## Contents
| path | content |
|---|---|
| `tables/games.parquet` | games of the tracking period, status, fixed split / fold / parity |
| `tables/rosters.parquet`, `tables/attack_direction.parquet` | rosters; attack direction per team and period |
| `tables/pbp.parquet` | typed 2015-16 play-by-play (all 1,230 games) |
| `tables/ghost_v1.parquet`, `tables/ghost_v1_l2release.parquet` | half-court possession windows (original / corrected shot release) |
| `tables/ledger.parquet`, `tables/event_possession.parquet` | possession ledger; event → possession mapping |
| `tables/shot_clock.parquet` | per-frame shot clock readings (observed or rule-imputed; no positions) |
| `events/<type>.parquet` | `screen_candidate`, `pass`, `handoff`, `shot_release`, `possession_touch` events |
| `external/raptor_2015_16.parquet` | FiveThirtyEight RAPTOR, CC BY 4.0 — credit FiveThirtyEight |

## Not included
Player / ball tracks (`frames`, `frame_index`), the per-frame ball handler and distance
snapshots: they are, or are computed directly from, the raw SportVU coordinates, whose licence
is unclear. They can be rebuilt from the public raw logs (github.com/linouk23/NBA-Player-Movements)
with the nbacore code. The point coordinates kept in this package, if any, are listed per
file in `MANIFEST.json` (`coordinate_columns`; removed ones in `dropped_coordinate_columns`). External tables from Basketball-Reference, NBA.com or
Kaggle sources are not redistributed; the nbacore repository documents how to obtain them.

## Sources and terms
Underlying data © NBA / NBA.com (SportVU tracking, play-by-play). Derived tables are shared for
research reproducibility; RAPTOR is © FiveThirtyEight under CC BY 4.0.
"""


if __name__ == "__main__":
    main()
