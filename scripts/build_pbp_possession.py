"""Rebuild only the L3 ``pbp_possession`` table of an L3 build (pbp + the build's ledger).

Used when ``nbacore.ledger.pbp_possession_map`` gains columns (2026-09-27:
``oreb_listed_after_putback`` / ``putback_event_num`` / ``missed_shot_event_num``)
without recomputing the ledger. Rewrites ``<l3>/_pbp_possession/<game_id>.parquet`` and
``<l3>/pbp_possession.parquet``.

Usage:
    uv run python scripts/build_pbp_possession.py --base v1.5 --l3 scratch/l3_v13
"""

from __future__ import annotations

import argparse

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.ledger import pbp_possession_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v1.5")
    ap.add_argument("--l3", required=True)
    a = ap.parse_args()
    l3 = paths.data_root() / a.l3
    pbp = L.pbp(a.base)
    gids = sorted(p.stem for p in (l3 / "_ledger").glob("*.parquet"))
    parts = []
    for gid in gids:
        led = pl.read_parquet(l3 / "_ledger" / f"{gid}.parquet")
        m = pbp_possession_map(pbp.filter(pl.col("game_id") == gid), led)
        m.write_parquet(l3 / "_pbp_possession" / f"{gid}.parquet")
        parts.append(m)
    out = pl.concat(parts, how="vertical_relaxed")
    out.write_parquet(l3 / "pbp_possession.parquet", compression="zstd")
    print(
        out.height,
        "rows,",
        len(gids),
        "games,",
        int(out["oreb_listed_after_putback"].sum()),
        "flagged",
    )


if __name__ == "__main__":
    main()
