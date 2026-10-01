"""L3 check of the foul state / free-throw links (``nbacore.fouls``) on all games.

Reports the share of free throws linked to a foul, free throws per foul detail, and the
penalty consistency (in penalty → defensive non-shooting common fouls get free throws).
Writes ``reports/l3_fouls_check.json``.

Usage:
    uv run python scripts/l3_fouls_check.py [--workers 4]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore.fouls import _analyse

REPO = Path(__file__).resolve().parents[1]


def run(g: str) -> tuple[list[dict], list[dict]]:
    return _analyse(L.pbp(game_id=g))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    games = L.games().filter(pl.col("status") == "ok")["game_id"].to_list()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, games, chunksize=8))
    f = pl.DataFrame([r for x in res for r in x[0]], infer_schema_length=None)
    ft = pl.DataFrame([r for x in res for r in x[1]], infer_schema_length=None)
    linked = ft.filter(pl.col("linked_foul_event_num").is_not_null())
    per_detail = (
        f.group_by("foul_detail")
        .agg(
            pl.len().alias("n"),
            (pl.col("n_free_throws") > 0).mean().alias("with_ft"),
            pl.col("n_free_throws").mean().alias("ft_per_foul"),
        )
        .sort("n", descending=True)
    )
    nonshoot = f.filter(
        pl.col("foul_detail").is_in(["personal", "personal_block", "personal_take", "loose_ball"])
    )
    stats = {
        "games": len(games),
        "fouls": f.height,
        "free_throws": ft.height,
        "ft_linked": linked.height,
        "ft_linked_frac": linked.height / ft.height,
        "ft_linked_frac_technical": ft.filter(pl.col("ft_kind") == "technical")[
            "linked_foul_event_num"
        ]
        .is_not_null()
        .mean(),
        "ft_linked_frac_other": ft.filter(pl.col("ft_kind") != "technical")["linked_foul_event_num"]
        .is_not_null()
        .mean(),
        "nonshooting_in_penalty_with_ft": nonshoot.filter(pl.col("in_bonus_for_fouled_team"))[
            "n_free_throws"
        ]
        .gt(0)
        .mean(),
        "nonshooting_not_in_penalty_with_ft": nonshoot.filter(~pl.col("in_bonus_for_fouled_team"))[
            "n_free_throws"
        ]
        .gt(0)
        .mean(),
        "per_detail": per_detail.to_dicts(),
        "unlinked_examples": ft.filter(pl.col("linked_foul_event_num").is_null())
        .head(15)
        .to_dicts(),
    }
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k != "unlinked_examples"}, indent=1, default=str
        )
    )
    (REPO / "reports" / "l3_fouls_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
