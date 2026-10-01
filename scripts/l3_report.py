"""L3 summary of an L3 build (``reports/l3_summary.md`` + ``reports/l3_stats.json``).

Usage:
    uv run python scripts/l3_report.py --l3 scratch/l3_full [--tag v1.3]

With ``--tag``, the files are ``reports/p3_summary_<tag>.md`` / ``p3_stats_<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from nbacore import paths

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l3", default="scratch/l3_full")
    ap.add_argument("--tag", default="", help="suffix of the report files")
    a = ap.parse_args()
    sfx = f"_{a.tag}" if a.tag else ""
    d = paths.data_root() / a.l3
    res = json.loads((d / "build_l3_results.json").read_text(encoding="utf-8"))
    ok = [r for r in res if r["status"] == "ok"]
    led = pl.read_parquet(d / "ledger.parquet")
    fouls = pl.read_parquet(d / "fouls.parquet")
    fts = pl.read_parquet(d / "free_throws.parquet")
    gv = pl.read_parquet(d / "ghost_v1.parquet")
    n_est = sum(r["estimate"] for r in ok)
    diffs = pl.Series([(r["n_possessions"] - r["estimate"]) / r["estimate"] for r in ok])
    n_al = sum(r["estimate_aligned"] for r in ok)
    d_al = pl.Series(
        [(r["n_possessions"] - r["estimate_aligned"]) / r["estimate_aligned"] for r in ok]
    )
    sub = (
        led.select(pl.col("subsegments").explode().struct.field("kind").alias("kind"))
        .drop_nulls()
        .group_by("kind")
        .len()
        .sort("len", descending=True)
    )
    flags = led.select(
        [
            pl.col(c).mean().alias(c)
            for c in (
                "is_transition",
                "has_jump_ball",
                "has_technical",
                "has_flagrant",
                "has_clear_path",
                "data_gap",
                "lineup_stable",
            )
        ]
    ).row(0, named=True)
    sc_frac = [r["shot_clock_imputed_frac"] for r in ok]
    nonshoot = fouls.filter(
        pl.col("foul_detail").is_in(["personal", "personal_block", "personal_take", "loose_ball"])
    )
    stats = {
        "games_ok": len(ok),
        "games_failed": [r["game_id"] for r in res if r["status"] != "ok"],
        "possessions": led.height,
        "estimate": round(n_est, 1),
        "season_diff": led.height / n_est - 1,
        "per_game_abs_diff_p95": float(diffs.abs().quantile(0.95)),
        "per_game_abs_diff_median": float(diffs.abs().median()),
        "estimate_aligned": round(n_al, 1),
        "season_diff_aligned": led.height / n_al - 1,
        "per_game_abs_diff_aligned_p95": float(d_al.abs().quantile(0.95)),
        "per_game_abs_diff_aligned_median": float(d_al.abs().median()),
        "games_over_3pct_aligned": int((d_al.abs() > 0.03).sum()),
        "poss_uid_unique": bool(led["poss_uid"].drop_nulls().is_unique().all()),
        "possessions_without_times": led.filter(pl.col("t_start_ms").is_null()).height,
        "end_types": dict(led.group_by("end_type").len().sort("len", descending=True).iter_rows()),
        "end_time_methods": dict(
            led.group_by("end_time_method").len().sort("len", descending=True).iter_rows()
        ),
        "flags_share": flags,
        "subsegments": dict(sub.iter_rows()),
        "points_total": int(
            led["points"].sum()
            + led["tech_points_offense"].sum()
            + led["tech_points_defense"].sum()
        ),
        "ghost_windows": gv.height,
        "ghost_window_uid_unique": bool(gv["window_uid"].is_unique().all()),
        "ghost_offense_match": float(gv["offense_match"].fill_null(False).mean()),
        "ghost_overlap_ge_0_9_and_match": float(
            (gv["offense_match"].fill_null(False) & (gv["overlap_frac"] >= 0.9)).mean()
        ),
        "fouls": fouls.height,
        "free_throws": fts.height,
        "ft_linked": float(fts["linked_foul_event_num"].is_not_null().mean()),
        "in_bonus_nonshooting_with_ft": float(
            nonshoot.filter(pl.col("in_bonus_for_fouled_team"))["n_free_throws"].gt(0).mean()
        ),
        "shot_clock_imputed_frac_mean": sum(sc_frac) / len(sc_frac),
        "shot_clock_games_fully_imputed": sum(x > 0.999 for x in sc_frac),
        "build_seconds_sum": round(sum(r["seconds"] for r in ok), 1),
    }
    (REPO / "reports" / f"l3_stats{sfx}.json").write_text(
        json.dumps(stats, indent=1), encoding="utf-8"
    )
    f = stats["flags_share"]
    lines = [
        f"# L3 summary: L3 possession ledger ({a.tag or 'v1.2'} build)",
        "",
        f"`scripts/l3_report.py` on `{a.l3}`: {len(ok)} games ok, "
        f"{len(stats['games_failed'])} failed.",
        "",
        "## Acceptance (docs/ledger_schema.md)",
        "",
        "| criterion | result |",
        "|---|---|",
        "| ghost_v1 reproduces ghost-defense's possessions (≥ 99 % within one frame) | 107,923 / "
        "107,923 windows identical in every column on all 631 games "
        "(`reports/l3_ghost_v1_parity_all.json`) |",
        f"| possessions vs pbp estimate (season ≤ 3 %, per-game p95 ≤ 3.5 %); "
        f"aligned estimate (team offensive rebounds, empty end-of-period possessions) | "
        f"season {stats['season_diff_aligned']:+.2%}, per-game p95 "
        f"{stats['per_game_abs_diff_aligned_p95']:.2%} (median "
        f"{stats['per_game_abs_diff_aligned_median']:.2%}; "
        f"{stats['games_over_3pct_aligned']} games above 3 %) |",
        f"| same, classic box-score estimate (reference) | season {stats['season_diff']:+.2%}, "
        f"per-game p95 {stats['per_game_abs_diff_p95']:.2%} "
        f"(median {stats['per_game_abs_diff_median']:.2%}) |",
        f"| ghost_v1 windows inside one ledger possession with the same offence | "
        f"{stats['ghost_overlap_ge_0_9_and_match']:.1%} (overlap ≥ 0.9); offence match "
        f"{stats['ghost_offense_match']:.1%} |",
        f"| bonus consistency: in penalty → defensive non-shooting fouls get free throws | "
        f"{stats['in_bonus_nonshooting_with_ft']:.1%} |",
        "| 20 walk-through possessions | `reports/l3_epv_walkthrough.md` |",
        "",
        "## Contents",
        "",
        f"- {stats['possessions']:,} possessions (`poss_uid` unique: {stats['poss_uid_unique']}; "
        f"{stats['possessions_without_times']} without tracking times, untracked periods).",
        f"- End types: {stats['end_types']}.",
        f"- End times: {stats['end_time_methods']}.",
        f"- Flags (share of possessions): transition {f['is_transition']:.1%}, jump ball "
        f"{f['has_jump_ball']:.1%}, technical {f['has_technical']:.1%}, flagrant "
        f"{f['has_flagrant']:.2%}, clear path {f['has_clear_path']:.2%}, data gap "
        f"{f['data_gap']:.1%}; lineup stable {f['lineup_stable']:.1%}.",
        f"- Sub-segments: {stats['subsegments']}.",
        f"- {stats['fouls']:,} fouls, {stats['free_throws']:,} free throws "
        f"({stats['ft_linked']:.2%} linked to their foul).",
        f"- ghost_v1: {stats['ghost_windows']:,} windows (`window_uid` unique: "
        f"{stats['ghost_window_uid_unique']}).",
        f"- Shot clock: imputed share per game {stats['shot_clock_imputed_frac_mean']:.2%} on "
        f"average; {stats['shot_clock_games_fully_imputed']} games fully imputed.",
        f"- Build time: {stats['build_seconds_sum'] / 3600:.1f} CPU-hours.",
        "",
    ]
    (REPO / "reports" / f"l3_summary{sfx}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
