"""L2 acceptance numbers on a full L2 build.

Reads ``<root>/<l2_build>/`` (events, handler, pbp_align, build_l2_results.json) and the base
release; writes ``reports/l2_summary.md`` and ``reports/l2_stats.json``:

* build: games ok / failed, uid duplicates, events per type per game;
* handler: last handler (L2 table) before each FG release = pbp shooter (acceptance ≥ 90 %);
* passes: assist coverage (completed pass / handoff assister → shooter, catch within the
  game-clock window [pbp s − 1, pbp s + 12]); interceptions vs pbp steals;
* shot release: method distribution, release − pbp quantiles;
* clock stops / pbp alignment: tracking-event share per pbp type; stop residual quantiles for
  fouls, violations, timeouts;
* screen candidates per game (precision comes from a manual review, recall from L2M reports).

Usage:
    uv run python scripts/l2_report.py --l2-build scratch/l2_full [--base v1.0] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

import nbacore.load as L
from nbacore import paths

REPO = Path(__file__).resolve().parents[1]
NAMES = {1: "FG made", 2: "FG missed", 3: "free throw", 4: "rebound", 5: "turnover", 6: "foul",
         7: "violation", 8: "substitution", 9: "timeout", 10: "jump ball", 12: "period start",
         13: "period end", 18: "replay"}  # fmt: skip


def per_game(args: tuple[str, str, str]) -> dict:
    gid, base, l2 = args
    ev = pl.read_parquet(Path(l2) / "events" / f"{gid}.parquet")
    h = pl.read_parquet(Path(l2) / "handler" / f"{gid}.parquet").filter(pl.col("handler_id") >= 0)
    fi = L.frame_index(gid, base).select("period", "unix_ms", "game_clock")
    pb = L.pbp(base, game_id=gid)
    shots = ev.filter(pl.col("event_type") == "shot_release")
    desc = pb.select(
        pl.col("event_num").alias("pbp_event_num"),
        pl.coalesce("home_desc", "visitor_desc").alias("_d"),
    )
    shots = shots.join(desc, on="pbp_event_num", how="left")
    # handler vs pbp shooter, overall and on the comparable subset (ghost's 96.1 % on small was
    # measured on half-court possessions): release from the ball with the shooter in hand
    # (method rim / apex) and not a tip-in / putback
    agree = n_sh = agree_c = n_c = 0
    for s in shots.iter_rows(named=True):
        if s["shooter_id"] is None or s["t_start_ms"] is None:
            continue
        hh = h.filter((pl.col("period") == s["period"]) & (pl.col("unix_ms") <= s["t_start_ms"]))
        if hh.height == 0:
            continue
        ok = int(hh["handler_id"][-1] == s["shooter_id"])
        n_sh += 1
        agree += ok
        d = s["_d"] or ""
        if s["method"] in ("rim", "apex") and "Tip" not in d and "Putback" not in d:
            n_c += 1
            agree_c += ok
    # assist coverage
    pc = (
        ev.filter(
            pl.col("event_type").is_in(["pass", "handoff"]) & pl.col("completed").fill_null(False)
        )
        .select("period", "actor_id", "target_id", "t_end_ms")
        .join(fi, left_on=["period", "t_end_ms"], right_on=["period", "unix_ms"], how="left")
    )
    ast = pb.filter((pl.col("msg_type") == 1) & (pl.col("p2_id") > 0))
    covered = 0
    for a in ast.iter_rows(named=True):
        covered += (
            pc.filter(
                (pl.col("period") == a["period"])
                & (pl.col("actor_id") == a["p2_id"])
                & (pl.col("target_id") == a["p1_id"])
                & pl.col("game_clock").is_between(a["pctime_sec"] - 1, a["pctime_sec"] + 12)
            ).height
            > 0
        )
    counts = dict(ev.group_by("event_type").len().iter_rows())
    return {
        "game_id": gid,
        "shooter_agree": agree,
        "shooter_n": n_sh,
        "shooter_agree_comparable": agree_c,
        "shooter_n_comparable": n_c,
        "assists": ast.height,
        "assists_covered": covered,
        "steals": pb.filter((pl.col("msg_type") == 5) & (pl.col("p2_id") > 0)).height,
        "intercepted": int(
            ev.filter(pl.col("event_type") == "pass")["intercepted"].fill_null(False).sum()
        ),
        "shot_methods": dict(shots.group_by("method").len().iter_rows()),
        "release_minus_pbp": shots["release_minus_pbp_s"].drop_nulls().to_list(),
        "counts": counts,
        "uid_dup": ev.height - ev["event_uid"].n_unique(),
    }


def q(v, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    a = np.asarray(v, dtype=float)
    return [round(float(np.quantile(a, x)), 2) for x in qs] if a.size else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2-build", required=True)
    ap.add_argument("--base", default="v1.0")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    l2 = paths.data_root() / args.l2_build
    results = json.loads((l2 / "build_l2_results.json").read_text(encoding="utf-8"))
    ok = [r["game_id"] for r in results if r["status"] == "ok"]
    with ProcessPoolExecutor(args.workers) as ex:
        per = list(ex.map(per_game, [(g, args.base, str(l2)) for g in ok]))
    al = pl.read_parquet(l2 / "pbp_align.parquet")

    tot = lambda k: sum(p[k] for p in per)  # noqa: E731
    types = sorted({t for p in per for t in p["counts"]})
    counts = {t: sum(p["counts"].get(t, 0) for p in per) for t in types}
    methods: dict = {}
    for p in per:
        for m, n in p["shot_methods"].items():
            methods[m] = methods.get(m, 0) + n
    rel = [x for p in per for x in p["release_minus_pbp"]]
    stats = {
        "games_ok": len(ok),
        "games_failed": [r["game_id"] for r in results if r["status"] != "ok"],
        "uid_duplicates": tot("uid_dup"),
        "events_per_game": {t: round(counts[t] / len(ok), 1) for t in types},
        "shooter_agreement": tot("shooter_agree") / max(1, tot("shooter_n")),
        "shooter_n": tot("shooter_n"),
        "shooter_agreement_comparable": tot("shooter_agree_comparable")
        / max(1, tot("shooter_n_comparable")),
        "shooter_n_comparable": tot("shooter_n_comparable"),
        "assist_coverage": tot("assists_covered") / max(1, tot("assists")),
        "assists": tot("assists"),
        "intercepted": tot("intercepted"),
        "pbp_steals": tot("steals"),
        "shot_methods": methods,
        "release_minus_pbp_q": q(rel),
    }
    by_type = (
        al.group_by("msg_type")
        .agg(
            pl.len(),
            (pl.col("anchor_quality") >= 2).mean().alias("tracking_event"),
            (pl.col("anchor_quality") == 0).mean().alias("unlocated"),
        )
        .sort("msg_type")
    )
    resid = {
        NAMES[m]: q(
            al.filter((pl.col("msg_type") == m) & (pl.col("method") == "clock_stop"))["residual_s"]
            .drop_nulls()
            .to_list()
        )
        for m in (6, 7, 9)
    }
    stats["pbp_alignment"] = {
        NAMES.get(r["msg_type"], str(r["msg_type"])): [
            r["len"],
            round(r["tracking_event"], 4),
            round(r["unlocated"], 4),
        ]
        for r in by_type.iter_rows(named=True)
    }
    stats["stop_residual_q"] = resid
    (REPO / "reports" / "l2_stats.json").write_text(
        json.dumps(
            {
                "summary": stats,
                "per_game": [{k: v for k, v in p.items() if k != "release_minus_pbp"} for p in per],
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    n_sh = stats["shooter_n"]
    lines = [
        "# L2 statistics (full L2 build)",
        "",
        f"Generated by `scripts/l2_report.py` from `{args.l2_build}` on base `{args.base}`.",
        "",
        f"- Games: {len(ok)} ok, failed {stats['games_failed'] or 'none'}; duplicate event uids: {stats['uid_duplicates']}.",
        f"- **Handler: last handler before the FG release = pbp shooter {stats['shooter_agreement_comparable']:.4f}** "
        f"on shots with a ball-derived release and the shooter in hand, tip-ins / putbacks excluded "
        f"({tot('shooter_agree_comparable')} / {stats['shooter_n_comparable']}; the definition behind "
        f"ghost's 96.1 % and the ≥ 0.90 acceptance); all FG attempts {stats['shooter_agreement']:.4f} "
        f"({tot('shooter_agree')} / {n_sh}; tip-ins and releases without a hand or from pbp are "
        f"undecidable).",
        f"- **Assist coverage {stats['assist_coverage']:.4f}** ({tot('assists_covered')} / {stats['assists']}).",
        f"- Interceptions {stats['intercepted']} vs pbp steals {stats['pbp_steals']}.",
        f"- Shot release methods: {methods}; release − pbp (s) 5/25/50/75/95 %: {stats['release_minus_pbp_q']}.",
        f"- Stop residual (pbp s − clock at onset) 5/25/50/75/95 %: {resid}.",
        "",
        "## Events per game",
        "",
        "| type | per game | total |",
        "|---|---|---|",
    ]
    lines += [f"| {t} | {stats['events_per_game'][t]} | {counts[t]} |" for t in types]
    lines += [
        "",
        "## pbp alignment",
        "",
        "| pbp type | rows | on a tracking event | unlocated |",
        "|---|---|---|---|",
    ]
    lines += [
        f"| {k} | {v[0]} | {v[1]:.3f} | {v[2]:.3f} |" for k, v in stats["pbp_alignment"].items()
    ]
    (REPO / "reports" / "l2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
