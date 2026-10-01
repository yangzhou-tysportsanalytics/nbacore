"""ghost_v1 with FG windows ending at the corrected (v1.3 L2) shot release.

Per game: ``ghost_v1_windows`` with ``SegmentConfig(shot_cfg=ShotTimeConfig(shooter_only=True))``
(same code as ``ghost_v1``; only the release rule differs, so crossings, 24 s cropping, the
transition flag and the next window's start follow the new release), keyed by the same
``window_uid`` and mapped to the ledger of ``--l3``. Output under ``<data root>/<out>/``:
``_ghost_v1_l2release/<game_id>.parquet`` and ``ghost_v1_l2release.parquet``.

Report ``reports/l3_ghost_v1_l2release.json`` (against ``<l3>/ghost_v1.parquet``): windows kept /
added / dropped, windows whose end moves and the shift distribution, ``t_terminal`` = L2
``shot_release.t_release_ms`` agreement, and the L2 handler at the new release vs the pbp shooter.

Usage:
    uv run python scripts/build_ghost_v1_l2release.py --base v1.2 --l2 scratch/l2_v13c \
        --l3 scratch/l3_v13 --out scratch/l3_v14 [--workers 3] [--games tiny]
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

import nbacore.load as L
from nbacore import paths
from nbacore.possession.shots import ShotTimeConfig
from nbacore.views.ghost_v1 import SegmentConfig, ghost_v1_windows

REPO = Path(__file__).resolve().parents[1]
CFG = SegmentConfig(shot_cfg=ShotTimeConfig(shooter_only=True))
PART = "_ghost_v1_l2release"


def run_game(gid: str, base: str, l2: str, l3: str, out: str) -> dict:
    t0 = time.time()
    try:
        fr, fi = L.frames(gid, base), L.frame_index(gid, base).sort("period", "unix_ms")
        pbp = L.pbp(base, game_id=gid)
        home = int(L.games(base).filter(pl.col("game_id") == gid)["home_team_id"][0])
        led = pl.read_parquet(Path(l3) / "_ledger" / f"{gid}.parquet")
        shots = pl.read_parquet(Path(l2) / "events" / f"{gid}.parquet").filter(
            pl.col("event_type") == "shot_release"
        )
        win, rej = ghost_v1_windows(
            fr, fi, pbp, L.attack_direction(base, game_id=gid), home, led, shots, CFG
        )
        win.write_parquet(Path(out) / PART / f"{gid}.parquet")
        return {
            "game_id": gid,
            "status": "ok",
            "seconds": round(time.time() - t0, 1),
            "n": win.height,
            "rejects": rej,
        }
    except Exception as e:  # noqa: BLE001 - recorded per game
        return {
            "game_id": gid,
            "status": f"failed:{type(e).__name__}",
            "error": f"{e}\n{traceback.format_exc(limit=6)}",
        }


def report(new: pl.DataFrame, l2: Path, l3: Path, gids: list[str]) -> dict:
    old = pl.read_parquet(l3 / "ghost_v1.parquet").filter(pl.col("game_id").is_in(gids))
    j = old.join(new, on="window_uid", how="inner", suffix="_n")
    moved = j.filter(pl.col("t_end") != pl.col("t_end_n"))
    shift = (moved["t_end"] - moved["t_end_n"]) / 1000
    fg = new.filter(pl.col("terminal_type").is_in(["fg_made", "fg_missed"]))
    rel = pl.concat(
        [
            pl.read_parquet(l2 / "events" / f"{g}.parquet")
            .filter(pl.col("event_type") == "shot_release")
            .select("game_id", pl.col("pbp_event_num").alias("terminal_event_num"), "t_release_ms")
            for g in gids
        ]
    )
    fgj = fg.join(rel, on=["game_id", "terminal_event_num"], how="left")
    hand = pl.concat(
        [
            pl.read_parquet(
                l2 / "handler" / f"{g}.parquet", columns=["game_id", "unix_ms", "handler_id"]
            )
            for g in gids
        ]
    )
    # last real handler (handler_id -1 = none: the ball is already leaving the hand at the
    # release frame) within 1 s before the release
    hand = hand.filter(pl.col("handler_id") >= 0).sort("unix_ms")
    fh = (
        fg.filter(pl.col("terminal_player_id").is_not_null())
        .sort("t_terminal")
        .join_asof(
            hand,
            left_on="t_terminal",
            right_on="unix_ms",
            by="game_id",
            strategy="backward",
            tolerance=1000,
        )
    )
    q = [0.1, 0.5, 0.9]
    return {
        "games": len(gids),
        "windows_ghost_v1": old.height,
        "windows_l2release": new.height,
        "kept": j.height,
        "dropped": old.height - j.height,
        "added": new.height - j.height,
        "end_moved": moved.height,
        "end_moved_frac_of_fg": round(moved.height / max(1, fg.height), 4),
        "end_moved_earlier": int((shift > 0).sum()),
        "end_moved_later": int((shift < 0).sum()),
        "shift_s_quantiles_earlier": {
            str(x): float(shift.filter(shift > 0).quantile(x) or 0) for x in q
        },
        "start_moved": j.filter(pl.col("t_start") != pl.col("t_start_n")).height,
        "is_transition_changed": j.filter(
            pl.col("is_transition") != pl.col("is_transition_n")
        ).height,
        "fg_windows": fg.height,
        "t_terminal_equals_l2_release": round(
            float((fgj["t_terminal"] == fgj["t_release_ms"]).mean()), 4
        ),
        "shooter_inferred_is_pbp_shooter": round(
            float((fg["shooter_inferred_id"] == fg["terminal_player_id"]).fill_null(False).mean()),
            4,
        ),
        "last_handler_1s_before_release_is_pbp_shooter": round(
            float((fh["handler_id"] == fh["terminal_player_id"]).fill_null(False).mean()), 4
        ),
        "no_handler_in_1s_before_release": round(float(fh["handler_id"].is_null().mean()), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v1.2")
    ap.add_argument("--l2", required=True)
    ap.add_argument("--l3", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--games", default="all", help="all | tiny | small")
    args = ap.parse_args()
    root = paths.data_root()
    l2, l3, out = root / args.l2, root / args.l3, root / args.out
    (out / PART).mkdir(parents=True, exist_ok=True)
    games = L.games(args.base).filter(pl.col("status") == "ok")
    if args.games != "all":
        names = json.loads((paths.raw_dir() / f"manifest_{args.games}.json").read_text())[
            "archives"
        ]
        games = games.filter(pl.col("archive_name").is_in(names))
    gids = sorted(games["game_id"].to_list())
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_game, g, args.base, str(l2), str(l3), str(out)) for g in gids]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            results.append(r)
            print(f"[{i}/{len(gids)}] {r['game_id']}: {r['status']} {r.get('n', '')}", flush=True)
    bad = [r for r in results if r["status"] != "ok"]
    if bad:
        raise SystemExit(f"{len(bad)} failed: {bad[:2]}")
    new = pl.concat([pl.read_parquet(out / PART / f"{g}.parquet") for g in gids])
    new.write_parquet(out / "ghost_v1_l2release.parquet", compression="zstd")
    stats = report(new, l2, l3, gids)
    stats["build_seconds"] = round(time.time() - t0)
    name = (
        "l3_ghost_v1_l2release.json"
        if args.games == "all"
        else f"p3_ghost_v1_l2release_{args.games}.json"
    )
    (REPO / "reports" / name).write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
