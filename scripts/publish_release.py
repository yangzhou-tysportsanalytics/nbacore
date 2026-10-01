"""Publish an immutable data release.

* v1.0-style (``--build`` only): copies the L1 + L4 tables of every ``ok`` game from an L1 build.
* Incremental (``--base vX.Y --l2-build ...`` and / or ``--l3-build ...``): takes every file of
  the base release (hard links when possible — the files are read-only, so sharing them keeps
  both releases immutable) and adds the L2 tables of an L2 build (``handler/``, ``events/``,
  ``pbp_align.parquet``) and / or the L3 tables of an L3 build (``ledger.parquet``,
  ``fouls.parquet``, ``free_throws.parquet``, ``ghost_v1.parquet``, ``shot_clock/``) and / or
  the L5 tables of ``--l5-build`` (``external/<name>.parquet`` + ``external/SOURCES.json`` from
  the raw ``provenance.json`` files). A layer given as a build *replaces* that layer of the base:
  its base files are not linked (a hard link would share the file with the base release), and a
  new L2 needs a new L3 when the base has one (L3 is derived from L2).

Then writes ``MANIFEST.json`` and ``CHANGELOG.md``, makes every file read-only, verifies, and
(unless ``--no-latest``) points ``releases/latest.txt`` at the new version.

Refuses when the version already exists, when the code working tree has uncommitted changes, or
when a build is incomplete. Tag the code afterwards: ``git tag data-<version>``.

Usage:
    uv run python scripts/publish_release.py --version v1.0 --build scratch/l1_full
    uv run python scripts/publish_release.py --version v1.1 --base v1.0 --l2-build scratch/l2_full
    uv run python scripts/publish_release.py --version v1.2 --base v1.1 --l3-build scratch/l3_full
    uv run python scripts/publish_release.py --version v1.4 --base v1.3 --l3-add scratch/l3_v13
    uv run python scripts/publish_release.py --version v1.3 --base v1.2 --l2-build scratch/l2_v13         --l3-build scratch/l3_v13 --l5-build scratch/l5
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import polars as pl

from nbacore import paths
from nbacore.events.build import all_params
from nbacore.io.listing import sha256_file
from nbacore.release import (
    MANIFEST,
    config_hash,
    git_state,
    l1_config,
    l3_params,
    make_read_only,
    verify_release,
    write_manifest,
)

REPO = Path(__file__).resolve().parents[1]
TABLES = ["games", "rosters", "pbp", "attack_direction", "tracking_events"]
PER_GAME = ["frames", "frame_index"]
L2_PER_GAME = ["handler", "events"]
L3_TABLES = ["ledger", "fouls", "free_throws", "ghost_v1", "event_possession", "pbp_possession"]
L3_PER_GAME = ["shot_clock"]
L3_OPTIONAL = ["ghost_v1_l2release"]  # copied when the L3 build has it (v1.4)
# L5 table -> raw/external/<source dir> holding its provenance.json
L5_TABLES = {
    "shots_2015_16": "shots_domsamangy",
    "player_bio_2015_16": "player_bio_kaggle",
    "box_games_2015_16": "boxscores_eoinamoore_v515",
    "box_players_2015_16": "boxscores_eoinamoore_v515",
    "bbref_player_map_2015_16": "bbref_sumitrodatta",
    "advanced_2015_16": "bbref_sumitrodatta",
    "awards_2015_16": "bbref_sumitrodatta",
    "raptor_2015_16": "raptor_538",
    "referees_2015_16": "referees_kaggle",
}
LAYER_PATHS = {  # release paths of a layer (prefixes), replaced when the layer is rebuilt
    "L2": ["handler/", "events/", "event_frames/", "pbp_align.parquet"],
    "L3": [f"{t}/" for t in L3_PER_GAME] + [f"{t}.parquet" for t in L3_TABLES + L3_OPTIONAL],
    "L5": ["external/"],
}


def _changelog_without_drafts() -> str:
    """CHANGELOG.md without the sections of unpublished versions (headings marked DRAFT)."""
    out, skip = [], False
    for line in (REPO / "CHANGELOG.md").read_text(encoding="utf-8").splitlines(keepends=True):
        if line.startswith("## "):
            skip = "DRAFT" in line
        if not skip:
            out.append(line)
    return "".join(out)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--build", help="L1 build dir relative to the data root (full release)")
    ap.add_argument("--base", help="previous release to extend")
    ap.add_argument("--l2-build", help="L2 build dir relative to the data root")
    ap.add_argument("--l3-build", help="L3 build dir relative to the data root")
    ap.add_argument("--l5-build", help="dir with the L5 tables (<name>.parquet), rel. to data root")
    ap.add_argument(
        "--l3-add",
        help="dir with optional L3 tables (L3_OPTIONAL) to add; the base L3 stays linked",
    )
    ap.add_argument("--no-latest", action="store_true")
    args = ap.parse_args()
    if bool(args.build) == bool(args.base):
        raise SystemExit("give either --build (full release) or --base with --l2/--l3/--l5-build")
    if args.base and not (args.l2_build or args.l3_build or args.l5_build or args.l3_add):
        raise SystemExit("--base needs --l2-build, --l3-build, --l3-add and / or --l5-build")
    if args.l3_add and (args.l3_build or args.l2_build):
        raise SystemExit("--l3-add only extends the base L3; not with --l2-build / --l3-build")

    version = paths.resolve_version(args.version)
    dest = paths.releases_dir() / version
    if dest.exists():
        raise SystemExit(f"{dest} exists; published releases are immutable")
    commit, clean = git_state(REPO)
    if not clean:
        raise SystemExit("code working tree has uncommitted changes; commit before publishing")
    tmp = dest.with_name(dest.name + ".partial")
    if tmp.exists():
        shutil.rmtree(tmp)

    cfg = l1_config()
    meta: dict = {"version": version, "code_commit": commit}
    if args.build:
        build = paths.data_root() / args.build
        games = pl.read_parquet(build / "games.parquet")
        ok_ids = games.filter(pl.col("status") == "ok")["game_id"].to_list()
        for sub in PER_GAME:
            missing = [g for g in ok_ids if not (build / sub / f"{g}.parquet").exists()]
            if missing:
                raise SystemExit(
                    f"build incomplete: {len(missing)} {sub} files missing, e.g. {missing[:3]}"
                )
        for sub in PER_GAME:
            (tmp / sub).mkdir(parents=True)
            for g in ok_ids:
                shutil.copy2(build / sub / f"{g}.parquet", tmp / sub / f"{g}.parquet")
        for t in TABLES:
            shutil.copy2(build / f"{t}.parquet", tmp / f"{t}.parquet")
        results = json.loads((build / "build_results.json").read_text(encoding="utf-8"))
        meta.update(
            layers=["L1", "L4"],
            inputs={
                "github_listing_sha256": sha256_file(paths.listing_json()),
                "pbp_csv_sha256": sha256_file(paths.pbp_csv()),
                "n_archives": len(results),
            },
            build_seconds_sum=round(sum(r.get("seconds", 0) for r in results), 1),
        )
    else:
        base_dir = paths.release_dir(args.base)
        base_manifest = json.loads((base_dir / MANIFEST).read_text(encoding="utf-8"))
        if verify_release(base_dir):
            raise SystemExit(f"base release {args.base} does not verify")
        games = pl.read_parquet(base_dir / "games.parquet")
        ok_ids = games.filter(pl.col("status") == "ok")["game_id"].to_list()
        rebuilt = [
            layer
            for layer, given in (
                ("L2", args.l2_build),
                ("L3", args.l3_build),
                ("L5", args.l5_build),
            )
            if given
        ]
        if "L2" in rebuilt and "L3" in base_manifest["layers"] and "L3" not in rebuilt:
            raise SystemExit("a new L2 needs a new L3 (--l3-build): L3 is derived from L2")
        replaced = tuple(pre for layer in rebuilt for pre in LAYER_PATHS[layer])
        for rel in base_manifest["files"]:
            if rel != "CHANGELOG.md" and not rel.startswith(replaced):
                _link_or_copy(base_dir / rel, tmp / rel)
        meta.update(
            layers=list(base_manifest["layers"]),
            base_release=args.base,
            base_code_commit=base_manifest["code_commit"],
            inputs=base_manifest["inputs"],
        )
        cfg = base_manifest.get("config", cfg)
        if args.l2_build:
            l2 = paths.data_root() / args.l2_build
            results = json.loads((l2 / "build_l2_results.json").read_text(encoding="utf-8"))
            failed = [r["game_id"] for r in results if r["status"] != "ok"]
            for sub in L2_PER_GAME:
                missing = [g for g in ok_ids if not (l2 / sub / f"{g}.parquet").exists()]
                if missing:
                    raise SystemExit(
                        f"L2 build incomplete: {len(missing)} {sub} files missing, "
                        f"e.g. {missing[:3]}"
                    )
            for sub in L2_PER_GAME:
                for g in ok_ids:
                    _copy(l2 / sub / f"{g}.parquet", tmp / sub / f"{g}.parquet")
            if (l2 / "event_frames").is_dir():  # optional
                ef = l2 / "event_frames"
                missing = [g for g in ok_ids if not (ef / f"{g}.parquet").exists()]
                if missing:
                    raise SystemExit(
                        f"event_frames incomplete: {len(missing)} missing, e.g. {missing[:3]}"
                    )
                for g in ok_ids:
                    _copy(ef / f"{g}.parquet", tmp / "event_frames" / f"{g}.parquet")
            shutil.copy2(l2 / "pbp_align.parquet", tmp / "pbp_align.parquet")
            meta.update(
                layers=sorted(set(meta["layers"]) | {"L2"}),
                l2_params=all_params(),
                l2_failed_games=failed,
                l2_build_seconds_sum=round(sum(r.get("seconds", 0) for r in results), 1),
            )
            cfg = {**cfg, "l2": all_params()}
        if args.l3_build:
            l3 = paths.data_root() / args.l3_build
            results = json.loads((l3 / "build_l3_results.json").read_text(encoding="utf-8"))
            failed = [r["game_id"] for r in results if r["status"] != "ok"]
            if failed:
                raise SystemExit(f"L3 build has failed games: {failed[:5]}")
            for sub in L3_PER_GAME:
                missing = [g for g in ok_ids if not (l3 / sub / f"{g}.parquet").exists()]
                if missing:
                    raise SystemExit(
                        f"L3 build incomplete: {len(missing)} {sub} files missing, "
                        f"e.g. {missing[:3]}"
                    )
                for g in ok_ids:
                    _copy(l3 / sub / f"{g}.parquet", tmp / sub / f"{g}.parquet")
            for t in L3_TABLES:
                shutil.copy2(l3 / f"{t}.parquet", tmp / f"{t}.parquet")
            for t in L3_OPTIONAL:
                if (l3 / f"{t}.parquet").exists():
                    shutil.copy2(l3 / f"{t}.parquet", tmp / f"{t}.parquet")
            meta.update(
                layers=sorted(set(meta["layers"]) | {"L3"}),
                l3_params=l3_params(),
                l3_build_seconds_sum=round(sum(r.get("seconds", 0) for r in results), 1),
            )
        if args.l3_add:
            if "L3" not in base_manifest["layers"]:
                raise SystemExit("--l3-add needs a base release with L3")
            add = paths.data_root() / args.l3_add
            added = [t for t in L3_OPTIONAL if (add / f"{t}.parquet").exists()]
            if not added:
                raise SystemExit(f"no L3_OPTIONAL table in {add}")
            for t in added:
                if (tmp / f"{t}.parquet").exists():
                    (
                        tmp / f"{t}.parquet"
                    ).unlink()  # a link to the base: replace, never write through
                shutil.copy2(add / f"{t}.parquet", tmp / f"{t}.parquet")
            meta.update(l3_added_tables=added)
        if args.l5_build:
            l5 = paths.data_root() / args.l5_build
            sources = {}
            for t, src in L5_TABLES.items():
                _copy(l5 / f"{t}.parquet", tmp / "external" / f"{t}.parquet")
                prov = json.loads(
                    (paths.raw_dir() / "external" / src / "provenance.json").read_text(
                        encoding="utf-8"
                    )
                )
                sources[t] = {"source_dir": src, "provenance": prov}
            (tmp / "external" / "SOURCES.json").write_text(
                json.dumps(sources, indent=1, ensure_ascii=False), encoding="utf-8"
            )
            meta.update(layers=sorted(set(meta["layers"]) | {"L5"}), l5_tables=sorted(L5_TABLES))
            cfg = {**cfg, "l3": l3_params()}
    (tmp / "CHANGELOG.md").write_text(_changelog_without_drafts(), encoding="utf-8")
    meta.update(
        config=cfg,
        config_hash=config_hash(cfg),
        games={
            "status_counts": games.group_by("status").len().sort("status").rows(),
            "split_counts": games.group_by("split").len().drop_nulls().sort("split").rows(),
            "ok_game_ids": ok_ids,
        },
    )
    write_manifest(tmp, meta)
    make_read_only(tmp)
    problems = verify_release(tmp)
    if problems:
        raise SystemExit(f"verification failed: {problems[:5]}")
    tmp.rename(dest)
    if not args.no_latest:
        (paths.releases_dir() / "latest.txt").write_text(version + "\n", encoding="utf-8")
    print(f"published {dest}: {len(ok_ids)} games; now tag the code: git tag data-{version}")


if __name__ == "__main__":
    main()
