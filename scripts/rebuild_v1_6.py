"""Rebuild nbacore data version v1.6 from the public raw sources, step by step, and verify it.

Runs the build scripts of a checkout of the code tag ``data-v1.6`` against a separate data root,
records wall time, peak memory (whole process tree) and data-root size after every step, and
compares the result with the published file hashes. Only the Python standard library is needed;
peak memory is recorded when ``psutil`` is importable (``uv run --with psutil ...``).

    git clone https://github.com/yangzhou-tysportsanalytics/nbacore.git nbacore
    git -C nbacore worktree add ../nbacore-v1.6 data-v1.6
    cd nbacore-v1.6 && uv sync --locked && cd ..
    python nbacore/scripts/rebuild_v1_6.py --code nbacore-v1.6 --data-root /big/disk/nba_data \
        [--reference reference_v1.6_files.json] [--from-step l2] [--only verify]

Steps (names usable with ``--from-step`` / ``--only``), in order:

    check     versions; the code checkout is data-v1.6; raw / external inputs and their checksums
    download  636 SportVU archives (git blob SHA-1 checked) + 2015-16 play-by-play (sha256 checked)
    l1        L1 for all games                                     -> scratch/l1
    v1.0      publish release v1.0 (L1 + L4 split)
    l2        L2 events, handler, pbp alignment                    -> scratch/l2
    frames    per-event distance snapshots (L2)                    -> scratch/l2
    l3        ledger, fouls, free throws, ghost_v1, shot clock     -> scratch/l3
    l3extra   event_possession, pbp_possession, ghost_v1_l2release -> scratch/l3
    v1.2      publish v1.2 = v1.0 + L2 + L3 (the L5 build scripts read this version name)
    l5a       shots, player bio, box scores (read v1.2)            -> scratch/l5
    stage     read-only staging folder releases/v1.4 = v1.2 + the box-score tables (the
              Basketball-Reference and referee builds read box scores from a version named v1.4)
    l5b       advanced metrics, awards, id map, RAPTOR, referees    -> scratch/l5
              (the referee build reads ``L2M_stats_nba.csv`` from the code directory; it is
              copied there from ``raw/external/l2m_atlhawksfanatic``)
    v1.6      publish v1.6 = v1.2 + L5
    verify    every file of releases/v1.6 against the reference hashes (MANIFEST / CHANGELOG
              excluded); for a differing parquet file, rows and a content hash are compared

The L5 source files are not redistributed: place them under ``<data root>/raw/external/<source>/``
as listed in ``docs/rebuild.md``; ``check`` copies the published provenance records
(``docs/reference/l5_sources/<source>/provenance.json``) next to them when missing and verifies every
file against its sha256 before anything is built.

Build scripts rewrite some tracked report files in the code checkout, and publishing refuses a
modified checkout: before each publish step the regenerated reports are kept under
``<data root>/rebuild_reports`` and the checkout is restored. ``stage`` leaves ``releases/v1.4``
in place (it is not a release: no MANIFEST); delete it after the run if disk space matters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

TAG_COMMIT = "bfe7662ccf7c24f690070055aa022b6358524dd5"  # public data-v1.6
PBP_SHA256 = "b10a0e3ff6870d9a7559281e910ebaed12fd244938049b7384480be64dd0d21f"
L5_SOURCES = [
    "shots_domsamangy",
    "player_bio_kaggle",
    "boxscores_eoinamoore_v515",
    "bbref_sumitrodatta",
    "raptor_538",
    "referees_kaggle",
    "l2m_atlhawksfanatic",  # L2M reports (MIT): official ids for the referee table
]
STAGED_TABLES = ["box_games_2015_16", "box_players_2015_16"]
# published provenance records (source, files, sha256, terms) of the L5 sources
REPO = Path(__file__).resolve().parents[1]
SOURCES_REF = next(
    (
        d
        for d in (
            REPO / "docs/reference/l5_sources",
            REPO / "public_templates/docs/reference/l5_sources",
        )
        if d.exists()
    ),
    REPO / "docs/reference/l5_sources",
)


def sh(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_size(root: Path) -> int:
    total = 0
    for dp, _, fs in os.walk(root):
        for f in fs:
            try:
                total += os.stat(os.path.join(dp, f)).st_size
            except OSError:
                pass
    return total


class PeakMemory:
    """Peak resident memory of a process and its children (needs psutil; else None)."""

    def __init__(self, pid: int):
        self.peak: int | None = None
        try:
            import psutil
        except ImportError:
            return
        self.peak = 0
        self._stop = threading.Event()

        def poll() -> None:
            try:
                root = psutil.Process(pid)
            except psutil.Error:
                return
            while not self._stop.is_set():
                total = 0
                try:
                    for p in [root, *root.children(recursive=True)]:
                        try:
                            total += p.memory_info().rss
                        except psutil.Error:
                            pass
                except psutil.Error:
                    break
                self.peak = max(self.peak or 0, total)
                time.sleep(1.0)

        self._t = threading.Thread(target=poll, daemon=True)
        self._t.start()

    def stop(self) -> None:
        if self.peak is not None:
            self._stop.set()
            self._t.join(timeout=5)


class Runner:
    def __init__(self, code: Path, root: Path, log: Path):
        self.code, self.root, self.log = code, root, log
        self.env = dict(
            os.environ, NBA_DATA_ROOT=str(root), PYTHONUTF8="1", PYTHONIOENCODING="utf-8"
        )
        self.records: list[dict] = json.loads(log.read_text()) if log.exists() else []

    def restore_reports(self) -> None:
        """Build scripts rewrite tracked report JSON files in the code checkout, and publishing
        refuses a modified checkout: keep the regenerated reports under <data root>/rebuild_reports
        and restore the committed ones."""
        changed = subprocess.run(
            ["git", "diff", "--name-only"], cwd=self.code, capture_output=True, text=True
        ).stdout.split()
        if not changed:
            return
        keep = self.root / "rebuild_reports"
        for rel in changed:
            dst = keep / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.code / rel, dst)
        subprocess.run(["git", "checkout", "--", *changed], cwd=self.code, check=True)
        print(f"kept {len(changed)} regenerated report(s) in {keep}; checkout restored: {changed}")

    def run(self, step: str, *cmds: list[str], fn=None) -> None:
        print(f"\n=== {step} ===", flush=True)
        t0, peak = time.time(), 0
        start = datetime.now(UTC).isoformat(timespec="seconds")
        if fn is not None:
            fn()
        if step in ("v1.0", "v1.2", "v1.6"):
            self.restore_reports()
        for cmd in cmds:
            full = ["uv", "run", "--locked", "python", *cmd]
            print("$ " + " ".join(full), flush=True)
            p = subprocess.Popen(full, cwd=self.code, env=self.env)
            mem = PeakMemory(p.pid)
            rc = p.wait()
            mem.stop()
            if mem.peak is not None:
                peak = max(peak, mem.peak)
            if rc != 0:
                raise SystemExit(f"step {step} failed (exit {rc}): {' '.join(cmd)}")
        rec = {
            "step": step,
            "start_utc": start,
            "seconds": round(time.time() - t0, 1),
            "peak_rss_gb": round(peak / 1e9, 2) if peak else None,
            "data_root_gb": round(tree_size(self.root) / 1e9, 2),
        }
        self.records = [r for r in self.records if r["step"] != step] + [rec]
        self.log.write_text(json.dumps(self.records, indent=1))
        print(f"--- {step}: {rec}", flush=True)


def check(code: Path, root: Path) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=code, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=code,
        capture_output=True,
        text=True,
    ).stdout.strip()
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip()
    py = subprocess.run(
        ["uv", "run", "--locked", "python", "--version"], cwd=code, capture_output=True, text=True
    ).stdout.strip()
    print(f"code {code} at {head}; {py}; {uv}; {platform.platform()}")
    if head != TAG_COMMIT:
        raise SystemExit(f"the code checkout must be data-v1.6 ({TAG_COMMIT}), found {head}")
    if dirty:
        raise SystemExit("the code checkout has uncommitted changes")
    problems, n_checked = [], 0
    for src in L5_SOURCES:
        prov = root / "raw" / "external" / src / "provenance.json"
        if not prov.exists() and (SOURCES_REF / src / "provenance.json").exists():
            prov.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(SOURCES_REF / src / "provenance.json", prov)
            print(f"provenance record copied from {SOURCES_REF / src}")
        if not prov.exists():
            problems.append(f"missing {prov}")
            continue
        files = json.loads(prov.read_text(encoding="utf-8")).get("files", {})
        for name, meta in files.items():
            want = meta.get("sha256") if isinstance(meta, dict) else None
            f = prov.parent / Path(name).name
            if want is None:
                continue
            if not f.exists():
                problems.append(f"missing {f}")
            elif sh(f) != want:
                problems.append(f"sha256 differs: {f}")
            else:
                n_checked += 1
    if problems:
        raise SystemExit("L5 sources not ready:\n  " + "\n  ".join(problems))
    print(f"L5 sources: {n_checked} files present with the sha256 of their provenance.json")


def check_pbp(root: Path) -> None:
    pbp = root / "raw" / "pbp" / "2015-16_pbp.csv"
    got = sh(pbp)
    if got != PBP_SHA256:
        raise SystemExit(
            f"{pbp} sha256 {got} differs from the one v1.6 was built from ({PBP_SHA256})"
        )
    print(f"pbp sha256 ok ({PBP_SHA256[:12]}…)")


def place_l2m(code: Path, root: Path) -> None:
    """build_external_referees reads ``L2M_stats_nba.csv`` from the working directory."""
    src = root / "raw" / "external" / "l2m_atlhawksfanatic" / "L2M_stats_nba.csv"
    dst = code / "L2M_stats_nba.csv"
    if not dst.exists():
        shutil.copy2(src, dst)
    print(f"L2M reports available at {dst} (untracked)")


def stage_v14(root: Path) -> None:
    """releases/v1.4 = hard links to releases/v1.2 + the box-score tables from scratch/l5."""
    src, dst = root / "releases" / "v1.2", root / "releases" / "v1.4"
    if dst.exists():
        print(f"{dst} exists; kept")
        return
    for dp, _, fs in os.walk(src):
        out = dst / Path(dp).relative_to(src)
        out.mkdir(parents=True, exist_ok=True)
        for f in fs:
            if f in ("MANIFEST.json", "CHANGELOG.md"):
                continue
            try:
                os.link(os.path.join(dp, f), out / f)
            except OSError:
                shutil.copy2(os.path.join(dp, f), out / f)
    (dst / "external").mkdir(exist_ok=True)
    for t in STAGED_TABLES:
        shutil.copy2(root / "scratch" / "l5" / f"{t}.parquet", dst / "external" / f"{t}.parquet")
    (dst / "STAGING.txt").write_text(
        "Not a release: read-only staging for build_external_bbref / build_external_referees.\n"
    )
    print(f"staged {dst}")


def content_hash(path: Path) -> str:
    """Row-order-free content hash of a parquet table: sha256 of its rows as JSON lines, sorted
    (nested columns included; needs polars, i.e. run inside the nbacore environment)."""
    import polars as pl

    lines = sorted(pl.read_parquet(path).write_ndjson().splitlines())
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode())
        h.update(b"\n")
    return h.hexdigest()


def write_reference(release: Path, out: Path) -> None:
    """Reference file for ``verify``: sha256 and rows of every file of a published release (from
    its MANIFEST.json), plus ``content_sha256`` for the flat tables (top level and external/)."""
    files = json.loads((release / "MANIFEST.json").read_text(encoding="utf-8"))["files"]
    ref = {}
    for name, meta in sorted(files.items()):
        if name in ("MANIFEST.json", "CHANGELOG.md"):
            continue
        rec = {"sha256": meta["sha256"], "rows": meta.get("rows")}
        if name.endswith(".parquet") and (name.count("/") == 0 or name.startswith("external/")):
            rec["content_sha256"] = content_hash(release / name)
        ref[name] = rec
    out.write_text(json.dumps({"version": release.name, "files": ref}, indent=0), encoding="utf-8")
    print(f"wrote {out}: {len(ref)} files")


def verify(root: Path, reference: Path) -> None:
    ref = json.loads(reference.read_text(encoding="utf-8"))
    ref_files = ref.get("files", ref)
    rel = root / "releases" / "v1.6"
    skip = {"MANIFEST.json", "CHANGELOG.md"}
    same, differ, missing = 0, [], []
    for name, meta in sorted(ref_files.items()):
        if name in skip:
            continue
        f = rel / name
        if not f.exists():
            missing.append(name)
        elif sh(f) == meta["sha256"]:
            same += 1
        else:
            differ.append(name)
    extra = sorted(
        p.relative_to(rel).as_posix()
        for p in rel.rglob("*")
        if p.is_file() and p.name not in skip and p.relative_to(rel).as_posix() not in ref_files
    )
    same_content = []
    for name in list(differ):
        want = ref_files[name].get("content_sha256")
        if want and content_hash(rel / name) == want:
            differ.remove(name)
            same_content.append(name)
    print(
        f"identical {same}, same content in another row order {len(same_content)}, "
        f"different {len(differ)}, missing {len(missing)}, extra {len(extra)}"
    )
    for name in same_content:
        print(f"  row order only: {name}")
    report = {
        "identical": same,
        "same_content_row_order": same_content,
        "different": differ,
        "missing": missing,
        "extra": extra,
    }
    out = root / "rebuild_verify_v1.6.json"
    out.write_text(json.dumps(report, indent=1))
    for name in differ[:50]:
        print(f"  differs: {name} (rows in reference: {ref_files[name].get('rows')})")
    print(f"report: {out}")


STEPS = [
    "check", "download", "l1", "v1.0", "l2", "frames", "l3", "l3extra",
    "v1.2", "l5a", "stage", "l5b", "v1.6", "verify",
]  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True, help="checkout of the code tag data-v1.6")
    ap.add_argument("--data-root", required=True, help="empty or partly built data root")
    ap.add_argument(
        "--reference", help="published file hashes (MANIFEST.json of v1.6 or the export)"
    )
    ap.add_argument("--workers-l1", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--from-step", choices=STEPS, default="check")
    ap.add_argument("--only", choices=STEPS)
    ap.add_argument(
        "--write-reference",
        metavar="RELEASE_DIR",
        help="maintainers: write the reference file for --reference from a published release",
    )
    a = ap.parse_args()
    if a.write_reference:
        write_reference(Path(a.write_reference), Path(a.reference or "reference_v1.6_files.json"))
        return
    code, root = Path(a.code).resolve(), Path(a.data_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    r = Runner(code, root, root / "rebuild_log_v1.6.json")
    todo = [a.only] if a.only else STEPS[STEPS.index(a.from_step) :]
    w1, w = str(a.workers_l1), str(a.workers)
    for step in todo:
        if step == "check":
            r.run(step, fn=lambda: check(code, root))
        elif step == "download":
            r.run(step, ["scripts/download_raw.py", "--config", "large"], fn=None)
            check_pbp(root)
        elif step == "l1":
            r.run(step, ["scripts/build_l1.py", "--all", "--out", "scratch/l1", "--workers", w1])
        elif step == "v1.0":
            r.run(
                step, ["scripts/publish_release.py", "--version", "v1.0", "--build", "scratch/l1"]
            )
        elif step == "l2":
            r.run(
                step,
                ["scripts/build_l2.py", "--base", "v1.0", "--out", "scratch/l2", "--workers", w],
            )
        elif step == "frames":
            r.run(
                step,
                [
                    "scripts/build_event_frames.py",
                    "--l2-build",
                    "scratch/l2",
                    "--base",
                    "v1.0",
                    "--workers",
                    w,
                ],
            )
        elif step == "l3":
            r.run(
                step,
                [
                    "scripts/build_l3.py",
                    "--base",
                    "v1.0",
                    "--l2",
                    "scratch/l2",
                    "--out",
                    "scratch/l3",
                    "--workers",
                    w,
                ],
            )
        elif step == "l3extra":
            r.run(
                step,
                ["scripts/build_event_possession.py", "--l2", "scratch/l2", "--l3", "scratch/l3"],
                ["scripts/build_pbp_possession.py", "--base", "v1.0", "--l3", "scratch/l3"],
                ["scripts/build_ghost_v1_l2release.py", "--base", "v1.0", "--l2", "scratch/l2",
                 "--l3", "scratch/l3", "--out", "scratch/l3", "--workers", w],
            )  # fmt: skip
        elif step == "v1.2":
            r.run(step, ["scripts/publish_release.py", "--version", "v1.2", "--base", "v1.0",
                         "--l2-build", "scratch/l2", "--l3-build", "scratch/l3"])  # fmt: skip
        elif step == "l5a":
            r.run(
                step,
                ["scripts/build_external_shots.py"],
                ["scripts/build_external_player_bio.py"],
                ["scripts/build_external_boxscores.py"],
            )
        elif step == "stage":
            r.run(step, fn=lambda: stage_v14(root))
        elif step == "l5b":
            r.run(
                step,
                ["scripts/build_external_bbref.py"],
                ["scripts/build_external_referees.py"],
                fn=lambda: place_l2m(code, root),
            )
        elif step == "v1.6":
            r.run(step, ["scripts/publish_release.py", "--version", "v1.6", "--base", "v1.2",
                         "--l5-build", "scratch/l5"])  # fmt: skip
        elif step == "verify":
            if not a.reference:
                print("no --reference given: verification skipped")
                continue
            r.run(step, fn=lambda: verify(root, Path(a.reference)))
    print(json.dumps(r.records, indent=1))


if __name__ == "__main__":
    sys.exit(main())
