"""Publishing and verifying immutable data releases.

A release directory holds the tables plus ``MANIFEST.json`` (sha256, bytes and row count per
file, code commit, configuration and its hash, inputs, machine) and ``CHANGELOG.md``. Files are
made read-only after publishing; a published version is never modified.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from nbacore import __version__, splits
from nbacore.clean import dedup
from nbacore.io.listing import sha256_file
from nbacore.possession import direction

MANIFEST = "MANIFEST.json"


def l1_config() -> dict:
    """Every parameter that shapes the L1 + L4 tables (hashed into the manifest)."""
    return {
        "dedup_entity_key": dedup.ENTITY_KEY,
        "frame_gap_ms": dedup.FRAME_GAP_MS,
        "clock_jump_s": dedup.CLOCK_JUMP_S,
        "clock_frozen_abs_dgc_s": 0.005,
        "direction_shot_msg_types": list(direction.SHOT_MSG_TYPES),
        "direction_tol_s": 1.0,
        "split_seed": splits.SPLIT_SEED,
        "split_train_frac": splits.TRAIN_FRAC,
        "split_val_frac": splits.VAL_FRAC,
        "n_folds": splits.N_FOLDS,
    }


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def git_state(repo: Path) -> tuple[str, bool]:
    """(HEAD commit, working tree clean?) of the code repository."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return head, dirty == ""


def file_record(path: Path) -> dict:
    rec = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    if path.suffix == ".parquet":
        rec["rows"] = pq.read_metadata(path).num_rows
    return rec


def machine_info() -> dict:
    import numpy
    import polars

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "polars": polars.__version__,
        "numpy": numpy.__version__,
        "nbacore": __version__,
    }


def write_manifest(release: Path, meta: dict) -> dict:
    files = sorted(p for p in release.rglob("*") if p.is_file() and p.name != MANIFEST)
    manifest = dict(meta)
    manifest["created_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
    manifest["machine"] = machine_info()
    manifest["files"] = {p.relative_to(release).as_posix(): file_record(p) for p in files}
    (release / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def make_read_only(release: Path) -> None:
    for p in release.rglob("*"):
        if p.is_file():
            os.chmod(p, stat.S_IREAD)


def verify_release(release: Path) -> list[str]:
    """Problems found (empty = the release matches its manifest and is read-only)."""
    problems: list[str] = []
    manifest = json.loads((release / MANIFEST).read_text(encoding="utf-8"))
    listed = set(manifest["files"])
    present = {
        p.relative_to(release).as_posix()
        for p in release.rglob("*")
        if p.is_file() and p.name != MANIFEST
    }
    for rel in sorted(present - listed):
        problems.append(f"not in manifest: {rel}")
    for rel in sorted(listed - present):
        problems.append(f"missing: {rel}")
    for rel in sorted(listed & present):
        p = release / rel
        if sha256_file(p) != manifest["files"][rel]["sha256"]:
            problems.append(f"sha256 mismatch: {rel}")
        if os.access(p, os.W_OK):
            problems.append(f"writable: {rel}")
    return problems


def l3_params() -> dict:
    """Parameters of the L3 build (ledger, fouls, ghost_v1, shot clock) for the manifest."""
    from dataclasses import asdict

    from nbacore import ledger, ledger_detail, rules_2015_16, shot_clock
    from nbacore.events import pbp_align
    from nbacore.views.ghost_v1 import SegmentConfig

    rules = {k: v for k, v in vars(rules_2015_16).items() if k.isupper()}
    rules = {k: sorted(v) if isinstance(v, frozenset) else v for k, v in rules.items()}
    seg = asdict(SegmentConfig())
    return {
        "rules_2015_16": rules,
        "ledger": {
            "anchor_clock_window_s": list(pbp_align.ANCHOR_CLOCK_WINDOW_S),
            "hole_ms": ledger.HOLE_MS,
            "switch_frames": ledger.SWITCH_FRAMES,
        },
        "ledger_detail": {
            "min_run_ms": ledger_detail.MIN_RUN_MS,
            "transition_s": ledger_detail.TRANSITION_S,
            "live_gap_clock_s": ledger_detail.LIVE_GAP_CLOCK_S,
        },
        "ghost_v1": seg,
        "shot_clock": {
            "stuck_24_s": shot_clock.STUCK_24_S,
            "inbound_touch_ms": shot_clock.INBOUND_TOUCH_MS,
            "rim_mode": "hold",
            "live_lag_ms": 0,
            "dedupe_rebound_s": 6.0,
        },
    }
