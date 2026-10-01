"""Tests for release manifests (nbacore.release) and the reader (nbacore.load) on a toy release."""

from __future__ import annotations

import os
import stat

import polars as pl
import pytest

from nbacore import load, paths
from nbacore.release import config_hash, l1_config, make_read_only, verify_release, write_manifest

GID = "0021500001"


@pytest.fixture
def toy_release(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    rel = tmp_path / "releases" / "v1.0"
    (rel / "frames").mkdir(parents=True)
    (rel / "frame_index").mkdir()
    pl.DataFrame({"game_id": [GID, "0021500002"], "status": ["ok", "empty_archive"]}).write_parquet(
        rel / "games.parquet"
    )
    pl.DataFrame(
        {"game_id": [GID] * 3, "unix_ms": [0, 40, 80], "x": [1.0, 2.0, 3.0]}
    ).write_parquet(rel / "frames" / f"{GID}.parquet")
    pl.DataFrame({"game_id": [GID, GID, "0021500009"], "event_num": [1, 2, 1]}).write_parquet(
        rel / "pbp.parquet"
    )
    manifest = write_manifest(rel, {"version": "v1.0"})
    make_read_only(rel)
    (tmp_path / "releases" / "latest.txt").write_text("v1.0\n", encoding="utf-8")
    yield rel, manifest
    for p in rel.rglob("*"):  # let pytest clean up
        if p.is_file():
            os.chmod(p, stat.S_IREAD | stat.S_IWRITE)


def test_manifest_and_verify(toy_release):
    rel, manifest = toy_release
    assert manifest["files"][f"frames/{GID}.parquet"]["rows"] == 3
    assert set(manifest["files"]) == {"games.parquet", "pbp.parquet", f"frames/{GID}.parquet"}
    assert verify_release(rel) == []
    # tampering is detected
    p = rel / "pbp.parquet"
    os.chmod(p, stat.S_IREAD | stat.S_IWRITE)
    p.write_bytes(p.read_bytes() + b"x")
    problems = verify_release(rel)
    assert any("sha256 mismatch: pbp.parquet" in s for s in problems)
    assert any("writable: pbp.parquet" in s for s in problems)


def test_load(toy_release):
    assert load.manifest()["version"] == "v1.0"
    assert load.games("v1.0").height == 2
    assert load.frames(21500001, "latest")["x"].to_list() == [1.0, 2.0, 3.0]  # id zero-padded
    assert load.frames(GID, columns=["x"]).columns == ["x"]
    assert load.pbp(game_id=GID).height == 2
    with pytest.raises(FileNotFoundError, match="status"):
        load.frames("0021500002")
    with pytest.raises(FileNotFoundError):
        load.rosters()
    with pytest.raises(ValueError):
        load.games("1.0")


def test_config_hash_stable():
    cfg = l1_config()
    assert cfg["frame_gap_ms"] == 60 and cfg["clock_jump_s"] == 1.0 and cfg["split_seed"] == 9
    assert config_hash(cfg) == config_hash(dict(reversed(list(cfg.items()))))
