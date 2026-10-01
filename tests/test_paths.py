"""Tests for nbacore.paths (data root resolution and release versions)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nbacore import paths


def test_default_root(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    assert paths.data_root() == Path("nba_data")


def test_env_root_and_layout(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    assert paths.data_root() == tmp_path
    assert paths.sportvu_dir() == tmp_path / "raw" / "sportvu"
    assert paths.pbp_csv() == tmp_path / "raw" / "pbp" / "2015-16_pbp.csv"
    assert paths.listing_json() == tmp_path / "raw" / "github_listing.json"
    assert paths.external_dir("shots") == tmp_path / "raw" / "external" / "shots"
    assert paths.products_dir() == tmp_path / "products"
    assert paths.scratch_dir("a", "b") == tmp_path / "scratch" / "a" / "b"
    assert paths.private_dir("video_index") == tmp_path / "private" / "video_index"


def test_release_versions(monkeypatch, tmp_path):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path))
    with pytest.raises(FileNotFoundError):
        paths.release_dir()  # nothing published yet
    assert paths.release_dir("v1.0") == tmp_path / "releases" / "v1.0"
    (tmp_path / "releases").mkdir()
    (tmp_path / "releases" / "latest.txt").write_text("v1.2\n", encoding="utf-8")
    assert paths.release_dir() == tmp_path / "releases" / "v1.2"
    for bad in ["1.0", "v1", "v1.0.0", "../v1.0"]:
        with pytest.raises(ValueError):
            paths.resolve_version(bad)
