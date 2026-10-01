"""Locations inside the shared data root.

The data root lives outside every git repository and is read-only for consumer projects. It is
given by the environment variable ``NBA_DATA_ROOT`` (default ``nba_data`` in the working directory)::

    <root>/raw/sportvu/<archive>.7z        append-only
    <root>/raw/pbp/2015-16_pbp.csv
    <root>/raw/github_listing.json
    <root>/raw/external/<source>/...
    <root>/releases/vX.Y/...               immutable once published
    <root>/releases/latest.txt             version string (no symlinks on Windows)
    <root>/products/<project>_<name>/<version>/...
    <root>/private/<name>/...              restricted metadata (e.g. video index); never released
    <root>/scratch/                        development data, may be deleted
"""

from __future__ import annotations

import os
import re
from pathlib import Path

ENV_VAR = "NBA_DATA_ROOT"
DEFAULT_DATA_ROOT = "nba_data"
VERSION_RE = re.compile(r"^v\d+\.\d+$")


def data_root() -> Path:
    """Root of the shared data directory (``$NBA_DATA_ROOT`` or ``nba_data`` in the working directory)."""
    return Path(os.environ.get(ENV_VAR) or DEFAULT_DATA_ROOT)


def raw_dir() -> Path:
    return data_root() / "raw"


def sportvu_dir() -> Path:
    """Raw SportVU ``.7z`` archives (append-only)."""
    return raw_dir() / "sportvu"


def pbp_csv() -> Path:
    return raw_dir() / "pbp" / "2015-16_pbp.csv"


def listing_json() -> Path:
    """Cached GitHub directory listing of the SportVU archives."""
    return raw_dir() / "github_listing.json"


def external_dir(source: str | None = None) -> Path:
    d = raw_dir() / "external"
    return d / source if source else d


def releases_dir() -> Path:
    return data_root() / "releases"


def resolve_version(version: str = "latest") -> str:
    """``"latest"`` -> the version named in ``releases/latest.txt``; else validate the name."""
    if version == "latest":
        pointer = releases_dir() / "latest.txt"
        if not pointer.exists():
            raise FileNotFoundError(f"no published release: {pointer} does not exist")
        version = pointer.read_text(encoding="utf-8").strip()
    if not VERSION_RE.match(version):
        raise ValueError(f"bad data version {version!r}; expected 'vMAJOR.MINOR'")
    return version


def release_dir(version: str = "latest") -> Path:
    """Directory of a published release (``version`` like ``"v1.0"`` or ``"latest"``)."""
    return releases_dir() / resolve_version(version)


def products_dir() -> Path:
    return data_root() / "products"


def private_dir(*parts: str) -> Path:
    """Restricted metadata (e.g. the broadcast video index). Never part of a release, never
    distributed; consumers read it by path, not through ``nbacore.load``."""
    return data_root().joinpath("private", *parts)


def scratch_dir(*parts: str) -> Path:
    """Development scratch space under the data root (never read by consumers)."""
    return data_root().joinpath("scratch", *parts)
