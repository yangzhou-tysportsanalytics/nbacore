"""The GitHub directory listing of the SportVU archives, and archive integrity checks.

The listing (``raw/github_listing.json``, from the GitHub contents API) gives, per archive,
``name``, ``size`` and ``sha`` -- the *git blob* SHA-1 of the file, i.e.
``sha1(b"blob <size>\\0" + content)``. Every archive in ``raw/sportvu`` is checked against it, so
copies from other projects and fresh downloads are verified the same way.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from nbacore import paths


def load_listing(path: str | Path | None = None) -> list[dict]:
    """Listing entries (``name``, ``size``, ``sha``, ...) sorted by name."""
    p = Path(path) if path is not None else paths.listing_json()
    items = json.loads(p.read_text(encoding="utf-8"))
    return sorted(items, key=lambda x: x["name"])


def git_blob_sha1(path: str | Path) -> str:
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(path: str | Path, item: dict) -> bool:
    """True when ``path`` has the listing's size and git blob SHA-1."""
    path = Path(path)
    return (
        path.exists()
        and path.stat().st_size == int(item["size"])
        and git_blob_sha1(path) == item["sha"]
    )
