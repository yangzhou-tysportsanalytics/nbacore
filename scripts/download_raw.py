"""Download raw SportVU game archives (.7z) and the season play-by-play CSV.

Reproduces the HuggingFace ``dcayton/nba_tracking_data_15_16`` config subsets without going
through the (fragile, script-based) HF loader: the HF script samples ``random.seed(9);
random.sample(listing, n)`` from the GitHub directory listing, so we do the same on the
name-sorted listing (n = 5 / 25 / 100 / all for tiny / small / medium / large).

Destinations default to the shared data root (``nbacore.paths``, ``$NBA_DATA_ROOT``). Archives
already present are not downloaded again. ``--extract`` unpacks into ``scratch/extracted/``,
never next to the append-only raw archives.

Usage:
    uv run python scripts/download_raw.py --config tiny [--extract] [--dest <root>/raw/sportvu]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from nbacore import paths
from nbacore.io.listing import verify_archive

GITHUB_API = (
    "https://api.github.com/repos/linouk23/NBA-Player-Movements/contents/"
    "data/2016.NBA.Raw.SportVU.Game.Logs"
)
RAW_BASE = (
    "https://github.com/linouk23/NBA-Player-Movements/raw/master/"
    "data/2016.NBA.Raw.SportVU.Game.Logs"
)
PBP_URL = (
    "https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv"
)
CONFIG_SIZES = {"tiny": 5, "small": 25, "medium": 100, "large": None}
HF_SEED = 9


def get_listing(cache: Path) -> list[dict]:
    if cache.exists():
        return json.loads(cache.read_text())
    r = requests.get(GITHUB_API, timeout=60)
    r.raise_for_status()
    items = r.json()
    if not isinstance(items, list):
        raise RuntimeError(f"unexpected listing: {items}")
    items = sorted(items, key=lambda x: x["name"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items))
    return items


def select_games(items: list[dict], config: str) -> list[dict]:
    n = CONFIG_SIZES[config]
    if n is None:
        return list(items)
    rng = random.Random(HF_SEED)
    return rng.sample(items, n)


def download(url: str, dest: Path, retries: int = 3) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fp:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fp.write(chunk)
                tmp.replace(dest)
            return
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            print(f"retry {dest.name}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=list(CONFIG_SIZES), default="tiny")
    ap.add_argument("--dest", default=str(paths.sportvu_dir()))
    ap.add_argument("--pbp-dest", default=str(paths.pbp_csv()))
    ap.add_argument("--extract", action="store_true", help="also extract each .7z")
    ap.add_argument("--skip-pbp", action="store_true")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    items = get_listing(paths.listing_json())
    games = select_games(items, args.config)
    manifest = {
        "config": args.config,
        "seed": HF_SEED,
        "n_games": len(games),
        "archives": [g["name"] for g in games],
    }
    (dest.parent / f"manifest_{args.config}.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"config={args.config}: {len(games)} archives, "
        f"{sum(g['size'] for g in games) / 1e6:.0f} MB compressed"
    )

    if not args.skip_pbp:
        pbp = Path(args.pbp_dest)
        pbp.parent.mkdir(parents=True, exist_ok=True)
        download(PBP_URL, pbp)

    n_bad = 0
    for g in tqdm(games, desc="download"):
        out = dest / g["name"]
        if out.exists() and not verify_archive(out, g):
            # never silently keep a corrupt archive; set it aside and fetch again
            out.replace(out.with_suffix(".7z.bad"))
        download(RAW_BASE + "/" + g["name"], out)
        if not verify_archive(out, g):
            n_bad += 1
            out.replace(out.with_suffix(".7z.bad"))
            print(f"verification failed (size / git blob sha1): {g['name']}", file=sys.stderr)
    if n_bad:
        raise SystemExit(f"{n_bad} archives failed verification (kept as .7z.bad)")

    if args.extract:
        from nbacore.io import extract_archive

        for g in tqdm(games, desc="extract"):
            arc = dest / g["name"]
            extract_archive(arc, paths.scratch_dir("extracted", arc.stem))


if __name__ == "__main__":
    main()
