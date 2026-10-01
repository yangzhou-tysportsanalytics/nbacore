# nbacore

A versioned data layer over the public 2015-16 NBA SportVU tracking release: raw game logs →
deduplicated 25 Hz frames (L1) → events such as shot releases, passes, dribbles and screen /
drive / cut / post-up candidates (L2) → a possession ledger with team-foul state, free throws and a
per-frame shot clock (L3) → fixed game splits (L4) → external tables linked to NBA ids (L5).
Releases are immutable and hashed; consumers pin a data version and the matching code tag.

**No tracking data is included in this repository.** The raw tracking logs and the play-by-play
are downloaded from public sources by `scripts/download_raw.py`; everything else is rebuilt
locally. The raw SportVU coordinates have no clear licence, so neither they nor any derived
frame-level table is redistributed.

## Install

```bash
uv sync                                  # Python >= 3.11
export NBA_DATA_ROOT=/path/to/nba_data   # default: ./nba_data
```

## Rebuild the data

```bash
uv run python scripts/download_raw.py --config large          # all archives + 2015-16 pbp
uv run python scripts/build_l1.py --all --out scratch/l1_full --workers 4
uv run python scripts/publish_release.py --version v1.0 --build scratch/l1_full
uv run python scripts/build_l2.py --base v1.0 --out scratch/l2 --workers 3
uv run python scripts/build_event_frames.py --l2-build scratch/l2 --base v1.0
uv run python scripts/build_l3.py --base v1.0 --l2 scratch/l2 --out scratch/l3 --workers 3
uv run python scripts/build_event_possession.py --l2 scratch/l2 --l3 scratch/l3
uv run python scripts/build_ghost_v1_l2release.py --base v1.0 --l2 scratch/l2 --l3 scratch/l3 --out scratch/l3
uv run python scripts/publish_release.py --version v1.1 --base v1.0 --l2-build scratch/l2 --l3-build scratch/l3
```

A full build of the 631 usable games takes several hours on a 16 GB machine. The L5 external
tables need source files that must be obtained manually (some require an account); see
`docs/external_sources.md`. Release history: `CHANGELOG.md`.

## Derived data packages

Coordinate-free derived tables are attached to the GitHub releases:

| package | release | MANIFEST.json sha256 | zip sha256 |
|---|---|---|---|
| `nbacore-derived-v1.5-shot-p1` | `data-v1.5` | `6a7080088b3d00dde818551bf2d5c5b54251440d27283fabd9d923b0372ac4e0` | `e742442c68b0701d31cf65152a4c1b4b7abc0a93b1ab88da695a58935b9d8f31` |

Contents: games and splits, rosters, play-by-play, half-court possession windows, possession
ledger, screen / pass / handoff / shot / touch events, per-frame shot-clock readings, RAPTOR
(CC BY 4.0, FiveThirtyEight). No player or ball tracks; the only point coordinates are shot
release locations.

## Use

```python
import nbacore.load as L

L.games("v1.5")  # games, status, split / fold / parity
L.frames("0021500001", "v1.5")  # 25 Hz frames of one game
L.events("0021500001", "shot_release", "v1.5")
L.ledger("v1.5")  # one row per team possession
```

Start with `docs/DATA_GUIDE.md` (layers, loaders, keys, caveats); definitions are in
`docs/data_schema.md`, `docs/event_definitions.md`, `docs/ledger_schema.md`, `docs/splits.md`,
external sources in `docs/external_sources.md`.

## Tests

```bash
uv run pytest            # data tests skip when the data root or reference outputs are absent
```

## Sources and credits

- SportVU logs: github.com/linouk23/NBA-Player-Movements (public copy of the 2015-16 release);
  play-by-play: github.com/sumitrodatta/nba-alt-awards. Underlying data © NBA / NBA.com.
- Parts of the possession segmentation and shot-release code come from the ghost-defense
  project (github.com/yangzhou-tysportsanalytics/ghosting_defence).
- External tables credit their sources in `docs/external_sources.md`; RAPTOR is CC BY 4.0
  (FiveThirtyEight), the referee sources are CC BY-SA 4.0.

## Licence

Code: MIT (see `LICENSE`). Documentation: CC BY 4.0. No tracking data is licensed or distributed
here.
