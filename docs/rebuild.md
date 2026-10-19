# Rebuilding data version v1.6 from public sources

This rebuilds the complete data release `v1.6`, including L1–L4, `pbp_possession` and the L5
external tables, from the public raw sources. It uses the code of tag `data-v1.6`
(commit `bfe7662`).

**Verified on 2026-10-19.** A fresh clone ran the chain below into an empty data root, and every
file was compared with the published hashes in `docs/reference/v1.6_files.json`:

| result | files |
|---|---|
| byte-identical | 3,808 |
| identical content, different row order (`external/bbref_player_map_2015_16.parquet`) | 1 |
| different / missing / extra | 0 |

With the published provenance records, `external/SOURCES.json` (the provenance of the L5 tables)
will differ in its free-text notes. Everything else should match.

## Requirements

- **Software:** git, uv (tested with 0.12.19) and the locked environment of the tag (Python 3.12.10).
  On Windows, set `PYTHONUTF8=1`: the L1 build prints characters that the default console
  encoding cannot write. The orchestrator sets this itself.
- **Memory:** 16 GB. The L1 step peaked at 10.0 GB with 4 workers. With less memory, use
  `--workers-l1 2`.
- **Disk:** about 80 GB free on the disk of the data root.
  - 61 GB of this is the extracted game JSON in `scratch/extracted`, needed only during the L1 build.
  - Releases share unchanged files through hard links, so keep the data root on one NTFS / ext4 volume.

## 1. Code

```bash
git clone https://github.com/yangzhou-tysportsanalytics/nbacore.git nbacore
git -C nbacore worktree add ../nbacore-v1.6 data-v1.6     # the code that builds the data
cd nbacore-v1.6 && uv sync --locked && cd ..
```

The orchestrator `scripts/rebuild_v1_6.py` and the reference files live on `main` (`nbacore/`).
The build itself runs the scripts of the `data-v1.6` worktree.

## 2. Sources

- **SportVU logs and play-by-play.** These are downloaded by the `download` step from the public
  GitHub copies; nothing to prepare. Each archive is checked against the git blob SHA-1 of the
  listing. The play-by-play file must have sha256 `b10a0e3f…dd0d21f`; the orchestrator stops if
  the upstream file has changed.
- **L5 sources.** These are not redistributed here: obtain each file yourself and place it under
  `<data root>/raw/external/<source>/`. Some require a Kaggle account. The published provenance
  records in `docs/reference/l5_sources/<source>/provenance.json` list the exact files, URLs,
  sizes, sha256 values and terms. The `check` step copies them next to your files and verifies
  every sha256 before anything is built.

| source folder | files | where | terms |
|---|---|---|---|
| `shots_domsamangy` | `NBA_2016_Shots.csv.zip` | github.com/DomSamangy/NBA_Shots_04_25 (pinned commit in the record) | see record |
| `player_bio_kaggle` | the two Kaggle zips named in the record | Kaggle (account) | see record |
| `boxscores_eoinamoore_v515` | `Games.csv`, `PlayerStatistics.csv` | Kaggle `eoinamoore/historical-nba-data-and-player-box-scores`, version 515 | NBA.com data, Kaggle compilation |
| `bbref_sumitrodatta` | the CSV files named in the record | github.com/sumitrodatta/bball-reference-datasets (pinned commit) | Basketball-Reference data; not redistributed |
| `raptor_538` | `modern_RAPTOR_by_player.csv` | github.com/fivethirtyeight/data (pinned commit) | CC BY 4.0 |
| `referees_kaggle` | `2012-18_officialBoxScore.csv.zip`, `officials.csv` | Kaggle (account) | CC BY-SA 4.0 |
| `l2m_atlhawksfanatic` | `L2M_stats_nba.csv` | github.com/atlhawksfanatic/L2M @ `330ab64a` | MIT |

The NBA Last Two Minute reports (`l2m_atlhawksfanatic`) map referee names to NBA official ids.
The referee build reads the file from the code directory; the orchestrator copies it there.

## 3. Run

```bash
cd nbacore-v1.6
uv run --locked --with psutil python ../nbacore/scripts/rebuild_v1_6.py \
    --code . --data-root /path/to/nba_data_v16 \
    --reference ../nbacore/docs/reference/v1.6_files.json
```

`psutil` is only used to record peak memory. A failed run can be resumed with
`--from-step <step>`; one step can be repeated with `--only <step>`. Every step is appended to
`<data root>/rebuild_log_v1.6.json` (start time, seconds, peak memory, data-root size), and the
comparison is written to `<data root>/rebuild_verify_v1.6.json`.

| step | what | time (measured) | peak memory |
|---|---|---|---|
| `check` | code is `data-v1.6`, clean; L5 files and their sha256 | seconds | — |
| `download` | 636 archives (3.6 GB) + play-by-play | 30–45 min to download; 30 s to verify files already present | 0.1 GB |
| `l1` | L1 for all games (`build_l1.py --all`, 4 workers) | 27.5 min | 10.0 GB |
| `v1.0` | publish release v1.0 (L1 + fixed split) | 30 s | 0.1 GB |
| `l2` | L2 events, ball handler, play-by-play alignment (3 workers) | 80 min | 3.3 GB |
| `frames` | per-event distance snapshots | 10 min | 2.9 GB |
| `l3` | possession ledger, fouls, free throws, ghost_v1, shot clock | 27 min | 2.1 GB |
| `l3extra` | `event_possession`, `pbp_possession`, `ghost_v1_l2release` | 8.5 min | 6.6 GB |
| `v1.2` | publish v1.2 = v1.0 + L2 + L3 | 50 s | 0.1 GB |
| `l5a` | shots, player bio, box scores (read release v1.2) | 15 s | 1.7 GB |
| `stage` | staging folder `releases/v1.4` (see below) | 1 s | — |
| `l5b` | advanced metrics, awards, id map, RAPTOR, referees | 5 s | 0.1 GB |
| `v1.6` | publish v1.6 = v1.2 + L5 | 25 s | 0.1 GB |
| `verify` | compare with `docs/reference/v1.6_files.json` | 10 s | — |

Total: about 2 h 40 min of compute on a 22-thread laptop, plus the download.

## Notes on the chain

- **Intermediate version names.** The L5 build scripts of `data-v1.6` read releases named `v1.2`
  (shots, player bio, box scores) and `v1.4` (Basketball-Reference and referees read the box-score
  tables from it).
  - `publish_release.py --l5-build` needs all nine L5 tables at once. So `stage` creates a
    read-only folder `releases/v1.4`: hard links to `v1.2` plus the two box-score tables, and a
    `STAGING.txt` instead of a MANIFEST.
  - It is not a release; delete it after the run if you like.
- **Report files.** Several build scripts rewrite report JSON files that are tracked in the
  repository, and `publish_release.py` refuses a modified checkout. Before each publish, the
  orchestrator copies the regenerated reports to `<data root>/rebuild_reports/` and restores the
  committed ones.
- **What the data root contains afterwards:** `releases/v1.0`, `v1.2`, `v1.6` (and the staging
  `v1.4`), `latest.txt` pointing to `v1.6`, and `scratch/`.
  - `scratch/` can be deleted, except that a later rebuild of individual layers starts from it.
  - Consumers read `v1.6` with `nbacore.load` and `NBA_DATA_ROOT` set to this data root.
- **Data versions and code.** Earlier data versions were built with earlier code. The internal
  tags `data-v1.0` … `data-v1.4` are not public. Rebuilding with the `data-v1.6` code gives the v1.6
  content for every layer; the intermediate releases `v1.0` and `v1.2` of this chain are build
  steps, not the historical releases of those names.
