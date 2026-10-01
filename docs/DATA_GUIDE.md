# nbacore data guide

How to use the nbacore data releases and what to watch out for. Column-level definitions are in
`data_schema.md` (raw data and L1), `event_definitions.md` (L2), `ledger_schema.md` (L3),
`splits.md` (L4) and `external_sources.md` (L5).

## 1. Setup

```python
# pyproject.toml: nbacore = { git = "https://github.com/yangzhou-tysportsanalytics/nbacore", tag = "data-v1.5" }
import nbacore.load as L

V = "v1.5"  # pin the data version in your configuration; do not use "latest"
L.games(V)  # start here: which games have tracking data, and the train / val / test split
```

- Releases live under `$NBA_DATA_ROOT/releases/vX.Y/`; they are read-only and immutable, and
  `MANIFEST.json` records the hash of every file.
- Every data version has a matching code tag `data-vX.Y`; pin both together.
- Frame-level coordinates are never committed or redistributed. External tables are for local
  research use only (section 5).

## 2. Game coverage

| scope | games | notes |
|---|---|---|
| 2015-16 regular season | 1,230 | play-by-play, box scores, shot locations, referees and advanced metrics cover every game |
| with tracking data (`games.status == "ok"`) | **631** | 2015-10-27 to 2016-01-23; layers L1–L4 exist for these games only |
| missing or broken in the tracking period | 32 | `missing_archive` 27, `empty_archive` 4, `failed:SchemaError` 1 |

Fixed split (L4, `splits.md`): `split` (train 455 / val 50 / test 126, by date, test = the last
20 %), `fold` (0–4, for cross-fitting), `parity` (odd / even games).

## 3. Layers and loaders

All times are on one tracking axis, `unix_ms` (milliseconds). `game_clock` is the game clock
(seconds, counting down); `period` is the quarter or overtime number.

### L1 tracking frames (since v1.0, unchanged)
| loader | grain | notes |
|---|---|---|
| `frames(gid)` | one row per frame and entity (the ball has `player_id == -1`) | court coordinates in feet (x 0–94, y 0–50), deduplicated, 25 Hz |
| `frame_index(gid)` | one row per frame | `gap`, `clock_jump`, `clock_frozen`, `segment_id` (continuity segment) |
| `rosters()`, `attack_direction()`, `pbp()`, `tracking_events()` | — | rosters; attack direction per team and period; season play-by-play; raw event windows |

### L2 events (since v1.1)
| loader | grain | notes |
|---|---|---|
| `events(gid, types)` | one row per event (long table) | types: `shot_release`, `pass`, `handoff`, `dribble`, `possession_touch`, `ball_flight`, `rim_contact`, `rebound_landing`, `inbound`, `clock_stop`, and the candidates `screen_candidate`, `drive_candidate`, `cut_candidate`, `postup_candidate`; key `event_uid` |
| `handler(gid)` | one row per frame | ball handler `handler_id` (`-1` = nobody), `control_team_id` |
| `pbp_align()` | one row per pbp event | the event's place on the tracking axis; `method`, `anchor_quality` |
| `event_frames(gid)` | shot releases, pass releases, catches | offence × defence and ball distances |

### L3 possession ledger (since v1.2)
| loader | grain | notes |
|---|---|---|
| `ledger()` / `epv_full()` | one row per team possession | sub-segments, both lineups, flags; key `poss_uid` |
| `pbp_possession()` | one row per pbp event | the possession it belongs to and its role (v1.3+); use it to slice the play-by-play by possession instead of event-number ranges |
| `event_possession()` | one row per offensive event | possession and `offense_match` |
| `fouls()`, `free_throws()`, `foul_state_at(gid, t)` | per foul / free throw | team fouls, penalty state, free throw ↔ foul links |
| `ghost_v1()`; `ghost_v1(release="l2")` (v1.4+) | one row per half-court window | the ghost-defense half-court possessions, key `window_uid`; the second variant ends shot windows at the corrected shot release |
| `with_post_release(s)` | half-court windows | windows extended to s seconds (≤ 3) after the release |
| `shot_clock(gid)` | one row per frame | never null; `shot_clock_imputed` marks rule-based values |

### L5 external tables (since v1.3, extended in v1.5; `external_sources.md`)
| loader | content | source / terms |
|---|---|---|
| `shots()` | official shot locations of the whole season, linked to the play-by-play | NBA.com, compiled by DomSamangy |
| `player_bio()` | height, weight; wingspan where a draft-combine measurement exists | Kaggle (wingspan data CC BY 4.0) |
| `box_players()`, `box_games()` | box scores | NBA.com, compiled by eoinamoore |
| `advanced()`, `awards()` | advanced metrics; award voting, All-NBA teams, All-Star selections | Basketball-Reference, compiled by sumitrodatta |
| `raptor()` | RAPTOR, WAR | FiveThirtyEight, **CC BY 4.0** |
| `referees()` | the three officials of every game (NBA official ids) | Kaggle (wyattowalsh, pablote), **CC BY-SA 4.0** |

Code helpers that are not part of a data release:
- `nbacore.grid.resample(...)`: resampling onto a fixed-rate grid (`shot_clock=False` skips the shot clock);
- `nbacore.geometry`: distances and speeds;
- `nbacore.video`: a private broadcast-video index (metadata only).

## 4. Keys

| key | format / notes |
|---|---|
| `game_id` | 10-character string, e.g. `0021500333` |
| `player_id`, `team_id`, `official_id` | NBA ids (integers); every L5 table is mapped to them |
| `poss_uid` | `<game>:<period>:<t_start_ms>`; **stable within one release only**; across releases join on `(game_id, poss_seq)` |
| `event_uid` | contains a time; changes when an event time is corrected; match across releases with a time tolerance |
| `window_uid` | `<game>:<terminal_event_num>`; independent of the shot release time, stable across releases |
| `screen_group_uid` | candidates of one screener with contacts within 1 s: one physical screen (v1.3+) |

## 5. Caveats

1. **Candidate events favour recall.** Filter them for your use:
   - on a manual review of 191 decided screen candidates, 48 % were real screens (on-ball 75 %,
     off-ball 37 %);
   - recall against the screen fouls of the NBA Last Two Minute reports is 76 % (off-ball 59 %);
   - since v1.3 candidates are generated for the possession's offence only;
2. **Shot release:**
   - about 5 % of releases remain implausible (ball > 10 ft from the shooter at the release),
     mostly with `method` `pbp`, `*_nohand`, `*_anyhand`; filter on `method` if needed;
   - `ghost_v1` keeps the original ghost-defense release rule; use `release="l2"` to match L2.
3. **Tracking defects:**
   - in about 2.4 % of frames two players share identical coordinates (an identity merge in the
     source, `geometry.shared_xy`); distances involving them are null, not 0;
   - `frame_index.gap` marks holes, the ledger's `data_gap` marks holes during live play.
4. **Shot clock:** 12 games have no shot clock at all and other games have stuck or missing frames;
   all are filled by rule (median error 0.65 s, p95 10.6 s). Use `shot_clock_imputed` for
   sensitivity analyses.
5. **Play-by-play quirks:**
   - event numbers are not always in time order; use `nbacore.ledger.order_pbp`;
   - offensive rebounds are often listed after the putback they led to; the ledger already
     counts such rows in the scoring possession;
   - game 0021500916 has a spurious "start of period 5" row.
6. **Box scores:** 206 appearances of a few seconds have `minutes` = 0; count games played with
   `comment` null.
7. **RAPTOR** has an on-off component but is not a strict RAPM.
8. **Terms of the external data:**
   - local research use only, no redistribution; credit the sources in outputs;
   - credit FiveThirtyEight for RAPTOR; the referee tables are CC BY-SA, so derived data must keep
     that licence;
   - per-source details are in `external/SOURCES.json` of each release.

## 6. Versions

| version | content |
|---|---|
| v1.0 | L1 tracking frames, L4 fixed split |
| v1.1 | L2 events |
| v1.2 | L3 possession ledger, fouls and free throws, ghost_v1 windows, per-frame shot clock |
| v1.3 | L2 fixes (shot release, free-throw alignment, candidates of the offence only), screen kinds and groups, `pbp_possession`; L5 shot locations, player bio, box scores |
| v1.4 | `ghost_v1(release="l2")` |
| v1.5 | L5 advanced metrics, awards, RAPTOR, referees |

Full notes: `CHANGELOG.md`.
