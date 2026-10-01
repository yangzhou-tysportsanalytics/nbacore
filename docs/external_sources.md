# External sources (L5)

nbacore adds a small set of public 2015-16 tables to the tracking data: shot locations for the
whole regular season, player body measurements, box scores, advanced metrics and awards, game
officials, and rule constants from the 2015-16 rulebook. This document records, for each table,
where it comes from, under which terms, how it was processed and how it was checked.

## General policy

- The terms of every source were checked before anything was fetched.
- Raw copies are kept **locally only**, each with a `provenance.json` (URL, commit or version,
  date, sha256, terms summary). They are never committed to git and never redistributed.
- Releases carry only normalised tables or derived constants, with the source named. Any output
  built on these tables must credit the sources as listed below.
- Use is non-commercial research on a one-off historical snapshot; no source is updated
  periodically or offered as a database.
- Not used: the NBA stats API and NBA.com pages accessed by script. They reject non-browser
  clients, and nbacore does not work around access controls.

Loaders: `nbacore.load.shots / player_bio / box_games / box_players / advanced / awards / raptor /
referees`.

## `shots_2015_16` — shot locations, whole regular season

- **Source**: GitHub `DomSamangy/NBA_Shots_04_25` (https://github.com/DomSamangy/NBA_Shots_04_25),
  one zipped CSV per season 2003-04 to 2024-25 (last commit 2025-06-09). Columns include
  TEAM_ID, PLAYER_ID, GAME_ID, GAME_DATE, EVENT_TYPE, SHOT_MADE, ACTION_TYPE, SHOT_TYPE,
  BASIC_ZONE, ZONE_NAME, ZONE_RANGE, LOC_X, LOC_Y, SHOT_DISTANCE, QUARTER, MINS_LEFT, SECS_LEFT.
- **Terms**: the repository has no licence (the GitHub licence API returns Not Found; the README
  states no licence or attribution requirement) and names NBA.com as the data source. Without a
  licence, copyright is reserved by default, so there is no explicit right to redistribute. The
  data are NBA statistics, covered by the NBA.com Terms of Use, section 9 ("NBA Statistics"):
  prominent credit to NBA.com; only legitimate news reporting or private, non-commercial use; no
  gambling, fantasy or commercial products; no complete, regularly updated statistics database.
- **Credit**: "Data: NBA.com (compiled by DomSamangy/NBA_Shots_04_25)".
- **Obtained**: `NBA_2015_Shots.csv.zip` and `NBA_2016_Shots.csv.zip` (about 3.9 MB each)
  downloaded by script from raw.githubusercontent.com. The README does not say whether files are
  named by start or end year, so the 2015-16 file was identified by GAME_DATE (2015-10-27 to
  2016-04-13) and GAME_ID (prefix 00215).
- **Processing** (`scripts/build_external_shots.py`): one row per field-goal attempt with
  `game_id, period, clock_s, player_id, team_id, made, shot_value, action_type, zone_basic,
  zone_name, zone_range, shot_distance_ft`, offence-relative feet `lat_ft` (lateral, basket at 0)
  and `depth_ft` (from the attacking baseline, basket at 5.25), the linked `pbp_event_num` (same
  game, period, clock second and shooter; several attempts of one shooter in one second are
  paired in order) and, for tracked games, the SportVU court position `x_court_ft, y_court_ft`
  (using the attack direction of the shooter's team in that period).
- **Coverage and checks** (`reports/l5_shots_check.json`): 207,893 attempts in all 1,230
  regular-season games; 100 % linked to a pbp field-goal row; made / missed agrees with the pbp
  for 99.89 %; field-goal attempts per game equal to the pbp in 1,135 of 1,230 games (made shots
  in 1,193); season totals 207,893 vs 208,049 pbp attempts. On the tracked games the official
  location was compared with the tracking release location, which also fixed the sign of the
  lateral axis.

## `player_bio_2015_16` — height, weight, wingspan

- **Sources**:
  - height and weight: Kaggle `justinas/nba-players-data`
    (https://www.kaggle.com/datasets/justinas/nba-players-data), `all_seasons.csv` (1.92 MB):
    one row per player and season 1996–2022 with height (cm), weight (kg) and basic statistics,
    compiled by its author from the NBA Stats API. It has **no player id**.
  - wingspan: Kaggle `tymoteuszdobrucki/nba-anthropometric`
    (https://www.kaggle.com/datasets/tymoteuszdobrucki/nba-anthropometric), draft-combine
    measurements for draft years 2000–2023.
- **Terms**: the justinas licence field reads "Other (specified in description)" and the
  description gives no licence text, so it is used for local research only and not
  redistributed; as NBA statistics, NBA.com Terms of Use section 9 applies (credit, private
  non-commercial use). The anthropometric dataset is **CC BY 4.0**.
- **Credit**: "NBA.com / NBA Stats, compiled by justinas (Kaggle)"; "NBA draft combine data,
  compiled by tymoteuszdobrucki (Kaggle), CC BY 4.0".
- **Obtained**: downloaded manually from Kaggle (account required).
- **Processing and id matching**: 2015-16 rows only. Names are normalised (accents removed,
  suffixes dropped, case folded) and matched together with the team to the `player_id` of the
  2015-16 box scores; unmatched players are listed and resolved through a manual mapping table.
  Columns: `player_id, height_cm, weight_kg, wingspan_cm, source_height, source_wingspan,
  match_method`. Name normalisation fixes (2026-09-27): "JR" in "JR Smith" is no longer dropped
  as a "Jr." suffix; the apostrophe in O'Brien is removed instead of treating "O" as an initial;
  characters such as ß and ı are transliterated first. Rebuilding changed no existing value and
  matched one more wingspan.
- **Coverage** (`reports/l5_player_bio_check.json`): of the 450 players on the tracking rosters,
  height and weight 100 %, wingspan 67 % (only players measured at a draft combine).
- **Caveat**: wingspan is missing for players drafted before 2000 or never measured.

## `box_games_2015_16`, `box_players_2015_16` — box scores

- **Source**: Kaggle eoinamoore, "NBA Dataset: Box Scores and Stats (1947 - Today)", version 515,
  files `Games.csv` and `PlayerStatistics.csv`.
- **Terms**: published by the uploader under **CC0**; the underlying data are NBA.com
  statistics (NBA.com Terms of Use section 9: credit, private non-commercial use).
- **Credit**: "NBA.com, compiled by the eoinamoore Kaggle dataset".
- **Obtained**: version 515 downloaded from Kaggle (account required); sha256 recorded in
  `provenance.json`. Built by `scripts/build_external_boxscores.py`; released with v1.3.

| table | rows | columns |
|---|---|---|
| `box_games_2015_16` | 1,230 games | `game_id` (10-digit NBA id), `game_date`, home / away team ids and scores, `arena_id` |
| `box_players_2015_16` | 31,423 player-games, 25,873 with playing time | `game_id, player_id, player_name, team_id, opponent_team_id, home, starter, starting_position, minutes, pts, fgm, fga, fg3m, fg3a, ftm, fta, oreb, dreb, reb, ast, stl, blk, tov, pf, plus_minus, comment` |

- The source has no officials, attendance or arena name for 2015-16, so these columns are absent.
- `comment` holds the reason a player did not play, e.g. "DNP - Coach's Decision".
- `player_id` uses the same ids as the tracking rosters: all 450 roster players are found.
- **Checks against pbp** (`reports/l5_boxscores_check.json`):
  - final score: 1,229 / 1,230 games agree with the last pbp score. The exception, 0021500916
    (not tracked), has an extra "start of period 5" row after Q4 in pbp with a wrong score
    (112-114); the game ended in regulation and the box score (117-115) is correct. Take care
    with this row when using pbp scores.
  - per player-game FGA / FGM / 3PM / FTA / FTM / points against pbp counts: agreement above
    99.9 % for each; of 25,903 player-games only slightly more than ten rows differ, at most 2
    per game.

## `advanced_2015_16`, `awards_2015_16`, `raptor_2015_16`, `bbref_player_map_2015_16`

| source | files | licence / terms | credit |
|---|---|---|---|
| GitHub `sumitrodatta/bball-reference-datasets` (https://github.com/sumitrodatta/bball-reference-datasets, commit 76a70b4), compiled from Basketball-Reference | `Advanced.csv`, `Player Award Shares.csv`, `End of Season Teams.csv`, `End of Season Teams (Voting).csv`, `All-Star Selections.csv` | no licence file; the underlying data belong to Sports Reference and fall under its terms. Local research use only, not redistributed | "Basketball-Reference, compiled by sumitrodatta" |
| GitHub `fivethirtyeight/data`, folder `nba-raptor` (https://github.com/fivethirtyeight/data, commit 6d9b327) | `modern_RAPTOR_by_player.csv` | **CC BY 4.0** | "FiveThirtyEight RAPTOR" |

- **Obtained**: downloaded from GitHub at the commits above; sha256 and terms in
  `provenance.json`. Built by `scripts/build_external_bbref.py`.
- Both sources label 2015-16 as `season == 2016`.
- `bbref_player_map_2015_16`: `bbref_id → player_id` (NBA id), matched by normalised name + team
  against the 2015-16 box scores: 476 / 476 matched, `player_id` unique.
- `advanced_2015_16`: one row per player and team stint; traded players also get a season-total
  row (`team = "TOT"`, `is_total`). Columns: `g, gs, mp, per, ts_percent, x3p_ar, f_tr,
  orb/drb/trb/ast/stl/blk/tov/usg_percent, ows, dws, ws, ws_48, obpm, dbpm, bpm, vorp`. Team codes
  converted to NBA three-letter codes (BRK→BKN, CHO→CHA, PHO→PHX).
- `awards_2015_16`: long table, `category` one of
  - `award_voting`: MVP / ROY / DPOY / SMOY / MIP points, share, first-place votes, winner;
  - `season_team`: All-NBA / All-Defensive / All-Rookie teams with voting, `detail` = 1st / 2nd /
    3rd, vote-getters not selected have `winner = false`;
  - `all_star`: `detail` = East / West; `winner = false` = injured and replaced.
- `raptor_2015_16`: `poss, mp`, offence / defence / total of `raptor_box`, `raptor_onoff` and
  RAPTOR, `war_total / reg_season / playoffs`, `predator_*`, `pace_impact`. 478 players; 2 (John
  Holland, Dorell Wright) have no regular-season box score row and a null `player_id`, presumably
  playoff-only appearances (RAPTOR seasons include the playoffs); not verified.
- **Checks** (`reports/l5_advanced_awards_check.json`):
  - all 450 tracking-roster players have advanced metrics;
  - games played agree with the box scores in 99.8 %, counting a player-game as played when it
    has no DNP comment. The box scores contain 206 appearances of a few seconds with `minutes` = 0:
    count games played with an empty `comment`, not with `minutes > 0`;
  - 2015-16 winners match the public record: MVP Stephen Curry, ROY Karl-Anthony Towns, DPOY
    Kawhi Leonard, SMOY Jamal Crawford, MIP CJ McCollum.

## `referees_2015_16` — game officials

| source | content | role |
|---|---|---|
| Kaggle wyattowalsh "NBA Database", `officials.csv`, from the NBA API game summaries | `game_id` (NBA id), `official_id` (NBA id), name, jersey number | **primary**: 1,124 / 1,230 games of 2015-16 |
| Kaggle pablote "NBA Enhanced Box Score and Standings", `2012-18_officialBoxScore.csv` | one row per game and official, names only, located by date and teams | fills the 106 missing games; matched to the game id by date + home / away team, to the official id by name |

- **Terms**: both are **CC BY-SA 4.0** (attribution; derived data under the same licence). Used
  locally for research and not redistributed.
- **Credit**: "wyattowalsh / pablote (Kaggle), data from NBA.com".
- **Obtained**: downloaded manually from Kaggle (account required); sha256 in `provenance.json`.
  Built by `scripts/build_external_referees.py`.
- **Columns**: `game_id, official_id, official_name, jersey_num, source` (`wyattowalsh` /
  `pablote`), one row per game and official.
- **Coverage**: all 1,230 games (all 631 tracked games), 66 officials; 1,229 games with 3
  officials, 0021501064 with 2.
- **Checks** (`reports/l5_referees_check.json`):
  - against the officials named in the NBA Last Two Minute reports: 410 / 410 games identical;
  - the two sources agree completely in 1,121 of the 1,124 games they share; name variants are
    mapped (e.g. "Steven Anderson" in pablote, "Steve Anderson" in the NBA source);
  - the 3 disagreements are all an extra official in pablote: 0021501103 and 0021501207 (4
    listed), 0021501064 (3 listed, 2 in the NBA source). Possibly a mid-game replacement; not
    verified. The table keeps the NBA-source record.

## Rule constants (`nbacore.rules_2015_16`)

- **Source**: *Official Rules of the National Basketball Association 2015-2016*, official PDF
  https://ak-static-int.nba.com/wp-content/uploads/sites/3/2015/12/2015-16-NBA-Rule-Book.pdf
  (inside cover 10-02-2015; 67 pages; PDF page number = printed page number; sha256 `6d1d3f5c…`).
- **Terms**: NBA.com Terms of Use, section 1: material may be downloaded to a single computer
  for personal, non-commercial use, keeping copyright notices; no publishing, reposting or
  modification for public or commercial purposes. The PDF is on an NBA-operated domain and
  carries an NBA copyright, so these terms are assumed to apply.
- **Obtained**: downloaded manually in a browser (the server rejects non-browser requests with
  403). Third-party uploads of the 2015-16 edition are not used. The current-season rules on
  official.nba.com differ (e.g. the 14-second reset after an offensive rebound applies only from
  2018-19); 2016-17 editions serve only for cross-checking.
- **Use**: the PDF stays local, outside git and every release. Only the extracted values and
  their citations (Rule–Section–article, page) are distributed; rule values are facts. Quote
  the rulebook text only briefly and with a citation.

| constant | value | citation |
|---|---|---|
| `PERIOD_S` | 720 | 5-II-a, p. 21 |
| `OT_PERIOD_S` | 300 | 5-II-b, p. 21 |
| `TWO_MINUTE_S` | 120 ("2:00 or less") | 5-II-f, p. 21 |
| `BACKCOURT_S` | 8 | 4-V-f, p. 18 |
| `FREE_THROW_LIMIT_S` | 10 | 4-IV, p. 18 |
| `SHOT_CLOCK_S` | 24 | 7-II-d, p. 28 |
| `SHOT_CLOCK_TENTHS_BELOW_S` | 5 (tenths shown from 4.9) | 7-I, p. 27 |
| `SHOT_CLOCK_OFF_GAME_CLOCK_S` | 24: after a change of possession with ≤ 24 s left in the period the shot clock does not run | 7-II-i, p. 28 |
| `RESET_FULL_S` | 24 on change of possession; ball touching the ring of the team in possession; foul or violation with a backcourt throw-in; jump balls not caused by a defensive held ball; flagrant and punching fouls | 7-IV-c(1)–(6), p. 29 |
| `OREB_RESET_S` | **24** (offensive rebound after rim contact; the 14-s reset only from 2018-19) | 7-IV-c(2), p. 29 |
| — | no reset: air ball, defence knocks the ball out of bounds, technical / delay warning on the offence, suspension of play, retossed jump ball, jump ball from a defensive held ball | 7-IV-b(1)–(6), p. 29 |
| `FRONTCOURT_RESET_MIN_S` | 14: remaining or 14, whichever is greater, on a defensive personal foul with a frontcourt throw-in, defensive three seconds, technical / delay warning on the defence, kicked / punched ball (frontcourt throw-in), infection control, jump ball retained after a defensive violation | 7-IV-d(1)–(6), p. 29 |
| — | double foul: 24 with a backcourt throw-in, else remaining or 14, whichever is greater | 12B-VI-c, p. 47 |
| `TEAM_FOUL_QUOTA` | 4 team fouls per regulation period without penalty (penalty from the 5th) | 12B-V-a, a(1), p. 46 |
| `TEAM_FOUL_QUOTA_OT` | 3 per overtime period (penalty from the 4th) | 12B-V-a(2), a(4), p. 46 |
| `LAST_TWO_MINUTES_FREE_FOULS` | 1: a team below its quota at 2:00 may commit one team foul in the last two minutes without penalty | 12B-V-a(3), p. 46 |
| `PENALTY_FREE_THROWS` | 2 (one free throw plus a penalty free throw) | 12B-V-a, p. 46; 12B-I PENALTIES (5), p. 45; 12B-VIII-a(4), p. 48 |

`shot_clock_after_defensive_foul(remaining_s, frontcourt_throw_in)` implements 7-IV-c(3) and
7-IV-d(1). `in_penalty(period, game_clock_s, fouls_before, last2_fouls_before)`: in penalty if
the team already has its quota of team fouls in the period, or, with 2:00 or less on the clock,
it already committed a team foul in the last two minutes.

Team fouls (`TEAM_FOUL_DETAILS` / `NON_TEAM_FOUL_DETAILS`, keyed by the `foul_detail` of
`pbp_align`): personal (incl. block, take, inbound), shooting (12B-I PENALTIES, p. 44), loose
ball (12B-VIII-a(1), p. 48), flagrant 1 / 2 (12B-IV-a, b, p. 46), punching (12B-IX-a, p. 48),
clear path and away from play (own penalties, included in the team total; 12B-V-a(5), p. 47)
count. Offensive and offensive charge (12B-VII-a(3), p. 47; 12B-I PENALTIES, p. 44), double
personal (12B-VI-b, p. 47) and technical fouls of all kinds, incl. defensive three seconds and
delay of game, do **not** count (technicals: *inferred* — 12B-V counts common fouls charged as
team fouls; technicals are penalised under Rule 12A, pp. 39-43).

**Consistency with the data** (`scripts/l5_rulebook_check.py` → `reports/l5_rulebook_check.json`;
all 631 games, shot-clock checks on the 619 games with a shot clock):

- **Offensive rebound → 24** (7-IV-c(2)): after a missed FG with a detected rim contact and an
  offensive rebound, the tracked shot clock jumps to ~24 in 89.2 % (6,283 / 7,040), is already 24
  at the release in 2.9 % (tips after an earlier rim contact), and goes to 13–15 s in **1** case —
  the 14-second reset of later seasons does not apply. No reset in 7.6 %; without a detected rim
  contact (air balls, 7-IV-b(5), or missed detection) no reset in 36 %.
- **Defensive non-shooting foul, no free throws** (7-IV-c(3), 7-IV-d(1)): frontcourt throw-in →
  max(remaining, 14) in 83.4 % (4,222 / 5,060); backcourt → 24 in 86.0 % (472 / 549). The rest
  are mostly fouls linked to the wrong clock stop or shot-clock readings changed by the operator
  during the stop; not a sign of a different rule.
- **Penalty situation** (12B-V-a): fouls predicted in the penalty got free throws in 99.9 %
  (2,994 / 2,996); fouls predicted outside it got free throws in 1.9 % (157 / 8,131), mostly
  clear-path or fast-break fouls (12B-I PENALTIES (6), (8)) that pbp records as plain personal
  fouls. Overall agreement 98.6 %.
