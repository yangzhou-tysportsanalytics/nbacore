# Data schema (verified on real files, 2026-09-25)

> This description was first written in the ghost-defense project, from which nbacore's loading
> and cleaning code was migrated. Measurements were made on its `tiny` / `small` subsets; the
> per-game statistics script and its output live in ghost-defense. All numbers below are subset measurements; they are not re-checked here on the
> full set of 634 usable games.

Everything below was measured on the `tiny` subset (5 games) unless stated otherwise.

## 1. Acquisition

* **We do not use the HuggingFace loader.** `dcayton/nba_tracking_data_15_16` is a
  *script-based* dataset: `nba_tracking_data_15_16.py` scrapes the GitHub HTML directory
  listing with a regex at import time, needs `datasets<3` + `trust_remote_code`, and its
  `_generate_examples` (a) silently drops every tracking event whose `eventId` does not
  match exactly one pbp row, and (b) **drops the `unix_ms` timestamp** of every moment.
  We need `unix_ms` for de-duplication, so we replicate the merge ourselves.
* Sources actually used:
  * tracking: `https://github.com/linouk23/NBA-Player-Movements/raw/master/data/2016.NBA.Raw.SportVU.Game.Logs/<name>.7z`
    (636 archives, 3.79 GB compressed, ~6 MB each, one JSON per archive ~60 MB).
  * play-by-play: `https://github.com/sumitrodatta/nba-alt-awards/raw/main/Historical/PBP%20Data/2015-16_pbp.csv`
    (94.6 MB, 568 333 rows, 1 230 games = the whole regular season).
* Subsets: `scripts/download_raw.py --config {tiny,small,medium,large}` reproduces the HF
  sampling (`random.Random(9).sample(name_sorted_listing, n)`, n = 5/25/100/all). The HF
  script samples from the scraped HTML listing, which is also name-sorted, so the subsets
  should coincide; this is *not* guaranteed and does not matter for our results.
* Naming quirk: 51 archives carry a bogus prefix, e.g.
  `2016.NBA.Raw.SportVU.Game.Logs12.05.2015.CHA.at.CHI.7z`. The JSON inside is named by
  game id (`0021500292.json`), so we always key on `gameid` from the JSON.
* tiny = games 0021500115 (TOR@PHI 2015-11-11), 0021500230 (MIA@NYK 11-27),
  0021500292 (CHA@CHI 12-05), 0021500333 (MIA@IND 12-11), 0021500648 (LAC@NYK 2016-01-22).

## 2. Raw tracking JSON (verified)

```
game   = {"gameid": "0021500115", "gamedate": "2015-11-11", "events": [event, ...]}
event  = {"eventId": "12",                       # STRING; == pbp EVENTNUM
          "visitor": team, "home": team,          # full 13-man rosters, identical in every event
          "moments": [moment, ...]}               # may be []  (see §4.3)
team   = {"name": "Toronto Raptors", "teamid": 1610612761, "abbreviation": "TOR",
          "players": [{"lastname","firstname","playerid": int,"jersey": str,"position": str}]}
moment = [period: int, unix_ms: int, game_clock: float, shot_clock: float|None, None, rows]
rows   = [[team_id, player_id, x, y, z], ...]     # usually 11 rows; ball row team_id == player_id == -1
```

| field | type | unit / range | notes |
|---|---|---|---|
| `period` | int | 1–4 in tiny (OT = 5+) | |
| `unix_ms` | int | ms since epoch | strictly increasing inside an event; **the reliable time axis** |
| `game_clock` | float | s remaining in period, 720 → 0 | official clock; can be *corrected upward* by officials (§4.5) |
| `shot_clock` | float or None | s, 24 → 0 | None while the clock is off/stopped (§4.6) |
| `x`, `y` | float | feet; court 94 × 50, origin one corner | observed range x ∈ [−4.8, 99.9], y ∈ [−4.7, 52.0]; ~9–10 % of frames have some entity out of the [0,94]×[0,50] box |
| `z` | float | feet | ball height 0–16.7 ft; **players always 0** |
| `team_id` | int | NBA team id | −1 for ball |
| `player_id` | int | NBA player id | −1 for ball; every on-court id is in the roster |

Sampling: `unix_ms` deltas are 40 ms (mode), with 39/41 ms jitter → **25 Hz** confirmed.

The `home`/`visitor` block is repeated identically in every event (1 roster variant per game).

## 3. Play-by-play CSV (NBA stats format)

34 raw columns; `nbacore.io.load_pbp` keeps 28 (snake_case). Important ones:

| raw column | ours | notes |
|---|---|---|
| `GAME_ID` | `game_id` (str) | CSV stores it as int → **leading zeros lost**; restored with `zfill(10)` |
| `EVENTNUM` | `event_num` | equals tracking `eventId` |
| `EVENTMSGTYPE` | `msg_type`, `msg_type_name` | 1 FG made, 2 FG missed, 3 FT, 4 rebound, 5 turnover, 6 foul, 7 violation, 8 substitution, 9 timeout, 10 jump ball, 11 ejection, 12 period begin, 13 period end, 18 instant replay |
| `EVENTMSGACTIONTYPE` | `action_type` | shot/foul subtype (e.g. 1 jump shot, 5 layup, 79 pull-up …) |
| `PERIOD`, `PCTIMESTRING` | `period`, `pctime_str`, `pctime_sec` | game clock at the event, whole seconds |
| `HOMEDESCRIPTION`, `VISITORDESCRIPTION` | `home_desc`, `visitor_desc` | text, one side non-null; `NEUTRALDESCRIPTION` is always null in this file |
| `SCORE` | `score_str`, `score_visitor`, `score_home` | **format `"<visitor> - <home>"`**, only on scoring rows (26 % of rows); verified against final margins |
| `SCOREMARGIN` | `score_margin` | home − visitor, `"TIE"` → 0, only on scoring rows |
| `PERSONnTYPE` | `pn_type` | 4 = home player, 5 = visitor player, 0 none, 1 official, 2/3 team |
| `PLAYERn_ID/_NAME/_TEAM_ID` | `pn_id`, `pn_name`, `pn_team_id` | `TEAM_ID` is float in CSV (NaN when absent) |
| `WCTIMESTRING` | `wctime_str` | wall clock, minute resolution |

Season totals: FG made 94 065, FG missed 113 984, FT 57 469, rebounds 127 964, turnovers 35 383,
fouls 51 325, substitutions 55 030, timeouts 16 508, jump balls 2 079; periods 1–8 present
(OT games), `NEUTRALDESCRIPTION` never populated.

## 4. Irregularities (all handled downstream)

1. **Duplicated frames across events (massive).** Each event's `moments` start ≈2.4 s
   (median; 90 %: 10 s) before the pbp event time and run ≈16 s (90 %: 20 s) past it, so
   consecutive events overlap heavily. Raw moments per game 188k–227k vs **unique
   `(period, unix_ms)` frames 76k–86k → 54–62 % duplicates.** Duplicates carry identical
   content, except case 2 below. Dedup key: `(game_id, period, unix_ms)` at frame level or
   `(…, team_id, player_id)` at entity level. `game_clock` adds nothing to the key.
2. **Split frames.** In 0021500230, 74 timestamps appear twice with *different* content: a
   1-row moment holding only the ball and a 10-row moment holding only the players. →
   dedup at **entity level** (union of rows per timestamp), not frame level.
3. **Events without coordinates.** 0–65 per game (0–14 % of events). By pbp type they are
   mostly substitutions, timeouts, fouls, period boundaries, but a few FG/rebound events too.
   Also, pbp `EVENTNUM 0` (period begin) never exists in tracking, and each game has 5–8
   tracking events with ids not present in pbp (the HF loader drops those).
4. **Frames with ≠ 11 entities.** 86–486 per game (≤0.6 %): 10 rows = ball missing
   (`no-ball frames`), rarely 7/9 rows = players missing. The ball is *not* guaranteed to be
   row 0 — always select by `team_id == -1`.
5. **Game-clock corrections.** Inside ~25 % of events the game clock *increases* somewhere.
   Almost always < 1 s (official adjustment after a stoppage), but 0021500292 Q3 has a
   +16.5 s and a +24.65 s correction across two consecutive events (clock had run while it
   should have been stopped; officials reset it). `unix_ms` stays monotone. → use `unix_ms`
   as the time axis; treat `game_clock` as a noisy label; split possessions at large jumps.
6. **Shot clock null.** 1.8–7.4 % of unique frames. In 0021500115 all of them are at
   `game_clock ≤ 24` (shot clock switched off). In 0021500648 most occur in 2–7 s stretches
   during which the game clock is frozen (dead balls, free throws, inbounds) — i.e. the shot
   clock feed is simply off while the clock is stopped. None of the 12 games known to lack a
   shot clock is in tiny; that list was still to be checked on `large` when this was written.
7. **Time gaps.** Within a period, `unix_ms` gaps > 1 s occur ~80 times per game (up to
   3 min: timeouts, free-throw sequences not tracked). Continuity must be re-established
   after dedup: a new "segment" starts whenever the gap exceeds 1 frame.
8. **Out-of-bounds coordinates** are common (players stepping out, ball in flight) and are
   kept.
9. **pbp clock readings are late for shots.** Measured on 3 100 FG attempts (small): the ball
   reaches the rim a median 0.6 s *before* the pbp second, and the ball-derived release is a
   median 1.8 s before it (5 %: 3.6 s; occasionally the pbp is early). Non-shot events
   (turnovers, fouls) are presumably similar but cannot be checked from the ball alone.
   Consequence: shot windows end at the ball-derived release, not at the pbp time.

## 5. Deviations from the initial assumptions about the data

| assumption | reality |
|---|---|
| use HF `dcayton/nba_tracking_data_15_16` configs | HF loader is fragile and drops `unix_ms`; we download the same raw files directly and reproduce the seeded subsets |
| pbp merged in by HF | we merge ourselves on `(game_id, eventId == EVENTNUM)`; a handful of events per game exist on one side only |
| "few events without coordinates" | 0–14 % of events per game |
| ball in first row | true in 99.4 % of frames; select by `team_id == -1` |
| 11 rows per moment | 10-row (no ball) and split (1 + 10) frames exist |
| `eventid` int | it is a string in the JSON |
| pbp `GAME_ID` | int in CSV, leading zeros lost |
| game clock monotone within event | ~25 % of events contain upward corrections, one of 24.65 s |
| 12 listed games lack shot clock | shot-clock nulls are also common in *other* games during stopped clock; the listed games were still to be verified on `large` |

Note: the `(game_id, eventId == EVENTNUM)` merge is kept only as provenance. Released tables
align pbp to tracking by the game clock, never by the tracking `eventId`: in 415 of 2,572
game-periods (16 %) fewer than half of the pbp events fall inside their own tracking event's
game-clock range, because the moments are attached to the wrong `eventId`.

## 6. Generated per-game table (tiny)

| game | date | events | no-coord events | raw moments | unique frames | dup frac | no-ball frames | shot-clock null | events w/ clock rewind | max rewind (s) | frac frames any OOB | players used | pbp rows | evt in pbp | final score (V-H) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0021500648 | 2016-01-22 | 440 | 19 | 188253 | 85934 | 0.544 | 264 | 6320 | 111 | 0.830 | 0.105 | 24 | 434 | 433/440 | 116 - 88 |
| 0021500115 | 2015-11-11 | 462 | 39 | 208024 | 79894 | 0.616 | 86 | 1476 | 137 | 0.890 | 0.087 | 22 | 460 | 459/462 | 119 - 103 |
| 0021500230 | 2015-11-27 | 450 | 0 | 227289 | 86046 | 0.621 | 331 | 1877 | 168 | 1.480 | 0.105 | 23 | 443 | 442/450 | 97 - 78 |
| 0021500333 | 2015-12-11 | 427 | 18 | 198506 | 77915 | 0.607 | 137 | 2838 | 110 | 0.800 | 0.100 | 19 | 423 | 422/427 | 83 - 96 |
| 0021500292 | 2015-12-05 | 468 | 65 | 189777 | 76416 | 0.597 | 486 | 3273 | 106 | 24.650 | 0.077 | 20 | 464 | 463/468 | 102 - 96 |

Unique frames per game ≈ 80k × 40 ms ≈ 53 min of tracked time (48 min of play + some dead-ball
tracking). Extrapolated to 631 games: ≈ 50 M unique frames, ≈ 550 M entity rows.
