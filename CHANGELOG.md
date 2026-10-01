# Changelog

Data releases are immutable; each has a matching code tag `data-vX.Y`. MINOR versions add tables or
columns, or fix bugs without changing the meaning of existing fields.

## v1.5 (2026-09-27): external tables

New L5 tables: advanced metrics and awards (Basketball-Reference via sumitrodatta), RAPTOR
(FiveThirtyEight, CC BY 4.0), a Basketball-Reference → NBA id map, and the officials of every
game (Kaggle, CC BY-SA 4.0). Player bio rebuilt with an improved name matcher (values unchanged).
All other files identical to v1.4.

## v1.4 (2026-09-27): ghost_v1 with the corrected shot release

New table `ghost_v1_l2release.parquet` (`L.ghost_v1(release="l2")`): the ghost-defense half-court
windows with shot windows ending at the corrected L2 release. 13.4 % of shot windows end
0.24–0.96 s earlier, none later. All other files identical to v1.3.

## v1.3 (2026-09-27): L2 fixes and the first external tables

- Shot release: only the pbp shooter's hands count as "in hand" (a ball in flight passing over a
  teammate had been taken as still in hand). Ball > 10 ft from the shooter at the release:
  14.8 % → 5.1 %.
- Free-throw alignment: flights are assigned from the end of a trip; free throws without a
  tracked flight are placed inside the stop (`ft_stop_est`). Before, 24 % of last free throws
  fell back to a clock match before the foul.
- Screen / cut / drive / post-up candidates are generated for the possession's offence only.
- Screen candidates: `screen_kind` (set / moving), `user_def_id`, `screened_def_is_user_def`,
  `screen_group_uid`; passes and handoffs: `n_no_handler_frames`.
- L3: new table `pbp_possession` (every pbp row → possession and role). 8,198 `poss_uid` changed.
- L5: official shot locations (all 1,230 games), player height / weight / wingspan, box scores.

## v1.2 (2026-09-26): possession ledger

New layer L3: `ledger` (122,768 team possessions with sub-segments, lineups and flags), `fouls`
and `free_throws` (team-foul and penalty state, free throw ↔ foul links), `ghost_v1` (the
ghost-defense half-court windows, identical on all 631 games), per-frame `shot_clock` (never null,
rule-imputed where missing) and `event_possession`. Possessions vs the play-by-play estimate
aligned to the ledger's definition: +0.77 % over the season, per-game p95 2.57 %. Free throws
linked to their foul: 99.89 %. Shot-clock imputation: median error 0.65 s, p95 10.6 s.

## v1.1 (2026-09-26): events

New layer L2: per-game event tables (`possession_touch`, `ball_flight`, `pass`, `handoff`,
`shot_release`, `rim_contact`, `rebound_landing`, `clock_stop`, `inbound`, `dribble`, and screen /
drive / cut / post-up candidates), a 25 Hz ball-handler table, `pbp_align` (every pbp event on the
tracking axis, by the game clock only) and distance snapshots at key frames. Last handler before
the release = pbp shooter for 95.1 % of shots with a ball-derived release; fouls anchored on a
clock stop 94 %, residual −0.92 / −0.22 / +0.49 s (5 / 50 / 95 %).

## v1.0 (2026-09-25): tracking frames and fixed split

Cleaned 25 Hz tracking (`frames`, `frame_index`), rosters, attack direction, raw tracking events,
typed play-by-play of the whole season, and `games` with the fixed `split` / `fold` / `parity`.
663 games in the tracking period: 631 usable, 4 empty archives, 1 without any moment, 27 without
an archive. Split: train 455 / val 50 / test 126 (test from 2016-01-04).
