# L3 tables of release v1.2: possession ledger, fouls, free throws, ghost_v1, shot clock

Rule constants: `nbacore.rules_2015_16` (values and rulebook citations in `external_sources.md`).
Built by `scripts/build_l3.py` from the L1 tables and the L2 build (v1.1). Times are tracking
`unix_ms` (ms).

Loaders: `nbacore.load.ledger / epv_full / fouls / free_throws / foul_state_at / ghost_v1 /
with_post_release / shot_clock / event_possession`.

## `ledger.parquet` — one row per team possession

A possession runs from gaining the ball to losing it: it ends with a made FG (after an
and-one's free throws), a made last free throw, a defensive rebound (team rebounds included), a
turnover (offensive fouls included), a jump ball won by the other side, the end of the period, or
an unrecorded change (a FG attempt / free throw / turnover of the team not in possession).
Offensive rebounds, technical free throws and dead-ball team rebounds do not end it. Possessions
of a period are contiguous on the tracking axis.

| column | type | meaning |
|---|---|---|
| `game_id`, `period` | str, i8 | |
| `poss_seq` | i32 | order in the game |
| `poss_uid` | str | `<game_id>:<period>:<t_start_ms>` — stable key; `:k` suffix for zero-length duplicates |
| `offense_team_id`, `defense_team_id` | i64 | |
| `attacks_left` | bool | offence attacks the left basket (x = 0) in this period |
| `t_start_ms`, `t_end_ms` | i64 | start = end of the previous possession (period start: first frame) |
| `start_type` | str | `period_start`, `jump_ball`, `made_basket_inbound`, `made_ft_inbound`, `defensive_rebound`, `turnover`, `unrecorded_change`, `free_throws` |
| `end_type` | str | `made_fg`, `made_ft`, `defensive_rebound`, `turnover`, `jump_ball_lost`, `period_end`, `game_end`, `unrecorded_change` |
| `start_event_num`, `end_event_num` | i32 | pbp events |
| `end_time_method` | str | how `t_end_ms` was set: the `pbp_align` method of the end event (`shot_release`, `control`, `ft_flight`, `pass`, `clock_stop`, `clock_match`), `handler_switch` (sustained control of the new offence in the L2 handler table, for defensive rebounds without a tracked control and unrecorded changes), `clock` / `clock_fallback:*` / `after_gap` (fallbacks), `period_last_frame`; `:clamped` when forced monotone |
| `n_fga`, `n_fta`, `n_oreb`, `points` | i16 | points exclude technical free throws |
| `tech_points_offense`, `tech_points_defense` | i16 | technical free throws made during the possession |
| `score_margin_offense_start` | i16 | offence − defence after the start event |
| `offense_player_ids`, `defense_player_ids` | list[i64] | on court at the first frame |
| `lineup_stable` | bool | the same ten players at the last frame |
| `subsegments` | list[struct] | `{kind, t_start_ms, t_end_ms, ref_event_uid}` sorted by time; kinds: `backcourt`, `frontcourt` (ball half; runs < 0.5 s merged), `shot` (release → rim contact / landing; ref = shot event uid), `offensive_rebound` (control instant), `second_chance` (offensive rebound → next release or end), `free_throws` (one per trip; ref = `<game_id>:pbp:<foul event_num>`), `inbound` (release → catch), `dead_ball` (clock stop, clipped) |
| `t_first_frontcourt_ms`, `t_first_shot_ms` | i64 | |
| `is_transition` | bool | first frontcourt entry → first release (or end) < 4 s (the threshold of ghost-defense's half-court view) |
| `n_backcourt_returns`, `backcourt_return_ms`, `backcourt_return_live_ms` | i16, i32, i32 | v1.6: number and total duration of `backcourt` runs after the first `frontcourt` run; the last excludes the time inside `dead_ball` sub-segments. Null when the possession never reached the frontcourt |
| `has_jump_ball`, `has_technical`, `has_flagrant`, `has_clear_path` | bool | pbp events of the possession |
| `data_gap` | bool | a tracking hole during live play (the game clock ran > 0.5 s across it) |
| `ghost_v1_window_uids` | list[str] | ghost_v1 windows linked to the possession |

`epv_full` = these columns in the column order used for expected-possession-value models;
`drop_backcourt_subsegments=True` removes the backcourt entries.

**Possession-count acceptance.** Ledger possessions are compared with the box-score estimate
FGA − OREB + TOV + 0.44 · FTA, aligned to the ledger's definition: team offensive rebounds are
included, and possessions that end with the period without a shot, free throw or turnover are
added. Acceptance: |season total difference| ≤ 3 % against the aligned estimate **and** the 95th
percentile of the per-game |difference| ≤ 3.5 %; the classic estimate is reported alongside.

**Offence agreement with tracking** (v1.5, all 631 games; `reports/l3_definitions_check_v1.5.json`):
on running-clock frames with a ball handler (35.8 M frames) the handler's team is the ledger offence
in 97.4 %; the shooter's team is the ledger offence for 99.68 % of 103,236 shot releases. Every
running-clock frame lies in some possession (possessions tile each period by construction).

**How `backcourt` / `frontcourt` are decided** (v1.6 note). Per ball frame of the
possession, the half is the side of the midcourt line (x = 47 ft) the *ball* is on, relative to
the basket the offence attacks; runs shorter than 0.5 s are merged into the previous run. This is
a ball-position rule, not the rulebook's backcourt status (which also needs the player's feet),
and it runs through dead balls. So a `backcourt` run after the first `frontcourt` run is not a
backcourt violation. On v1.5 (122,726 possessions with sub-segments) 3.1 % have one; of the
3,919 such runs:
- 66 % lie mostly inside `dead_ball` sub-segments: the ball is carried across during a stoppage;
- 2 % start within 4 s of a shot of the offence (long rebounds, tip-outs);
- 33 % are other live play. Among those, the offence controls the ball in 74 % (e.g. deflections
  recovered in the backcourt; the frontcourt run before the return reaches a median 25 ft past
  midcourt, so these are not mainly near-line crossings). The defence controls it in 26 %: the
  possession has in fact changed before the ledger end (end anchored late, about 0.3 % of
  possessions).

Use `backcourt_return_live_ms > 0` to find the live cases. An analysis that treats every
backcourt run after the first frontcourt entry as live play of the possession is not affected by
the flag.

## `fouls.parquet` — one row per pbp foul

| column | meaning |
|---|---|
| `game_id, event_num, period, pctime_sec, unix_ms` | `unix_ms` from `pbp_align` |
| `committed_by_team`, `player_id` | |
| `foul_class`, `foul_detail` | pbp action type mapping (`pbp_align.FOUL_DETAIL`) |
| `counts_as_team_foul` | 12B; offensive, double and technical fouls do not count |
| `team_fouls_before`, `team_fouls_in_period_after` | fouling team, this period |
| `fouls_in_last2min_before`, `fouls_in_last2min_after` | same, game clock ≤ 2:00 |
| `in_bonus_for_fouled_team` | the fouling team was in the penalty before this foul (12B-V-a) |
| `fouls_to_give` | team fouls the fouling team could still commit without penalty |
| `opp_team_fouls_before`, `opp_in_penalty`, `opp_fouls_to_give` | the other team, before the foul |
| `n_free_throws` | free throws linked to the foul |

`nbacore.load.foul_state_at(game_id, unix_ms)`: both teams' state at a tracking time.

## `free_throws.parquet` — one row per free throw

`game_id, event_num, period, pctime_sec, unix_ms, team_id, player_id, ft_n, ft_m, is_last_ft,
ft_kind` (regular | technical | flagrant | clear_path), `made`, `linked_foul_event_num`.

## `ghost_v1.parquet` — ghost-defense's half-court windows

ghost-defense's window rules (`segment_game`) migrated into nbacore as a reference view on
nbacore inputs. The columns of ghost-defense `POSSESSION_SCHEMA` (identical rows to
ghost-defense's own output on all 631 games), plus `window_uid = <game_id>:<terminal_event_num>`
(stable), `poss_uid` (ledger possession overlapped most), `overlap_frac`, `offense_match`.
Consistency check: every ghost window lies inside one ledger possession with the same offence;
exceptions are explained one by one.
`nbacore.views.ghost_v1.ghost_v1_windows(..., SegmentConfig(include_second_chance=True))`
starts windows that never left the frontcourt at the last offensive rebound (optional; not in
the release table). `with_post_release(seconds ≤ 3)` adds `t_end_post_ms` = release + seconds
for shot windows.

## `shot_clock/<game_id>.parquet` — per frame of `frame_index`

| column | meaning |
|---|---|
| `period, unix_ms` | |
| `shot_clock_filled` | f32; never null |
| `fill_method` | `observed`, `off_game_clock`, `carry_forward` (as `shot_clock_filled`), or `imputed:<missing / suspect / stuck_24 / unavailable>` |
| `shot_clock_imputed` | True where the rule-based value is used (every frame of the 12 games without a shot clock) |

Rule-based value (`shot_clock_resets` / `shot_clock_imputed`): resets to 24 at possession
starts, rim contacts (held until the first control), flagrant fouls and backcourt throw-ins after
defensive fouls; max(remaining, 14) for frontcourt throw-ins, defensive three seconds, defensive
technicals and kicked balls; the clock starts at the inbound catch. Backtest on 619 games:
median |error| 0.65 s, 62 % within 1 s, 75 % within 2 s, p95 10.6 s
(`reports/l3_shot_clock_backtest.json`).

## `pbp_possession.parquet` — pbp row → ledger possession (v1.3+)

One row per play-by-play row of a tracked game (`nbacore.ledger.pbp_possession_map`). Use it to
slice the play-by-play by possession; event numbers are not always in time order, so ranges such
as `start_event_num < event_num <= end_event_num` miss or misplace rows.

| column | meaning |
|---|---|
| `game_id`, `event_num` | the pbp row |
| `poss_seq`, `poss_uid` | the ledger possession it belongs to; null for rows before any offence is known |
| `role` | `end_event` (the row that ends the possession), `start_event` (starts it, when it is not also the end of the previous one), `inside`, `between_possessions` (e.g. a period start before the tip) |
| `oreb_listed_after_putback`, `putback_event_num`, `missed_shot_event_num` | v1.6: the offensive rebound is listed after the putback it led to; the putback's and the missed shot's event numbers |

A row that ends a possession (made last free throw, defensive rebound, turnover, lost jump ball)
belongs to it; rows the ledger treats as part of the scoring possession (a putback rebound listed
after the make, and-one free throws) belong to it too.

## `event_possession.parquet` — L2 event → ledger possession

One row per offensive-action event of v1.1 (`screen_candidate`, `cut_candidate`,
`drive_candidate`, `postup_candidate`, `pass`, `handoff`, `dribble`, `possession_touch`,
`shot_release`, `inbound`): `game_id, event_uid, event_type, period, t_key_ms` (screens:
`t_contact_ms`; shots: release − 1 ms; else `t_start_ms`), `team_id` (the event's team),
`poss_uid`, `offense_team_id` (ledger offence), `offense_match` (event team = ledger offence).

Why: the L2 handler's controlling team is sometimes the defence for over a second; candidates
generated then have screener and user on the defence. In a manual review of 224 screen
candidates, candidates with `offense_match` true were real screens in 48 % (82 / 171), with it
false in 10 % (2 / 20). Season-wide `offense_match` is false for 16.4 % of screen candidates,
6.5 % of cut candidates, 5.3 % of passes, 1.5 % of dribbles. **Filter candidates on
`offense_match`.** The handler itself is to be constrained by the ledger in a later version.
