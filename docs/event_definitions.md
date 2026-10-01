# L2 event definitions (v1.1)

Every event: rule, parameters (also in the release MANIFEST), validation method and numbers.
All events are computed at 25 Hz on the L1 frames of the release, per continuity segment of the
frame index; times are `unix_ms` (the game clock is only a label: officials can correct it
upward). pbp is aligned by the game clock only, never by the tracking `eventId`, whose moment
windows are attached to the wrong pbp event in about 16 % of game-periods.

## Handler (`handler/<game_id>.parquet`, `possession_touch` events)

- Code: `nbacore.events.handler.game_handler`, `HandlerL2Config`.
- Rule:
  1. Candidate per frame = nearest player (xy) of either team within `max_dist_ft` of the ball
     while the ball is below `max_ball_z_ft` (`cand_id`); `cand_ambiguous` when the nearest and
     second-nearest are at the identical distance (identity merges: two players reported at the
     same position).
  2. Controlling team (`control_team_id`): run-length on the candidate's team; a run of the
     other team takes control only if it lasts ≥ `team_switch_s` or follows ≥ `min_hold_s`
     without a candidate; the first run of a segment takes control if ≥ `min_hold_s`.
  3. Handler candidate = nearest player of the controlling team within `max_dist_ft`; sticky
     rule: runs < `min_hold_s` inherit the previous handler unless followed by ≥ `min_hold_s`
     without a candidate (catch-and-shoot, tip pass).
- Parameters: `max_dist_ft` 3.0, `max_ball_z_ft` 10.0, `min_hold_s` 0.4, `team_switch_s` 1.0,
  `hz` 25.
- Validation (small, 25 games, ghost-defense possession windows, 929,492 ball frames):
  frame agreement with ghost-defense's offence-only handler 97.9 %; last handler before the
  ball-derived release = pbp shooter 95.0 % (ghost-defense 96.2 %). Full-data numbers: after the
  v1.1 build.

## Ball flight (`ball_flight` events)

- Code: `nbacore.events.flight.ball_flights`, `FlightConfig`.
- Rule: a frame is *free* when the ball is tracked and has no handler candidate (no player within
  3 ft in xy, or ball above 10 ft) — the handler test, so flight and handler never disagree. A
  flight is a run of ≥ `min_flight_frames` free frames inside one continuity segment.
  `from_id` = handler on the frame before; `to_id` = first handler candidate after, in the same
  segment (null if none); `reaches_rim` = ball within `rim_xy_ft` of a hoop centre between
  `rim_z_lo` and `rim_z_hi` (the rim zone of the shot-release rule).
- Parameters: `min_flight_frames` 2, `rim_xy_ft` 2.0, `rim_z_lo` 8.0, `rim_z_hi` 11.5.
- Checks (tiny, 2 games): ~1,000 flights per game, 209–222 reaching the rim (pbp FGA 161–176
  plus free throws and tips); teammate-to-teammate flights not reaching the rim 612–653 per game
  (league ≈ 600 passes per game for both teams).

## Pass / handoff (`pass`, `handoff` events)

- Code: `nbacore.events.passes.passes`, `PassConfig`.
- Rule: a transfer is a change of the sticky handler P → R ≠ P within a segment. `t_release_ms` =
  P's last frame as candidate, `t_catch_ms` = R's first frame as candidate.
  - `kind`: longest free run between them ≥ `min_flight_frames` → `pass`; otherwise by the P–R
    distance at the catch: ≤ `handoff_max_ft` → `handoff`, ≤ `ambiguous_max_ft` → `ambiguous`,
    farther → `ambiguous` + `flight_missing`.
  - Flags for an alternative pass / handoff definition: `gap_3_5ft` (0 free frames, 3–5 ft) and
    `short_toss` (1–5 free frames, ≤ 8 ft); `n_free_frames` and `max_free_run` give both
    definitions' flight counts.
  - Not passes: transfers inside a **shot window**, from the start of a flight that reaches the rim
    until the first handler run ≥ `control_s` (tips, rebounds, the ball through the net); and
    transfers to the other team without a flight (strips).
  - Other team across a flight → `intercepted`. A flight that nobody catches in the segment (and
    that does not reach the rim) → incomplete pass (`receiver_id` null).
  - `dead_ball`: the game clock is frozen at the catch (or release, if incomplete) and still
    frozen `dead_ball_s` later — a ball handed around during a stoppage. Such transfers are
    neither `completed` nor `intercepted`. Inbound passes stay live: the clock starts right after
    the catch.
  - `event_uid = <game_id>:<event_type>:<t_key_ms>:<passer_id>`, key time = release (pass) or
    catch (handoff), rounded to the 40 ms frame; `nbacore.events.match_events(old, new,
    tol_ms=200)` maps uids between releases.
- Parameters: `min_flight_frames` 2, `handoff_max_ft` 3.0, `ambiguous_max_ft` 5.0,
  `short_toss_max_frames` 5, `short_toss_max_ft` 8.0, `converge_max_s` 2.0, `control_s` 0.4,
  `dead_ball_s` 1.0.
- Reconciled with shot events (`mark_after_made_fg`, `mark_shots`):
  - `after_made_fg`: from a made FG's release until the other team first holds the ball (≤ 10 s),
    transfers released by the scoring team are dead ball (courtesy toss, net retrieval).
  - `is_shot`: a "pass" released by the shooter within 200 ms of a FG release is the shot itself
    (air ball, block, a shot not reaching the rim zone).
  Both clear `completed` / `intercepted`.
- Validation (small, 25 games, `scripts/l2_pass_validate.py`): 16,443 rows (15,006 `pass`,
  301 `handoff`, 1,136 `ambiguous`), uids unique.
  - **Assist coverage 85.1 %** (935 / 1,099 assisted made FGs have a completed pass or handoff
    assister → shooter, catch within the game-clock window [pbp s − 1, pbp s + 12]).
    Main misses (before reconciliation): shooter or assister never handler (72), someone between
    them (32), transfer inside a shot window (17), no tracking or a tracking hole (30).
  - Interceptions 706 vs pbp steals 391, after removing 786 dead-ball transfers (263 after made
    FGs) and 155 shots. The rest are possession changes through a flight that pbp does not
    credit as a steal (deflected balls recovered, long passes picked off without a steal
    credit) or rebound scrambles; `pbp_align` adds the link to the pbp turnover so that a
    pbp-confirmed turnover can be required.

## Shot release, rim contact, rebound (`shot_release`, `rim_contact`, `rebound_landing`)

- Code: `nbacore.events.shots.shot_events`, `ShotEventConfig`.
- Rule: terminal pbp events are walked in order as in ghost-defense `segment_game`; each is placed
  with `locate_clock` (game clock, bin 1 s) after the previous terminal. FG attempts are refined
  with `refine_shot_release` (search window −5 s … +2 s around the pbp time, lower bound = the
  previous terminal): first rim contact, else the closest apex above 9 ft; release = last frame
  with the ball within 3 ft of a shooting-team player and not at rim height inside 4 ft of the
  hoop. After the release: `t_rim_ms` (first rim-zone frame within 3 s), `t_control_ms` /
  `control_id` (first handler run ≥ 0.4 s in the same segment — rebounder after a miss,
  usually the inbounder after a make), `offensive_rebound`, and the rebound landing (first frame
  after the rim contact with the ball below 9 ft while falling). `blocked` from pbp (PLAYER3).
  `release_minus_pbp_s` is stored.
- Parameters: `ShotTimeConfig` (before 5 s, after 2 s, rim 2 ft / 8–11.5 ft, apex 9 ft,
  hand 3 ft, rim zone 4 ft, hand z 8.5 ft), `clock_bin_s` 1.0, `rim_search_s` 3.0,
  `control_s` 0.4, `landing_z_ft` 9.0.
- **v1.3**: `ShotTimeConfig(shooter_only=True)`: only the pbp shooter's hands count as "in hand"
  (fallback to any shooting-team player, method suffix `_anyhand`). Under the earlier rule (any
  shooting-team player's hands) a ball in flight passing over a teammate counted as in hand and
  the release was placed mid-air: ball > 10 ft from the shooter at the release 14.6 % → 5.7 %
  (10 games, `reports/l5_shot_release_check.json`; 0 % for `rim` / `apex`); 87 % of releases
  unchanged, the rest 0.2–1.1 s earlier. A height cap (`hand_z_ft` 10) was tried and rejected:
  jump shots leave the hand at 10–12 ft, so it moved correct releases 80–120 ms earlier.
  ghost_v1 keeps the earlier any-hand release (parity with ghost-defense).
- Validation (small, 25 games, 4,174 FG attempts, `scripts/l2_shot_validate.py`):
  - **`t_release_ms` = ghost-defense `t_terminal` for 3,098 / 3,100** shot possessions (also a
    data test on tiny, `tests/test_l2_parity.py`). The 2 others are pbp rows inserted late
    (0021500504: Q4 events 513 and 518 numbered among OT events). Walking terminal events by
    event number, ghost-defense refines them onto the release of the earlier event 426; nbacore
    keeps the release for the row whose pbp time is closest and marks the others
    `duplicate_release` (no release), so no two FG rows share a release.
  - Methods: rim 81.9 %, rim_nohand 6.9 %, apex 5.6 %, pbp 2.7 %, unlocated 2.7 %.
  - Release − pbp (s): 5 % −4.00, 25 % −2.48, median −1.88, 75 % −1.24, 95 % 0.00 (so a −3 s
    window would miss > 5 % of releases).
  - Rim contact found 89.0 %; offensive rebound rate 25.8 % of misses with a control;
    inferred shooter = pbp shooter 83.9 %.

## Clock stop (`clock_stop` events)

- Code: `nbacore.events.clock_stop.clock_stops`, `StopConfig`.
- Rule: for a run of frames a..b with `clock_frozen` (clock equal to the previous frame's), frame
  a−1 is the first frame showing the stopped value (`t_start_ms`), a−2 the last running frame
  (`t_last_running_ms`, `clock_left_s` above the stop value).
  - `onset_exact` when a−2, a−1 are contiguous and the clock was running at a−2.
  - Otherwise the stop happened inside a tracking hole: the public data keep windows around pbp
    events, so tracking typically resumes with the clock already stopped. A running clock
    advances in real time, so the onset is estimated as
    `t_onset_est_ms = t_last_running_ms + clock_left_s · 1000` (capped at `t_start_ms`).
  - `t_end_ms` = restart (first running frame), `ends_at_gap` if the run ends at a hole;
    `duration_s` from the onset estimate to the last stopped frame; kept if ≥ `min_stop_s`;
    `micro_freeze` if < `micro_s`; `clock_correction` if a `clock_jump` lies within 1 s.
  - Window columns (for foul analysis) on [onset − 3 s, onset) and [onset, onset + 1 s).
  - `stop_cause_hint` from the pbp events anchored on the stop (foul > violation > turnover >
    jump ball > timeout > replay > ejection), else a made FG in the last 2:00 of Q4 / OT released
    ≤ 5 s before the onset, else `unknown`.
- Parameters: `min_stop_s` 0.2, `micro_s` 1.0, `correction_s` 1.0, `pre_s` 3.0, `post_s` 1.0,
  `missing_player_s` 0.2.
- Checks (small, 25 games): 1,791 stops, 14.0 % with an exact onset; for the others the last
  running frame is a median 0.25–0.33 s of game clock before the stop (90 %: ≤ 0.8 s). Pre-onset
  3 s tracked share median 0.93. Freezes shorter than 0.2 s: 26 in 25 games (2 frames: 21,
  3 frames: 2, 5 frames: 3). Causes: foul 978, turnover 200, timeout 75, made FG last 2:00 56,
  violation 46, jump ball 9, replay 1, unknown 426.

## pbp alignment (`pbp_align.parquet`)

- Code: `nbacore.events.pbp_align.align_game`. Game clock only (see the introduction).
- Anchors: FG → shot release; free throw → the n-th rim-reaching flight of the shooter inside the
  stop at that clock; rebound after a FG miss → first control after the shot (matching player or
  team), else the pbp rebounder catching the ball at the end of a flight ≤ 6 s after the release;
  rebound after a missed last free throw → first handler after the FT flight; rebound after a
  missed free throw that is not the last → the free throw itself (`ft_dead_ball`: dead ball,
  pbp books a team rebound); turnover → catch of a live interception by the other team (passer = turnover player,
  catch clock in [pbp s − 1, pbp s + 2]); fouls, violations, turnovers, timeouts, jump balls,
  replays, ejections → stop onset (stops with pbp s − 0.5 ≤ clock < pbp s + 1.5; several → nearest
  pbp s + 0.5); else `locate_clock`.
- `anchor_quality` 0–4 and `match_level`, `residual_s`, `n_stops_within_2s`,
  `stop_event_uid`, `linked_event_uid`; `foul_class` / `foul_detail` (action types checked
  against the descriptions; 10 and 16 have none, every row names two players of opposite teams
  → double personal / double technical); `ft_n`, `ft_m` from the description. Passes get
  `pbp_turnover_event_num` when a turnover is anchored on them.
- Validation (small, 25 games, 11,573 pbp events, `scripts/l2_align_validate.py`), share
  anchored on a tracking event (quality ≥ 2) / unlocated: FG 94.9 % / 2.4 % (made), 94.3 % /
  2.5 % (missed); fouls 95.0 % / 0.3 %; violations 94.3 % / 0; free throws 79.3 % / 0;
  rebounds 74.1 % / 2.2 % (after the free-throw and flight fallbacks; was 63.3 %); turnovers
  67.6 % / 2.1 % (26.4 % on an interception); timeouts 49.7 % / 1.8 %; substitutions clock match
  only; jump balls 44.7 % unlocated, period starts 60.2 % unlocated (tip-offs are untracked).
  **Stop residual (pbp s − clock at the onset), 5 / 50 / 95 %: fouls −0.93 / −0.24 / 0.42 s
  (n = 998); violations −0.92 / −0.23 / 0.43 (n = 50); timeouts −5.9 / −0.32 / 0.32 (n = 170).**
  `residual_s` is in game-clock seconds; negative = the tracked clock at the anchor shows more time
  left than the pbp second (the pbp clock is floored to whole seconds).

## `shot_clock_filled` (`nbacore.shot_clock`)

- `fill_method`: `observed` (0–24 s), `suspect` (tracked value outside [0, 24], not used),
  `off_game_clock` (missing while the game clock < 24 s or < the last observed shot clock → game
  clock), `carry_forward` (missing while the game clock is frozen → last observed value),
  `missing` (running clock with > 24 s left, not filled), `unavailable` (the 12 games without a
  shot clock; rule-based rebuild in v1.2).
- Check: 0021500115 observed 78,418 frames, off_game_clock 1,476; 0021500040 unavailable 73,161.

## Grids and distances (functions, not stored)

- `nbacore.grid.resample(frames, frame_index, poss, hz, n_steps, flip_to_left, post_release_s,
  handler, handler_mode)` wraps ghost-defense `resample_possession`; default n_steps =
  (24 s + post_release_s) · hz + 1; adds `handler_slot` (L2 handler, or ghost-defense's with
  `handler_mode="ghost"`), `ball_in_flight`, `shot_clock_filled`, `fill_method`.
  Tested on real possessions: at 5 Hz with the ghost-defense handler identical to ghost-defense
  `resample_possession` + `handler_on_grid`; 10 / 25 Hz steps k·hz/5 equal the 5 Hz steps.
- `nbacore.geometry`: `pair_distances` (opposite-team pairs, 25 Hz long table),
  `ball_distances`, `player_speeds` (5-frame centred mean, central differences, null across
  holes), `grid_distances` ((T, 5, 5) offence × defence, (T, 10) ball, ball speed). Pairs involving
  a player whose x, y equals another player's in that frame (identity merge) get
  `dist_ft` null and `xy_shared` True.

## Screen candidate (`screen_candidate` events)

- Code: `nbacore.events.screens.screen_candidates`, `ScreenConfig`. Loose on purpose: a superset
  of several screen definitions, with the raw features each needs.
- Rule: per frame, with the controlling team T of the L2 handler table, a triple (screener S,
  user U, defender D) qualifies when S, U ∈ T, S ≠ U, S not the handler, S's speed
  < `set_speed_fts`; D is U's nearest opponent and within `set_dist_ft` of S; U within
  `user_dist_ft` of S. A run of qualifying frames with the same (S, U, D) in one segment is a
  candidate if the set run around it (S slow and within `set_dist_ft` of D) lasts ≥ `min_set_s`.
  `t_contact_ms` = minimum S–D distance in the run; candidates with the same S and D within
  `dup_window_s` share `dup_group` (several users of one physical screen).
- **v1.3**: T is the **ledger offence** (the handler's controlling team is ignored where it
  disagrees; v1.2 had 16.4 % candidates of the defence). `screen_kind`: `set` (the rule above) or
  `moving` (second pass: S slower than `moving_speed_fts` 9.0 and closing ≥ `moving_approach_ft`
  1.0 on D over the 0.5 s before contact, never set; a moving candidate of the same (S, U, D)
  within `dup_window_s` of a set one is dropped). `user_def_id`: U's most frequent nearest
  opponent over `user_def_window_s` (−1.0, −0.2) s before the contact; `screened_def_is_user_def`
  (≈ 70 %: the screened defender is often a helper or the screener's own defender).
  `screen_group_uid` = `<game>:screen_group:<first contact ms>:<screener>`: all candidates of one
  screener with contacts within `dup_window_s` of the group's first (one physical screen;
  ≈ 1.5 candidates per group).
- Fields:
  - core: `screener_id`, `user_id`, `screened_def_id`, `screener_def_id` = nearest opponent of S
    at contact other than D, `t_contact_ms`, `min_dist_ft`, `on_ball`, `x`, `y`;
  - timing and geometry: `t_set_start_ms`, `t_approach_ms`, `t_pass_ms`, `t_end_ms`, speeds,
    `min_dist_screener_user_ft`, `approach_angle_deg`, `side_passed`, screener displacement in the
    1.5 s after contact and its component toward the basket, both defenders' distance to U at
    contact + 1 s, `ball_in_flight_at_contact`;
  - contact motion: screener mean / max speed, displacement and its component along the screened
    defender's travel over [−0.5, 0], [−0.3, 0], [0, +0.3] s; screened defender's mean speed and
    approach angle;
  - `xy_shared_at_contact` (identity merge at the contact — distances and speeds unreliable).
- `t_pass_ms` is searched from contact − 1.5 s (not only after the contact): the defender trails
  U, so U usually passes the screener before the defender reaches it. `event_uid`
  actor = `<screener_id>-<user_id>`.
- Speeds: positions smoothed by a centred 5-frame mean, central differences.
- Parameters: `set_speed_fts` 6.0, `set_dist_ft` 6.0, `user_dist_ft` 8.0, `min_set_s` 0.2,
  `approach_speed_fts` 2.0, `smooth_frames` 5, `dup_window_s` 1.0.
- Checks (0021500115): 1,328 candidates in 950 physical screens (337 on-ball); `t_pass_ms` found
  for 68.4 %; 21 with an identity merge at the contact. ≈ 3–4 × the usual number of screens per
  game — within the intended tolerance (≤ ~3 candidates per true screen); precision from a
  manual check of 50 possessions, recall from 412 screens
  in the NBA Last Two Minute (L2M) reports.


## Drive / cut candidates (`drive_candidate`, `cut_candidate` events)

- Code: `nbacore.events.motion.motion_candidates`, `MotionConfig`. Loose on purpose.
- Radial speed = decrease rate of the smoothed distance to the basket the team attacks (ft/s).
  With the controlling team T (L2 handler table): **drive** = T's handler, radial speed
  > `min_radial_fts` for ≥ `min_s`, start ≥ `drive_min_start_ft` from the basket **or** outside
  the paint (union of two candidate definitions); **cut** = a T player who is not the handler,
  same speed rule, no start distance. A run inside one segment is a candidate.
- Fields: `def_id` (nearest opponent at the start), `t_start_ms / t_peak_ms / t_end_ms`,
  distances to the basket, peak / mean radial speed, `def_offset_*` (player's minus defender's
  distance to the basket: > 0 = defender between him and the basket), `def_dist_*`,
  `start_frontcourt`. Drives: `start_in_paint`, `speed_max_fts`, `second_def_dist_end_ft`,
  `t_paint_ms`, `t_restricted_ms` (4 ft), `end_reason` (shot / pass / turnover / foul / none, from
  shot releases, passes and foul stops of the driver up to 0.5 s after the end). Cuts:
  `heading_change_deg`, `ball_dist_start_ft`, `ball_minus_cutter_to_basket_ft`, `caught` /
  `t_catch_ms` (completed pass to the cutter in the cut or ≤ 0.5 s after), `screen_uid` (a screen
  candidate with this user whose contact lies in the 1.5 s before the start).
- Parameters: `min_radial_fts` 5.0, `min_s` 0.3, `drive_min_start_ft` 10.0, `link_after_s` 0.5,
  `screen_link_ft` 8.0, `screen_link_s` 1.5, `smooth_frames` 5.
- Checks (0021500115): 436 drives (229 starting in the frontcourt; ends: pass 155, shot 49, foul
  6, turnover 3, none 223), 1,304 cuts (523 in the frontcourt; 149 caught; 204 after a screen
  candidate). Backcourt runs (bringing the ball up) are included on purpose; filter with
  `start_frontcourt`.

## Dribble (`dribble` events) and touch summaries (`possession_touch`)

- Code: `nbacore.events.dribble.dribbles`, `DribbleConfig`.
- A dribble is a local minimum of ball z inside a possession touch with `z_min` ≤ 1.0 ft,
  prominence ≥ 1.0 ft (smaller of the highest z within 0.3 s before / after, minus `z_min`),
  3-frame vertical velocity < 0 before and > 0 after, ≥ 0.2 s after the previous dribble of the
  touch, ball within 4 ft of the handler. Nothing is inferred across missing-ball frames
  (`ball_gap` on the touch).
- `possession_touch` rows gain `n_dribbles`, `live_duration_s` (frames with a running clock),
  `ball_gap`, `ends_with` (shot / pass / handoff / turnover / none: the handler's shot release or
  pass within ± 0.2 s of the touch end).
- Check (0021500115): 1,733 dribbles; per touch median 0, max 21; touches end with a pass 427,
  shot 105, turnover 14, handoff 12, none 409. Totals are to be compared with official tracking
  stats.

## Post-up candidate (`postup_candidate` events)

- Code: `nbacore.events.motion.postups`, `PostUpConfig`.
- The handler within 18 ft of the basket, speed < 4 ft/s, nearest opponent within 4 ft, for
  ≥ 1.0 s (no upper limit), inside one segment. `def_id` = the opponent nearest most often;
  mean distance to the basket, mean speed, mean defender distance, `def_between_frac` (share of
  frames with the defender closer to the basket), `n_dribbles`, `end_reason`
  (shot / pass / turnover / exit, within 1 s after the end).
- Check (0021500115): 47 candidates, duration median 1.4 s, defender between in 89 % of frames.

## Inbound (`inbound` events)

- Code: `nbacore.events.passes.inbounds`.
- A completed pass released with the ball outside the court or within `edge_ft` = 2 ft of a
  boundary line, caught inside, and either released with a frozen game clock (`context =
  stoppage`) or ≤ 10 s after a made FG by the other team (`after_made_fg`, the clock may run).
  `t_entry_ms`, `x_entry`, `y_entry` = first frame after the release with the ball inside.
- Check (0021500115): 84 inbounds (42 after stoppages, 42 after made FGs; 61 stops and 83 made
  FGs in the game — the rest fall in tracking holes).

## Event-frame distances (`event_frames/<game_id>.parquet`)

- Code: `nbacore.geometry.event_frame_distances`; built by `scripts/build_event_frames.py` after
  the L2 build.
- One row per key frame: shot release, pass / handoff release, pass / handoff catch. Offence =
  the shooter's / passer's team; `off_ids`, `def_ids` (sorted, as present in the frame),
  `off_def_dist_ft` (row-major offence × defence), `ball_dist_off_ft`, `ball_dist_def_ft`;
  distances involving an identity merge are null, `xy_shared` flags the frame.
- Check (0021500115): 1,445 snapshots (174 shot releases, 631 + 612 pass releases / catches,
  14 + 14 handoffs), 4.4 s per game.
