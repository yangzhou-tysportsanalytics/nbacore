"""Passes and handoffs (L2).

Input: the L2 handler table (``nbacore.events.handler``), ball flights
(``nbacore.events.flight``) and L1 frames of one game.

A **transfer** is a change of the (sticky) handler from P to R ≠ P inside one continuity segment.
Between P's last in-hand frame (``t_release_ms``: last frame with P as candidate) and R's first
frame as candidate (``t_catch_ms``) the ball is either free or with nobody in particular.

``kind`` (the definition used for possession-value models):
* a **free-flight** run of ≥ ``min_flight_frames`` between the two → ``pass``;
* no free flight: passer–receiver distance at the transfer ≤ ``handoff_max_ft`` → ``handoff``;
  ≤ ``ambiguous_max_ft`` → ``ambiguous``; farther → ``ambiguous`` with ``flight_missing``.
Flags for the action-segmentation definition: ``gap_3_5ft`` (0 free frames, 3–5 ft) and
``short_toss`` (1–5 free frames, ≤ 8 ft); ``n_free_frames`` / ``max_free_run`` give both
definitions' flight counts.

Transfers across a flight that reaches the rim are shots / rebounds, not passes. Transfers to the
other team across a flight are passes with ``intercepted``. Transfers to the other team without a
flight (strips) are not passes. A flight from P that ends with no handler in the segment (out of
bounds, tracking hole) is an incomplete pass (``receiver_id`` null) unless it reaches the rim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.io.raw import BALL_ID

PASS_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,  # pass | handoff (kind "ambiguous" is reported as event_type "pass")
    "kind": pl.Utf8,  # pass | handoff | ambiguous
    "passer_id": pl.Int64,
    "passer_team_id": pl.Int64,
    "receiver_id": pl.Int64,  # null: nobody caught it (incomplete)
    "receiver_team_id": pl.Int64,
    "t_release_ms": pl.Int64,
    "t_catch_ms": pl.Int64,
    "x_release": pl.Float32,  # ball at release
    "y_release": pl.Float32,
    "x_catch": pl.Float32,  # ball at catch
    "y_catch": pl.Float32,
    "completed": pl.Boolean,  # caught by a teammate
    "intercepted": pl.Boolean,  # caught by the other team
    "deflected": pl.Boolean,  # a defender was the nearest candidate at some frame in between
    "flight_s": pl.Float32,  # t_catch - t_release (s)
    "max_ball_z": pl.Float32,
    "n_free_frames": pl.Int32,  # free frames between release and catch (action-segmentation "flight frames")
    "max_free_run": pl.Int32,  # longest run of free frames (a flight for ``kind`` if >= 2)
    "transfer_dist_ft": pl.Float32,  # passer - receiver at the catch frame
    "min_dist_ft": pl.Float32,  # min passer - receiver within +-0.5 s of the catch
    "t_converge_ms": pl.Int64,  # handoffs: start of monotone approach (<= 2 s before catch)
    "gap_3_5ft": pl.Boolean,
    "short_toss": pl.Boolean,
    "flight_missing": pl.Boolean,
    "cand_ambiguous": pl.Boolean,  # an identity merge touched the passer or receiver frames
    "event_uid": pl.Utf8,
}


@dataclass
class PassConfig:
    min_flight_frames: int = 2
    handoff_max_ft: float = 3.0
    ambiguous_max_ft: float = 5.0
    short_toss_max_frames: int = 5
    short_toss_max_ft: float = 8.0
    converge_max_s: float = 2.0
    control_s: float = 0.4  # a real control after a shot: handler held this long
    dead_ball_s: float = 1.0  # clock still frozen this long after the catch -> dead ball


class _Positions:
    """Player xy by (period, unix_ms) for fast lookups."""

    def __init__(self, frames: pl.DataFrame):
        p = frames.filter(pl.col("team_id") != BALL_ID).select(
            "player_id", "period", "unix_ms", "x", "y"
        )
        self._d = {}
        for (pid,), g in p.sort(["player_id", "period", "unix_ms"]).group_by(
            ["player_id"], maintain_order=True
        ):
            self._d[int(pid)] = (
                g["period"].to_numpy(),
                g["unix_ms"].to_numpy(),
                g["x"].to_numpy().astype(float),
                g["y"].to_numpy().astype(float),
            )

    def at(self, pid: int, period: int, t: np.ndarray) -> np.ndarray:
        """(len(t), 2) positions; NaN where the player is not tracked at that exact frame."""
        out = np.full((t.size, 2), np.nan)
        if pid not in self._d:
            return out
        per, u, x, y = self._d[pid]
        m = per == period
        u, x, y = u[m], x[m], y[m]
        k = np.searchsorted(u, t).clip(0, max(u.size - 1, 0))
        ok = (u.size > 0) & (u[k] == t) if u.size else np.zeros(t.size, bool)
        out[ok] = np.c_[x[k[ok]], y[k[ok]]]
        return out


def _runs(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def passes(
    handler: pl.DataFrame,
    flights: pl.DataFrame,
    frames: pl.DataFrame,
    frame_index: pl.DataFrame | None = None,
    cfg: PassConfig | None = None,
) -> pl.DataFrame:
    """Passes / handoffs of one game. With ``frame_index``, ``dead_ball`` marks transfers after
    which the game clock stays frozen for ``dead_ball_s`` (ball handed around during a stoppage);
    those are neither ``completed`` nor ``intercepted``."""
    cfg = cfg or PassConfig()
    gid = handler["game_id"][0]
    ball = frames.filter(pl.col("team_id") == BALL_ID).select("period", "unix_ms", "x", "y", "z")
    h = handler.join(ball, on=["period", "unix_ms"], how="left").sort(["period", "unix_ms"])
    t = h["unix_ms"].to_numpy()
    per = h["period"].to_numpy()
    seg = h["segment_id"].to_numpy()
    hid = h["handler_id"].to_numpy()
    cand = h["cand_id"].to_numpy()
    amb = h["cand_ambiguous"].to_numpy()
    free = (h["has_ball"] & (h["cand_id"] < 0)).to_numpy()
    bx, by, bz = (h[c].to_numpy().astype(float) for c in ("x", "y", "z"))
    team_of = dict(
        frames.filter(pl.col("team_id") != BALL_ID)
        .select("player_id", "team_id")
        .unique()
        .iter_rows()
    )
    pos = _Positions(frames)
    rows = []
    # handled runs: (start, end) index ranges with a constant handler >= 0
    idx = np.flatnonzero(hid >= 0)
    starts = ends = np.zeros(0, dtype=int)
    if idx.size:
        brk = (
            np.flatnonzero((np.diff(idx) > 1) | (np.diff(hid[idx]) != 0) | (np.diff(seg[idx]) != 0))
            + 1
        )
        starts, ends = idx[np.r_[0, brk]], idx[np.r_[brk - 1, idx.size - 1]]

    # shot / rebound windows: from the start of a flight that reaches the rim until the first
    # real control afterwards (a handler run >= control_s in the same segment). Handler changes
    # inside are tips, rebounds, the ball dropping through the net - not passes.
    min_ctrl = max(1, int(round(cfg.control_s * 25)))
    t_index = {int(u): i for i, u in enumerate(t.tolist())}
    windows = []
    for f in flights.filter(pl.col("reaches_rim")).iter_rows(named=True):
        i_end = t_index[int(f["t_end_ms"])]
        k = np.searchsorted(starts, i_end + 1)
        t_ctrl = None
        while k < starts.size and seg[starts[k]] == seg[i_end]:
            if ends[k] - starts[k] + 1 >= min_ctrl:
                t_ctrl = int(t[starts[k]])
                break
            k += 1
        windows.append(
            (int(f["period"]), int(f["t_start_ms"]), t_ctrl if t_ctrl else int(t[i_end]))
        )

    def in_shot_window(p: int, t0: int, t1: int) -> bool:
        return any(p == wp and ws <= t1 and t0 < we for wp, ws, we in windows)

    if idx.size:
        for k in range(starts.size - 1):
            a_end, b_start = ends[k], starts[k + 1]
            P, R = int(hid[a_end]), int(hid[b_start])
            if P == R or seg[a_end] != seg[b_start]:
                continue
            # P's last in-hand frame and R's first candidate frame
            rel = a_end
            while rel > starts[k] and cand[rel] != P:
                rel -= 1
            cat = b_start
            while cat < ends[k + 1] and cand[cat] != R:
                cat += 1
            if cand[rel] != P or cand[cat] != R:
                continue
            p = int(per[rel])
            if in_shot_window(p, int(t[rel]), int(t[cat])):
                continue
            between = slice(rel + 1, cat)
            fr = free[between]
            n_free, max_run = int(fr.sum()), _runs(fr)
            has_flight = max_run >= cfg.min_flight_frames
            same_team = team_of.get(P) == team_of.get(R)
            if not same_team and not has_flight:
                continue  # strip / steal without a flight: not a pass
            rows.append(
                _row(
                    gid,
                    p,
                    P,
                    R,
                    team_of,
                    rel,
                    cat,
                    t,
                    bx,
                    by,
                    bz,
                    cand,
                    amb,
                    between,
                    n_free,
                    max_run,
                    has_flight,
                    same_team,
                    pos,
                    cfg,
                )
            )
    # incomplete passes: flights from a handler that nobody catches in the segment (no rim)
    inc = flights.filter(
        pl.col("from_id").is_not_null() & pl.col("to_id").is_null() & ~pl.col("reaches_rim")
    )
    done = {(r[4], r[8]) for r in rows}  # (passer, t_release) already reported as a transfer
    for f in inc.iter_rows(named=True):
        i0 = t_index[int(f["t_start_ms"])] - 1
        P = int(f["from_id"])
        rel = i0
        while rel > 0 and cand[rel] != P and seg[rel] == seg[i0]:
            rel -= 1
        if cand[rel] != P or (P, int(t[rel])) in done:
            continue
        if in_shot_window(int(f["period"]), int(t[rel]), int(f["t_end_ms"])):
            continue
        i1 = t_index[int(f["t_end_ms"])]
        between = slice(rel + 1, i1 + 1)
        fr = free[between]
        # a flight split in two by a short touch walks back to the same release: report it once
        done.add((P, int(t[rel])))
        rows.append(
            _row(
                gid,
                int(f["period"]),
                P,
                None,
                team_of,
                rel,
                None,
                t,
                bx,
                by,
                bz,
                cand,
                amb,
                between,
                int(fr.sum()),
                _runs(fr),
                True,
                None,
                pos,
                cfg,
            )
        )
    out = pl.DataFrame(rows, schema=PASS_SCHEMA, orient="row")
    if frame_index is not None:
        # dead ball: the game clock is frozen at the catch (or, if nobody caught it, at the
        # release) and still shows the same value dead_ball_s later. An inbound pass is released
        # with a frozen clock but the clock starts right after the catch, so it stays live.
        gc = frame_index.select("period", "unix_ms", "game_clock").sort(["period", "unix_ms"])
        ref = out.with_row_index("_i").select(
            "_i",
            "period",
            pl.coalesce("t_catch_ms", "t_release_ms").alias("_t0"),
        )
        ref = ref.with_columns((pl.col("_t0") + int(cfg.dead_ball_s * 1000)).alias("_t1"))
        a0 = ref.sort(["period", "_t0"]).join_asof(
            gc.rename({"unix_ms": "_t0", "game_clock": "_g0"}),
            on="_t0",
            by="period",
            check_sortedness=False,
        )
        a1 = ref.sort(["period", "_t1"]).join_asof(
            gc.rename({"unix_ms": "_t1", "game_clock": "_g1"}),
            on="_t1",
            by="period",
            check_sortedness=False,
        )
        dead = (
            a0.select("_i", "_g0")
            .join(a1.select("_i", "_g1"), on="_i")
            .select(
                "_i",
                ((pl.col("_g1") - pl.col("_g0")).abs() < 0.005).fill_null(False).alias("dead_ball"),
            )
        )
        out = (
            out.with_row_index("_i")
            .join(dead, on="_i", how="left")
            .drop("_i")
            .with_columns(
                (pl.col("completed") & ~pl.col("dead_ball")).alias("completed"),
                (pl.col("intercepted") & ~pl.col("dead_ball")).alias("intercepted"),
            )
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Boolean).alias("dead_ball"))
    return out.sort(["period", "t_release_ms"])


def _row(
    gid,
    p,
    P,
    R,
    team_of,
    rel,
    cat,
    t,
    bx,
    by,
    bz,
    cand,
    amb,
    between,
    n_free,
    max_run,
    has_flight,
    same_team,
    pos,
    cfg,
):
    t_rel = int(t[rel])
    t_cat = int(t[cat]) if cat is not None else None
    dist = min_dist = None
    t_conv = None
    if R is not None:
        w = np.arange(max(rel, cat - int(0.5 * 25)), cat + int(0.5 * 25) + 1)
        w = w[w < t.size]
        pp, pr = pos.at(P, p, t[w]), pos.at(R, p, t[w])
        d = np.hypot(*(pp - pr).T)
        dc = d[np.flatnonzero(w == cat)[0]] if (w == cat).any() else np.nan
        dist = None if np.isnan(dc) else float(dc)
        min_dist = None if np.all(np.isnan(d)) else float(np.nanmin(d))
    if not has_flight:
        if dist is None:
            kind, flight_missing = "ambiguous", True
        elif dist <= cfg.handoff_max_ft:
            kind, flight_missing = "handoff", False
        elif dist <= cfg.ambiguous_max_ft:
            kind, flight_missing = "ambiguous", False
        else:
            kind, flight_missing = "ambiguous", True
        # start of the monotone approach of giver and receiver before the transfer
        lo = max(0, cat - int(cfg.converge_max_s * 25))
        ww = np.arange(lo, cat + 1)
        dd = np.hypot(*(pos.at(P, p, t[ww]) - pos.at(R, p, t[ww])).T)
        j = dd.size - 1
        while j > 0 and not np.isnan(dd[j - 1]) and dd[j - 1] >= dd[j]:
            j -= 1
        t_conv = int(t[ww[j]])
    else:
        kind, flight_missing = "pass", False
    gap_3_5 = n_free == 0 and dist is not None and cfg.handoff_max_ft < dist <= cfg.ambiguous_max_ft
    short_toss = (
        1 <= n_free <= cfg.short_toss_max_frames
        and dist is not None
        and dist <= cfg.short_toss_max_ft
    )
    seg_c = cand[between]
    passer_team = team_of.get(P)
    deflected = bool(np.any([(c >= 0) and (team_of.get(int(c)) != passer_team) for c in seg_c]))
    zmax = float(np.nanmax(bz[rel : (cat if cat is not None else between.stop) + 1]))
    end = cat if cat is not None else between.stop - 1
    event_type = "handoff" if kind == "handoff" else "pass"
    t_key = t_cat if kind == "handoff" else t_rel
    return (
        gid,
        p,
        event_type,
        kind,
        P,
        passer_team,
        R,
        team_of.get(R) if R is not None else None,
        t_rel,
        t_cat,
        float(bx[rel]),
        float(by[rel]),
        float(bx[cat]) if cat is not None else None,
        float(by[cat]) if cat is not None else None,
        bool(R is not None and same_team),
        bool(R is not None and same_team is False),
        deflected,
        (t_cat - t_rel) / 1000 if t_cat is not None else (int(t[end]) - t_rel) / 1000,
        zmax,
        n_free,
        max_run,
        dist,
        min_dist,
        t_conv,
        bool(gap_3_5),
        bool(short_toss),
        bool(flight_missing),
        bool(amb[rel] or (cat is not None and amb[cat])),
        f"{gid}:{event_type}:{t_key}:{P}",
    )


def mark_after_made_fg(
    passes_df: pl.DataFrame, shots: pl.DataFrame, handler: pl.DataFrame, max_s: float = 10.0
) -> pl.DataFrame:
    """Dead ball after a made field goal (the clock may still run): from the release of a made FG
    until the other team first holds the ball (capped at ``max_s``), a transfer released by the
    scoring team is a courtesy toss / net retrieval, not play. Sets ``after_made_fg`` and clears
    ``completed`` / ``intercepted`` for those rows (``dead_ball`` also becomes True)."""
    made = shots.filter(pl.col("made") & pl.col("t_release_ms").is_not_null()).select(
        "period", "t_release_ms", "shooter_team_id"
    )
    hh = handler.filter(pl.col("handler_id") >= 0).select("period", "unix_ms", "handler_team_id")
    per = passes_df["period"].to_numpy()
    t = passes_df["t_release_ms"].to_numpy()
    team = passes_df["passer_team_id"].to_numpy()
    flag = np.zeros(passes_df.height, dtype=bool)
    for mp, mt, st in made.iter_rows():
        other = hh.filter(
            (pl.col("period") == mp) & (pl.col("unix_ms") > mt) & (pl.col("handler_team_id") != st)
        )
        t_end = min(
            int(other["unix_ms"].min()) if other.height else np.iinfo(np.int64).max,
            mt + int(max_s * 1000),
        )
        flag |= (per == mp) & (t >= mt) & (t < t_end) & (team == st)
    return passes_df.with_columns(pl.Series("after_made_fg", flag)).with_columns(
        (pl.col("dead_ball").fill_null(False) | pl.col("after_made_fg")).alias("dead_ball"),
        (pl.col("completed") & ~pl.col("after_made_fg")).alias("completed"),
        (pl.col("intercepted") & ~pl.col("after_made_fg")).alias("intercepted"),
    )


def mark_shots(passes_df: pl.DataFrame, shots: pl.DataFrame, tol_ms: int = 200) -> pl.DataFrame:
    """A "pass" released by a shooter within ``tol_ms`` of a FG release is the shot itself (air
    ball, blocked shot, a shot that never reaches the rim zone): ``is_shot`` True, and it is
    neither ``completed`` nor ``intercepted``."""
    s = shots.filter(pl.col("t_release_ms").is_not_null()).select(
        "period",
        pl.col("t_release_ms").alias("_ts"),
        pl.coalesce("shooter_inferred_id", "shooter_id").alias("_sid"),
    )
    j = (
        passes_df.with_row_index("_i")
        .join(s, left_on=["period", "passer_id"], right_on=["period", "_sid"], how="left")
        .with_columns(
            ((pl.col("t_release_ms") - pl.col("_ts")).abs() <= tol_ms).fill_null(False).alias("_m")
        )
        .group_by("_i")
        .agg(pl.col("_m").any().alias("is_shot"))
    )
    return (
        passes_df.with_row_index("_i")
        .join(j, on="_i", how="left")
        .drop("_i")
        .with_columns(
            (pl.col("completed") & ~pl.col("is_shot")).alias("completed"),
            (pl.col("intercepted") & ~pl.col("is_shot")).alias("intercepted"),
        )
    )


INBOUND_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,
    "inbounder_id": pl.Int64,
    "receiver_id": pl.Int64,
    "team_id": pl.Int64,
    "context": pl.Utf8,  # stoppage | after_made_fg
    "t_release_ms": pl.Int64,
    "t_entry_ms": pl.Int64,  # first frame after the release with the ball inside the court
    "t_catch_ms": pl.Int64,
    "x_release": pl.Float32,
    "y_release": pl.Float32,
    "x_entry": pl.Float32,
    "y_entry": pl.Float32,
    "pass_uid": pl.Utf8,
    "event_uid": pl.Utf8,
}


def inbounds(
    passes_df: pl.DataFrame,
    frames: pl.DataFrame,
    frame_index: pl.DataFrame,
    shots: pl.DataFrame | None = None,
    edge_ft: float = 2.0,
    after_made_s: float = 10.0,
) -> pl.DataFrame:
    """Inbound passes: completed passes released with the ball outside the
    court or within ``edge_ft`` of a boundary line (the inbounder's arms reach over the line),
    caught inside the court by a teammate, and either released with a frozen game clock (after a
    stoppage: the clock starts when the ball is touched inside) or within ``after_made_s`` after a
    made field goal by the other team (baseline inbound, the clock may run). ``context`` says
    which. ``t_entry_ms`` / ``x_entry`` / ``y_entry``: first frame after the release with the ball
    inside the court."""
    from nbacore.court import LENGTH, WIDTH

    def near_or_out(x, y, tol):
        return (x < tol) | (x > LENGTH - tol) | (y < tol) | (y > WIDTH - tol)

    def inside(x, y):
        return (x >= 0) & (x <= LENGTH) & (y >= 0) & (y <= WIDTH)

    fz = frame_index.select("period", pl.col("unix_ms").alias("t_release_ms"), "clock_frozen")
    p = passes_df.filter(
        pl.col("completed")
        & near_or_out(pl.col("x_release"), pl.col("y_release"), edge_ft)
        & inside(pl.col("x_catch"), pl.col("y_catch"))
    ).join(fz, on=["period", "t_release_ms"], how="left")
    made = (
        shots.filter(pl.col("made") & pl.col("t_release_ms").is_not_null())
        .select("period", "t_release_ms", "shooter_team_id")
        .rows()
        if shots is not None
        else []
    )
    ball = (
        frames.filter(pl.col("team_id") == BALL_ID)
        .select("period", "unix_ms", "x", "y")
        .sort(["period", "unix_ms"])
    )
    rows = []
    for r in p.iter_rows(named=True):
        if r["clock_frozen"]:
            context = "stoppage"
        elif any(
            mp == r["period"]
            and st != r["passer_team_id"]
            and 0 <= r["t_release_ms"] - mt <= after_made_s * 1000
            for mp, mt, st in made
        ):
            context = "after_made_fg"
        else:
            continue
        b = ball.filter(
            (pl.col("period") == r["period"])
            & pl.col("unix_ms").is_between(r["t_release_ms"], r["t_catch_ms"])
        ).filter(inside(pl.col("x"), pl.col("y")) & ~near_or_out(pl.col("x"), pl.col("y"), 0.0))
        if b.height:
            e = b.row(0, named=True)
        else:
            e = {"unix_ms": r["t_catch_ms"], "x": r["x_catch"], "y": r["y_catch"]}
        rows.append(
            {
                "game_id": r["game_id"],
                "period": r["period"],
                "event_type": "inbound",
                "inbounder_id": r["passer_id"],
                "receiver_id": r["receiver_id"],
                "team_id": r["passer_team_id"],
                "context": context,
                "t_release_ms": r["t_release_ms"],
                "t_entry_ms": int(e["unix_ms"]),
                "t_catch_ms": r["t_catch_ms"],
                "x_release": r["x_release"],
                "y_release": r["y_release"],
                "x_entry": e["x"],
                "y_entry": e["y"],
                "pass_uid": r["event_uid"],
                "event_uid": f"{r['game_id']}:inbound:{int(r['t_release_ms'])}:{r['passer_id']}",
            }
        )
    return pl.DataFrame(rows, schema=INBOUND_SCHEMA)
