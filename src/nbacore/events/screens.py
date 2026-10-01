"""Screen candidates, on- and off-ball (L2). Loose on purpose (high recall); consumers filter.

Per frame, with the controlling team T (``control_team_id`` of the L2 handler table), a triple
(screener S, user U, defender D) qualifies when

* S and U are players of T, S ≠ U, S is not the handler, S's speed < ``set_speed_fts``;
* D is U's nearest opponent in that frame and within ``set_dist_ft`` of S;
* U is within ``user_dist_ft`` of S.

A run of qualifying frames with the same (S, U, D) inside one continuity segment is a candidate
if the *set* run around it (S slow and within ``set_dist_ft`` of D, U not required) lasts
≥ ``min_set_s``. ``t_contact_ms`` = frame of minimum S–D distance inside the candidate run;
``t_set_start_ms`` = start of the set run. Candidates of the same S and D whose contacts lie
within ``dup_window_s`` share a ``dup_group`` (the same physical screen, several users).
``screen_group_uid`` (v1.3) is coarser: all candidates of the same screener whose contacts lie
within ``dup_window_s`` of the group's first contact, whatever U and the screened defender (one
physical screen; on the v1.2 review 1.5 candidates per screen).

Speeds come from positions smoothed by a centred ``smooth_frames`` moving average.

v1.3: a second pass adds ``screen_kind = "moving"`` candidates (screener below
``moving_speed_fts`` and closing on the defender, but never set); ``user_def_id`` is U's
defender over the matchup window before the contact. The handler table passed in is constrained
by the ledger offence (``nbacore.events.build``), so candidates of the defence are no longer
generated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from nbacore.court import HOOP_LEFT, HOOP_RIGHT
from nbacore.io.raw import BALL_ID

FPS = 25


@dataclass
class ScreenConfig:
    set_speed_fts: float = 6.0
    set_dist_ft: float = 6.0
    user_dist_ft: float = 8.0
    min_set_s: float = 0.2
    approach_speed_fts: float = 2.0
    smooth_frames: int = 5
    dup_window_s: float = 1.0
    # v1.3: "moving" screens (a recall check on screen fouls: illegal screens were missed
    # because the screener never stayed below set_speed_fts): screener slower than
    # moving_speed_fts and closing on the screened defender by >= moving_approach_ft over the
    # 0.5 s before contact
    moving_speed_fts: float = 9.0
    moving_approach_ft: float = 1.0
    user_def_window_s: tuple[float, float] = (-1.0, -0.2)  # matchup window for user_def_id


class _Dense:
    """Players' smoothed positions and velocities on the frame grid of one period."""

    def __init__(self, frames: pl.DataFrame, t: np.ndarray, cfg: ScreenConfig):
        p = frames.filter(pl.col("team_id") != BALL_ID)
        self.ids = sorted(int(i) for i in p["player_id"].unique())
        self.team = dict(p.select("player_id", "team_id").unique().iter_rows())
        col = {pid: k for k, pid in enumerate(self.ids)}
        pos = {int(u): i for i, u in enumerate(t.tolist())}
        n, m = t.size, len(self.ids)
        self.xy = np.full((n, m, 2), np.nan)
        self.raw_xy = np.full((n, m, 2), np.nan)
        for pid, u, x, y in p.select("player_id", "unix_ms", "x", "y").iter_rows():
            i = pos.get(int(u))
            if i is not None:
                self.raw_xy[i, col[int(pid)]] = (x, y)
        k = cfg.smooth_frames // 2
        for j in range(m):
            for c in (0, 1):
                v = self.raw_xy[:, j, c]
                s = pl.Series(v).rolling_mean(cfg.smooth_frames, center=True, min_samples=k + 1)
                self.xy[:, j, c] = s.to_numpy()
        self.xy[np.isnan(self.raw_xy)] = np.nan
        dt = np.gradient(t.astype(float)) / 1000.0
        self.v = np.gradient(self.xy, axis=0) / dt[:, None, None]
        self.speed = np.linalg.norm(self.v, axis=-1)
        self.col = col


SCREEN_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_type": pl.Utf8,
    "screener_id": pl.Int64,
    "user_id": pl.Int64,
    "screened_def_id": pl.Int64,
    "screener_def_id": pl.Int64,
    "offense_team_id": pl.Int64,
    "handler_id": pl.Int64,  # handler at contact (-1 none / ball in flight)
    "on_ball": pl.Boolean,
    "ball_in_flight_at_contact": pl.Boolean,
    "t_set_start_ms": pl.Int64,
    "t_approach_ms": pl.Int64,
    "t_contact_ms": pl.Int64,
    "t_pass_ms": pl.Int64,
    "t_end_ms": pl.Int64,
    "min_dist_ft": pl.Float32,  # screener - screened defender at contact
    "min_dist_screener_user_ft": pl.Float32,  # in [contact - 1 s, contact + 1 s]
    "x": pl.Float32,  # screener at contact (raw court coordinates)
    "y": pl.Float32,
    "screener_speed_fts": pl.Float32,
    "user_speed_fts": pl.Float32,
    "approach_angle_deg": pl.Float32,  # user's approach vs screener -> basket axis
    "side_passed": pl.Utf8,  # left / right of the screener seen from the basket
    "screener_dx_1p5s": pl.Float32,
    "screener_dy_1p5s": pl.Float32,
    "screener_to_basket_1p5s_ft": pl.Float32,  # displacement toward the basket (roll > 0)
    "screened_def_to_user_1s_ft": pl.Float32,
    "screener_def_to_user_1s_ft": pl.Float32,
    # screener motion over three windows around the contact
    "scr_mean_speed_m05_0": pl.Float32,
    "scr_max_speed_m05_0": pl.Float32,
    "scr_disp_m05_0_ft": pl.Float32,
    "scr_disp_along_def_m05_0_ft": pl.Float32,
    "scr_mean_speed_m03_0": pl.Float32,
    "scr_max_speed_m03_0": pl.Float32,
    "scr_disp_m03_0_ft": pl.Float32,
    "scr_disp_along_def_m03_0_ft": pl.Float32,
    "scr_mean_speed_0_p03": pl.Float32,
    "scr_max_speed_0_p03": pl.Float32,
    "scr_disp_0_p03_ft": pl.Float32,
    "scr_disp_along_def_0_p03_ft": pl.Float32,
    "def_mean_speed_m05_0": pl.Float32,
    "def_approach_angle_deg": pl.Float32,  # screened defender's travel vs defender -> screener
    # S, U or D shares its exact x, y with another player at the contact (a source identity
    # merge): distances and speeds around the contact are unreliable
    "xy_shared_at_contact": pl.Boolean,
    # v1.3: set = the screener stayed below set_speed_fts for min_set_s (the v1.1 rule); moving =
    # only below moving_speed_fts and closing on the defender
    "screen_kind": pl.Utf8,
    # v1.3: U's defender = the opponent nearest to U most often in
    # user_def_window_s before the contact (the "nearest opponent at contact" screened_def_id
    # was wrong in 14 % of the reviewed real screens)
    "user_def_id": pl.Int64,
    "screened_def_is_user_def": pl.Boolean,
    "dup_group": pl.Int32,
    "screen_group_uid": pl.Utf8,
    "event_uid": pl.Utf8,
}


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > 1) + 1
    return [(int(a[0]), int(a[-1])) for a in np.split(idx, brk)]


def _angle(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if not (na > 1e-6 and nb > 1e-6) or np.isnan(na) or np.isnan(nb):
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def screen_candidates(
    frames: pl.DataFrame,
    handler: pl.DataFrame,
    direction: pl.DataFrame,
    cfg: ScreenConfig | None = None,
) -> pl.DataFrame:
    cfg = cfg or ScreenConfig()
    gid = handler["game_id"][0]
    dir_map = {
        (r["team_id"], r["period"]): r["attacks_left"] for r in direction.iter_rows(named=True)
    }
    rows: list[dict] = []
    for (period,), h in handler.sort(["period", "unix_ms"]).group_by(
        ["period"], maintain_order=True
    ):
        rows += _period(
            gid, int(period), frames.filter(pl.col("period") == period), h, dir_map, cfg
        )
    out = pl.DataFrame(rows, schema=SCREEN_SCHEMA)
    return _dup_groups(out, cfg)


def _period(gid, period, frames, h, dir_map, cfg) -> list[dict]:
    t = h["unix_ms"].to_numpy()
    seg = h["segment_id"].to_numpy()
    ctrl = h["control_team_id"].to_numpy()
    hid = h["handler_id"].to_numpy()
    free = (h["has_ball"] & (h["cand_id"] < 0)).to_numpy()
    D = _Dense(frames, t, cfg)
    ids = np.array(D.ids)
    team = np.array([D.team[i] for i in D.ids])
    n = t.size
    min_set = max(1, int(round(cfg.min_set_s * FPS)))
    rows = []
    for T in np.unique(team):
        off = np.flatnonzero(team == T)
        opp = np.flatnonzero(team != T)
        hoop = HOOP_LEFT if dir_map.get((int(T), period), True) else HOOP_RIGHT
        # nearest opponent of every offensive player per frame
        d_uo = np.linalg.norm(
            D.raw_xy[:, off, None, :] - D.raw_xy[:, None, opp, :], axis=-1
        )  # (n, o, d)
        d_uo = np.where(np.isnan(d_uo), np.inf, d_uo)
        nearest = opp[np.argmin(d_uo, axis=2)]  # (n, o) column index into ids
        has_near = np.isfinite(np.min(d_uo, axis=2))
        in_ctrl = ctrl == T
        for s, kind in [(s_, k_) for k_ in ("set", "moving") for s_ in off]:
            sid = ids[s]
            cap = cfg.set_speed_fts if kind == "set" else cfg.moving_speed_fts
            slow = (D.speed[:, s] < cap) & in_ctrl & (hid != sid)
            strict = (D.speed[:, s] < cfg.set_speed_fts) & in_ctrl & (hid != sid)
            d_sd_all = np.linalg.norm(D.raw_xy[:, [s], :] - D.raw_xy[:, opp, :], axis=-1)  # (n, d)
            for ui, u in enumerate(off):
                if u == s:
                    continue
                d_us = np.linalg.norm(D.raw_xy[:, s] - D.raw_xy[:, u], axis=-1)
                dcol = nearest[:, ui]
                d_sd = d_sd_all[np.arange(n), np.searchsorted(opp, dcol)]
                cond = (
                    slow & has_near[:, ui] & (d_sd <= cfg.set_dist_ft) & (d_us <= cfg.user_dist_ft)
                )
                cond &= ~np.isnan(d_sd) & ~np.isnan(d_us)
                for a, b in _runs(cond):
                    # split the run where the defender or the segment changes
                    cuts = (
                        np.flatnonzero(
                            (np.diff(dcol[a : b + 1]) != 0) | (np.diff(seg[a : b + 1]) != 0)
                        )
                        + 1
                    )
                    for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, b - a + 1], strict=True):
                        i0, i1 = a + lo, a + hi - 1
                        dj = int(dcol[i0])
                        k = i0 + int(np.nanargmin(d_sd[i0 : i1 + 1]))
                        # set run of S near D around the contact (user not required)
                        dcol_j = np.searchsorted(opp, dj)
                        setm = slow & (d_sd_all[:, dcol_j] <= cfg.set_dist_ft) & (seg == seg[k])
                        s0 = k
                        while s0 > 0 and setm[s0 - 1]:
                            s0 -= 1
                        s1 = k
                        while s1 + 1 < n and setm[s1 + 1]:
                            s1 += 1
                        if s1 - s0 + 1 < min_set:
                            continue
                        if kind == "moving":
                            # not a set screen (those come from the first pass) ...
                            setm_s = strict & (d_sd_all[:, dcol_j] <= cfg.set_dist_ft)
                            r0 = k
                            while r0 > 0 and setm_s[r0 - 1] and seg[r0 - 1] == seg[k]:
                                r0 -= 1
                            r1 = k
                            while r1 + 1 < n and setm_s[r1 + 1] and seg[r1 + 1] == seg[k]:
                                r1 += 1
                            if setm_s[k] and r1 - r0 + 1 >= min_set:
                                continue
                            # ... and the screener closes on the defender before the contact
                            kb = _idx_at(t, t[k] - 500)
                            closing = d_sd_all[kb, dcol_j] - d_sd_all[k, dcol_j]
                            if not (closing >= cfg.moving_approach_ft):
                                continue
                        row = _row(
                            gid,
                            period,
                            D,
                            ids,
                            t,
                            hid,
                            free,
                            seg,
                            hoop,
                            int(T),
                            s,
                            u,
                            dj,
                            k,
                            s0,
                            opp,
                            cfg,
                        )
                        row["screen_kind"] = kind
                        rows.append(row)
    return _drop_moving_duplicates(rows, cfg)


def _drop_moving_duplicates(rows: list[dict], cfg: ScreenConfig) -> list[dict]:
    """A moving candidate of the same (screener, user, screened defender) within dup_window_s
    of a set candidate is the same screen: keep the set one."""
    sets: dict[tuple, list[int]] = {}
    for r in rows:
        if r["screen_kind"] == "set":
            key = (r["screener_id"], r["user_id"], r["screened_def_id"])
            sets.setdefault(key, []).append(r["t_contact_ms"])
    win = cfg.dup_window_s * 1000
    out = []
    for r in rows:
        if r["screen_kind"] == "moving":
            ts = sets.get((r["screener_id"], r["user_id"], r["screened_def_id"]), [])
            if any(abs(r["t_contact_ms"] - x) <= win for x in ts):
                continue
        out.append(r)
    return out


def _shared(xy_k: np.ndarray, cols) -> bool:
    """Any of ``cols`` at exactly the same x, y as another player in this frame."""
    ok = ~np.isnan(xy_k).any(axis=1)
    for c in cols:
        if ok[c] and (np.all(xy_k[ok] == xy_k[c], axis=1).sum() > 1):
            return True
    return False


def _idx_at(t: np.ndarray, u: float) -> int:
    return int(np.clip(np.searchsorted(t, u), 0, t.size - 1))


def _win_motion(D, t, j, k, lo_s, hi_s, dir_vec):
    i0, i1 = _idx_at(t, t[k] + lo_s * 1000), _idx_at(t, t[k] + hi_s * 1000)
    if i1 <= i0:
        return None, None, None, None
    sp = D.speed[i0 : i1 + 1, j]
    disp = D.xy[i1, j] - D.xy[i0, j]
    along = None
    if dir_vec is not None and np.linalg.norm(dir_vec) > 1e-6 and not np.isnan(disp).any():
        along = float(disp @ dir_vec / np.linalg.norm(dir_vec))
    return (
        float(np.nanmean(sp)) if np.isfinite(sp).any() else None,
        float(np.nanmax(sp)) if np.isfinite(sp).any() else None,
        float(np.linalg.norm(disp)) if not np.isnan(disp).any() else None,
        along,
    )


def _row(gid, period, D, ids, t, hid, free, seg, hoop, T, s, u, dj, k, s0, opp, cfg) -> dict:
    sid, uid, did = int(ids[s]), int(ids[u]), int(ids[dj])
    # screener's own defender: nearest opponent to S at contact other than D
    dd = np.linalg.norm(D.raw_xy[k, opp] - D.raw_xy[k, s], axis=-1)
    dd = np.where((opp == dj) | np.isnan(dd), np.inf, dd)
    sdef = int(ids[opp[np.argmin(dd)]]) if np.isfinite(dd).any() else None
    ps = D.raw_xy[k, s]
    # approach: last time before contact the screener's speed toward the contact point > 2 ft/s
    ta = None
    for i in range(k - 1, max(-1, k - 2 * FPS), -1):
        vec = ps - D.xy[i, s]
        nv = np.linalg.norm(vec)
        if nv > 1e-6 and not np.isnan(nv) and (D.v[i, s] @ vec) / nv > cfg.approach_speed_fts:
            ta = int(t[i])
            break
    if ta is None:
        ta = int(t[k]) - 2000
    # user's approach direction over the 0.5 s before contact
    i_b = _idx_at(t, t[k] - 500)
    appr = D.xy[k, u] - D.xy[i_b, u]
    axis = hoop - ps  # screener -> basket
    # pass: first time in [contact - 1.5 s, contact + 3 s] U crosses the line through S
    # perpendicular to the approach direction. The search starts before the contact because the
    # defender trails U: U usually passes the screener before the defender hits it.
    tp = None
    if np.linalg.norm(appr) > 1e-6 and not np.isnan(appr).any():
        i0 = _idx_at(t, t[k] - 1500)
        while i0 < k and (seg[i0] != seg[k] or np.isnan(D.raw_xy[i0, u]).any()):
            i0 += 1
        sign0 = np.sign((D.raw_xy[i0, u] - ps) @ appr)
        for i in range(i0 + 1, min(t.size, k + 3 * FPS)):
            if seg[i] != seg[k] or np.isnan(D.raw_xy[i, u]).any():
                break
            if sign0 != 0 and np.sign((D.raw_xy[i, u] - ps) @ appr) != sign0:
                tp = int(t[i])
                break
    t_end = int(min(t[k] + 3000, max(t[k] + 1500, (tp or 0) + 1000)))
    w = (t >= t[k] - 1000) & (t <= t[k] + 1000)
    d_su = np.linalg.norm(D.raw_xy[w, s] - D.raw_xy[w, u], axis=-1)
    i15 = _idx_at(t, t[k] + 1500)
    disp = D.xy[i15, s] - D.xy[k, s]
    i1 = _idx_at(t, t[k] + 1000)
    # side passed: sign of the cross product basket->screener x screener->user at the pass
    side = None
    if tp is not None:
        ip = _idx_at(t, tp)
        va, vb = ps - hoop, D.raw_xy[ip, u] - ps
        cr = va[0] * vb[1] - va[1] * vb[0]
        side = "left" if cr > 0 else "right"
    # U's defender over the matchup window before the contact
    iu0 = _idx_at(t, t[k] + cfg.user_def_window_s[0] * 1000)
    iu1 = _idx_at(t, t[k] + cfg.user_def_window_s[1] * 1000)
    near = []
    for i in range(iu0, max(iu0, iu1) + 1):
        du = np.linalg.norm(D.raw_xy[i, opp] - D.raw_xy[i, u], axis=-1)
        if np.isfinite(du).any():
            near.append(int(opp[int(np.nanargmin(du))]))
    udef = int(ids[max(set(near), key=near.count)]) if near else None
    dvec = D.xy[k, dj] - D.xy[_idx_at(t, t[k] - 500), dj]  # screened defender's travel
    m05 = _win_motion(D, t, s, k, -0.5, 0.0, dvec)
    m03 = _win_motion(D, t, s, k, -0.3, 0.0, dvec)
    p03 = _win_motion(D, t, s, k, 0.0, 0.3, dvec)
    dm = _win_motion(D, t, dj, k, -0.5, 0.0, None)

    def f(v):
        return None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)

    h_at = int(hid[k])
    return {
        "game_id": gid,
        "period": period,
        "event_type": "screen_candidate",
        "screener_id": sid,
        "user_id": uid,
        "screened_def_id": did,
        "user_def_id": udef,
        "screened_def_is_user_def": None if udef is None else udef == did,
        "screener_def_id": sdef,
        "offense_team_id": T,
        "handler_id": h_at,
        "on_ball": h_at == uid,
        "ball_in_flight_at_contact": bool(free[k]),
        "t_set_start_ms": int(t[s0]),
        "t_approach_ms": min(ta, int(t[k])),
        "t_contact_ms": int(t[k]),
        "t_pass_ms": tp,
        "t_end_ms": t_end,
        "min_dist_ft": f(np.linalg.norm(D.raw_xy[k, s] - D.raw_xy[k, dj])),
        "min_dist_screener_user_ft": f(np.nanmin(d_su)) if np.isfinite(d_su).any() else None,
        "x": f(ps[0]),
        "y": f(ps[1]),
        "screener_speed_fts": f(D.speed[k, s]),
        "user_speed_fts": f(D.speed[k, u]),
        "approach_angle_deg": _angle(appr, axis),
        "side_passed": side,
        "screener_dx_1p5s": f(disp[0]),
        "screener_dy_1p5s": f(disp[1]),
        "screener_to_basket_1p5s_ft": f(disp @ axis / np.linalg.norm(axis))
        if not np.isnan(disp).any()
        else None,
        "screened_def_to_user_1s_ft": f(np.linalg.norm(D.raw_xy[i1, dj] - D.raw_xy[i1, u])),
        "screener_def_to_user_1s_ft": (
            f(np.linalg.norm(D.raw_xy[i1, D.col[sdef]] - D.raw_xy[i1, u]))
            if sdef is not None
            else None
        ),
        "scr_mean_speed_m05_0": f(m05[0]),
        "scr_max_speed_m05_0": f(m05[1]),
        "scr_disp_m05_0_ft": f(m05[2]),
        "scr_disp_along_def_m05_0_ft": f(m05[3]),
        "scr_mean_speed_m03_0": f(m03[0]),
        "scr_max_speed_m03_0": f(m03[1]),
        "scr_disp_m03_0_ft": f(m03[2]),
        "scr_disp_along_def_m03_0_ft": f(m03[3]),
        "scr_mean_speed_0_p03": f(p03[0]),
        "scr_max_speed_0_p03": f(p03[1]),
        "scr_disp_0_p03_ft": f(p03[2]),
        "scr_disp_along_def_0_p03_ft": f(p03[3]),
        "def_mean_speed_m05_0": f(dm[0]),
        "def_approach_angle_deg": _angle(dvec, ps - D.xy[_idx_at(t, t[k] - 500), dj]),
        "xy_shared_at_contact": _shared(D.raw_xy[k], (s, u, dj)),
        "dup_group": None,
        "screen_group_uid": None,
        "event_uid": f"{gid}:screen_candidate:{int(t[k])}:{sid}",
    }


def _dup_groups(out: pl.DataFrame, cfg: ScreenConfig) -> pl.DataFrame:
    if out.height == 0:
        return out
    o = out.sort(["period", "screener_id", "screened_def_id", "t_contact_ms"]).with_row_index("_i")
    grp = np.zeros(o.height, dtype=np.int32)
    g = -1
    prev = None
    for i, (p, s, d, tc) in enumerate(
        o.select("period", "screener_id", "screened_def_id", "t_contact_ms").iter_rows()
    ):
        if prev is None or (p, s, d) != prev[:3] or tc - prev[3] > cfg.dup_window_s * 1000:
            g += 1
            prev = (p, s, d, tc)
        grp[i] = g
    o = o.with_columns(pl.Series("dup_group", grp)).drop("_i")
    o = o.sort(["period", "screener_id", "t_contact_ms"])
    sg = []
    prev = None
    for gid, p, s, tc in o.select("game_id", "period", "screener_id", "t_contact_ms").iter_rows():
        if prev is None or (p, s) != prev[:2] or tc - prev[2] > cfg.dup_window_s * 1000:
            prev = (p, s, tc)
        sg.append(f"{gid}:screen_group:{prev[2]}:{s}")
    o = o.with_columns(pl.Series("screen_group_uid", sg, dtype=pl.Utf8))
    # uid actor = screener-user: several users of one physical screen share screener and contact
    return o.with_columns(
        pl.format(
            "{}:screen_candidate:{}:{}-{}", "game_id", "t_contact_ms", "screener_id", "user_id"
        ).alias("event_uid")
    ).sort(["period", "t_contact_ms"])
