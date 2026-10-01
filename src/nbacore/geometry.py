"""Distances between players and to the ball (L2 functions).

Nothing here is stored per frame; call the functions on the window you need.

* ``shared_xy(frames)``: players whose (x, y) equals another player's in the same frame (a
  source identity merge). Distances involving them are **null**, not 0.
* ``pair_distances(frames, t0, t1)``: 25 Hz long table, one row per frame and pair of players of
  opposite teams (``id_a`` in the lower team id, ``id_b`` in the other), ``dist_ft``.
* ``ball_distances(frames, t0, t1)``: one row per frame and player, ``ball_dist_ft`` (xy).
* ``player_speeds(frames, t0, t1, smooth_frames)``: central differences on unix time after a
  centred moving average of ``smooth_frames`` frames (ft/s); null across tracking holes.
* ``grid_distances(arrays)``: on a resampled possession (``nbacore.grid``): offence × defence
  (T, 5, 5), ball to the 10 players (T, 10), ball speed (T,).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nbacore.clean.dedup import FRAME_GAP_MS
from nbacore.io.raw import BALL_ID


def _window(frames: pl.DataFrame, t0: int | None, t1: int | None) -> pl.DataFrame:
    f = frames
    if t0 is not None:
        f = f.filter(pl.col("unix_ms") >= t0)
    if t1 is not None:
        f = f.filter(pl.col("unix_ms") <= t1)
    return f


def shared_xy(frames: pl.DataFrame) -> pl.DataFrame:
    """(period, unix_ms, player_id) of players sharing their exact x, y with another player."""
    p = frames.filter(pl.col("team_id") != BALL_ID)
    return (
        p.with_columns(pl.len().over(["period", "unix_ms", "x", "y"]).alias("_n"))
        .filter(pl.col("_n") > 1)
        .select("period", "unix_ms", "player_id", pl.lit(True).alias("xy_shared"))
    )


def _players(frames: pl.DataFrame) -> pl.DataFrame:
    p = frames.filter(pl.col("team_id") != BALL_ID).select(
        "period", "unix_ms", "team_id", "player_id", "x", "y"
    )
    return p.join(
        shared_xy(frames), on=["period", "unix_ms", "player_id"], how="left"
    ).with_columns(pl.col("xy_shared").fill_null(False))


def pair_distances(
    frames: pl.DataFrame, t0: int | None = None, t1: int | None = None
) -> pl.DataFrame:
    p = _players(_window(frames, t0, t1))
    teams = sorted(int(t) for t in p["team_id"].unique())
    if len(teams) != 2:
        return pl.DataFrame()
    a = p.filter(pl.col("team_id") == teams[0])
    b = p.filter(pl.col("team_id") == teams[1])
    j = a.join(b, on=["period", "unix_ms"], suffix="_b")
    return j.select(
        "period",
        "unix_ms",
        pl.col("player_id").alias("id_a"),
        pl.col("player_id_b").alias("id_b"),
        pl.col("team_id").alias("team_a"),
        pl.col("team_id_b").alias("team_b"),
        pl.when(pl.col("xy_shared") | pl.col("xy_shared_b"))
        .then(None)
        .otherwise(((pl.col("x") - pl.col("x_b")) ** 2 + (pl.col("y") - pl.col("y_b")) ** 2).sqrt())
        .alias("dist_ft"),
        (pl.col("xy_shared") | pl.col("xy_shared_b")).alias("xy_shared"),
    ).sort(["period", "unix_ms", "id_a", "id_b"])


def ball_distances(
    frames: pl.DataFrame, t0: int | None = None, t1: int | None = None
) -> pl.DataFrame:
    f = _window(frames, t0, t1)
    ball = f.filter(pl.col("team_id") == BALL_ID).select(
        "period", "unix_ms", pl.col("x").alias("bx"), pl.col("y").alias("by")
    )
    return (
        _players(f)
        .join(ball, on=["period", "unix_ms"], how="inner")
        .select(
            "period",
            "unix_ms",
            "team_id",
            "player_id",
            pl.when(pl.col("xy_shared"))
            .then(None)
            .otherwise(
                ((pl.col("x") - pl.col("bx")) ** 2 + (pl.col("y") - pl.col("by")) ** 2).sqrt()
            )
            .alias("ball_dist_ft"),
            "xy_shared",
        )
        .sort(["period", "unix_ms", "player_id"])
    )


def player_speeds(
    frames: pl.DataFrame, t0: int | None = None, t1: int | None = None, smooth_frames: int = 5
) -> pl.DataFrame:
    """Speed (ft/s) per frame and player: centred moving average over ``smooth_frames`` frames,
    then central differences on unix time; null where a neighbour frame is across a hole."""
    p = (
        _window(frames, t0, t1)
        .filter(pl.col("team_id") != BALL_ID)
        .sort(["player_id", "period", "unix_ms"])
    )
    over = ["player_id", "period"]
    sm = p.with_columns(
        pl.col("x").rolling_mean(smooth_frames, center=True, min_samples=1).over(over).alias("xs"),
        pl.col("y").rolling_mean(smooth_frames, center=True, min_samples=1).over(over).alias("ys"),
    )
    nxt, prv = (lambda c: pl.col(c).shift(-1).over(over)), (lambda c: pl.col(c).shift(1).over(over))
    dt = (nxt("unix_ms") - prv("unix_ms")) / 1000
    ok = ((nxt("unix_ms") - pl.col("unix_ms")) <= FRAME_GAP_MS) & (
        (pl.col("unix_ms") - prv("unix_ms")) <= FRAME_GAP_MS
    )
    v = (((nxt("xs") - prv("xs")) ** 2 + (nxt("ys") - prv("ys")) ** 2).sqrt()) / dt
    return sm.select(
        "period", "unix_ms", "player_id", pl.when(ok).then(v).otherwise(None).alias("speed_fts")
    )


def grid_distances(arrays) -> dict[str, np.ndarray]:
    """``arrays``: a resampled possession (``nbacore.grid.GridArrays`` / ``PossessionArrays``)."""
    off, dfn, ball = arrays.off_xy, arrays.def_xy, arrays.ball
    off_def = np.linalg.norm(off[:, :, None, :] - dfn[:, None, :, :], axis=-1)  # (T, 5, 5)
    players = np.concatenate([off, dfn], axis=1)  # (T, 10, 2)
    ball_dist = np.linalg.norm(players - ball[:, None, :2], axis=-1)  # (T, 10)
    return {
        "off_def_dist": off_def,
        "ball_dist": ball_dist,
        "ball_speed": np.linalg.norm(arrays.ball_v, axis=-1),
    }


EVENT_FRAME_SCHEMA: dict[str, pl.DataType] = {
    "game_id": pl.Utf8,
    "period": pl.Int8,
    "event_uid": pl.Utf8,
    "event_type": pl.Utf8,
    "frame_role": pl.Utf8,  # release | catch
    "unix_ms": pl.Int64,
    "offense_team_id": pl.Int64,
    "off_ids": pl.List(pl.Int64),  # sorted player ids present in the frame (up to 5)
    "def_ids": pl.List(pl.Int64),
    "off_def_dist_ft": pl.List(pl.Float32),  # row-major len(off_ids) x len(def_ids); null on merges
    "ball_dist_off_ft": pl.List(pl.Float32),
    "ball_dist_def_ft": pl.List(pl.Float32),
    "xy_shared": pl.Boolean,  # an identity merge in this frame
}


def event_frame_distances(frames: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    """Distance snapshots at key event frames: shot releases, pass / handoff releases
    and catches. Offence = the team of the shooter / passer."""
    keys = []
    for r in events.filter(
        pl.col("event_type").is_in(["shot_release", "pass", "handoff"])
    ).iter_rows(named=True):
        if r["team_id"] is None:
            continue
        keys.append(
            (r["period"], r["t_start_ms"], r["event_uid"], r["event_type"], "release", r["team_id"])
        )
        if r["event_type"] != "shot_release" and r["t_end_ms"] is not None:
            keys.append(
                (r["period"], r["t_end_ms"], r["event_uid"], r["event_type"], "catch", r["team_id"])
            )
    if not keys:
        return pl.DataFrame(schema=EVENT_FRAME_SCHEMA)
    k = pl.DataFrame(
        keys,
        schema={
            "period": pl.Int8,
            "unix_ms": pl.Int64,
            "event_uid": pl.Utf8,
            "event_type": pl.Utf8,
            "frame_role": pl.Utf8,
            "offense_team_id": pl.Int64,
        },  # fmt: skip
        orient="row",
    )
    f = frames.join(k.select("period", "unix_ms").unique(), on=["period", "unix_ms"], how="inner")
    shared = (
        shared_xy(f)
        .select("period", "unix_ms")
        .unique()
        .with_columns(pl.lit(True).alias("xy_shared"))
    )
    by_frame = {(p, u): g for (p, u), g in f.group_by(["period", "unix_ms"])}
    rows = []
    gid = frames["game_id"][0]
    shared_set = {(p, u) for p, u, _ in shared.iter_rows()}
    for p, u, uid, et, role, off in k.iter_rows():
        g = by_frame.get((p, u))
        if g is None:
            continue
        pl_ = g.filter(pl.col("team_id") != BALL_ID)
        ball = g.filter(pl.col("team_id") == BALL_ID)
        o = pl_.filter(pl.col("team_id") == off).sort("player_id")
        d = pl_.filter(pl.col("team_id") != off).sort("player_id")
        oxy = o.select("x", "y").to_numpy().astype(float)
        dxy = d.select("x", "y").to_numpy().astype(float)
        merged_ids = set()
        if (p, u) in shared_set:
            xy = pl_.with_columns(pl.len().over(["x", "y"]).alias("_n")).filter(pl.col("_n") > 1)
            merged_ids = set(xy["player_id"].to_list())
        dm = (
            np.linalg.norm(oxy[:, None, :] - dxy[None, :, :], axis=-1)
            if len(oxy) and len(dxy)
            else np.zeros((0, 0))
        )
        oid, did = o["player_id"].to_list(), d["player_id"].to_list()
        flat = [
            None if (oid[i] in merged_ids or did[j] in merged_ids) else float(dm[i, j])
            for i in range(len(oid))
            for j in range(len(did))
        ]
        if ball.height:
            bxy = ball.select("x", "y").to_numpy()[0].astype(float)
            bo = [
                None if pid in merged_ids else float(np.hypot(*(oxy[i] - bxy)))
                for i, pid in enumerate(oid)
            ]
            bd = [
                None if pid in merged_ids else float(np.hypot(*(dxy[i] - bxy)))
                for i, pid in enumerate(did)
            ]
        else:
            bo, bd = [None] * len(oid), [None] * len(did)
        rows.append(
            {
                "game_id": gid,
                "period": p,
                "event_uid": uid,
                "event_type": et,
                "frame_role": role,
                "unix_ms": u,
                "offense_team_id": off,
                "off_ids": oid,
                "def_ids": did,
                "off_def_dist_ft": flat,
                "ball_dist_off_ft": bo,
                "ball_dist_def_ft": bd,
                "xy_shared": (p, u) in shared_set,
            }  # fmt: skip
        )
    return pl.DataFrame(rows, schema=EVENT_FRAME_SCHEMA)
