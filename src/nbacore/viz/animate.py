"""Animate tracking frames (players + ball + court) to mp4/gif via matplotlib.

Input is the *long* frame table from ``nbacore.io`` restricted to one contiguous stretch
(e.g. one event). Frames are deduplicated by (period, unix_ms) and sorted by time here so
the caller does not have to.
"""

from __future__ import annotations

from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib import animation  # noqa: E402

from nbacore.io.raw import BALL_ID  # noqa: E402
from nbacore.viz.court import draw_court  # noqa: E402

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

HOME_COLOR = "#1f77b4"
VISITOR_COLOR = "#d62728"
BALL_COLOR = "#ff7f0e"


def _fmt_clock(sec: float) -> str:
    sec = max(sec, 0.0)
    return f"{int(sec // 60):02d}:{sec % 60:04.1f}"


def frames_to_arrays(frames: pl.DataFrame) -> dict:
    """Dedup + sort a long frame table and pivot to dense arrays.

    Returns dict with keys: unix_ms (T,), period (T,), game_clock (T,), shot_clock (T,),
    ball (T, 3) with NaN when absent, player_ids (P,), team_ids (P,), xy (T, P, 2) NaN when absent.
    """
    f = frames.unique(subset=["period", "unix_ms", "team_id", "player_id"], keep="first").sort(
        ["period", "unix_ms"]
    )
    keys = (
        f.select(["period", "unix_ms", "game_clock", "shot_clock"])
        .unique(subset=["period", "unix_ms"], keep="first")
        .sort(["period", "unix_ms"])
    )
    T = keys.height
    key_index = {
        (p, u): i
        for i, (p, u) in enumerate(
            zip(keys["period"].to_list(), keys["unix_ms"].to_list(), strict=True)
        )
    }
    players = (
        f.filter(pl.col("team_id") != BALL_ID)
        .select(["team_id", "player_id"])
        .unique()
        .sort(["team_id", "player_id"])
    )
    pid_index = {pid: i for i, pid in enumerate(players["player_id"].to_list())}
    P = players.height
    xy = np.full((T, P, 2), np.nan, dtype=np.float32)
    ball = np.full((T, 3), np.nan, dtype=np.float32)
    per = f["period"].to_numpy()
    ux = f["unix_ms"].to_numpy()
    tid = f["team_id"].to_numpy()
    pid = f["player_id"].to_numpy()
    x = f["x"].to_numpy()
    y = f["y"].to_numpy()
    z = f["z"].to_numpy()
    for i in range(f.height):
        t = key_index[(per[i], ux[i])]
        if tid[i] == BALL_ID:
            ball[t] = (x[i], y[i], z[i])
        else:
            xy[t, pid_index[pid[i]]] = (x[i], y[i])
    return {
        "unix_ms": keys["unix_ms"].to_numpy(),
        "period": keys["period"].to_numpy(),
        "game_clock": keys["game_clock"].to_numpy(),
        "shot_clock": keys["shot_clock"].to_numpy(),
        "ball": ball,
        "player_ids": players["player_id"].to_numpy(),
        "team_ids": players["team_id"].to_numpy(),
        "xy": xy,
    }


def animate_frames(
    frames: pl.DataFrame,
    out_path: str | Path,
    roster: pl.DataFrame | None = None,
    home_team_id: int | None = None,
    fps: int = 25,
    stride: int = 1,
    title: str = "",
    dpi: int = 100,
    trail: int = 0,
    annotations: list[dict] | None = None,
    notes: str = "",
) -> Path:
    """Render ``frames`` to ``out_path`` (.mp4 via ffmpeg, or .gif via pillow).

    ``annotations`` (optional): dicts ``{"t_from_ms", "t_to_ms", "labels": {player_id: str}}``;
    while a frame's unix time is inside [t_from_ms, t_to_ms] the label is drawn above that
    player. ``notes``: fixed text in the upper-left corner. Without them the output is unchanged.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = frames_to_arrays(frames)
    idx = np.arange(0, len(arr["unix_ms"]), stride)
    jersey = {}
    abbr = {}
    if roster is not None:
        for r in roster.iter_rows(named=True):
            jersey[r["player_id"]] = r["jersey"]
            abbr[r["team_id"]] = r["abbreviation"]
    if home_team_id is None:
        home_team_id = int(arr["team_ids"][0])
    colors = [HOME_COLOR if t == home_team_id else VISITOR_COLOR for t in arr["team_ids"]]

    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=dpi)
    draw_court(ax, color="0.3")
    scat = ax.scatter(
        arr["xy"][0, :, 0], arr["xy"][0, :, 1], s=260, c=colors, edgecolors="white", zorder=3
    )
    ball_sc = ax.scatter([arr["ball"][0, 0]], [arr["ball"][0, 1]], s=60, c=BALL_COLOR, zorder=4)
    labels = [
        ax.text(
            0,
            0,
            jersey.get(int(p), ""),
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            zorder=5,
        )
        for p in arr["player_ids"]
    ]
    pid_index = {int(p): j for j, p in enumerate(arr["player_ids"])}
    ann_txt = [
        ax.text(
            0,
            0,
            "",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="black",
            zorder=6,
        )
        for _ in arr["player_ids"]
    ]
    if notes:
        ax.text(0.5, 51.5, notes, ha="left", va="top", fontsize=6, family="monospace", zorder=6)
    trails = None
    if trail > 0:
        trails = [ax.plot([], [], color=c, lw=1, alpha=0.4, zorder=2)[0] for c in colors]
    clock_txt = ax.text(47.0, -1.0, "", ha="center", va="top", fontsize=11)
    ax.set_ylim(-5, 52)
    ttl = title
    if roster is not None and home_team_id is not None:
        teams = sorted(abbr.items(), key=lambda kv: kv[0] != home_team_id)
        if len(teams) == 2:
            ttl = f"{title}  {teams[1][1]} (red) at {teams[0][1]} (blue)"
    ax.set_title(ttl, fontsize=10)

    def update(k: int):
        t = idx[k]
        pts = arr["xy"][t]
        scat.set_offsets(pts)
        bx, by, bz = arr["ball"][t]
        ball_sc.set_offsets([[bx, by]])
        ball_sc.set_sizes([60 + 12 * (0 if np.isnan(bz) else bz)])
        for lab, (px, py) in zip(labels, pts, strict=True):
            lab.set_position((px, py))
        if annotations:
            now = int(arr["unix_ms"][t])
            active: dict[int, list[str]] = {}
            for a in annotations:
                if a["t_from_ms"] <= now <= a["t_to_ms"]:
                    for pid, lab in a["labels"].items():
                        active.setdefault(pid_index.get(int(pid), -1), []).append(lab)
            for j, txt in enumerate(ann_txt):
                labs = active.get(j)
                if labs and not np.isnan(pts[j]).any():
                    txt.set_position((pts[j, 0], pts[j, 1] + 1.3))
                    txt.set_text(",".join(labs))
                else:
                    txt.set_text("")
        if trails is not None:
            t0 = max(0, t - trail)
            for line, p in zip(trails, range(pts.shape[0]), strict=True):
                line.set_data(arr["xy"][t0 : t + 1, p, 0], arr["xy"][t0 : t + 1, p, 1])
        sc = arr["shot_clock"][t]
        sc_s = "--" if sc is None or np.isnan(sc) else f"{sc:4.1f}"
        clock_txt.set_text(
            f"Q{arr['period'][t]}  {_fmt_clock(float(arr['game_clock'][t]))}   shot {sc_s}"
        )
        return (scat, ball_sc, clock_txt, *labels)

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    if out_path.suffix.lower() == ".gif":
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=1800))
    plt.close(fig)
    return out_path
