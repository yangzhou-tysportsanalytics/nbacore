"""L5: full-season 2015-16 shot locations (docs/external_sources.md).

Source: DomSamangy/NBA_Shots_04_25 ``NBA_2016_Shots.csv.zip`` (NBA.com data; provenance in
``<data root>/raw/external/shots_domsamangy/provenance.json``). Coordinates in the source are
offence-relative feet: ``LOC_X`` lateral (−25 … 25, hoop at 0), ``LOC_Y`` from the attacking
baseline (hoop at 5.25).

Output ``<data root>/scratch/l5/shots_2015_16.parquet`` (L5 build, released with a later version): one row per FG attempt of the regular season
(1,230 games) — ``game_id, period, clock_s, player_id, team_id, made, shot_value, action_type,
zone_basic, zone_name, zone_range, shot_distance_ft, lat_ft, depth_ft`` (offence-relative), the
linked ``pbp_event_num`` (same game, period, clock second and shooter; several attempts of one
shooter in one second are paired in order), and for tracked games the SportVU court position
``x_court_ft, y_court_ft`` (attack direction of the shooter's team in that period).

Checks (``reports/l5_shots_check.json``): FGA / FGM per game vs pbp for all games, link rate,
and for tracked games the distance between the official location and the tracked release
location (``shot_release`` ``x_release, y_release``), which also fixes the sign of the lateral
axis.

Usage:
    uv run python scripts/build_external_shots.py
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

import nbacore.load as L
from nbacore import paths

REPO = Path(__file__).resolve().parents[1]
SRC = paths.raw_dir() / "external" / "shots_domsamangy" / "NBA_2016_Shots.csv.zip"
HOOP_DEPTH = 5.25
COURT_LEN, COURT_W = 94.0, 50.0


def load_source() -> pl.DataFrame:
    z = zipfile.ZipFile(SRC)
    df = pl.read_csv(io.BytesIO(z.read("NBA_2016_Shots.csv")), infer_schema_length=20000)
    return df.select(
        pl.col("GAME_ID").cast(pl.Utf8).str.zfill(10).alias("game_id"),
        pl.col("QUARTER").cast(pl.Int8).alias("period"),
        (pl.col("MINS_LEFT") * 60 + pl.col("SECS_LEFT")).cast(pl.Float32).alias("clock_s"),
        pl.col("PLAYER_ID").cast(pl.Int64).alias("player_id"),
        pl.col("TEAM_ID").cast(pl.Int64).alias("team_id"),
        pl.col("SHOT_MADE").alias("made"),
        pl.when(pl.col("SHOT_TYPE").str.starts_with("3"))
        .then(3)
        .otherwise(2)
        .cast(pl.Int8)
        .alias("shot_value"),
        pl.col("ACTION_TYPE").alias("action_type"),
        pl.col("BASIC_ZONE").alias("zone_basic"),
        pl.col("ZONE_NAME").alias("zone_name"),
        pl.col("ZONE_RANGE").alias("zone_range"),
        pl.col("SHOT_DISTANCE").cast(pl.Float32).alias("shot_distance_ft"),
        pl.col("LOC_X").cast(pl.Float32).alias("lat_ft"),
        pl.col("LOC_Y").cast(pl.Float32).alias("depth_ft"),
    )


def link_pbp(shots: pl.DataFrame, pbp: pl.DataFrame) -> pl.DataFrame:
    """Pair shots with pbp FG rows on (game, period, clock second, shooter), in order."""
    fg = pbp.filter(pl.col("msg_type").is_in([1, 2])).select(
        "game_id",
        pl.col("period").cast(pl.Int8),
        pl.col("pctime_sec").cast(pl.Float32).alias("clock_s"),
        pl.col("p1_id").cast(pl.Int64).alias("player_id"),
        pl.col("event_num").alias("pbp_event_num"),
        (pl.col("msg_type") == 1).alias("pbp_made"),
    )
    key = ["game_id", "period", "clock_s", "player_id"]
    s = shots.with_columns(pl.int_range(pl.len()).over(key).alias("_k"))
    f = fg.sort("pbp_event_num").with_columns(pl.int_range(pl.len()).over(key).alias("_k"))
    out = s.join(f, on=[*key, "_k"], how="left")
    # second pass: unmatched shots against unused pbp rows one second apart
    used = set(out["pbp_event_num"].drop_nulls().to_list())
    miss = out.filter(pl.col("pbp_event_num").is_null())
    if miss.height:
        rest = fg.filter(
            ~pl.struct("game_id", "pbp_event_num").is_in(
                out.filter(pl.col("pbp_event_num").is_not_null())
                .select("game_id", "pbp_event_num")
                .to_struct()
                .implode()
            )
        )
        cand = miss.drop("pbp_event_num", "pbp_made").join(
            rest.rename({"clock_s": "pbp_clock"}), on=["game_id", "period", "player_id"]
        )
        cand = cand.filter((pl.col("pbp_clock") - pl.col("clock_s")).abs() <= 1.0).sort(
            (pl.col("pbp_clock") - pl.col("clock_s")).abs()
        )
        picked, taken = {}, set()
        for r in cand.iter_rows(named=True):
            k = (r["game_id"], r["period"], r["clock_s"], r["player_id"], r["_k"])
            u = (r["game_id"], r["pbp_event_num"])
            if k in picked or u in taken:
                continue
            picked[k], taken = (r["pbp_event_num"], r["pbp_made"]), taken | {u}
        del used
        fill = pl.DataFrame(
            [(*k, v[0], v[1]) for k, v in picked.items()],
            schema={
                "game_id": pl.Utf8,
                "period": pl.Int8,
                "clock_s": pl.Float32,
                "player_id": pl.Int64,
                "_k": pl.Int64,
                "_ev": pl.Int32,
                "_pm": pl.Boolean,
            },
            orient="row",
        )
        out = out.join(fill, on=[*key, "_k"], how="left").with_columns(
            pl.coalesce("pbp_event_num", "_ev").alias("pbp_event_num"),
            pl.coalesce("pbp_made", "_pm").alias("pbp_made"),
        )
        out = out.drop("_ev", "_pm")
    return out.drop("_k")


def court_xy(shots: pl.DataFrame, direction: pl.DataFrame, lat_sign: float) -> pl.DataFrame:
    """SportVU court position for tracked games (x along the length, y across)."""
    d = direction.select(
        "game_id", pl.col("team_id").cast(pl.Int64), pl.col("period").cast(pl.Int8), "attacks_left"
    )
    s = shots.join(d, on=["game_id", "team_id", "period"], how="left")
    left = pl.col("attacks_left")
    return s.with_columns(
        pl.when(left.is_null())
        .then(None)
        .when(left)
        .then(pl.col("depth_ft"))
        .otherwise(COURT_LEN - pl.col("depth_ft"))
        .alias("x_court_ft"),
        pl.when(left.is_null())
        .then(None)
        .when(left)
        .then(COURT_W / 2 + lat_sign * pl.col("lat_ft"))
        .otherwise(COURT_W / 2 - lat_sign * pl.col("lat_ft"))
        .alias("y_court_ft"),
    ).drop("attacks_left")


def main() -> None:
    shots = load_source()
    pbp = L.pbp("v1.2")
    shots = link_pbp(shots, pbp)
    stats: dict = {"rows": shots.height, "games": shots["game_id"].n_unique()}
    stats["linked"] = int(shots["pbp_event_num"].is_not_null().sum())
    stats["linked_frac"] = stats["linked"] / shots.height
    stats["made_agrees_with_pbp"] = float(
        (
            shots.filter(pl.col("pbp_made").is_not_null())["made"]
            == shots.filter(pl.col("pbp_made").is_not_null())["pbp_made"]
        ).mean()
    )
    fg = (
        pbp.filter(pl.col("msg_type").is_in([1, 2]))
        .group_by("game_id")
        .agg(pl.len().alias("pbp_fga"), (pl.col("msg_type") == 1).sum().alias("pbp_fgm"))
    )
    per = (
        shots.group_by("game_id")
        .agg(pl.len().alias("fga"), pl.col("made").sum().alias("fgm"))
        .join(fg, on="game_id", how="full", coalesce=True)
        .fill_null(0)
    )
    stats["games_fga_equal"] = int((per["fga"] == per["pbp_fga"]).sum())
    stats["games_fgm_equal"] = int((per["fgm"] == per["pbp_fgm"]).sum())
    stats["games_in_pbp"] = per.height
    stats["fga_total"] = [int(per["fga"].sum()), int(per["pbp_fga"].sum())]
    stats["worst_games_fga_diff"] = (
        per.with_columns(
            (pl.col("fga").cast(pl.Int64) - pl.col("pbp_fga").cast(pl.Int64)).alias("d")
        )
        .sort(pl.col("d").abs(), descending=True)
        .head(5)
        .to_dicts()
    )

    # tracked games: official vs tracked release location; choose the lateral sign
    ok = L.games("v1.2").filter(pl.col("status") == "ok")["game_id"].to_list()
    rel = []
    for g in ok:
        e = L.events(g, "shot_release", "v1.2").select(
            pl.lit(g).alias("game_id"),
            pl.col("pbp_event_num").cast(pl.Int32),
            "x_release",
            "y_release",
        )
        rel.append(e)
    rel = pl.concat(rel).drop_nulls()
    direction = L.attack_direction("v1.2")
    best = None
    for sign in (1.0, -1.0):
        c = court_xy(shots.filter(pl.col("game_id").is_in(ok)), direction, sign).join(
            rel, on=["game_id", "pbp_event_num"]
        )
        dist = (
            (c["x_court_ft"] - c["x_release"]) ** 2 + (c["y_court_ft"] - c["y_release"]) ** 2
        ).sqrt()
        med = float(dist.median())
        if best is None or med < best[1]:
            best = (sign, med, dist)
    sign, med, dist = best
    stats["lateral_sign"] = sign
    stats["tracked_compared"] = int(dist.len())
    stats["official_vs_tracked_release_ft"] = {
        str(q): float(dist.quantile(q)) for q in (0.5, 0.75, 0.9, 0.95)
    }
    stats["within_3ft"] = float((dist <= 3).mean())

    out = court_xy(shots, direction, sign)
    dest = paths.scratch_dir("l5") / "shots_2015_16.parquet"  # L5 build; released later
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.drop("pbp_made").sort(
        "game_id", "period", pl.col("clock_s"), descending=[False, False, True]
    ).write_parquet(dest)
    stats["output"] = str(dest)
    (REPO / "reports" / "l5_shots_check.json").write_text(
        json.dumps(stats, indent=1, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in stats.items() if k != "worst_games_fga_diff"}, indent=1, default=str
        )
    )
    print(np.round(np.array([r["d"] for r in stats["worst_games_fga_diff"]]), 1))


if __name__ == "__main__":
    main()
