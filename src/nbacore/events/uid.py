"""Stable event keys across releases.

``event_uid = "<game_id>:<event_type>:<t_key_ms>:<actor>"``. ``match_events`` maps the uids of
two releases: same game, type and actor, key times within ``tol_ms``; each old event is matched
to the nearest unused new event. Unmatched events appear with a null partner (to be listed in
the CHANGELOG when a definition changes).
"""

from __future__ import annotations

import polars as pl


def split_uid(uids: pl.Series | pl.Expr) -> pl.Expr:
    """Expression parts of an ``event_uid`` column: game_id, event_type, t_key_ms, actor."""
    e = uids if isinstance(uids, pl.Expr) else pl.lit(uids)
    parts = e.str.split(":")
    return pl.struct(
        parts.list.get(0).alias("game_id"),
        parts.list.get(1).alias("event_type"),
        parts.list.get(2).cast(pl.Int64).alias("t_key_ms"),
        parts.list.get(3).alias("actor"),
    )


def match_events(old: pl.DataFrame, new: pl.DataFrame, tol_ms: int = 200) -> pl.DataFrame:
    """``old`` / ``new``: tables with an ``event_uid`` column. Returns ``old_uid, new_uid,
    dt_ms`` for every event of either side (null partner when unmatched)."""

    def parts(df: pl.DataFrame, side: str) -> pl.DataFrame:
        return (
            df.select(pl.col("event_uid").alias(f"{side}_uid"))
            .unique()
            .with_columns(split_uid(pl.col(f"{side}_uid")).alias("_p"))
            .unnest("_p")
            .rename({"t_key_ms": f"t_{side}"})
        )

    o, n = parts(old, "old"), parts(new, "new")
    cand = (
        o.join(n, on=["game_id", "event_type", "actor"], how="inner")
        .with_columns((pl.col("t_new") - pl.col("t_old")).abs().alias("dt_ms"))
        .filter(pl.col("dt_ms") <= tol_ms)
        .sort("dt_ms")
    )
    used_old, used_new, pairs = set(), set(), []
    for ou, nu, dt in cand.select("old_uid", "new_uid", "dt_ms").iter_rows():
        if ou in used_old or nu in used_new:
            continue
        used_old.add(ou)
        used_new.add(nu)
        pairs.append((ou, nu, dt))
    schema = {"old_uid": pl.Utf8, "new_uid": pl.Utf8, "dt_ms": pl.Int64}
    matched = pl.DataFrame(pairs, schema=schema, orient="row")
    only_old = o.filter(~pl.col("old_uid").is_in(list(used_old))).select(
        "old_uid",
        pl.lit(None, dtype=pl.Utf8).alias("new_uid"),
        pl.lit(None, dtype=pl.Int64).alias("dt_ms"),
    )
    only_new = n.filter(~pl.col("new_uid").is_in(list(used_new))).select(
        pl.lit(None, dtype=pl.Utf8).alias("old_uid"),
        "new_uid",
        pl.lit(None, dtype=pl.Int64).alias("dt_ms"),
    )
    return pl.concat([matched, only_old, only_new])
