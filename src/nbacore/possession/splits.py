"""Train / validation / test split by game and by date.

* Games are sorted by date (ties by game id); the first ``train_frac`` of games are the training
  pool, the rest is the test set (later dates -> test, so nothing from the future leaks in).
* Validation games are drawn from the training pool at random (seeded), ``val_frac`` of it.

The split is a function of the *set of games available*, so it is recomputed for every data
config (tiny/small/medium/large) and stored next to the possessions.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def make_game_splits(
    games: pl.DataFrame,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
) -> pl.DataFrame:
    """``games`` needs columns game_id, game_date (str YYYY-MM-DD). Returns game_id, game_date, split."""
    g = games.select(["game_id", "game_date"]).unique().sort(["game_date", "game_id"])
    n = g.height
    n_train_pool = int(round(train_frac * n))
    split = np.array(["test"] * n, dtype=object)
    split[:n_train_pool] = "train"
    rng = np.random.default_rng(seed)
    pool_idx = np.arange(n_train_pool)
    n_val = int(round(val_frac * n_train_pool))
    if n_val > 0:
        val_idx = rng.choice(pool_idx, size=n_val, replace=False)
        split[val_idx] = "val"
    return g.with_columns(pl.Series("split", split.tolist(), dtype=pl.Utf8))
