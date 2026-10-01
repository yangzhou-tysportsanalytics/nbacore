"""L4: the fixed game split, computed once over every usable game.

* ``split``: usable games sorted by (game_date, game_id); the first 80 % form the training pool,
  the rest is ``test``; ``val`` = 10 % of the training pool drawn at random (seed 9). This is
  ``nbacore.possession.splits.make_game_splits`` (ghost-defense's rule, same seed) applied to all
  usable games instead of the games that happen to be downloaded.
* ``fold``: 0..4, every usable game assigned by a seeded random permutation (seed 9) of the
  games sorted by game_id, balanced (fold = rank % 5). Independent of ``split``.
* ``parity``: 0/1 = position in the (game_date, game_id) order modulo 2 (odd/even games for
  reliability analyses).

Games whose ``status`` is not ``ok`` get nulls.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from nbacore.possession.splits import make_game_splits

SPLIT_SEED = 9
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
N_FOLDS = 5


def assign_splits(games: pl.DataFrame) -> pl.DataFrame:
    """Add ``split``, ``fold``, ``parity`` to the games table (usable = ``status == "ok"``)."""
    usable = games.filter(pl.col("status") == "ok").select(["game_id", "game_date"])
    s = make_game_splits(usable, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, seed=SPLIT_SEED)
    s = s.with_columns(pl.int_range(pl.len()).mod(2).cast(pl.Int8).alias("parity"))

    by_id = usable.sort("game_id")
    perm = np.random.default_rng(SPLIT_SEED).permutation(by_id.height)
    fold = np.empty(by_id.height, dtype=np.int8)
    fold[perm] = np.arange(by_id.height) % N_FOLDS
    f = by_id.select("game_id").with_columns(pl.Series("fold", fold, dtype=pl.Int8))

    out = games.join(s.select(["game_id", "split", "parity"]), on="game_id", how="left")
    out = out.join(f, on="game_id", how="left")
    return out.sort(["game_id", "archive_name"], nulls_last=True)
