import polars as pl

from nbacore.possession.splits import make_game_splits


def test_game_splits_by_date():
    games = pl.DataFrame(
        {
            "game_id": [f"00215000{i:02d}" for i in range(20)],
            "game_date": [f"2015-11-{1 + i:02d}" for i in range(20)],
        }
    )
    s = make_game_splits(games, train_frac=0.8, val_frac=0.25, seed=1)
    assert s.height == 20
    counts = dict(
        zip(*s["split"].value_counts().sort("split").to_dict(as_series=False).values(), strict=True)
    )
    assert counts == {"test": 4, "train": 12, "val": 4}
    # all test games are later than every train/val game
    last_train = s.filter(pl.col("split") != "test")["game_date"].max()
    first_test = s.filter(pl.col("split") == "test")["game_date"].min()
    assert first_test > last_train
    # deterministic
    s2 = make_game_splits(games, train_frac=0.8, val_frac=0.25, seed=1)
    assert s["split"].to_list() == s2["split"].to_list()
