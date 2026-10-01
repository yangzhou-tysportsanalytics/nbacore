import numpy as np
import polars as pl
import pytest

from nbacore.ledger_detail import _half_runs, event_possession
from nbacore.views.post_release import with_post_release


def test_half_runs_merges_short_flickers():
    t = np.arange(0, 10_000, 40)
    front = t >= 4_000
    front[(t >= 6_000) & (t < 6_200)] = False  # 0.2 s flicker back
    runs = _half_runs(t, front)
    assert [r[0] for r in runs] == [False, True]
    assert runs[1][1] == 4_000 and runs[1][2] == int(t[-1])


def test_event_possession_end_event_belongs_to_ending_possession():
    pbp = pl.DataFrame(
        {
            "event_num": [1, 2, 3, 4],
            "period": [1, 1, 1, 1],
            "pctime_sec": [720.0, 700.0, 690.0, 680.0],
            "msg_type": [12, 1, 2, 4],
        }
    )
    poss = pl.DataFrame({"poss_seq": [0, 1], "period": [1, 1], "end_event_num": [2, None]})
    assert event_possession(pbp, poss) == {1: 0, 2: 0, 3: 1, 4: 1}


def test_with_post_release():
    w = pl.DataFrame(
        {"terminal_msg_type": [1, 6], "t_terminal": [1_000, 5_000], "t_end": [1_500, 5_500]}
    )
    out = with_post_release(w, 2.5)
    assert out["t_end_post_ms"].to_list() == [3_500, 5_500]
    with pytest.raises(ValueError):
        with_post_release(w, 4.0)
