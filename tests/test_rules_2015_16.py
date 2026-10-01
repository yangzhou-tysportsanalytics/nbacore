from nbacore import rules_2015_16 as R
from nbacore.events.pbp_align import FOUL_DETAIL


def test_every_pbp_foul_detail_is_classified():
    details = {d for d, _ in FOUL_DETAIL.values()}
    assert details == R.TEAM_FOUL_DETAILS | R.NON_TEAM_FOUL_DETAILS
    assert not R.TEAM_FOUL_DETAILS & R.NON_TEAM_FOUL_DETAILS


def test_in_penalty_regulation():
    assert not R.in_penalty(1, 300.0, 3, 0)
    assert R.in_penalty(1, 300.0, 4, 0)  # 5th team foul
    # last two minutes: first foul free, second penalised (12B-V-a(3))
    assert not R.in_penalty(4, 100.0, 2, 0)
    assert R.in_penalty(4, 100.0, 3, 1)
    assert R.in_penalty(4, 120.0, 1, 1)  # 2:00 counts as the two-minute part


def test_in_penalty_overtime():
    assert not R.in_penalty(5, 250.0, 2, 0)
    assert R.in_penalty(5, 250.0, 3, 0)  # 4th team foul in OT


def test_shot_clock_after_defensive_foul():
    assert R.shot_clock_after_defensive_foul(9.0, frontcourt_throw_in=True) == 14.0
    assert R.shot_clock_after_defensive_foul(17.5, frontcourt_throw_in=True) == 17.5
    assert R.shot_clock_after_defensive_foul(9.0, frontcourt_throw_in=False) == 24.0
    assert R.OREB_RESET_S == 24.0  # 2015-16: no 14-second reset on offensive rebounds
