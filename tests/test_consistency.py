"""Unit tests for the read-only redundant-channel consistency sweep.

Held-value pieces are built with the same expansion as the rule engine
(``channel_value_segments``: per-sample hold, optional TTL cap and the global
data horizon); the sweep compares only on the intersection where both
channels hold, strictly greater than the threshold, merging touching pieces.
"""

import pytest

from app.consistency import channel_consistency
from app.engine import Event, HeldValue, channel_value_segments


def ev(eid, ts, value, device="d", metric="m"):
    return Event(id=eid, device=device, metric=metric, ts=ts, value=value)


def pieces(values, start=0.0, step=60.0, horizon=None, ttl=None, device="d", metric="m"):
    events = [ev(i + 1, start + i * step, v, device, metric) for i, v in enumerate(values)]
    if horizon is None:
        horizon = start + (len(values) - 1) * step
    return channel_value_segments(events, horizon, ttl=ttl)


def seg_bounds(result):
    return [(s.start, s.end) for s in result.segments]


def test_basic_diff_within_common_hold():
    a = pieces([10, 20, 10])
    b = pieces([10, 25, 15])
    r = channel_consistency(a, b, 0, 180, 3.0)
    # [0,60) equal; [60,120) diff 5 > 3; [120,180) not held past horizon
    assert r.comparable_seconds == 120
    assert r.inconsistent_seconds == 60
    assert r.consistent_seconds == 60
    assert seg_bounds(r) == [(60, 120)]
    s = r.segments[0]
    assert s.max_diff == 5
    assert s.duration_seconds == 60
    assert (s.channel_1_sample["ts"], s.channel_1_sample["value"]) == (60, 20)
    assert (s.channel_2_sample["ts"], s.channel_2_sample["value"]) == (60, 25)
    assert r.to_dict()["inconsistency_rate"] == pytest.approx(0.5)


def test_strictly_greater_than_threshold():
    a = pieces([0, 0, 0])
    b = pieces([10, 5, 0])
    # diff == threshold (5) is NOT inconsistent; diff 10 on [0,60) is
    r = channel_consistency(a, b, 0, 120, 5.0)
    assert r.inconsistent_seconds == 60
    assert seg_bounds(r) == [(0, 60)]
    # zero threshold catches every nonzero difference
    r0 = channel_consistency(a, b, 0, 120, 0.0)
    assert r0.inconsistent_seconds == 120
    assert r0.segment_count == 1  # touching bad pieces merge into one


def test_touching_bad_pieces_merge_and_max_diff_witness_kept():
    a = pieces([0, 0, 0])
    b = pieces([10, 8, 2])
    r = channel_consistency(a, b, 0, 120, 5.0)
    # [0,60) diff 10 and [60,120) diff 8 both bad, touching at 60 -> one piece
    assert r.segment_count == 1
    s = r.segments[0]
    assert (s.start, s.end) == (0, 120)
    assert s.max_diff == 10
    assert s.channel_2_sample["value"] == 10
    # a consistent piece between two bad ones keeps them apart
    a2 = pieces([0, 0, 0, 0])
    b2 = pieces([10, 2, 10, 2])
    r2 = channel_consistency(a2, b2, 0, 180, 5.0)
    assert seg_bounds(r2) == [(0, 60), (120, 180)]
    assert r2.segment_count == 2


def test_window_clipping():
    a = pieces([0, 0], horizon=60)
    b = pieces([10, 10], horizon=60)
    # query a sub-window strictly inside the held piece
    r = channel_consistency(a, b, 10, 40, 5.0)
    assert r.comparable_seconds == 30
    assert r.inconsistent_seconds == 30
    assert seg_bounds(r) == [(10, 40)]


def test_pre_start_sample_is_carried_in():
    a = pieces([0, 0, 0], horizon=120)  # pieces [0,60), [60,120)
    b = pieces([8, 2, 2], horizon=120)
    # window starts at 30: the pre-start sample (b=8 on [0,60)) must be used
    r = channel_consistency(a, b, 30, 120, 5.0)
    assert seg_bounds(r) == [(30, 60)]
    assert r.comparable_seconds == 90
    assert r.inconsistent_seconds == 30
    assert r.segments[0].channel_2_sample["ts"] == 0


def test_ttl_gap_of_one_channel_is_not_comparable_or_extrapolated():
    # a: no TTL, constant 0; b: TTL 30 -> holds only [t, t+30) each minute.
    # horizon at 300 (data continues elsewhere) so the last sample still
    # reaches [240,270) even though these channels' own samples stop at 240.
    a = pieces([0, 0, 0, 0, 0], ttl=None, horizon=300)
    b = pieces([10, 10, 10, 10, 10], ttl=30.0, horizon=300)
    r = channel_consistency(a, b, 30, 300, 5.0)
    # at start=30, b's first piece [0,30) already expired; comparable only on
    # [60,90),[120,150),[180,210),[240,270) -> 120s, never extrapolated
    assert r.comparable_seconds == 120
    assert r.inconsistent_seconds == 120
    assert seg_bounds(r) == [(60, 90), (120, 150), (180, 210), (240, 270)]


def test_hold_ends_at_global_horizon_without_extrapolation():
    a = pieces([0, 0], horizon=120)
    b = pieces([10, 10], horizon=120)
    # query reaches past horizon: comparison stops at 120
    r = channel_consistency(a, b, 0, 300, 5.0)
    assert r.comparable_seconds == 120
    assert r.inconsistent_seconds == 120
    assert r.segments[0].end == 120


def test_staggered_pieces_intersect_only_on_overlap():
    # a holds [0,100), b holds [50,150): comparable only [50,100)
    a = [HeldValue(0, 100, sample_ts=0, value=0)]
    b = [HeldValue(50, 150, sample_ts=50, value=9)]
    r = channel_consistency(a, b, 0, 200, 5.0)
    assert r.comparable_seconds == 50
    assert r.inconsistent_seconds == 50
    assert seg_bounds(r) == [(50, 100)]


def test_value_change_mid_piece_creates_then_ends_segment():
    a = [HeldValue(0, 180, sample_ts=0, value=0)]
    # b: 20 on [0,60), 0 on [60,120), 20 on [120,180)
    b = [
        HeldValue(0, 60, sample_ts=0, value=20),
        HeldValue(60, 120, sample_ts=60, value=0),
        HeldValue(120, 180, sample_ts=120, value=20),
    ]
    r = channel_consistency(a, b, 0, 180, 5.0)
    assert seg_bounds(r) == [(0, 60), (120, 180)]
    assert r.comparable_seconds == 180
    assert r.inconsistent_seconds == 120


def test_no_comparable_time():
    a = [HeldValue(0, 30, sample_ts=0, value=0)]
    b = [HeldValue(60, 90, sample_ts=60, value=9)]
    r = channel_consistency(a, b, 0, 120, 5.0)
    assert r.comparable_seconds == 0
    assert r.inconsistent_seconds == 0
    assert r.segments == []
    assert r.to_dict()["inconsistency_rate"] == 0.0


def test_parameter_validation():
    a = pieces([0])
    b = pieces([1])
    with pytest.raises(ValueError):
        channel_consistency(a, b, 100, 100, 0.0)
    with pytest.raises(ValueError):
        channel_consistency(a, b, 100, 90, 0.0)
    with pytest.raises(ValueError):
        channel_consistency(a, b, 0, 60, -1.0)
    with pytest.raises(ValueError):
        channel_consistency(a, b, 0, 60, float("nan"))
