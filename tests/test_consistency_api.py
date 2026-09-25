"""End-to-end tests for the read-only redundant-channel consistency API.

The check reads existing events (and channel TTLs) on demand: it never
inserts/updates/deletes events, never writes fault results and never triggers
rule re-evaluation.
"""

import json
import os
import tempfile

import pytest

from app import db, services
from app.engine import to_epoch

BASE = to_epoch("2026-09-23T00:00:00Z")
P = 60.0


@pytest.fixture()
def scratch_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db.DB_PATH = tmp.name
    db._conn = None
    db.init_db()
    yield tmp.name
    for suffix in ("", "-wal", "-shm"):
        try:
            os.unlink(tmp.name + suffix)
        except OSError:
            pass


@pytest.fixture()
def client(scratch_db):
    from fastapi.testclient import TestClient
    from app.main import app

    return TestClient(app)


def _load_two_channels(vals_a, vals_b, device="d", metric_a="a", metric_b="b"):
    rows = [
        {"device": device, "metric": metric_a, "ts": BASE + k * P, "value": v}
        for k, v in enumerate(vals_a)
    ]
    rows += [
        {"device": device, "metric": metric_b, "ts": BASE + k * P, "value": v}
        for k, v in enumerate(vals_b)
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows), reevaluate=False)


# --------------------------------------------------------------------------- #
# service
# --------------------------------------------------------------------------- #


def test_service_segments_and_totals(scratch_db):
    # minute values: equal, bad(5), good(2), bad(8), equal
    _load_two_channels([0, 0, 0, 0, 0], [0, 5, 2, 8, 0])
    r = services.query_channel_consistency("d", "a", "d", "b", BASE, BASE + 4 * P, 3.0)
    assert r["comparable_seconds"] == 240
    assert r["inconsistent_seconds"] == 120
    assert r["consistent_seconds"] == 120
    assert [ (s["start"] - BASE, s["end"] - BASE) for s in r["segments"] ] == [
        (60, 120),
        (180, 240),
    ]
    worst = max(r["segments"], key=lambda s: s["max_diff"])
    assert worst["max_diff"] == 8
    assert worst["channel_1_sample"]["value"] == 0
    assert worst["channel_2_sample"]["value"] == 8
    assert worst["channel_2_sample"]["ts"] == BASE + 3 * P


def test_service_threshold_equality_is_consistent(scratch_db):
    _load_two_channels([0, 0, 0], [5, 5, 5])
    r = services.query_channel_consistency("d", "a", "d", "b", BASE, BASE + 2 * P, 5.0)
    assert r["segments"] == []
    assert r["inconsistent_seconds"] == 0
    assert r["comparable_seconds"] == 120


def test_service_ttl_caps_comparable_time(scratch_db):
    _load_two_channels([0] * 6, [10] * 6)
    db.upsert_channel_ttl("d", "b", 30)
    # window after the first minute: b holds 30s of each minute, a holds all
    r = services.query_channel_consistency("d", "a", "d", "b", BASE + P, BASE + 5 * P, 5.0)
    assert r["comparable_seconds"] == 120  # four 30s pieces
    assert r["inconsistent_seconds"] == 120
    assert r["segment_count"] == 4  # TTL gaps split them; nothing extrapolated
    assert r["channel_2"]["ttl_seconds"] == 30
    assert r["channel_1"]["ttl_seconds"] is None


def test_service_pre_start_sample_carried_in(scratch_db):
    _load_two_channels([0, 0, 0, 0], [9, 0, 0, 0])
    r = services.query_channel_consistency("d", "a", "d", "b", BASE + 30, BASE + 2 * P, 3.0)
    # the b=9 sample taken before window start is still held at start; at 60 b
    # switches to 0 and everything stays consistent to the window end
    assert [(s["start"] - BASE, s["end"] - BASE) for s in r["segments"]] == [(30, 60)]
    assert r["segments"][0]["channel_2_sample"]["ts"] == BASE
    assert r["comparable_seconds"] == 90


def test_service_validation_and_unknown_channels(scratch_db):
    _load_two_channels([0], [1])
    with pytest.raises(ValueError):
        services.query_channel_consistency("d", "a", "d", "a", BASE, BASE + P, 0)
    with pytest.raises(ValueError):
        services.query_channel_consistency("d", "a", "d", "b", BASE + P, BASE, 0)
    with pytest.raises(ValueError):
        services.query_channel_consistency("d", "a", "d", "b", BASE, BASE + P, -1)
    with pytest.raises(KeyError):
        services.query_channel_consistency("d", "a", "d", "nope", BASE, BASE + P, 0)
    with pytest.raises(KeyError):
        services.query_channel_consistency("x", "a", "d", "b", BASE, BASE + P, 0)


def test_service_channel_with_only_null_values_is_unknown(scratch_db):
    # a channel row with NULL value is not a usable measurement channel
    db.execute("INSERT INTO events (device, metric, ts, value) VALUES (?,?,?,?)",
               ("d", "nullm", BASE, None))
    services.import_events(
        json.dumps({"device": "d", "metric": "a", "ts": BASE, "value": 0}),
        reevaluate=False,
    )
    with pytest.raises(KeyError):
        services.query_channel_consistency("d", "a", "d", "nullm", BASE, BASE + P, 0)


def test_service_is_read_only(scratch_db):
    _load_two_channels([0, 0, 0, 0], [1, 1, 1, 1])
    db.execute("INSERT INTO rules (name, logic, duration, created_at) VALUES (?,?,?,?)",
               ("r", "AND", 0, 0))
    db.execute(
        "INSERT INTO fault_intervals (rule_id, start_ts, end_ts, duration_required, confirmed_at, evidence_json)"
        " VALUES (?,?,?,?,?,?)", (1, BASE, BASE + P, 0, BASE, "[]")
    )
    before_events = db.query_one("SELECT COUNT(*) AS c FROM events")["c"]
    before_faults = db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"]
    services.query_channel_consistency("d", "a", "d", "b", BASE, BASE + 3 * P, 0.5)
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == before_events
    assert db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"] == before_faults
    assert db.query_one("SELECT COUNT(*) AS c FROM rules")["c"] == 1


def test_service_across_devices(scratch_db):
    rows = [
        {"device": "pump-1", "metric": "temp", "ts": BASE, "value": 80},
        {"device": "pump-1", "metric": "temp", "ts": BASE + P, "value": 80},
        {"device": "pump-9", "metric": "temp", "ts": BASE, "value": 70},
        {"device": "pump-9", "metric": "temp", "ts": BASE + P, "value": 70},
    ]
    services.import_events("\n".join(json.dumps(r) for r in rows), reevaluate=False)
    r = services.query_channel_consistency(
        "pump-1", "temp", "pump-9", "temp", BASE, BASE + P, 5
    )
    assert r["inconsistent_seconds"] == 60
    assert r["segments"][0]["max_diff"] == 10


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def test_api_consistency_ok(scratch_db, client):
    _load_two_channels([0, 0, 0, 0, 0], [0, 5, 2, 8, 0])
    resp = client.get("/api/channel-consistency", params={
        "device_1": "d", "metric_1": "a",
        "device_2": "d", "metric_2": "b",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:04:00Z",
        "diff_threshold": 3,
    })
    assert resp.status_code == 200
    body = resp.json()
    assert body["comparable_seconds"] == 240
    assert body["inconsistent_seconds"] == 120
    assert body["segment_count"] == 2
    assert body["channel_1"]["device"] == "d"
    assert "horizon" in body


def test_api_same_channel_400(scratch_db, client):
    _load_two_channels([0], [1])
    resp = client.get("/api/channel-consistency", params={
        "device_1": "d", "metric_1": "a",
        "device_2": "d", "metric_2": "a",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:01:00Z",
        "diff_threshold": 1,
    })
    assert resp.status_code == 400


def test_api_unknown_channel_404(scratch_db, client):
    _load_two_channels([0], [1])
    resp = client.get("/api/channel-consistency", params={
        "device_1": "d", "metric_1": "a",
        "device_2": "d", "metric_2": "zz",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:01:00Z",
        "diff_threshold": 1,
    })
    assert resp.status_code == 404


def test_api_inverted_window_400(scratch_db, client):
    _load_two_channels([0], [1])
    resp = client.get("/api/channel-consistency", params={
        "device_1": "d", "metric_1": "a",
        "device_2": "d", "metric_2": "b",
        "start": "2026-09-23T00:02:00Z", "end": "2026-09-23T00:00:00Z",
        "diff_threshold": 1,
    })
    assert resp.status_code == 400


def test_api_validation_422(scratch_db, client):
    _load_two_channels([0], [1])
    base = {
        "device_1": "d", "metric_1": "a",
        "device_2": "d", "metric_2": "b",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:01:00Z",
    }
    # negative threshold rejected by Query(ge=0)
    assert client.get("/api/channel-consistency", params={**base, "diff_threshold": -1}).status_code == 422
    # missing params
    assert client.get("/api/channel-consistency", params=base).status_code == 422


def test_api_read_only(scratch_db, client):
    _load_two_channels([0, 0, 0], [9, 9, 9])
    before = db.query_one("SELECT COUNT(*) AS c FROM events")["c"]
    resp = client.get("/api/channel-consistency", params={
        "device_1": "d", "metric_1": "a",
        "device_2": "d", "metric_2": "b",
        "start": "2026-09-23T00:00:00Z", "end": "2026-09-23T00:02:00Z",
        "diff_threshold": 5,
    })
    assert resp.status_code == 200
    assert db.query_one("SELECT COUNT(*) AS c FROM events")["c"] == before
    assert db.query_one("SELECT COUNT(*) AS c FROM fault_intervals")["c"] == 0
