"""Append/query roundtrip, plus the staleness check the watchdog relies on."""

import pytest

from storage.metrics_store import MetricsStore


@pytest.fixture
def store(tmp_path):
    return MetricsStore(db_path=tmp_path / "metrics.db")


def test_append_and_query_since(store):
    store.append("job-1", step=0, payload={"loss": 2.5})
    store.append("job-1", step=1, payload={"loss": 2.1})
    rows = store.query_since("job-1", last_id=0)
    assert [r["loss"] for r in rows] == [2.5, 2.1]


def test_query_since_only_returns_new_rows(store):
    store.append("job-1", step=0, payload={"loss": 2.5})
    first_batch = store.query_since("job-1", last_id=0)
    store.append("job-1", step=1, payload={"loss": 2.1})
    second_batch = store.query_since("job-1", last_id=first_batch[-1]["id"])
    assert len(second_batch) == 1
    assert second_batch[0]["loss"] == 2.1


def test_seconds_since_last_metric_none_when_no_metrics(store):
    assert store.seconds_since_last_metric("never-ran") is None


def test_seconds_since_last_metric_is_small_right_after_append(store):
    store.append("job-1", step=0, payload={"loss": 1.0})
    age = store.seconds_since_last_metric("job-1")
    assert age is not None and age < 2.0
