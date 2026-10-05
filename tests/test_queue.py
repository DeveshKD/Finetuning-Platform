"""FIFO ordering and the claim-so-you-can't-double-pop behavior."""

import pytest

from common.job_spec import JobConfig, TrainingSchedule
from orchestrator.queue import JobQueue


@pytest.fixture
def queue(tmp_path):
    return JobQueue(db_path=tmp_path / "queue.db")


def _config(job_id):
    return JobConfig(
        job_id=job_id,
        base_model="google/gemma-2b",
        dataset_id="ds-1",
        schedule=TrainingSchedule(num_epochs=1),
    )


def test_fifo_order(queue):
    queue.enqueue(_config("job-1"))
    queue.enqueue(_config("job-2"))
    assert queue.pop_next().job_id == "job-1"
    assert queue.pop_next().job_id == "job-2"


def test_empty_queue_returns_none(queue):
    assert queue.pop_next() is None


def test_pop_claims_so_it_cant_be_popped_twice(queue):
    queue.enqueue(_config("job-1"))
    first = queue.pop_next()
    second = queue.pop_next()
    assert first.job_id == "job-1"
    assert second is None
