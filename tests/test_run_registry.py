"""RunRegistry backs GET /jobs, the idempotency check, and the watchdog's
state lookup — these are the paths that matter most."""

import pytest

from common.job_spec import JobConfig, TrainingSchedule
from common.job_state import JobState
from storage.run_registry import RunRegistry


@pytest.fixture
def registry(tmp_path):
    return RunRegistry(runs_dir=tmp_path / "runs")


def _config(job_id="job-1"):
    return JobConfig(
        job_id=job_id,
        base_model="google/gemma-2b",
        dataset_id="ds-1",
        schedule=TrainingSchedule(num_epochs=3),
    )


def test_create_and_get_run(registry):
    registry.create_run(_config())
    run = registry.get_run("job-1")
    assert run["state"] == JobState.QUEUED.value
    assert run["config"]["base_model"] == "google/gemma-2b"


def test_update_state_enforces_valid_transitions(registry):
    registry.create_run(_config())
    registry.update_state("job-1", JobState.PREPARING)
    assert registry.get_run("job-1")["state"] == JobState.PREPARING.value
    with pytest.raises(ValueError):
        registry.update_state("job-1", JobState.COMPLETED)  # illegal jump


def test_get_active_job_ignores_queued_and_terminal(registry):
    registry.create_run(_config("job-1"))
    assert registry.get_active_job() is None  # QUEUED doesn't count as active

    registry.update_state("job-1", JobState.PREPARING)
    active = registry.get_active_job()
    assert active["job_id"] == "job-1"

    registry.update_state("job-1", JobState.CANCELLED)
    assert registry.get_active_job() is None


def test_list_runs_most_recent_first(registry):
    registry.create_run(_config("job-1"))
    registry.create_run(_config("job-2"))
    runs = registry.list_runs()
    assert [r["job_id"] for r in runs] == ["job-2", "job-1"]
