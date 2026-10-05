"""The state machine is what keeps the orchestrator and watchdog honest — a
job should never silently skip TRAINING -> COMPLETED, and a stop request
must be able to land at any point while a job is actually running."""

import pytest

from common.job_state import JobState, assert_valid_transition


def test_legal_transition_is_allowed():
    assert_valid_transition(JobState.QUEUED, JobState.PREPARING)


def test_illegal_transition_raises():
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.QUEUED, JobState.COMPLETED)


@pytest.mark.parametrize(
    "state", [JobState.QUEUED, JobState.PREPARING, JobState.TRAINING, JobState.CHECKPOINTING, JobState.EVALUATING]
)
def test_cancel_is_reachable_from_every_active_state(state):
    assert_valid_transition(state, JobState.CANCELLED)


def test_persisting_cannot_be_cancelled():
    with pytest.raises(ValueError):
        assert_valid_transition(JobState.PERSISTING, JobState.CANCELLED)
