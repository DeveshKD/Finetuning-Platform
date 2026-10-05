"""Watchdog decision logic, exercised via _check_once() with fake
proc/metrics/registry doubles — no real subprocess, threading, or SQLite
needed to test the actual branching."""

from common.job_state import JobState
from orchestrator.watchdog import StallWatchdog


class FakeProc:
    def __init__(self, running=True):
        self._running = running
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self._running else 0

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class FakeMetrics:
    def __init__(self, age_s):
        self.age_s = age_s

    def seconds_since_last_metric(self, job_id):
        return self.age_s


class FakeRegistry:
    def __init__(self, state):
        self.state = state
        self.updated_to = None

    def get_run(self, job_id):
        return None if self.state is None else {"state": self.state}

    def update_state(self, job_id, new_state):
        self.updated_to = new_state
        self.state = new_state.value


def _watchdog(state, age_s, running=True, stall_timeout_s=60.0):
    return StallWatchdog(
        "job-1", FakeProc(running=running),
        stall_timeout_s=stall_timeout_s, poll_interval_s=0.01,
        metrics=FakeMetrics(age_s), registry=FakeRegistry(state),
    )


def test_stops_watching_once_process_exits():
    wd = _watchdog(state=JobState.TRAINING.value, age_s=5, running=False)
    assert wd._check_once() is True


def test_stops_watching_once_job_reaches_terminal_state():
    wd = _watchdog(state=JobState.COMPLETED.value, age_s=999)
    assert wd._check_once() is True


def test_does_not_flag_a_non_training_phase_as_stalled():
    wd = _watchdog(state=JobState.PREPARING.value, age_s=999)
    assert wd._check_once() is False
    assert wd.proc.terminated is False


def test_does_not_flag_recent_metrics_as_stalled():
    wd = _watchdog(state=JobState.TRAINING.value, age_s=5, stall_timeout_s=60.0)
    assert wd._check_once() is False
    assert wd.proc.terminated is False


def test_terminates_and_marks_failed_on_real_stall():
    wd = _watchdog(state=JobState.TRAINING.value, age_s=9999, stall_timeout_s=60.0)
    result = wd._check_once()
    assert result is True
    assert wd.proc.terminated is True
    assert wd.registry.updated_to == JobState.FAILED
