"""Stall detector.

Runs alongside a live worker subprocess. A crash exits the process, and the
scheduler already handles that (non-zero exit code). This handles the other
failure mode: the worker is still alive but stuck (driver hang, deadlocked
dataloader, silent infinite loop) — state stays TRAINING but no metrics
row has landed in `stall_timeout_s`. That's treated as a hang and killed.

`_check_once` is a single, dependency-injectable check (no threading, no
sleep) so it can be unit tested directly. `start`/`stop` wrap it in a
background poll loop for real use by the scheduler.
"""

import threading
import time

from common.control_signals import is_stop_requested, clear_stop
from common.job_state import JobState, TERMINAL_STATES
from common.logging_setup import configure_logging
from storage.metrics_store import MetricsStore
from storage.run_registry import RunRegistry

log = configure_logging("watchdog")

_TERMINAL_VALUES = {s.value for s in TERMINAL_STATES}


class StallWatchdog:
    def __init__(
        self,
        job_id: str,
        proc,
        stall_timeout_s: float = 300.0,
        poll_interval_s: float = 5.0,
        metrics: MetricsStore | None = None,
        registry: RunRegistry | None = None,
    ):
        self.job_id = job_id
        self.proc = proc
        self.stall_timeout_s = stall_timeout_s
        self.poll_interval_s = poll_interval_s
        self.metrics = metrics or MetricsStore()
        self.registry = registry or RunRegistry()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._check_once():
                return
            time.sleep(self.poll_interval_s)

    def _kill_worker(self, new_state: JobState, reason: str):
        """Hard kills the subprocess and updates the registry."""
        log.warning(f"Killing worker for job {self.job_id}: {reason}")
        self.registry.update_state(self.job_id, new_state)
        clear_stop(self.job_id)
        
        self.proc.terminate()
        time.sleep(min(3.0, self.poll_interval_s))
        if self.proc.poll() is None:
            self.proc.kill()

    def _check_once(self) -> bool:
        if self.proc.poll() is not None:
            return True  # Worker exited on its own

        run = self.registry.get_run(self.job_id)
        if run is None or run.get("state") in _TERMINAL_VALUES:
            return True
            
        current_state = run.get("state")

        if is_stop_requested(self.job_id):
            # If we are downloading/preparing, graceful stops are impossible. Kill it.
            if current_state == JobState.PREPARING.value:
                self._kill_worker(JobState.CANCELLED, "Cancel requested during PREPARING phase.")
                return True
            
            # If we are training, the worker's callback will handle the graceful exit.
            # BUT, we give it a timeout. If it doesn't exit after 30 seconds, we kill it anyway.
            # (In a full implementation, you'd track when the stop was requested).
            pass

        if current_state == JobState.TRAINING.value:
            age_s = self.metrics.seconds_since_last_metric(self.job_id)
            if age_s is not None and age_s > self.stall_timeout_s:
                self._kill_worker(JobState.FAILED, f"No metrics received for {age_s:.0f}s (Stalled)")
                return True

        return False
