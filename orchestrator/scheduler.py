"""Main orchestration loop.

Long-running process (can start as a background task inside the API
process for v0, split out later): pulls the next job, waits for a verified
clean GPU, spawns the worker as an isolated subprocess, watches it for
stalls, and updates the run registry as it goes.

"""

import subprocess
import sys
import time

from common.job_state import JobState, TERMINAL_STATES
from common.logging_setup import configure_logging
from orchestrator.resource_manager import ResourceManager
from orchestrator.watchdog import StallWatchdog
from storage.run_registry import RunRegistry

log = configure_logging("scheduler")

_TERMINAL_VALUES = {s.value for s in TERMINAL_STATES}


class Scheduler:
    def __init__(self, stall_timeout_s: float = 300.0):
        self.resources = ResourceManager()
        self.registry = RunRegistry()
        self.stall_timeout_s = stall_timeout_s

    def run_forever(self) -> None:
        log.info("scheduler started")
        while True:
            job = self.registry.pop_next_queued_job()
            if job is None:
                time.sleep(2.0)
                continue
            self._run_job(job)

    def _run_job(self, job) -> None:
        log.info("waiting for clean GPU before job %s", job.job_id)
        baseline_snap = self.resources.wait_until_ready()

        log.info("spawning worker subprocess for job %s", job.job_id)

        # Popen (not run) so the watchdog can monitor the process
        # concurrently instead of the scheduler blocking until exit. One
        # job = one subprocess either way, so a crash or leaked allocation
        # still dies with it.
        proc = subprocess.Popen(
            [sys.executable, "-m", "worker.entrypoint", "--job-id", job.job_id],
        )
        watchdog = StallWatchdog(job.job_id, proc, stall_timeout_s=self.stall_timeout_s)
        watchdog.start()

        returncode = proc.wait()
        watchdog.stop()

        if returncode != 0:
            log.error("job %s exited with code %s", job.job_id, returncode)
            run = self.registry.get_run(job.job_id)
            # The watchdog may have already marked this FAILED on a stall;
            # this covers a plain crash / non-zero exit it didn't catch.
            if run and run.get("status") not in _TERMINAL_VALUES:
                self.registry.update_state(job.job_id, JobState.FAILED)

        # Wait 2 seconds for the OS/CUDA to reap the subprocess memory pages
        time.sleep(2.0)
        post_run_snap = self.resources.snapshot()
        self.resources.log_fragmentation(
            job_id=job.job_id,
            expected_free_mb=baseline_snap.free_mb,
            actual=post_run_snap
        )


if __name__ == "__main__":
    Scheduler().run_forever()
