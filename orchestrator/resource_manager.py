"""VRAM gatekeeper — the one genuinely novel piece of this platform.

On a single GPU, the allocator does not always release memory cleanly
between back-to-back jobs. This module is what makes the orchestrator wait
for a *verified* clean slate instead of trusting that "process exited" ==
"memory reclaimed". The gap between expected-free and actually-free VRAM
after a job exits IS the fragmentation measurement this project set out to
make — log it, don't just gate on it.
"""

import subprocess
import time
from dataclasses import dataclass
from common.logging_setup import configure_logging
from storage.metrics_store import MetricsStore

log = configure_logging("resource_manager")

@dataclass
class VramSnapshot:
    free_mb: int
    total_mb: int


class ResourceManager:
    def __init__(
        self,
        min_free_mb: int = 2048,
        poll_interval_s: float = 2.0,
        timeout_s: float = 60.0,
    ):
        self.min_free_mb = min_free_mb
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self.metrics_store = MetricsStore()

    def snapshot(self) -> VramSnapshot:
        """Query VRAM via nvidia-smi rather than torch.cuda — this manager
        lives in the orchestrator process, which never imports torch, so it
        can gate a job's GPU usage without holding a CUDA context itself."""
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        free_str, total_str = out.decode().strip().splitlines()[0].split(",")
        return VramSnapshot(free_mb=int(free_str), total_mb=int(total_str))

    def wait_until_ready(self) -> VramSnapshot:
        """Block until free VRAM clears the threshold, or raise on timeout."""
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            snap = self.snapshot()
            if snap.free_mb >= self.min_free_mb:
                return snap
            time.sleep(self.poll_interval_s)
        raise TimeoutError(
            "GPU did not free enough VRAM before timeout — possible fragmentation"
        )

    def log_fragmentation(self, job_id: str, expected_free_mb: int, actual: VramSnapshot) -> None:
        """Measure and persist the gap between expected and actual VRAM recovery."""
        gap_mb = expected_free_mb - actual.free_mb
        
        # Persist it to SQLite for historical tracking
        self.metrics_store.log_fragmentation(
            job_id=job_id,
            expected_free_mb=expected_free_mb,
            actual_free_mb=actual.free_mb,
            gap_mb=gap_mb
        )
        
        # If we leaked more than 50MB of VRAM, something is fragmenting.
        if gap_mb > 50:
            log.warning(
                f"VRAM LEAK DETECTED [Job: {job_id}]: "
                f"Expected {expected_free_mb}MB free, got {actual.free_mb}MB. "
                f"Lost {gap_mb}MB to fragmentation/zombie context!"
            )
        else:
            log.info(f"VRAM recovered cleanly after {job_id} (Gap: {gap_mb}MB).")
