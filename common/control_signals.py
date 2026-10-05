"""File-based stop signal.

Simplest possible cross-process control channel: the API touches a flag
file for a job_id, and the worker's training loop checks for that file's
existence every step. No IPC library needed at this scale — a stat() call
per step is free compared to a training step.

control_dir is a parameter (not just a hardcoded path) so tests can point
it at a tmp directory instead of touching the real data/control/ folder.
"""

from pathlib import Path

CONTROL_DIR = Path("data/control")


def request_stop(job_id: str, control_dir: Path = CONTROL_DIR) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    (control_dir / f"{job_id}.stop").touch()


def is_stop_requested(job_id: str, control_dir: Path = CONTROL_DIR) -> bool:
    return (control_dir / f"{job_id}.stop").exists()


def clear_stop(job_id: str, control_dir: Path = CONTROL_DIR) -> None:
    flag = control_dir / f"{job_id}.stop"
    if flag.exists():
        flag.unlink()
