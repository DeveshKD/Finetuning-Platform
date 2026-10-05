"""Job submission and lookup endpoints.

POST /jobs           -> validates JobConfig, registers it, enqueues it (rejects a repeat job_id)
GET  /jobs           -> list all jobs with current state
GET  /jobs/active    -> the one job currently occupying the GPU, if any
GET  /jobs/{id}      -> full job detail, including eval results once available
POST /jobs/{id}/stop -> request a graceful stop; the worker checks this every step
"""

from fastapi import APIRouter, HTTPException

from common.control_signals import request_stop
from common.job_spec import JobConfig
from common.job_state import JobState
from storage.run_registry import RunRegistry

router = APIRouter()
# TODO: wire through a real DI container / lifespan hook
_registry = RunRegistry()

_TERMINAL_STATE_VALUES = {JobState.COMPLETED.value, JobState.FAILED.value, JobState.CANCELLED.value}


@router.post("/")
def submit_job(config: JobConfig) -> dict:
    # job_id doubles as the idempotency key. A duplicate submission is
    # almost always a retried or double-clicked request, not an
    # intentional second run — reject it instead of silently re-queueing.
    if _registry.get_run(config.job_id) is not None:
        raise HTTPException(status_code=409, detail=f"job_id {config.job_id} already exists")

    _registry.create_run(config)
    return {"job_id": config.job_id, "state": "queued"}


@router.get("/")
def list_jobs() -> list[dict]:
    return _registry.list_runs()


@router.get("/active")
def get_active_job() -> dict:
    """Cheap 'is anything running right now' check without listing every job —
    useful before submitting a new job on a single-GPU box."""
    active = _registry.get_active_job()
    return active or {"active": False}


@router.get("/{job_id}")
def get_job(job_id: str) -> dict:
    run = _registry.get_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail="job not found")
    return run


@router.post("/{job_id}/stop")
def stop_job(job_id: str) -> dict:
    run = _registry.get_run(job_id)
    if run is None:
        raise HTTPException(status_code=404, detail="job not found")
        
    current_state = run["status"]
    if current_state in _TERMINAL_STATE_VALUES:
        raise HTTPException(status_code=409, detail=f"job already in terminal state: {current_state}")

    # If it's just sitting in the queue, we can cancel it instantly
    if current_state == JobState.QUEUED.value:
        _registry.update_state(job_id, JobState.CANCELLED)
        return {"job_id": job_id, "state": "cancelled"}

    # If it's PREPARING or TRAINING, we write the stop file.
    # The Watchdog will nuke it if it's PREPARING.
    # The Worker will exit gracefully if it's TRAINING.
    request_stop(job_id)
    return {"job_id": job_id, "state": "stop_requested"}
