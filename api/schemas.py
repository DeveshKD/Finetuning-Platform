"""Request/response shapes that are API-specific (not part of the core
JobConfig contract shared with the worker).
"""

from pydantic import BaseModel

from common.job_state import JobState


class JobSummary(BaseModel):
    job_id: str
    base_model: str
    state: JobState


class JobDetail(JobSummary):
    eval_perplexity: float | None = None
    eval_judge_mean: float | None = None
