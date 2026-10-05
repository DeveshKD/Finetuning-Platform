"""JobConfig validation is the platform's contract — these tests are the
cheapest, highest-signal thing to have working before a live demo."""

import pytest
from pydantic import ValidationError

from common.job_spec import JobConfig, TrainingSchedule


def _base_kwargs(**overrides):
    kwargs = dict(
        job_id="job-1",
        base_model="google/gemma-2b",
        dataset_id="ds-1",
        schedule=TrainingSchedule(num_epochs=3),
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_job_config_parses():
    job = JobConfig(**_base_kwargs())
    assert job.lora.r == 16
    assert job.quantization.value == "nf4"


def test_schedule_requires_exactly_one_stop_condition():
    with pytest.raises(ValidationError):
        TrainingSchedule(num_epochs=3, max_steps=1000)  # both set
    with pytest.raises(ValidationError):
        TrainingSchedule(num_epochs=None, max_steps=None)  # neither set
