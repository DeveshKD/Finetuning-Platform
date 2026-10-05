"""Resume should be refused the moment base model, dataset, quantization,
or LoRA shape disagree — a silent mismatch is worse than a loud refusal."""

import pytest

from common.job_spec import JobConfig, TrainingSchedule
from common.provenance import assert_resume_compatible, check_resume_compatible


def _config(**overrides):
    kwargs = dict(
        job_id="job-a",
        base_model="google/gemma-2b",
        dataset_id="ds-1",
        schedule=TrainingSchedule(num_epochs=3),
    )
    kwargs.update(overrides)
    return JobConfig(**kwargs)


def test_identical_configs_are_compatible():
    original = _config(job_id="job-a")
    resumed = _config(job_id="job-b")
    assert check_resume_compatible(original, resumed) == []
    assert_resume_compatible(original, resumed)  # does not raise


def test_different_base_model_is_rejected():
    original = _config(job_id="job-a", base_model="google/gemma-2b")
    resumed = _config(job_id="job-b", base_model="google/gemma-9b")
    mismatches = check_resume_compatible(original, resumed)
    assert any(m.field == "base_model" for m in mismatches)
    with pytest.raises(ValueError):
        assert_resume_compatible(original, resumed)


def test_resume_target_cannot_be_own_job_id():
    with pytest.raises(Exception):
        _config(job_id="job-a", resume_from_job_id="job-a")
