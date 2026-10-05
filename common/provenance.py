"""Lightweight provenance check for checkpoint resume.

Before resuming a job from another run's checkpoint, verify the two runs
agree on everything that would make the resumed weights meaningless if
mismatched: base model, dataset, quantization, and LoRA shape. This is a
deliberately small version of what unsloth-studio's provenance.py does —
no cryptographic hashing, no dataset content pinning, just the fields
where a silent mismatch would corrupt training rather than error loudly.
"""

from dataclasses import dataclass

from common.job_spec import JobConfig


@dataclass
class ProvenanceMismatch:
    field: str
    original: str
    resumed: str


def check_resume_compatible(original: JobConfig, resumed: JobConfig) -> list[ProvenanceMismatch]:
    """Returns a list of mismatches; an empty list means it's safe to resume."""
    checks = [
        ("base_model", original.base_model, resumed.base_model),
        ("dataset_id", original.dataset_id, resumed.dataset_id),
        ("quantization", original.quantization.value, resumed.quantization.value),
        ("lora.r", str(original.lora.r), str(resumed.lora.r)),
        ("lora.alpha", str(original.lora.alpha), str(resumed.lora.alpha)),
        (
            "lora.target_modules",
            ",".join(sorted(original.lora.target_modules)),
            ",".join(sorted(resumed.lora.target_modules)),
        ),
    ]
    return [
        ProvenanceMismatch(field, orig, new)
        for field, orig, new in checks
        if orig != new
    ]


def assert_resume_compatible(original: JobConfig, resumed: JobConfig) -> None:
    mismatches = check_resume_compatible(original, resumed)
    if mismatches:
        details = "; ".join(f"{m.field}: {m.original!r} != {m.resumed!r}" for m in mismatches)
        raise ValueError(f"refusing to resume — provenance mismatch: {details}")
