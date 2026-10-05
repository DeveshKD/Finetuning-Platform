"""Defines JobConfig — the single source of truth for a fine-tuning run.

Everything the worker needs to reproduce a run (model, dataset, LoRA/quant
hyperparameters, training schedule, eval config) lives here as one validated
pydantic model. The API accepts this shape, the orchestrator queues it
untouched, and the worker deserializes it directly. No translation layer
between "what the user configured" and "what the worker runs".
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class QuantizationMode(str, Enum):
    NONE = "none"
    INT8 = "int8"
    NF4 = "nf4"  # QLoRA-style 4-bit


class LoraConfig(BaseModel):
    r: int = Field(16, ge=1, le=256, description="LoRA rank")
    alpha: int = Field(32, ge=1)
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


class TrainingSchedule(BaseModel):
    batch_size: int = Field(4, ge=1)
    gradient_accumulation_steps: int = Field(4, ge=1)
    learning_rate: float = Field(2e-4, gt=0)
    lr_scheduler: str = "cosine"
    num_epochs: Optional[int] = 3
    max_steps: Optional[int] = None
    warmup_steps: float = 0.03

    @model_validator(mode="after")
    def _require_one_stop_condition(self) -> "TrainingSchedule":
        if (self.num_epochs is None) == (self.max_steps is None):
            raise ValueError("set exactly one of num_epochs or max_steps")
        return self


class EvalConfig(BaseModel):
    run_perplexity: bool = True
    run_llm_judge: bool = True
    judge_model: str = "gemini-3.8-flash"  # or a locally hosted judge
    eval_dataset_id: Optional[str] = None
    holdout_fraction: float = 0.05


class JobConfig(BaseModel):
    job_id: str
    base_model: str
    dataset_id: str
    quantization: QuantizationMode = QuantizationMode.NF4
    lora: LoraConfig = Field(default_factory=LoraConfig)
    schedule: TrainingSchedule = Field(default_factory=TrainingSchedule)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    checkpoint_every_steps: int = 200
    resume_from_job_id: Optional[str] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def _resume_target_must_differ(self) -> "JobConfig":
        if self.resume_from_job_id == self.job_id:
            raise ValueError("resume_from_job_id cannot be the job's own job_id")
        return self
