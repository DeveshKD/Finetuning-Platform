"""Applies the job's LoRA config to a loaded (and possibly quantized) model."""

from peft import LoraConfig as PeftLoraConfig
from peft import get_peft_model

from common.job_spec import LoraConfig


def attach_lora(model, config: LoraConfig):
    peft_config = PeftLoraConfig(
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, peft_config)
