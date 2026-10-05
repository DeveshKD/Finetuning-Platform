"""Loads and quantizes the base model.

Wraps HF `from_pretrained` with the bitsandbytes config matching the job's
requested quantization mode, so quant behavior lives in exactly one place.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.job_spec import QuantizationMode


def _bnb_config(mode: QuantizationMode) -> BitsAndBytesConfig | None:
    if mode == QuantizationMode.NF4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="bfloat16",
        )
    if mode == QuantizationMode.INT8:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_and_quantize(base_model: str, mode: QuantizationMode):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires a CUDA-enabled PyTorch installation and an available GPU. "
            f"Detected torch {torch.__version__} with CUDA unavailable."
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=_bnb_config(mode),
        device_map="auto",
    )
    # TODO: peft.prepare_model_for_kbit_training(model) when mode != NONE
    return model, tokenizer
