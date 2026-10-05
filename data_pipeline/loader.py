import random
from datasets import load_dataset as hf_load_dataset

REQUIRED_FIELDS = {"instruction", "response"}

MIN_EVAL_ROWS = 16
MAX_EVAL_ROWS = 128
EVAL_SPLIT_SEED = 13

def validate_schema(record: dict) -> None:
    missing = REQUIRED_FIELDS - record.keys()
    if missing:
        raise ValueError(f"dataset record missing required fields: {missing}")

def compute_eval_split_size(num_rows: int, holdout_fraction: float) -> int:
    if num_rows < MIN_EVAL_ROWS:
        return 0
    size = int(num_rows * holdout_fraction)
    return max(MIN_EVAL_ROWS, min(size, MAX_EVAL_ROWS))

def split_eval_indices(num_rows: int, holdout_fraction: float) -> tuple[list[int], list[int]]:
    eval_size = compute_eval_split_size(num_rows, holdout_fraction)
    indices = list(range(num_rows))
    random.Random(EVAL_SPLIT_SEED).shuffle(indices)
    eval_indices = sorted(indices[:eval_size])
    train_indices = sorted(indices[eval_size:])
    return train_indices, eval_indices

def get_response_template(model_name: str) -> str:
    """Returns the exact string that precedes the assistant's response."""
    name = model_name.lower()
    if "gemma" in name:
        return "<start_of_turn>model\n"
    elif "llama-3" in name or "llama3" in name:
        return "<|start_header_id|>assistant<|end_header_id|>\n\n"
    elif "qwen" in name:
        return "<|im_start|>assistant\n"
    elif "phi-3" in name or "phi3" in name:
        return "<|assistant|>\n"
    elif "mistral" in name or "mixtral" in name:
        return "[/INST]"
    else:
        raise ValueError(f"Unsupported model family: {model_name}")

def load_dataset(dataset_id: str, tokenizer, model_name: str):
    dataset = hf_load_dataset(dataset_id, split="train")
    
    # 1. Get the template string and tokenize it so we can search for it
    response_template = get_response_template(model_name)
    # add_special_tokens=False is crucial so we don't accidentally add BOS tokens to our search array
    response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def format_tokenize_and_mask(example):
        instruction = example.get("instruction", example.get("prompt", ""))
        context = example.get("input", "")
        response = example.get("output", example.get("response", ""))

        user_prompt = f"{instruction}\n\n{context}" if context else instruction

        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": response}
        ]

        # Apply native chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        
        # Tokenize the full conversation
        tokenized = tokenizer(text, truncation=True, max_length=2048)
        input_ids = tokenized["input_ids"]
        
        # Create labels array (PyTorch uses this for loss calculation)
        labels = input_ids.copy()
        
        # Find the response template in the input_ids
        match_idx = -1
        temp_len = len(response_token_ids)
        for i in range(len(input_ids) - temp_len + 1):
            if input_ids[i:i+temp_len] == response_token_ids:
                match_idx = i + temp_len
                break
        
        # Mask everything before the assistant's response with -100
        if match_idx != -1:
            labels[:match_idx] = [-100] * match_idx
        else:
            # Safety fallback: if the template wasn't found, mask everything 
            # so we don't accidentally train on garbage data.
            labels = [-100] * len(labels)
            
        tokenized["labels"] = labels
        tokenized["text"] = text
        return tokenized

    # Map our new function over the dataset
    formatted_dataset = dataset.map(format_tokenize_and_mask)
    return formatted_dataset

def load_eval_split(dataset_id: str, holdout_fraction: float, tokenizer, model_name: str):
    dataset = load_dataset(dataset_id, tokenizer, model_name)
    if holdout_fraction <= 0:
        return dataset, None
        
    # We still have our existing split logic!
    from data_pipeline.loader import split_eval_indices
    train_idx, eval_idx = split_eval_indices(len(dataset), holdout_fraction)
    return dataset.select(train_idx), dataset.select(eval_idx)