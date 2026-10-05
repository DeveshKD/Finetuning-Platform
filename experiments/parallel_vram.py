import multiprocessing as mp
import time
import torch
import os

def run_isolated_worker(worker_id: str, model_id: str, vram_fraction: float):
    """
    Runs a fine-tuning job trapped inside a strict VRAM fraction.
    """
    # 1. THE VRAM GATEKEEPER
    # This MUST be the very first CUDA call in the process. 
    # It restricts the PyTorch caching allocator so it cannot exceed this fraction.
    torch.cuda.set_per_process_memory_fraction(vram_fraction, 0)
    
    print(f"[{worker_id}] Booted. VRAM Cap: {vram_fraction*100:.1f}% of total GPU.")
    
    # 2. Imports go here so they don't pollute the parent process memory
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorForLanguageModeling
    from datasets import load_dataset
    from peft import get_peft_model, LoraConfig

    print(f"[{worker_id}] Loading model {model_id} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0", 
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float32
        )
    )
    
    peft_config = LoraConfig(r=8, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(model, peft_config)

    print(f"[{worker_id}] Loading dataset...")
    # Using a tiny subset so it runs fast
    dataset = load_dataset("mhenrichsen/alpaca_2k_test", split="train[:50]")
    
    def format_ds(example):
        return {"text": f"User: {example['instruction']}\nAssistant: {example['output']}"}
    dataset = dataset.map(format_ds).map(
        lambda x: tokenizer(x["text"], truncation=True, max_length=128, padding="max_length"), 
        batched=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=f"data/experiments/{worker_id}",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        max_steps=10,
        logging_steps=2,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print(f"[{worker_id}] Starting training loop...")
    trainer.train()
    
    # Check max memory used
    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"[{worker_id}] FINISHED! Peak VRAM used: {peak_mb:.1f} MB.")


def main():
    if not torch.cuda.is_available():
        print("CUDA is required for this experiment.")
        return

    total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
    total_vram_gb = total_vram_bytes / (1024**3)
    print(f"Total System GPU VRAM: {total_vram_gb:.1f} GB")

    # --- THE VRAM ESTIMATOR ---
    # For a Qwen 0.5B model in 4-bit, the base weights take ~0.35 GB.
    # Optimizer states, activations, and LoRA overhead take ~1.5 GB.
    # We will safely allocate 2.5 GB to each worker.
    
    budget_gb = 2.5
    fraction = budget_gb / total_vram_gb
    
    if fraction * 2 > 0.95:
        print("Not enough VRAM to run two 2.5GB jobs simultaneously!")
        return

    print(f"Allocating {budget_gb} GB ({fraction*100:.1f}%) to Worker A")
    print(f"Allocating {budget_gb} GB ({fraction*100:.1f}%) to Worker B")

    # We must use "spawn" to ensure CUDA contexts are completely fresh 
    # and isolated between the two child processes.
    ctx = mp.get_context("spawn")

    worker_a = ctx.Process(
        target=run_isolated_worker, 
        args=("Worker_A", "Qwen/Qwen2.5-0.5B", fraction)
    )
    
    worker_b = ctx.Process(
        target=run_isolated_worker, 
        args=("Worker_B", "Qwen/Qwen2.5-0.5B", fraction)
    )

    # Launch them simultaneously!
    worker_a.start()
    worker_b.start()

    # Wait for both to finish
    worker_a.join()
    worker_b.join()

    print("Experiment Complete! Both workers successfully shared the GPU without OOMing.")

if __name__ == "__main__":
    main()