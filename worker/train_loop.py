import torch
from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, TrainerCallback
from pathlib import Path

from common.control_signals import clear_stop, is_stop_requested
from common.job_spec import JobConfig
from common.logging_setup import configure_logging
from worker.checkpoint import CheckpointManager
from worker.metrics_emitter import MetricsEmitter

log = configure_logging("train_loop")

class PlatformIntegrationCallback(TrainerCallback):
    """
    Hooks into the Hugging Face Trainer to emit metrics to SQLite and 
    check for API-driven stop signals.
    """
    def __init__(self, job_id: str, metrics_emitter: MetricsEmitter, checkpointer: CheckpointManager):
        self.job_id = job_id
        self.metrics_emitter = metrics_emitter
        self.checkpointer = checkpointer
        self.stopped_early = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        
        # Capture standard metrics
        loss = logs.get("loss")
        lr = logs.get("learning_rate")
        
        if loss is not None:
            # Capture VRAM fragmentation/usage if running on CUDA
            vram_mb = 0
            if torch.cuda.is_available():
                # Peak memory used by tensors during this logging step
                vram_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)
            
            self.metrics_emitter.emit(
                step=state.global_step,
                loss=loss,
                lr=lr,
                vram_mb=vram_mb
            )

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        model = kwargs["model"]
        
        self.checkpointer.maybe_save(step, model)
        
        if is_stop_requested(self.job_id):
            log.warning(f"Stop requested for job {self.job_id} at step {step}. Halting safely.")
            self.checkpointer.save_explicit(model, tag=f"step-{step}-stopped")
            clear_stop(self.job_id)
            
            self.stopped_early = True  # <--- MARK AS STOPPED
            control.should_training_stop = True

def get_response_template(model_name: str) -> str:
    """
    Returns the exact string that precedes the assistant's response for a given model family.
    The DataCollator uses this to find where to stop masking the user prompt with -100.
    """
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
        raise ValueError(f"Unsupported model family for completion-only masking: {model_name}")

def train(
    model,
    tokenizer,
    job_config: JobConfig,
    metrics_emitter: MetricsEmitter,
    checkpointer: CheckpointManager,
    resume_from: Path | None = None,
) -> bool:
    
    # 1. Load Dataset
    from data_pipeline.loader import load_dataset, load_eval_split
    
    # Normally we'd do: train_dataset = load_dataset(job_config.dataset_id)
    # But for a fast split, we'll just load the splits directly
    train_dataset, eval_dataset = load_eval_split(
        job_config.dataset_id, 
        job_config.eval.holdout_fraction,
        tokenizer,
        job_config.base_model
    )
    log.info(f"Loaded dataset: {len(train_dataset)} train rows, {len(eval_dataset) if eval_dataset else 0} eval rows.")

    # 2. Setup Data Collator
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # 3. Configure Training Arguments
    schedule = job_config.schedule
    args = TrainingArguments(
        output_dir=str(checkpointer.out_dir),
        per_device_train_batch_size=schedule.batch_size,
        gradient_accumulation_steps=schedule.gradient_accumulation_steps,
        learning_rate=schedule.learning_rate,
        lr_scheduler_type=schedule.lr_scheduler,
        num_train_epochs=schedule.num_epochs or 3.0,
        max_steps=schedule.max_steps or -1,
        warmup_steps=schedule.warmup_steps,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        bf16=True if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else False,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=True,
    )

    # 4. Initialize Trainer
    callback = PlatformIntegrationCallback(job_config.job_id, metrics_emitter, checkpointer)
    
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=args,
        data_collator=data_collator,
        callbacks=[callback],
    )

    # 5. Execute Training
    log.info("Starting training loop...")
    trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    return not callback.stopped_early