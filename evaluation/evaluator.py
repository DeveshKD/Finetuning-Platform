from dataclasses import dataclass
import torch
from common.job_spec import JobConfig
from evaluation.llm_judge import LLMJudge
from evaluation.perplexity import compute_perplexity
from common.logging_setup import configure_logging

log = configure_logging("evaluator")

@dataclass
class EvalResult:
    perplexity: float | None
    judge_scores: list[float]
    judge_mean: float | None

def run_evaluation(model, tokenizer, job_config: JobConfig) -> EvalResult:
    from data_pipeline.loader import load_eval_split
    
    # 1. Load the hold-out evaluation dataset
    _, eval_dataset = load_eval_split(
        job_config.dataset_id, 
        job_config.eval.holdout_fraction,
        tokenizer,
        job_config.base_model
    )
    
    if not eval_dataset or len(eval_dataset) == 0:
        log.warning("No eval dataset available. Skipping evaluation.")
        return EvalResult(None, [], None)

    # We will only evaluate up to 10 samples to save API costs and time
    eval_samples = eval_dataset.select(range(min(10, len(eval_dataset))))
    
    eval_texts = []
    judge_scores = []
    
    log.info(f"Generating responses for {len(eval_samples)} eval samples...")
    model.eval()
    
    # Initialize the LLM Judge
    judge = LLMJudge(job_config.eval.judge_model)

    with torch.no_grad():
        for example in eval_samples:
            # We need the raw prompt without the target response
            instruction = example.get("instruction", example.get("prompt", ""))
            context = example.get("input", "")
            reference = example.get("output", example.get("response", ""))
            
            user_prompt = f"{instruction}\n\n{context}" if context else instruction
            messages = [{"role": "user", "content": user_prompt}]
            
            # Format and tokenize
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            # Calculate perplexity text (full formatted string including response)
            eval_texts.append(example["text"])

            if job_config.eval.run_llm_judge:
                # Generate candidate response
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=256,                     # Give it more room to speak
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,                # Strongly discourage repeating tokens
                    temperature=0.7,                        # Add a little entropy to break loops
                    top_p=0.9,
                    do_sample=True                          # Required for temperature/top_p
                )
                
                candidate = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Ask the Judge to score it
                verdict = judge.score(user_prompt, candidate, reference)
                if verdict.score is not None:
                    log.info(f"Eval Score: {verdict.score}/10 | Rationale: {verdict.rationale}")
                    judge_scores.append(verdict.score)

    # Calculate final perplexity
    ppl = compute_perplexity(model, tokenizer, eval_texts) if job_config.eval.run_perplexity else None
    
    return EvalResult(
        perplexity=ppl,
        judge_scores=judge_scores,
        judge_mean=(sum(judge_scores) / len(judge_scores)) if judge_scores else None,
    )