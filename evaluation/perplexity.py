"""Perplexity on a held-out split."""
import math
import torch

def compute_perplexity(model, tokenizer, eval_texts: list[str]) -> float:
    if not eval_texts:
        return None
    losses = []
    model.eval()
    with torch.no_grad():
        for text in eval_texts:
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            out = model(**inputs, labels=inputs["input_ids"])
            losses.append(out.loss.item())
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)