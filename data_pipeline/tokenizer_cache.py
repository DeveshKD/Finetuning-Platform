"""Tokenizes a dataset once and caches the result, keyed by
(dataset_id, tokenizer name) so repeated runs against the same base model
skip re-tokenization.
"""

from pathlib import Path

CACHE_DIR = Path("data/tokenized_cache")


def tokenize_and_cache(dataset, tokenizer, dataset_id: str):
    cache_key = f"{dataset_id}__{tokenizer.name_or_path.replace('/', '_')}"
    cache_path = CACHE_DIR / cache_key
    if cache_path.exists():
        # TODO: load cached tokenized dataset from disk (e.g. datasets.load_from_disk)
        pass
    # TODO: tokenize, dataset.save_to_disk(cache_path), return
    raise NotImplementedError
