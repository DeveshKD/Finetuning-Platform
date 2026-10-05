"""The eval split has to be deterministic — a resumed run and its original
must score against the same held-out rows for eval numbers to be comparable."""

from data_pipeline.loader import compute_eval_split_size, split_eval_indices


def test_tiny_dataset_gets_no_eval_split():
    assert compute_eval_split_size(10, holdout_fraction=0.05) == 0


def test_eval_size_is_floored_and_capped():
    assert compute_eval_split_size(100, holdout_fraction=0.05) == 16   # floor
    assert compute_eval_split_size(100_000, holdout_fraction=0.05) == 128  # cap


def test_split_is_deterministic_across_calls():
    a = split_eval_indices(500, holdout_fraction=0.05)
    b = split_eval_indices(500, holdout_fraction=0.05)
    assert a == b


def test_split_covers_every_row_exactly_once():
    train_idx, eval_idx = split_eval_indices(200, holdout_fraction=0.1)
    assert sorted(train_idx + eval_idx) == list(range(200))
    assert set(train_idx).isdisjoint(eval_idx)
