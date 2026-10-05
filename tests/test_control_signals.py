"""File-based stop signal — the mechanism the /stop endpoint and the
training loop's per-step check are built on."""

from common.control_signals import clear_stop, is_stop_requested, request_stop


def test_not_requested_by_default(tmp_path):
    assert is_stop_requested("job-1", control_dir=tmp_path) is False


def test_request_then_clear(tmp_path):
    request_stop("job-1", control_dir=tmp_path)
    assert is_stop_requested("job-1", control_dir=tmp_path) is True

    clear_stop("job-1", control_dir=tmp_path)
    assert is_stop_requested("job-1", control_dir=tmp_path) is False


def test_clear_is_safe_when_nothing_to_clear(tmp_path):
    clear_stop("job-1", control_dir=tmp_path)  # should not raise
