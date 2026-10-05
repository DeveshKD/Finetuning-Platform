"""Exercises the VRAM gate's decision logic without needing a real GPU —
mock nvidia-smi's output and check wait_until_ready()'s behavior."""

from unittest.mock import patch

from orchestrator.resource_manager import ResourceManager


def _fake_nvidia_smi_output(free_mb: int, total_mb: int) -> bytes:
    return f"{free_mb}, {total_mb}\n".encode()


def test_ready_immediately_when_vram_already_free():
    rm = ResourceManager(min_free_mb=1000, poll_interval_s=0.01, timeout_s=1)
    with patch("subprocess.check_output", return_value=_fake_nvidia_smi_output(4000, 8000)):
        snap = rm.wait_until_ready()
    assert snap.free_mb == 4000


def test_times_out_when_vram_never_frees():
    rm = ResourceManager(min_free_mb=5000, poll_interval_s=0.01, timeout_s=0.05)
    with patch("subprocess.check_output", return_value=_fake_nvidia_smi_output(1000, 8000)):
        try:
            rm.wait_until_ready()
            assert False, "expected TimeoutError"
        except TimeoutError:
            pass
