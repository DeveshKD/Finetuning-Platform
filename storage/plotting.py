"""Per-run loss/VRAM plot.

Cheap demo polish: once a run has metrics, render its loss curve — and the
VRAM series, which is the direct output of the fragmentation question this
project set out to measure — as a PNG under the run's directory.

matplotlib is imported inside the function, not at module level, so this
file (and everything that imports it) stays importable even before
matplotlib is installed — only calling generate_loss_plot() needs it.
"""

from pathlib import Path

from storage.metrics_store import MetricsStore

RUNS_DIR = Path("data/runs")


def generate_loss_plot(job_id: str, out_dir: Path = RUNS_DIR) -> Path | None:
    series = MetricsStore().get_series(job_id)
    if not series:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [row["step"] for row in series]
    losses = [row.get("loss") for row in series]
    vram = [row.get("vram_mb") for row in series]

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, losses, color="tab:blue", label="loss")
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss", color="tab:blue")

    if any(v is not None for v in vram):
        ax2 = ax1.twinx()
        ax2.plot(steps, vram, color="tab:orange", alpha=0.6, label="VRAM (MB)")
        ax2.set_ylabel("VRAM (MB)", color="tab:orange")

    fig.tight_layout()
    out_path = out_dir / job_id / "loss_plot.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
