from pathlib import Path
import logging

log = logging.getLogger("checkpointer")

class CheckpointManager:
    def __init__(
        self,
        job_id: str,
        every_steps: int,
        out_dir: Path = Path("data/checkpoints"),
    ):
        self.job_id = job_id
        self.every_steps = every_steps
        self.out_dir = out_dir / job_id
        self.best_metric: float | None = None

    def maybe_save(self, step: int, model) -> None:
        if step > 0 and step % self.every_steps == 0:
            self.save_explicit(model, tag=f"step-{step}")

    def maybe_save_best(self, model, metric_value: float) -> None:
        if self.best_metric is None or metric_value < self.best_metric:
            self.best_metric = metric_value
            self.save_explicit(model, tag="best")

    def find_latest(self) -> Path | None:
        if not self.out_dir.exists():
            return None
        step_dirs = [
            p for p in self.out_dir.iterdir() if p.is_dir() and p.name.startswith("step-")
        ]
        if not step_dirs:
            return None
        return max(step_dirs, key=lambda p: int(p.name.split("-", 1)[1]))

    def find_best(self) -> Path | None:
        best = self.out_dir / "best"
        return best if best.exists() else None

    def save_explicit(self, model, tag: str) -> None:
        """Saves the PEFT adapter weights."""
        target = self.out_dir / tag
        target.mkdir(parents=True, exist_ok=True)
        log.info(f"Saving PEFT adapter to {target}")
        
        # This automatically saves adapter_model.safetensors and adapter_config.json
        model.save_pretrained(target)