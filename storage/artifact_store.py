"""Persists the final artifacts for a completed run: adapter weights, the
exact JobConfig used, and the eval results — everything needed to
reproduce or audit the run later.
"""

import json
from pathlib import Path

from common.job_spec import JobConfig
from evaluation.evaluator import EvalResult

RUNS_DIR = Path("data/runs")


class ArtifactStore:
    def save_run(
        self, job_id: str, model, job_config: JobConfig, eval_result: EvalResult
    ) -> None:
        run_dir = RUNS_DIR / job_id
        run_dir.mkdir(parents=True, exist_ok=True)

        print(f"Saving final PEFT adapter to {run_dir / 'adapter'}")
        model.save_pretrained(run_dir / "adapter")
        
        (run_dir / "config.json").write_text(job_config.model_dump_json(indent=2))
        (run_dir / "eval.json").write_text(json.dumps(eval_result.__dict__, indent=2))
