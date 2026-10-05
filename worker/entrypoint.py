"""Subprocess entrypoint for a single training job.

Invoked by the orchestrator as `python -m worker.entrypoint --job-id <id>`.
Owns the full lifecycle for one job: resolve resume target (if any) ->
load -> attach adapter -> train -> checkpoint -> evaluate -> persist. Runs
in isolation so its CUDA context (and any fragmented allocator state) dies
with the process when it exits.
"""

import argparse

from common.control_signals import clear_stop
from common.job_state import JobState
from common.logging_setup import configure_logging
from common.provenance import assert_resume_compatible
from evaluation.evaluator import run_evaluation
from storage.artifact_store import ArtifactStore
from storage.plotting import generate_loss_plot
from storage.run_registry import RunRegistry
from worker.checkpoint import CheckpointManager
from worker.metrics_emitter import MetricsEmitter
from worker.model_loader import load_and_quantize
from worker.peft_setup import attach_lora
from worker.train_loop import train

log = configure_logging("worker")


def _resolve_resume_path(registry: RunRegistry, job_config):
    """Returns a checkpoint dir to resume from, or None. Raises if the
    target job's config doesn't match closely enough to trust its weights,
    or if it has no checkpoint to resume from at all."""
    if not job_config.resume_from_job_id:
        return None

    original_config = registry.get_job_config(job_config.resume_from_job_id)
    assert_resume_compatible(original_config, job_config)

    source_checkpoints = CheckpointManager(
        job_config.resume_from_job_id, every_steps=job_config.checkpoint_every_steps
    )
    resume_path = source_checkpoints.find_latest() or source_checkpoints.find_best()
    if resume_path is None:
        raise FileNotFoundError(
            f"no checkpoint found for resume_from_job_id={job_config.resume_from_job_id}"
        )
    return resume_path


def main(job_id: str) -> None:
    registry = RunRegistry()
    job_config = registry.get_job_config(job_id)

    # Clear any stale stop flag left over from a prior process for this
    # job_id (shouldn't normally exist, but a crash right after a stop
    # request could leave one behind).
    clear_stop(job_id)

    resume_path = _resolve_resume_path(registry, job_config)
    if resume_path:
        log.info("job %s: resuming from %s", job_id, resume_path)

    metrics = MetricsEmitter(job_id)

    log.info("job %s: loading + quantizing %s", job_id, job_config.base_model)
    #registry.update_state(job_id, JobState.PREPARING)
    model, tokenizer = load_and_quantize(job_config.base_model, job_config.quantization)
    model = attach_lora(model, job_config.lora)

    log.info("job %s: training", job_id)
    registry.update_state(job_id, JobState.TRAINING)
    checkpointer = CheckpointManager(job_id, every_steps=job_config.checkpoint_every_steps)
    completed_normally = train(
        model, tokenizer, job_config,
        metrics_emitter=metrics, checkpointer=checkpointer, resume_from=resume_path,
    )

    all_metrics = metrics.store.get_series(job_id)
    if all_metrics:
        final_step = all_metrics[-1]["step"]
        final_loss = all_metrics[-1]["loss"]
        # Take up to 50 data points for the sparkline
        stride = max(1, len(all_metrics) // 50)
        sparkline = [m["loss"] for m in all_metrics[::stride]]
        registry.finalize_run(job_id, final_step, final_loss, sparkline)

    if not completed_normally:
        log.info("job %s: cancelled by user, skipping evaluation", job_id)
        registry.update_state(job_id, JobState.CANCELLED)
        return

    log.info("job %s: evaluating", job_id)
    registry.update_state(job_id, JobState.EVALUATING)
    eval_result = run_evaluation(model, tokenizer, job_config)

    log.info("job %s: persisting artifacts", job_id)
    registry.update_state(job_id, JobState.PERSISTING)
    ArtifactStore().save_run(job_id, model, job_config, eval_result)
    generate_loss_plot(job_id)
    
    registry.update_state(job_id, JobState.COMPLETED)
    log.info("job %s: completed", job_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    main(args.job_id)