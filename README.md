# Fine-tuning platform

A strict fine-tuning-only platform for LoRA/QLoRA training on a single GPU.
No auth, no multi-tenancy, no data recipes, no RAG/comparison chat, no
multimodal generation — one job type, done well: base model + dataset +
config in, an evaluated adapter out. Six extra features (watchdog, stop,
resume, dataset validation, idempotency, loss/VRAM plot) were added after
comparing this against unsloth-studio's training backend.

## Layout

```
api/              FastAPI control plane — submit/list/get/stop jobs, live metrics, plot, active-job check
orchestrator/     Job queue + VRAM-aware resource gate + stall watchdog + the scheduler loop that ties them together
worker/           Runs as an isolated subprocess, one per job: resolve resume -> load -> LoRA -> train -> checkpoint -> evaluate -> persist
evaluation/       Perplexity + LLM-as-judge, run automatically after training
data_pipeline/    Dataset ingestion, schema validation, deterministic eval-split sizing, tokenization caching
storage/          Run registry (job state + config + eval), metrics store (SQLite), artifact store, loss/VRAM plot generator
common/           Shared contracts: JobConfig, the job state machine, the resume-provenance check, the file-based stop signal
configs/          Example job submission payload
scripts/          Launchers for the API, scheduler, and a standalone worker run
tests/            38 tests — everything below that doesn't need a GPU is actually exercised
ui/               Not built yet — see ui/README.md for why and what it'd take
data/             Gitignored local storage: queue.db, metrics.db, runs/, checkpoints/, control/, tokenized_cache/
```

## How a job flows through the system

1. `POST /jobs` with a `JobConfig` — the API rejects a repeated `job_id`
   (idempotency), registers the run, and enqueues it.
2. The scheduler (`orchestrator/scheduler.py`) pops the next job, blocks on
   `ResourceManager.wait_until_ready()` until VRAM is verifiably free, then
   spawns `worker/entrypoint.py` as its own subprocess (`Popen`, not
   `run`) and starts a `StallWatchdog` alongside it.
3. If the job's `resume_from_job_id` is set, the worker first checks
   provenance (`common/provenance.py`) against the original run's config,
   then locates its latest/best checkpoint — refusing to resume on any
   mismatch (base model, dataset, quantization, LoRA shape).
4. The worker loads and quantizes the base model, attaches the LoRA
   adapter, and trains — emitting a metrics row every step, checkpointing
   periodically, and checking the file-based stop flag
   (`common/control_signals.py`) each step so `POST /jobs/{id}/stop` can
   land mid-run.
5. If training stalls (state stays TRAINING but no metric lands within the
   watchdog's timeout — a hang, not a crash), the watchdog terminates the
   subprocess and marks the job FAILED so the GPU isn't held forever.
6. On completion: evaluate (perplexity + LLM-as-judge), persist the
   adapter + config + eval results, and render the loss/VRAM plot.
   `GET /jobs/{id}/stream` (SSE) and `/metrics` (full series) read from the
   same metrics store live; `/plot` renders the PNG on demand.

## What's real vs. stubbed

Actually implemented and tested (38 tests, `pytest -q`, no GPU needed):

- `common/job_spec.py`, `job_state.py`, `provenance.py`, `control_signals.py`
- `orchestrator/queue.py`, `resource_manager.py`, `watchdog.py`
- `storage/run_registry.py`, `metrics_store.py`
- `data_pipeline/loader.py`'s eval-split sizing (`compute_eval_split_size`, `split_eval_indices`)
- `worker/checkpoint.py`'s `find_latest`/`find_best` lookup
- `worker/model_loader.py`, `peft_setup.py` — real quantization + LoRA wiring against `transformers`/`peft`/`bitsandbytes` (untested here since that needs the real libraries + a GPU)

Still a typed interface with a `TODO` where the logic goes:

- `worker/train_loop.py` — the actual `Trainer` wiring (steps 4–6 in the flow above)
- `data_pipeline/loader.py::load_dataset` / `load_eval_split` — the actual dataset load (the *sizing logic* they'll call is done and tested)
- `data_pipeline/tokenizer_cache.py`
- `evaluation/llm_judge.py`
- The `model.save_pretrained(...)` / adapter-loading calls in `checkpoint.py` and `train_loop.py`

## Where to start

Everything in "actually implemented" above already works and is tested —
you don't need to touch it before the call. Build in this order:

1. **`worker/train_loop.py`** — the one real gap left. Wire an actual
   `transformers.Trainer` (or `SFTTrainer`) using your existing 4-dataset
   training data, with a step-end callback that calls
   `metrics_emitter.emit(...)`, `checkpointer.maybe_save(...)`, and checks
   `is_stop_requested(...)`. This is the only piece that has to exist for
   `make worker JOB_ID=<id>` to run an actual training step end to end —
   everything upstream (queue, registry, resource gate, watchdog) and
   downstream (checkpoint/resume/plot) is already wired to call it.
2. **`data_pipeline/loader.py::load_dataset`** — load your existing
   datasets and run them through `validate_schema`. `load_eval_split` then
   falls out almost for free: call `split_eval_indices(len(dataset), ...)`
   (already implemented and tested) and `.select()` the two halves.
3. **`evaluation/llm_judge.py`** — port your existing LLM-as-judge prompt
   and scoring logic in; `evaluation/evaluator.py` already calls it in the
   right place.
4. **The two `model.save_pretrained(...)` calls** in `checkpoint.py`
   (`_save`) and `train_loop.py` (loading `resume_from`) — small, but
   nothing round-trips a real adapter to disk until these exist.

If the call is close and you have to cut something, cut (4) last-minute
and demo with checkpointing/resume as "wired but not yet round-tripping
real weights" — the queue → resource gate → watchdog → stop-signal flow
still runs and is the part worth showing live.

## Setup

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill in HF_TOKEN / JUDGE_MODEL_API_KEY if needed
make test              # 38 tests, all passing, no GPU needed
make api                       # starts the FastAPI control plane
make scheduler                 # starts the orchestrator loop (separate terminal)
make worker JOB_ID=<job_id>    # or run a single job manually, bypassing the scheduler
```
