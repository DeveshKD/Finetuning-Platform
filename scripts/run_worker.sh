#!/usr/bin/env bash
# Manual single-job run, bypassing the scheduler — useful while developing
# the worker in isolation.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m worker.entrypoint --job-id "${1:?usage: run_worker.sh <job_id>}"
