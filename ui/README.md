# UI (not scaffolded yet)

Deliberately left un-built for now the call is about the training
backend

When it's time to build this, it's a thin client over the existing API:

- **Job form** : POST /jobs (fields mirror `common/job_spec.py::JobConfig`)
- **Live dashboard** : subscribe to GET /jobs/{id}/stream (SSE), plot loss
  and VRAM over step
- **Run history** : GET /jobs, table of state + eval score
