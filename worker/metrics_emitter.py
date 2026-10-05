"""Writes per-step metrics so the API can stream them live.

Backed by the same SQLite metrics store the API's SSE endpoint reads from —
the worker writes, the API tails. No message broker needed at this scale.
"""

from storage.metrics_store import MetricsStore


class MetricsEmitter:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.store = MetricsStore()

    def emit(self, step: int, **metrics) -> None:
        self.store.append(self.job_id, step, metrics)
