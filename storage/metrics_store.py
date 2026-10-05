"""Local SQLite metrics store. One row per (job_id, step) metric emission.
Simple enough to tail for the SSE endpoint, durable enough to survive an
API restart mid-job, and queryable enough to back the watchdog's stall
check and the per-run plot.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path("data/metrics.db")


class MetricsStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    step INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    ts REAL NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_job ON metrics(job_id)")

            conn.execute(
                """CREATE TABLE IF NOT EXISTS fragmentation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    expected_free_mb INTEGER NOT NULL,
                    actual_free_mb INTEGER NOT NULL,
                    gap_mb INTEGER NOT NULL,
                    ts REAL NOT NULL
                )"""
            )

    def append(self, job_id: str, step: int, payload: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO metrics (job_id, step, payload, ts) VALUES (?, ?, ?, ?)",
                (job_id, step, json.dumps(payload), time.time()),
            )

    def query_since(self, job_id: str, last_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, step, payload, ts FROM metrics WHERE job_id = ? AND id > ? ORDER BY id",
                (job_id, last_id),
            ).fetchall()
        return [{"id": r[0], "step": r[1], "ts": r[3], **json.loads(r[2])} for r in rows]

    def get_series(self, job_id: str) -> list[dict]:
        """Full metrics history for a job — backs the plot and any 'give me
        everything so far' UI request."""
        return self.query_since(job_id, last_id=0)

    def log_fragmentation(self, job_id: str, expected_free_mb: int, actual_free_mb: int, gap_mb: int) -> None:
        """Persist the VRAM leak/fragmentation gap after a job exits."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fragmentation_logs (job_id, expected_free_mb, actual_free_mb, gap_mb, ts) VALUES (?, ?, ?, ?, ?)",
                (job_id, expected_free_mb, actual_free_mb, gap_mb, time.time()),
            )

    def seconds_since_last_metric(self, job_id: str) -> float | None:
        """None if the job has never emitted a metric yet — the watchdog
        treats that as 'still in PREPARING', not a stall."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ts FROM metrics WHERE job_id = ? ORDER BY id DESC LIMIT 1",
                (job_id,),
            ).fetchone()
        return None if row is None else time.time() - row[0]
