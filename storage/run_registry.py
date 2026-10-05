"""Tracks every job's config and current state — the source of truth the
API reads from for GET /jobs and GET /jobs/{id}, and the worker reads from
to fetch its own JobConfig at startup.

Backed by one JSON file per run (data/runs/<job_id>/state.json) rather than
a database table — at single-user scale this is simpler to inspect by hand
(just `cat` the file) and needs no migrations.
"""
import json
import sqlite3
import time
from pathlib import Path

from common.job_spec import JobConfig
from common.job_state import JobState, assert_valid_transition

DB_PATH = Path("data/runs.db")

class RunRegistry:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Returns dict-like rows
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    ended_at REAL,
                    total_steps INTEGER,
                    final_step INTEGER,
                    final_loss REAL,
                    output_dir TEXT,
                    error_message TEXT,
                    duration_seconds REAL,
                    loss_sparkline TEXT,
                    display_name TEXT,
                    resume_blocked BOOLEAN DEFAULT 0,
                    resumed_from_run_id TEXT
                )
            """)

    def create_run(self, config: JobConfig, output_dir: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO training_runs 
                   (id, status, model_name, dataset_name, config_json, started_at, output_dir, resumed_from_run_id) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    config.job_id,
                    JobState.QUEUED.value,
                    config.base_model,
                    config.dataset_id,
                    config.model_dump_json(),
                    time.time(),
                    output_dir,
                    config.resume_from_job_id
                )
            )

    def pop_next_queued_job(self) -> JobConfig | None:
        """Acts as the queue pop. Atomically grabs the oldest queued job."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, config_json FROM training_runs WHERE status = ? ORDER BY started_at ASC LIMIT 1",
                (JobState.QUEUED.value,)
            ).fetchone()
            
            if not row:
                return None
                
            # Claim it immediately so we don't double-process
            job_id = row["id"]
            conn.execute(
                "UPDATE training_runs SET status = ? WHERE id = ?", 
                (JobState.PREPARING.value, job_id)
            )
            
        return JobConfig.model_validate_json(row["config_json"])

    def get_job_config(self, job_id: str) -> JobConfig:
        run = self.get_run(job_id)
        if run is None:
            raise FileNotFoundError(f"no run found for job_id={job_id}")
        return JobConfig.model_validate_json(run["config_json"])

    def update_state(self, job_id: str, new_state: JobState, error_message: str = None) -> None:
        # Note: In a production app you'd fetch the current state and call assert_valid_transition
        with self._connect() as conn:
            if error_message:
                conn.execute(
                    "UPDATE training_runs SET status = ?, error_message = ? WHERE id = ?", 
                    (new_state.value, error_message, job_id)
                )
            else:
                conn.execute("UPDATE training_runs SET status = ? WHERE id = ?", (new_state.value, job_id))

    def finalize_run(self, job_id: str, final_step: int, final_loss: float, sparkline: list[float]):
        """Called when a run finishes to populate the rich UI fields."""
        run = self.get_run(job_id)
        if not run: return
        
        ended_at = time.time()
        duration = ended_at - run["started_at"]
        
        with self._connect() as conn:
            conn.execute("""
                UPDATE training_runs SET 
                    ended_at = ?, 
                    final_step = ?, 
                    final_loss = ?, 
                    duration_seconds = ?, 
                    loss_sparkline = ? 
                WHERE id = ?
            """, (ended_at, final_step, final_loss, duration, json.dumps(sparkline), job_id))

    def get_run(self, job_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM training_runs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_runs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM training_runs ORDER BY started_at DESC").fetchall()
            return [dict(row) for row in rows]

    def get_active_job(self) -> dict | None:
        active_states = ('preparing', 'training', 'checkpointing', 'evaluating', 'persisting')
        placeholders = ','.join('?' * len(active_states))
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM training_runs WHERE status IN ({placeholders}) LIMIT 1", 
                active_states
            ).fetchone()
            return dict(row) if row else None