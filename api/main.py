"""FastAPI entrypoint. No auth, no multi-tenancy — a single-user control
plane that sits in front of the orchestrator.

Run with: uvicorn api.main:app --reload
"""

from fastapi import FastAPI

from api.routes import jobs, metrics

app = FastAPI(title="Fine-tuning Platform API", version="0.1.0")

app.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
app.include_router(metrics.router, prefix="/jobs", tags=["metrics"])


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
