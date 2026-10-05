"""Live metrics streaming, full-series lookup, and the loss/VRAM plot.

GET /jobs/{id}/stream  — Server-Sent Events tailing the metrics store, for a
                          live dashboard's loss/VRAM graph during training
GET /jobs/{id}/metrics — the full series so far, for a one-shot chart or a
                          quick sanity check without opening an SSE connection
GET /jobs/{id}/plot    — renders (or re-renders) the loss/VRAM PNG
"""

import asyncio
import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from storage.metrics_store import MetricsStore
from storage.plotting import generate_loss_plot

router = APIRouter()
_store = MetricsStore()


@router.get("/{job_id}/stream")
async def stream_metrics(job_id: str):
    async def event_generator():
        last_seen_id = 0
        while True:
            rows = _store.query_since(job_id, last_seen_id)
            for row in rows:
                last_seen_id = row["id"]
                yield f"data: {json.dumps(row)}\n\n"
            # TODO: break the loop once the job registry reports a terminal state
            await asyncio.sleep(1.0)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/{job_id}/metrics")
def get_metrics_series(job_id: str) -> list[dict]:
    return _store.get_series(job_id)


@router.get("/{job_id}/plot")
def get_loss_plot(job_id: str):
    path = generate_loss_plot(job_id)
    if path is None:
        raise HTTPException(status_code=404, detail="no metrics recorded for this job yet")
    return FileResponse(path, media_type="image/png")
