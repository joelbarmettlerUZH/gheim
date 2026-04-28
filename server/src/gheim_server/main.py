"""FastAPI app exposing POST /v1/detect."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .auth import require_api_key
from .registry import DEFAULT_MODEL, registry


class DetectRequest(BaseModel):
    text: str = Field(..., min_length=0)
    model: str = Field(default=DEFAULT_MODEL)


class DetectedSpan(BaseModel):
    label: str
    start: int
    end: int
    score: float


class DetectResponse(BaseModel):
    model: str
    spans: list[DetectedSpan]
    elapsed_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up the default model unless explicitly disabled.
    if os.getenv("GHEIM_SKIP_WARMUP", "").lower() not in ("1", "true", "yes"):
        registry.warmup()
    yield


app = FastAPI(
    title="gheim-server",
    version=__version__,
    description="PII detection HTTP server.",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/detect", response_model=DetectResponse)
def detect(req: DetectRequest, _: None = Depends(require_api_key)) -> DetectResponse:
    try:
        pipe = registry.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not req.text:
        return DetectResponse(model=req.model, spans=[], elapsed_ms=0.0)

    t0 = time.perf_counter()
    raw: list[dict[str, Any]] = pipe(req.text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    spans = [
        DetectedSpan(
            label=str(item.get("entity_group") or item.get("entity")),
            start=int(item["start"]),
            end=int(item["end"]),
            score=float(item.get("score", 1.0)),
        )
        for item in raw
    ]
    return DetectResponse(model=req.model, spans=spans, elapsed_ms=elapsed_ms)


def run() -> None:
    """Console-script entry point: ``gheim-server`` runs uvicorn."""
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("gheim_server.main:app", host=host, port=port, log_level="info")
