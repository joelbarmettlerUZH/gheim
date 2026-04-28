"""API key auth — disabled if ``GHEIM_API_KEYS`` is not set."""
from __future__ import annotations

import os

from fastapi import HTTPException, Request, status


def _allowed_keys() -> set[str]:
    raw = os.getenv("GHEIM_API_KEYS", "").strip()
    if not raw:
        return set()
    return {k.strip() for k in raw.split(",") if k.strip()}


def require_api_key(request: Request) -> None:
    """Dependency: pass if no keys configured (open mode), otherwise require Bearer."""
    keys = _allowed_keys()
    if not keys:
        return
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = auth[7:].strip()
    if token not in keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid api key",
        )
