"""Test configuration for gheim-py.

Live tests (those marked ``@pytest.mark.live``) hit real models / network and are
skipped by default. Opt in by setting ``GHEIM_RUN_LIVE=1`` in the environment.
"""
from __future__ import annotations

import os

import pytest

LIVE_ENABLED = os.getenv("GHEIM_RUN_LIVE", "").lower() in ("1", "true", "yes")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: hits the real openai/privacy-filter model (downloads weights, slow). "
        "Skipped unless GHEIM_RUN_LIVE=1 is set.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if LIVE_ENABLED:
        return
    skip_live = pytest.mark.skip(reason="set GHEIM_RUN_LIVE=1 to run real-model tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
