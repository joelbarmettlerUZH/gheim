from .base import Detector, Span
from .local import LocalDetector
from .remote import RemoteDetector

__all__ = ["Detector", "LocalDetector", "RemoteDetector", "Span"]


def default_detector() -> Detector:
    """Pick a reasonable default: remote if a key is configured, else local."""
    import os

    if os.getenv("GHEIM_API_KEY"):
        return RemoteDetector()
    return LocalDetector()
