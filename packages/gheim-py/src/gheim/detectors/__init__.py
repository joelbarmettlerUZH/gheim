from .base import Detector, Span
from .calibrated import CalibratedDetector
from .local import LocalDetector
from .remote import RemoteDetector

__all__ = [
    "CalibratedDetector",
    "Detector",
    "LocalDetector",
    "RemoteDetector",
    "Span",
]


def default_detector() -> Detector:
    """Pick a reasonable default: remote if a key is configured, else
    a calibrated local model.

    The calibrated default (``o_bias=0.5``) recovers ~95% of the
    multi-entity-letter cases that the raw v1 model fails on, with
    ~0.4pp test_v1 strict-F1 cost. See eval/calibration_sweep.json.
    """
    import os

    if os.getenv("GHEIM_API_KEY"):
        return RemoteDetector()
    return CalibratedDetector()
