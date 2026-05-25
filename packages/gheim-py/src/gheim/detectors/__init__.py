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

    The calibrated default (``o_bias=0.5``) is Pareto-clean on the
    current gheim-ch-560m: +0.27pp test char-F1 with no probe cost
    (91.5% perfect at both 0.0 and 0.5). See eval/calibration_sweep.json.
    """
    import os

    if os.getenv("GHEIM_API_KEY"):
        return RemoteDetector()
    return CalibratedDetector()
