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
    current gheim-ch-560m: small precision/recall boost on both fp32
    (+0.27pp test char-F1) and q8 ONNX (+0.43pp test char-F1, +1.4pp
    forensic-probe perfect-rate), no observed cost on either backend.
    See eval/calibration_sweep.json and eval/calibration_sweep_q8.json.
    """
    import os

    if os.getenv("GHEIM_API_KEY"):
        return RemoteDetector()
    return CalibratedDetector()
