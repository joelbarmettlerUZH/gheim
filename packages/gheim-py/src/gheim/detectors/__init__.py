from .base import Detector, Span
from .calibrated import CalibratedDetector
from .composite import CompositeDetector, CustomPattern
from .local import LocalDetector
from .remote import RemoteDetector

__all__ = [
    "CalibratedDetector",
    "CompositeDetector",
    "CustomPattern",
    "Detector",
    "LocalDetector",
    "RemoteDetector",
    "Span",
]


def default_detector() -> Detector:
    """Pick a reasonable default: remote if a key is configured, else the
    composite detector (checksum regex front-end + calibrated local model).

    The calibrated model alone (``o_bias=0.5``) is Pareto-clean on the
    current gheim-ch-560m: +0.27pp test char-F1 with no probe cost
    (91.5% perfect at both 0.0 and 0.5). See eval/calibration_sweep.json.
    The composite adds what a token classifier cannot verify: checksummed
    IBAN / AHV / VAT / card numbers, phone numbers, labelled record
    numbers, and repeated occurrences of a name it found once. Use
    ``CalibratedDetector()`` for the model alone.
    """
    import os

    if os.getenv("GHEIM_API_KEY"):
        return RemoteDetector()
    return CompositeDetector(model=CalibratedDetector())
