"""The 33-class BIOES label space.

Single source of truth for category names, BIOES tag IDs, and the id2label /
label2id maps that the HF Trainer needs. The category names match what gheim's
`LABEL_TO_TAG` (packages/gheim-py/src/gheim/core/sentinels.py) expects, so the
fine-tuned model is a drop-in replacement.

BIOES (a.k.a. BIOLU) over 8 categories yields 4*8 + 1 = 33 classes:
  - O                     (outside any entity)
  - B-<cat>, I-<cat>, E-<cat>, S-<cat> for each of the 8 categories
"""
from __future__ import annotations

# Order matters: this MUST match the base model openai/privacy-filter's
# id2label exactly so the pretrained classifier-head weights line up. The base
# model uses alphabetical category ordering (account_number → secret).
CATEGORIES: tuple[str, ...] = (
    "account_number",
    "private_address",
    "private_date",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "secret",
)

PREFIXES: tuple[str, ...] = ("B", "I", "E", "S")

OUTSIDE = "O"


def _build_labels() -> list[str]:
    labels = [OUTSIDE]
    for cat in CATEGORIES:
        for p in PREFIXES:
            labels.append(f"{p}-{cat}")
    return labels


LABELS: list[str] = _build_labels()
NUM_LABELS: int = len(LABELS)  # 33

LABEL2ID: dict[str, int] = {lab: i for i, lab in enumerate(LABELS)}
ID2LABEL: dict[int, str] = {i: lab for lab, i in LABEL2ID.items()}


def category_of(label: str) -> str | None:
    """Return the category for a BIOES label, or None for O."""
    if label == OUTSIDE:
        return None
    return label.split("-", 1)[1]


def prefix_of(label: str) -> str | None:
    """Return B/I/E/S prefix for a label, or None for O."""
    if label == OUTSIDE:
        return None
    return label.split("-", 1)[0]


assert NUM_LABELS == 33, f"Expected 33 BIOES classes, got {NUM_LABELS}"
