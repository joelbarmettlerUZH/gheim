"""Language detection for Swiss-PII training data.

Uses Facebook's fasttext language identification model (lid218e via
``facebook/fasttext-language-identification``) — the standard
production-grade language identifier supporting 218 languages including
Romansh (``roh_Latn``). Replaces an earlier hand-tuned lingua + Romansh
keyword heuristic that had to be calibrated against false positives.

Output codes match our schema's ``language`` field:
  de_ch, fr_ch, it_ch, rm, gsw, en

Caveats:
  - GSW (Swiss German dialect, written form) is not distinct from standard
    German for any production LID — both fasttext and lingua collapse it
    to ``deu_Latn``. Detection cannot recover GSW; if a chunk's source
    path indicates GSW (e.g. a dedicated GSW corpus), pass that as
    ``subset`` so we can route it explicitly.
  - Very short text (< ~20 chars) gives noisy LID predictions; we fall
    back to the ``subset``-implied default in that regime.
  - The romansh subset of apertus-pretrain-swiss is identified by source
    path. We trust the path over fasttext when the path says romansh AND
    the text is too short for confident LID.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from .schema import Language

# fasttext lid218e returns NLLB-style language codes. Map the ones we
# care about to gheim's Language codes. Any other label (Spanish, Polish,
# etc.) falls through to subset default.
_NLLB_TO_GHEIM: dict[str, Language] = {
    "deu_Latn": "de_ch",
    "fra_Latn": "fr_ch",
    "ita_Latn": "it_ch",
    "roh_Latn": "rm",
    "eng_Latn": "en",
}

# Below this fasttext confidence we treat the verdict as unreliable and
# fall back to subset default. fasttext is sharply confident on text >=
# ~30 chars; below 0.5 indicates the model couldn't pin down a language.
_MIN_CONFIDENCE = 0.5

# Below this character length we don't even ask fasttext — too short to
# disambiguate and tends to surface random-looking labels at high
# confidence (the model trained on snippets can over-fire on common
# phrases).
_MIN_CHARS = 20


_NUMPY_PATCHED = False


def _patch_numpy_for_fasttext() -> None:
    """Newer numpy raised on ``np.array(obj, copy=False)``; fasttext's
    Python wrapper hasn't been updated. Drop the kwarg so it falls
    through to default copy semantics."""
    global _NUMPY_PATCHED
    if _NUMPY_PATCHED:
        return
    import numpy as np
    _orig: Any = np.array

    def _patched(obj: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("copy") is False:
            kwargs.pop("copy")
        return _orig(obj, *args, **kwargs)
    setattr(np, "array", _patched)  # noqa: B010
    _NUMPY_PATCHED = True


@lru_cache(maxsize=1)
def _model() -> Any:
    _patch_numpy_for_fasttext()
    import fasttext
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(
        repo_id="facebook/fasttext-language-identification",
        filename="model.bin",
    )
    return fasttext.load_model(path)


def _predict_top(text: str) -> tuple[str, float] | None:
    """Run fasttext on a single chunk. Returns (label, confidence) or None."""
    if len(text.strip()) < _MIN_CHARS:
        return None
    # fasttext's predict() can't handle newlines; collapse them.
    cleaned = " ".join(text.split())
    labels, scores = _model().predict(cleaned, k=1)
    if not labels:
        return None
    label = labels[0].removeprefix("__label__")
    return (label, float(scores[0]))


def detect(text: str, subset: str | None = None) -> Language:
    """Return the detected language as one of the 6 gheim Language codes.

    Decision tree:
      1. fasttext top-1 with confidence ≥ 0.5 and known mapping → that.
      2. subset == "romansh" and no confident verdict → rm.
      3. Else → subset default (de_ch for legal/parl/web, rm for romansh).

    Never raises on short or empty text — always returns a Language.
    """
    top = _predict_top(text)
    if top is not None:
        label, conf = top
        if conf >= _MIN_CONFIDENCE and label in _NLLB_TO_GHEIM:
            return _NLLB_TO_GHEIM[label]
    # Fall through: subset hint or default
    return _subset_default(subset)


def _subset_default(subset: str | None) -> Language:
    if subset == "romansh":
        return "rm"
    # entscheidsuche / curia_vista / fineweb / unknown: majority Swiss
    # legal/parl content is German.
    return "de_ch"
