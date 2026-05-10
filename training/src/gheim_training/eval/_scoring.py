"""Shared scoring helpers used by ``eval_on_ours`` and ``eval_on_external``.

Two backends produce different intermediate representations:

* HF / ONNX models predict per-token label IDs. We score them with
  seqeval at the token level (this is the headline F1 = 0.9161 metric)
  AND with character-aware F1 derived from the predicted spans.
* spaCy / Presidio emit character spans directly. Without a token grid
  there is no seqeval-equivalent; ``strict_span`` falls back to a
  set-match on ``(start, end, label)`` tuples (method
  ``char-set-strict``) and ``char`` F1 is computed identically to the
  HF path.

The output JSON shape is the same in both cases so ``report.py`` can
treat the files uniformly. The shape is documented at the top of
``eval_on_ours.py``.
"""
from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Any

from ._metrics import f1_pr

CharSpan = Annotated[tuple[int, int, str], "(char_start, char_end, label) span on the source text."]
PerCatCounters = Annotated[dict[str, dict[str, int]], "label -> {tp, fp, fn} counters."]


# ---------------------------------------------------------------------------
# Span / char-level counters
# ---------------------------------------------------------------------------

def _empty_counter() -> dict[str, int]:
    return {"tp": 0, "fp": 0, "fn": 0}


def char_level_counters(
    gold: list[CharSpan],
    pred: list[CharSpan],
) -> tuple[int, int, int, PerCatCounters]:
    """Per-character label-aware F1 counters.

    A ``(char_index, label)`` pair is in gold if any gold span covers
    that index with that label. The set intersection / differences give
    label-aware char-level TP / FP / FN.
    """
    gold_chars: set[tuple[int, str]] = set()
    for s, e, lab in gold:
        for c in range(s, e):
            gold_chars.add((c, lab))
    pred_chars: set[tuple[int, str]] = set()
    for s, e, lab in pred:
        for c in range(s, e):
            pred_chars.add((c, lab))

    tp_set = gold_chars & pred_chars
    fp_set = pred_chars - gold_chars
    fn_set = gold_chars - pred_chars

    per_cat: PerCatCounters = defaultdict(_empty_counter)
    for _, lab in tp_set:
        per_cat[lab]["tp"] += 1
    for _, lab in fp_set:
        per_cat[lab]["fp"] += 1
    for _, lab in fn_set:
        per_cat[lab]["fn"] += 1
    return len(tp_set), len(fp_set), len(fn_set), per_cat


def strict_set_counters(
    gold: list[CharSpan],
    pred: list[CharSpan],
) -> tuple[int, int, int, PerCatCounters]:
    """Exact ``(start, end, label)`` set match — equivalent to seqeval
    strict mode but computed on char spans (used for spaCy / Presidio
    backends that don't produce token-level predictions)."""
    gset = set(gold)
    pset = set(pred)
    tp_set = gset & pset
    fp_set = pset - gset
    fn_set = gset - pset
    per_cat: PerCatCounters = defaultdict(_empty_counter)
    for s in tp_set:
        per_cat[s[2]]["tp"] += 1
    for s in fp_set:
        per_cat[s[2]]["fp"] += 1
    for s in fn_set:
        per_cat[s[2]]["fn"] += 1
    return len(tp_set), len(fp_set), len(fn_set), per_cat


# ---------------------------------------------------------------------------
# BIOES / BIO label-sequence → char spans
# ---------------------------------------------------------------------------

def spans_from_label_sequence(
    seq: list[str],
    char_offsets: list[tuple[int, int]],
) -> list[CharSpan]:
    """Walk a BIOES/BIO label sequence and emit ``(char_start, char_end, label)``.

    ``char_offsets[i]`` is the ``(start, end)`` of the i-th token in
    the original text. We extract spans by grouping consecutive tokens
    of the same coarse category, ignoring B vs I vs E vs S distinctions
    for grouping (since :func:`merge_collapsed_spans` may have already
    mixed them).
    """
    out: list[CharSpan] = []
    cur_label: str | None = None
    cur_start: int = -1
    cur_end: int = -1
    for i, tag in enumerate(seq):
        prefix = ""
        if tag == "O" or "-" not in tag:
            bare = "O" if tag == "O" else tag
        else:
            prefix, bare = tag.split("-", 1)
        if cur_label is not None and (bare != cur_label or prefix in ("B", "S")):
            out.append((cur_start, cur_end, cur_label))
            cur_label = None
            cur_start = cur_end = -1
        if bare == "O":
            continue
        if i >= len(char_offsets):
            continue
        tok_s, tok_e = char_offsets[i]
        if cur_label is None:
            cur_label = bare
            cur_start = tok_s
            cur_end = tok_e
        else:
            cur_end = tok_e
    if cur_label is not None:
        out.append((cur_start, cur_end, cur_label))
    return out


def merge_collapsed_spans(seq: list[str]) -> list[str]:
    """When ``--remap-json`` collapses N source categories into one of
    ours, two consecutive ``B-X`` tokens (originally distinct
    fine-grained labels) get fragmented into separate spans under
    BIO/BIOES extraction. Demote a ``B-X`` to ``I-X`` when the previous
    token is in the same coarse category so seqeval extracts a single
    continuous span.
    """
    out = list(seq)
    for i in range(1, len(out)):
        if not out[i].startswith("B-"):
            continue
        cat = out[i][2:]
        prev = out[i - 1]
        if len(prev) > 2 and prev[1] == "-" and prev[2:] == cat:
            out[i] = "I-" + cat
    return out


# ---------------------------------------------------------------------------
# Label-space remap (model id-space → our 33-class BIOES id-space)
# ---------------------------------------------------------------------------

def build_pred_id_translation(
    model_id2label: dict[int, str],
    our_label2id: dict[str, int],
    our_id2label: dict[int, str],
    remap_path: Path | None,
) -> dict[int, int]:
    """Return a mapping from a model's predicted label IDs to our BIOES IDs.

    Three modes (mirrors the behaviour of the legacy ``test_eval`` /
    ``char_metric`` scripts so the JSON numbers stay bit-identical):

    1. No remap and label spaces match → identity mapping.
    2. No remap and label spaces differ → ``SystemExit``.
    3. ``remap_path`` given → translate via
       ``{their_bare_category: our_bare_category | "O"}``.
    """
    if remap_path is not None:
        remap: dict[str, str] = json.loads(remap_path.read_text())
        out: dict[int, int] = {}
        for tid, tlabel in model_id2label.items():
            if tlabel == "O":
                out[tid] = our_label2id["O"]
                continue
            if "-" not in tlabel:
                # Plain "PERSON" with no BIO prefix; treat as a span start (B-).
                prefix, bare = "B", tlabel
            else:
                prefix, bare = tlabel.split("-", 1)
            our_bare = remap.get(bare)
            if our_bare is None or our_bare == "O":
                out[tid] = our_label2id["O"]
                continue
            our_label = f"{prefix}-{our_bare}"
            out[tid] = our_label2id.get(our_label, our_label2id["O"])
        return out
    if model_id2label != our_id2label:
        diffs = [
            (i, model_id2label.get(i), our_id2label.get(i))
            for i in sorted(set(model_id2label) | set(our_id2label))
            if model_id2label.get(i) != our_id2label.get(i)
        ]
        raise SystemExit(
            f"label-space mismatch ({len(diffs)} differing IDs); cannot score this "
            "checkpoint without --remap-json. "
            f"First 5 diffs (id, model, ours): {diffs[:5]}"
        )
    return {i: i for i in our_id2label}


# ---------------------------------------------------------------------------
# Aggregator: builds the unified output JSON
# ---------------------------------------------------------------------------

class ScoreAccumulator:
    """Accumulates per-row counters and emits the unified ``metrics`` blob.

    Two scoring streams are tracked in parallel:

    * ``strict_span`` — either seqeval token-level (HF / ONNX) or
      char-set-strict (spaCy / Presidio). Method tag identifies which.
    * ``char`` — char-label-aware (always identical regardless of
      backend).
    """

    def __init__(self, strict_method: str) -> None:
        self.strict_method = strict_method  # "seqeval-token-level" or "char-set-strict"
        self._strict_overall: dict[str, int] = _empty_counter()
        self._char_overall: dict[str, int] = _empty_counter()
        self._strict_cat: PerCatCounters = defaultdict(_empty_counter)
        self._char_cat: PerCatCounters = defaultdict(_empty_counter)
        self._gold_cat_n: dict[str, int] = defaultdict(int)
        self._per_lang: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "strict": _empty_counter(),
            "char": _empty_counter(),
            "n_chunks": 0,
        })
        # When seqeval-token-level is in play, we also need the per-row
        # token sequences to compute per-language seqeval F1 (matching
        # the legacy test_eval.py behaviour). For char-set-strict
        # backends these stay empty and we fall back to per-row strict
        # counter aggregation, which is mathematically equivalent for
        # set-match metrics.
        self._token_true: list[list[str]] = []
        self._token_pred: list[list[str]] = []
        self._token_lang: list[str] = []
        self._cat_token_support: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Per-row updates
    # ------------------------------------------------------------------

    def add_token_row(
        self,
        true_seq: list[str],
        pred_seq: list[str],
        gold_spans: list[CharSpan],
        pred_spans: list[CharSpan],
        language: str,
    ) -> None:
        """Add one row from a token-level backend (HF / ONNX).

        ``true_seq`` and ``pred_seq`` are the BIOES strings for seqeval.
        ``gold_spans`` and ``pred_spans`` are the corresponding char
        spans for char F1.
        """
        self._token_true.append(true_seq)
        self._token_pred.append(pred_seq)
        self._token_lang.append(language)

        # char F1
        c_tp, c_fp, c_fn, c_pc = char_level_counters(gold_spans, pred_spans)
        self._char_overall["tp"] += c_tp
        self._char_overall["fp"] += c_fp
        self._char_overall["fn"] += c_fn
        for k, v in c_pc.items():
            self._char_cat[k]["tp"] += v["tp"]
            self._char_cat[k]["fp"] += v["fp"]
            self._char_cat[k]["fn"] += v["fn"]

        # per-language strict counters via char-set-strict (so per-lang
        # tables can be rendered without re-running seqeval per slice).
        s_tp, s_fp, s_fn, _ = strict_set_counters(gold_spans, pred_spans)
        self._per_lang[language]["strict"]["tp"] += s_tp
        self._per_lang[language]["strict"]["fp"] += s_fp
        self._per_lang[language]["strict"]["fn"] += s_fn
        self._per_lang[language]["char"]["tp"] += c_tp
        self._per_lang[language]["char"]["fp"] += c_fp
        self._per_lang[language]["char"]["fn"] += c_fn
        self._per_lang[language]["n_chunks"] += 1

        # gold span counts per category (for n_gold in per_category).
        for _s, _e, lab in gold_spans:
            self._gold_cat_n[lab] += 1

    def add_span_row(
        self,
        gold_spans: list[CharSpan],
        pred_spans: list[CharSpan],
        language: str,
    ) -> None:
        """Add one row from a span-emitting backend (spaCy / Presidio).

        Both ``strict_span`` and ``char`` are computed from the spans.
        """
        s_tp, s_fp, s_fn, s_pc = strict_set_counters(gold_spans, pred_spans)
        self._strict_overall["tp"] += s_tp
        self._strict_overall["fp"] += s_fp
        self._strict_overall["fn"] += s_fn
        for k, v in s_pc.items():
            self._strict_cat[k]["tp"] += v["tp"]
            self._strict_cat[k]["fp"] += v["fp"]
            self._strict_cat[k]["fn"] += v["fn"]

        c_tp, c_fp, c_fn, c_pc = char_level_counters(gold_spans, pred_spans)
        self._char_overall["tp"] += c_tp
        self._char_overall["fp"] += c_fp
        self._char_overall["fn"] += c_fn
        for k, v in c_pc.items():
            self._char_cat[k]["tp"] += v["tp"]
            self._char_cat[k]["fp"] += v["fp"]
            self._char_cat[k]["fn"] += v["fn"]

        self._per_lang[language]["strict"]["tp"] += s_tp
        self._per_lang[language]["strict"]["fp"] += s_fp
        self._per_lang[language]["strict"]["fn"] += s_fn
        self._per_lang[language]["char"]["tp"] += c_tp
        self._per_lang[language]["char"]["fp"] += c_fp
        self._per_lang[language]["char"]["fn"] += c_fn
        self._per_lang[language]["n_chunks"] += 1

        for _s, _e, lab in gold_spans:
            self._gold_cat_n[lab] += 1

    # ------------------------------------------------------------------
    # Finalisation
    # ------------------------------------------------------------------

    def finalize(self) -> dict[str, Any]:
        """Compute final F1s and return the ``metrics`` blob."""
        if self.strict_method == "seqeval-token-level":
            from seqeval.metrics import (
                classification_report,
                f1_score,
                precision_score,
                recall_score,
            )

            overall_strict = {
                "f1": float(f1_score(self._token_true, self._token_pred)),
                "precision": float(precision_score(self._token_true, self._token_pred)),
                "recall": float(recall_score(self._token_true, self._token_pred)),
                "method": self.strict_method,
            }
            rep = classification_report(
                self._token_true, self._token_pred,
                output_dict=True, zero_division=0,
            )
            per_cat_strict: dict[str, dict[str, float]] = {}
            for ent, sc in rep.items():
                if not isinstance(sc, dict):
                    continue
                if ent in ("micro avg", "macro avg", "weighted avg"):
                    continue
                per_cat_strict[ent] = {
                    "f1": float(sc.get("f1-score", 0.0)),
                    "precision": float(sc.get("precision", 0.0)),
                    "recall": float(sc.get("recall", 0.0)),
                    "n_gold": int(sc.get("support", 0)),
                }
        else:
            ov = f1_pr(self._strict_overall)
            overall_strict = {
                "f1": ov["f1"],
                "precision": ov["precision"],
                "recall": ov["recall"],
                "method": self.strict_method,
            }
            per_cat_strict = {}
            for cat in sorted(set(self._strict_cat) | set(self._gold_cat_n)):
                m = f1_pr(self._strict_cat.get(cat, _empty_counter()))
                per_cat_strict[cat] = {
                    "f1": m["f1"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "n_gold": self._gold_cat_n.get(cat, m["tp"] + m["fn"]),
                }

        ov_char = f1_pr(self._char_overall)
        overall_char = {
            "f1": ov_char["f1"],
            "precision": ov_char["precision"],
            "recall": ov_char["recall"],
            "method": "char-label-aware",
        }
        per_cat_char: dict[str, dict[str, float]] = {}
        for cat in sorted(set(self._char_cat) | set(self._gold_cat_n) | set(per_cat_strict)):
            m = f1_pr(self._char_cat.get(cat, _empty_counter()))
            per_cat_char[cat] = {
                "f1": m["f1"],
                "precision": m["precision"],
                "recall": m["recall"],
            }

        # Merge per-category strict & char into one table.
        per_category: dict[str, dict[str, dict[str, float]]] = {}
        for cat in sorted(set(per_cat_strict) | set(per_cat_char)):
            cell: dict[str, dict[str, float]] = {}
            if cat in per_cat_strict:
                cell["strict_span"] = per_cat_strict[cat]
            if cat in per_cat_char:
                cell["char"] = per_cat_char[cat]
            per_category[cat] = cell

        # Per-language: merge strict + char + n_chunks. For seqeval-token
        # backends, recompute per-language strict via seqeval for fidelity
        # with the legacy test_eval.py per-lang numbers.
        per_language: dict[str, dict[str, Any]] = {}
        if self.strict_method == "seqeval-token-level":
            from seqeval.metrics import f1_score, precision_score, recall_score

            langs = sorted(set(self._token_lang))
            for lang in langs:
                idx = [i for i, la in enumerate(self._token_lang) if la == lang]
                if not idx:
                    continue
                t_sub = [self._token_true[i] for i in idx]
                p_sub = [self._token_pred[i] for i in idx]
                strict_cell = {
                    "f1": float(f1_score(t_sub, p_sub)),
                    "precision": float(precision_score(t_sub, p_sub)),
                    "recall": float(recall_score(t_sub, p_sub)),
                    "method": self.strict_method,
                }
                char_pl = self._per_lang[lang]["char"]
                ch = f1_pr(char_pl)
                per_language[lang] = {
                    "strict_span": strict_cell,
                    "char": {
                        "f1": ch["f1"],
                        "precision": ch["precision"],
                        "recall": ch["recall"],
                        "method": "char-label-aware",
                    },
                    "n_chunks": self._per_lang[lang]["n_chunks"],
                }
        else:
            for lang in sorted(self._per_lang):
                pl = self._per_lang[lang]
                s = f1_pr(pl["strict"])
                c = f1_pr(pl["char"])
                per_language[lang] = {
                    "strict_span": {
                        "f1": s["f1"],
                        "precision": s["precision"],
                        "recall": s["recall"],
                        "method": self.strict_method,
                    },
                    "char": {
                        "f1": c["f1"],
                        "precision": c["precision"],
                        "recall": c["recall"],
                        "method": "char-label-aware",
                    },
                    "n_chunks": pl["n_chunks"],
                }

        return {
            "overall": {
                "strict_span": overall_strict,
                "char": overall_char,
            },
            "per_category": per_category,
            "per_language": per_language,
        }


# ---------------------------------------------------------------------------
# Misc helpers shared by both eval entry points
# ---------------------------------------------------------------------------

def gold_spans_from_row(row: dict) -> list[CharSpan]:
    """Extract ``(start, end, label)`` from a HF row's ``spans`` field,
    skipping any ``O``-labelled span and casting ints defensively."""
    out: list[CharSpan] = []
    for s in row.get("spans") or []:
        lab = s.get("label")
        if not lab or lab == "O":
            continue
        out.append((int(s["start"]), int(s["end"]), lab))
    return out


def write_result(
    out_path: Path,
    *,
    model_id: str,
    backend: str,
    dataset_id: str,
    dataset_split: str,
    n_chunks: int,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    """Serialise the unified eval JSON.

    ``extra`` is folded into the top-level dict for backend-specific
    provenance (e.g. ``onnx_file``, ``score_threshold``) without changing
    the contract that ``report.py`` consumes.
    """
    blob: dict[str, Any] = {
        "model_id": model_id,
        "backend": backend,
        "dataset_id": dataset_id,
        "dataset_split": dataset_split,
        "n_chunks": n_chunks,
        "metrics": metrics,
    }
    if extra:
        for k, v in extra.items():
            blob[k] = v
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2))


def iter_rows(raw: Iterable[dict]) -> Iterable[dict]:
    """Pass-through generator; isolated so the call sites can be wrapped
    by progress reporting in one place if we ever want it."""
    yield from raw


def load_test_dataset(
    dataset_dir: str | Path,
    split: str,
    languages: Sequence[str] | None = None,
) -> Any:
    """Load a HF Dataset split from disk and optionally filter by language.

    Returns a ``datasets.Dataset`` (return type widened to ``Any`` so
    callers don't need to add ``datasets`` to their type-checker
    classpath; the runtime contract is the standard HF Dataset API).
    """
    from datasets import load_from_disk

    dd = load_from_disk(str(dataset_dir))
    if split not in dd:
        raise SystemExit(f"split {split!r} not in {list(dd)}")
    raw: Any = dd[split]
    if languages:
        keep = {x.strip() for x in languages}
        raw = raw.filter(lambda r: r["language"] in keep,
                         desc=f"filter languages={sorted(keep)}")
    return raw
