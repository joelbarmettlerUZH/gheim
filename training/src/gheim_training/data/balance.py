"""Balance + dedup + synthetic injection into the publishable dataset.

Produces ``data/balanced.jsonl`` from:

- ``data/assembled.jsonl`` — LLM+regex labelled real-text corpus
  (~2.3M chunks, output of ``labelling/assemble.py``);
- 50k template-generated documents (KYC/HR/banking/medical) and gap-fill
  chunks under ``data/layer*.jsonl``;
- ~64k composer-rendered synthetic chunks (emails / docs / short-form
  narrative / form layouts / common-word disambiguation pairs /
  adversarial negatives) produced by ``synth/generate.py``.

Five phases run in order:

  E. Signal floor — drop spans with ``confidence < 0.5`` (single
     labeller, no other model agreed).
  A. Per-document cap = 30 chunks per source document. Synthetic sources
     bypass (they have no document structure; per-template diversity is
     bounded by Phase C cell caps instead).
  B. Per-(language, category, normalised value) cap — cell-aware (see
     ``_per_value_cap``): 30 for strong cells, 60 for medium, 100-200
     for data-thin cells (RM, EN). Synthetic sources have their own
     dedup namespace so they don't compete with real-text for budget.
  C. Per-(language, source, category) cell cap — tiered to balance
     across the corpus while preserving the natural source mix.
  D. Negatives — two-tier admission: (1) ALL adversarial-negative
     chunks are always admitted; (2) random sampling at 10%/lang from
     the remaining empty-span pool for naturalistic negatives.

Output schema is :class:`LabelledExample` — every span carries its full
``signals`` tuple and ``regex_subtype`` so downstream consumers can see
exactly which labellers agreed on each entity.

Run
---
    uv run python -m gheim_training.data.balance
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

from .labelling.schema import LabelledExample, LabelledSpan
from .labelling.schema import write_jsonl

ASSEMBLED_PATH = Path("data/assembled.jsonl")
# Gemma slot-fill gap-fill (synth/slot_fill_gemma.py)
SLOT_FILL_PATH = Path("data/layer_slot_fill.jsonl")
# Composer-rendered synthetic layers (synth/generate.py)
SYNTH_EMAILS_PATH = Path("data/layer_synth_emails.jsonl")
SYNTH_DOCS_PATH = Path("data/layer_synth_docs.jsonl")
SYNTH_MULTILANG_DOCS_PATH = Path("data/layer_synth_multilang_docs.jsonl")
SYNTH_SHORT_FORM_PATH = Path("data/layer_synth_short_form.jsonl")
SYNTH_FORMS_PATH = Path("data/layer_synth_forms.jsonl")
SYNTH_COMMON_WORD_PATH = Path("data/layer_synth_common_word.jsonl")
SYNTH_ADVERSARIAL_PATH = Path("data/layer_synth_adversarial.jsonl")
SYNTH_RM_SECRETS_PATH = Path("data/layer_synth_rm_secrets.jsonl")
OUT_PATH = Path("data/balanced.jsonl")

LANGS = ("de_ch", "fr_ch", "it_ch", "rm", "en")
CATS = (
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
)
REAL_SOURCES = ("fineweb", "entscheidsuche", "curia_vista", "romansh")
SYNTHETIC_SOURCES = (
    "synthetic_slot_fill",
    "synthetic_emails", "synthetic_docs", "synthetic_multilang_docs",
    "synthetic_short_form", "synthetic_forms", "synthetic_common_word",
    "synthetic_adversarial", "synthetic_rm_secrets",
)

# Phase E
CONFIDENCE_FLOOR = 0.5
# Phase A
PER_DOC_CAP = 30
# Phase B — base cap; cell-aware overrides via _per_value_cap() below.
PER_VALUE_CAP = 30
# Phase D
NEGATIVES_FRACTION = 0.10
SEED = 20260522


def _per_value_cap(lang: str, cat: str, subset: str) -> int:
    """Per-(lang, cat, value) cap for Phase B.

    Cell-aware tuning based on the predecessor build eval results (eval/v2_test_per_lang_cat.json):
    cells where the trained model underperformed get more headroom so the
    next training run sees additional context variations of the same
    entity. Strong cells stay at the default 30 to keep the dataset from
    bloating on already-easy patterns.

    Tiers:
    - 30 (default): strong cells — account_number/email/phone/url/secret
      in de_ch/fr_ch/it_ch all hit F1 ≥ 0.98, no benefit from more dupes.
    - 60: medium cells — person/address/date in the three Swiss langs.
    - 100: rm × any cat — RM is data-thin overall; let more through.
    - 200: en × any cat — corpus has only ~940 EN chunks; never the
      binding constraint.
    - 200: synthetic sources are deliberately diverse already, but the
      template-derived chunks can repeat values within a template_id.
      High cap here keeps name-pattern templates from being silently
      gutted (the synth_name_patterns generator uses Faker so repeats
      are rare anyway).
    """
    if subset in SYNTHETIC_SOURCES:
        return 200
    if lang == "en":
        return 200
    if lang == "rm":
        return 100
    medium_cats = ("private_person", "private_address", "private_date")
    if cat in medium_cats:
        return 60
    return PER_VALUE_CAP


def _cell_cap(lang: str, source: str, cat: str) -> int:
    """Per-(lang × source × cat) cap for Phase C. Caps scaled for the
    150-250k-total target the user set in the pipeline review.

    Notes:
    - English bottlenecks at ~940 chunks total in the corpus, regardless
      of cap. Cap is high purely to never gate EN.
    - secret is effectively uncapped — appears in <0.5% of chunks corpus-wide.
    """
    if cat == "secret":
        return 100_000  # effectively uncapped — secrets are very rare
    # Real-text sources
    if source in ("fineweb", "entscheidsuche", "curia_vista"):
        if lang in ("de_ch", "fr_ch"):
            major = (
                "private_person", "private_date", "private_address",
                "private_url", "private_phone",
            )
            # de_ch × person, fr_ch × date were the two ≥0.88-but-<0.93
            # weak cells in the predecessor build → bump person/address/date specifically
            # to 24k while keeping url/phone at 18k.
            weak_for_swiss = ("private_person", "private_address",
                              "private_date")
            if cat in weak_for_swiss:
                return 24_000
            return 18_000 if cat in major else 3_000
        if lang == "it_ch":
            major = (
                "private_person", "private_date", "private_address",
                "private_url", "private_phone",
            )
            return 12_000 if cat in major else 2_400
        if lang == "en":
            return 10_000  # keep all ~940 EN chunks
        if lang == "rm":
            return 600  # RM is overwhelmingly from `romansh`, almost none here
    if source == "romansh":
        if lang == "rm":
            # rm × person/address were the the predecessor build weak cells (F1 0.84/0.80)
            # — bump from 10k → 16k for more name + address variation.
            if cat in ("private_person", "private_address"):
                return 16_000
            return 10_000
        if lang == "it_ch":
            return 2_400  # the rm corpus has some it_ch leakage
        return 600  # de_ch/fr_ch/en in the rm corpus are scraps
    # Synthetic sources — synthetic_l1 sized so it ≈ 5-7% of the dataset.
    # ~6 spans/chunk × 4000-cap = ~700 chunks per major-cat cell.
    if source == "synthetic_multilang_docs":
        return 4_000
    if source == "synthetic_slot_fill":
        return 200  # keep nearly all of the 880 Layer-9 chunks
    if source == "synthetic_rm_secrets":
        # 800 chunks, ~1.2 spans/chunk, dominated by secret (800 spans).
        # cap=800 keeps all (rm × synthetic_rm_secrets × secret) and lets
        # the few co-occurring person/phone spans through. Closes the
        # (rm × secret) cell which was 1-span in the predecessor build.
        return 800
    # v3 synthetic layers — generated by data.v3.generate from 631
    # hand-written templates. Let through generously since they
    # target the specific failure patterns the predecessor build still misses.
    if source == "synthetic_emails":
        # 31k chunks (greet+body+sig compositions, all 5 langs).
        # Cap per (lang × cat) so emails don't dominate any single cell.
        return 4_000
    if source == "synthetic_docs":
        # 12.75k chunks (DE legal/medical/bank/HR docs, ~7 spans each).
        return 3_000
    if source == "synthetic_short_form":
        return 2_500
    if source == "synthetic_forms":
        # DE-only, ~4 spans/chunk, structured layouts. Cap loosely so
        # the model sees the form pattern variety.
        return 2_000
    if source == "synthetic_common_word":
        # 4.8k chunks (40 pos + 40 neg per surname × 8 surnames).
        # Cap to ensure even per-surname coverage in training.
        return 1_500
    if source == "synthetic_adversarial":
        # 2.8k DE adversarial-negative chunks. All have 0 spans, so
        # they don't contribute to per-cell cat counts; cap only on
        # the source axis.
        return 2_500
    return 1_000


# ---------------------------------------------------------- pass-1 metadata


class _SpanMeta:
    """Slim per-span record used for admission decisions (no full text)."""
    __slots__ = ("start", "end", "label", "value_norm", "confidence")

    def __init__(self, start: int, end: int, label: str, value_norm: str,
                 confidence: float) -> None:
        self.start = start
        self.end = end
        self.label = label
        self.value_norm = value_norm
        self.confidence = confidence


class _ChunkMeta:
    __slots__ = ("id", "language", "subset", "doc_id", "spans",
                 "_n_distinct_cats")

    def __init__(self, cid: str, lang: str, subset: str, doc_id: str,
                 spans: list[_SpanMeta]) -> None:
        self.id = cid
        self.language = lang
        self.subset = subset
        self.doc_id = doc_id
        self.spans = spans
        self._n_distinct_cats = len({sp.label for sp in spans})

    @property
    def n_distinct_cats(self) -> int:
        return self._n_distinct_cats


def _norm_value(v: str) -> str:
    return v.strip().casefold()


# -------------------------------------------------------------- loaders


def _load_assembled_metadata() -> list[_ChunkMeta]:
    """Pass 1 over v2_assembled.jsonl — keep only what we need to make
    admission decisions. Drops text + signals + regex_subtype to keep RAM
    modest (~250 MB for 2.3M chunks)."""
    out: list[_ChunkMeta] = []
    n_kept = 0
    n_dropped_by_floor = 0
    with ASSEMBLED_PATH.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            spans_raw = d.get("spans", [])
            # Phase E: drop spans below the confidence floor.
            spans_floor = []
            for sp in spans_raw:
                if sp["confidence"] >= CONFIDENCE_FLOOR:
                    spans_floor.append(_SpanMeta(
                        start=sp["start"], end=sp["end"], label=sp["label"],
                        value_norm=_norm_value(sp["value"]),
                        confidence=sp["confidence"],
                    ))
                else:
                    n_dropped_by_floor += 1
            out.append(_ChunkMeta(
                cid=d["id"], lang=d["language"], subset=d["subset"],
                doc_id=d.get("doc_id") or d["id"],
                spans=spans_floor,
            ))
            n_kept += 1
            if (i + 1) % 250_000 == 0:
                print(f"  pass-1 read {i + 1:,} chunks", flush=True)
    print(f"  pass-1: {n_kept:,} chunks loaded, "
          f"{n_dropped_by_floor:,} spans dropped by confidence floor "
          f"(<{CONFIDENCE_FLOOR})")
    return out


def _load_synthetic_metadata(path: Path, subset: str) -> list[_ChunkMeta]:
    """Load Layer 1 / Layer 9 → _ChunkMeta with subset stamped and
    template_id used as doc_id (so Phase A also caps per-template)."""
    out: list[_ChunkMeta] = []
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            text = d["text"]
            spans = []
            for sp in d.get("spans", []):
                val = text[sp["start"]:sp["end"]]
                spans.append(_SpanMeta(
                    start=sp["start"], end=sp["end"], label=sp["label"],
                    value_norm=_norm_value(val),
                    confidence=1.0,
                ))
            cid = f"{subset}__{i}"
            doc_id = d.get("template_id") or cid
            out.append(_ChunkMeta(
                cid=cid, lang=d["language"], subset=subset,
                doc_id=doc_id, spans=spans,
            ))
    return out


# -------------------------------------------------------- admission logic


def _phase_a_per_doc_cap(
    chunks: list[_ChunkMeta], rng: random.Random,
) -> set[str]:
    """Return chunk-ids surviving the per-doc cap of PER_DOC_CAP.

    Synthetic chunks bypass this phase: Layer 1 has only 6 distinct
    templates each repeated ~9k times, and a 30-per-doc cap would gate
    synthetic admission to 180 chunks (= 6 × 30) regardless of the
    Phase C cell caps. Per-template diversity for synthetic is handled
    instead by Phase C's per-(lang × subset × cat) cell caps.
    """
    by_doc: dict[str, list[_ChunkMeta]] = defaultdict(list)
    kept: set[str] = set()
    for c in chunks:
        if c.subset in SYNTHETIC_SOURCES:
            kept.add(c.id)  # bypass per-doc cap; see docstring
            continue
        by_doc[c.doc_id].append(c)
    n_capped = 0
    for doc_id, doc_chunks in by_doc.items():
        if len(doc_chunks) <= PER_DOC_CAP:
            kept.update(c.id for c in doc_chunks)
        else:
            sampled = rng.sample(doc_chunks, PER_DOC_CAP)
            kept.update(c.id for c in sampled)
            n_capped += len(doc_chunks) - PER_DOC_CAP
    print(f"  Phase A: kept {len(kept):,} chunks "
          f"({n_capped:,} dropped from over-cap docs; "
          f"synthetic bypassed per-doc cap)")
    return kept


def _phase_b_per_value_cap(
    chunks: list[_ChunkMeta], kept_ids: set[str],
) -> dict[str, set[tuple[int, int]]]:
    """Decide which SPANS survive the per-(lang, cat, value) cap of
    PER_VALUE_CAP. Returns {chunk_id: set of (start,end) of surviving
    spans}. Chunks with all spans dropped pass through as candidates for
    Phase D (negatives).

    The dedup is **partitioned by source-class**: real-text (the four
    LLM-labelled sources) shares one (lang, cat, value) budget, but each
    synthetic source has its own. Without this, Layer 1's Faker name
    pool gets starved — real-text "Anna Müller" appearances exhaust the
    30-budget before any synthetic chunk is evaluated. The synthetic
    layers are *intended* to repeat template entities, so they deserve
    their own per-entity budget.
    """
    real_count: Counter[tuple[str, str, str]] = Counter()
    syn_count: dict[str, Counter[tuple[str, str, str]]] = {
        s: Counter() for s in SYNTHETIC_SOURCES
    }
    surviving: dict[str, set[tuple[int, int]]] = {}
    n_dropped_real = 0
    n_dropped_syn = 0
    for c in chunks:
        if c.id not in kept_ids:
            continue
        surviving_here: set[tuple[int, int]] = set()
        if c.subset in SYNTHETIC_SOURCES:
            counter = syn_count[c.subset]
        else:
            counter = real_count
        for sp in c.spans:
            key = (c.language, sp.label, sp.value_norm)
            cap = _per_value_cap(c.language, sp.label, c.subset)
            if counter[key] < cap:
                surviving_here.add((sp.start, sp.end))
                counter[key] += 1
            else:
                if c.subset in SYNTHETIC_SOURCES:
                    n_dropped_syn += 1
                else:
                    n_dropped_real += 1
        surviving[c.id] = surviving_here
    print(f"  Phase B: kept span-survivors for {len(surviving):,} chunks "
          f"({n_dropped_real:,} real + {n_dropped_syn:,} synthetic spans "
          f"dropped as over-cap dupes; cell-aware caps)")
    return surviving


def _phase_c_cell_caps(
    chunks: list[_ChunkMeta],
    kept_ids: set[str],
    surviving: dict[str, set[tuple[int, int]]],
    rng: random.Random,
) -> tuple[set[str], Counter[tuple[str, str, str]]]:
    """Greedy admission with per-(lang, source, cat) cell caps. Prefers
    multi-category chunks first so the model sees rich co-occurrences,
    then by span density."""
    cell_count: Counter[tuple[str, str, str]] = Counter()
    admitted: set[str] = set()
    # Pre-shuffle within each (n_cats, n_spans) priority class to avoid
    # any spurious ordering bias from the input file.
    eligible = [
        c for c in chunks
        if c.id in kept_ids and surviving.get(c.id)
    ]
    # Surviving span counts after Phase B per chunk; precompute the
    # contributing-cells set per chunk too.
    def _cells_for(c: _ChunkMeta) -> Counter[tuple[str, str, str]]:
        surv = surviving[c.id]
        cells: Counter[tuple[str, str, str]] = Counter()
        for sp in c.spans:
            if (sp.start, sp.end) in surv:
                cells[(c.language, c.subset, sp.label)] += 1
        return cells

    # Cache cells per chunk so we don't recompute during sort.
    cells_by_chunk: dict[str, Counter[tuple[str, str, str]]] = {
        c.id: _cells_for(c) for c in eligible
    }
    # Distinct cats AFTER Phase B (some spans may have been dropped).
    def _key(c: _ChunkMeta) -> tuple[int, int, float]:
        cells = cells_by_chunk[c.id]
        distinct_cats = len({cat for (_, _, cat) in cells})
        n_spans = sum(cells.values())
        return (-distinct_cats, -n_spans, rng.random())

    eligible.sort(key=_key)

    n_rejected = 0
    for c in eligible:
        cells = cells_by_chunk[c.id]
        # Would admitting overflow any cell? Allow partial: if NO cell
        # would overflow, admit. (Stricter than v1; matches the design
        # commitment that per-cell caps are hard ceilings.)
        ok = all(
            cell_count[cell] + n <= _cell_cap(*cell)
            for cell, n in cells.items()
        )
        if ok:
            admitted.add(c.id)
            for cell, n in cells.items():
                cell_count[cell] += n
        else:
            n_rejected += 1

    print(f"  Phase C: admitted {len(admitted):,} positive chunks "
          f"({n_rejected:,} rejected by cell caps)")
    return admitted, cell_count


def _phase_d_negatives(
    chunks: list[_ChunkMeta],
    kept_a: set[str],
    admitted_positives: set[str],
    surviving: dict[str, set[tuple[int, int]]],
    rng: random.Random,
) -> set[str]:
    """Inject negatives at NEGATIVES_FRACTION of admitted positives per
    language. A chunk is a negative candidate if it (a) survived Phase A
    and (b) ended Phase B with zero surviving spans — and is not already
    a positive. Phase A must be respected so per-doc caps still hold for
    the negative pool.

    Two-tier emission:
    - Tier 1: ALL synthetic_adversarial chunks are admitted (these are
      hand-curated false-positive teachers; we never want to sample them
      down to random fractions).
    - Tier 2: random sampling at NEGATIVES_FRACTION from the remaining
      negative pool (real-text chunks where the model just didn't find
      any PII) for naturalistic negatives.
    """
    pos_per_lang: Counter[str] = Counter()
    for c in chunks:
        if c.id in admitted_positives:
            pos_per_lang[c.language] += 1
    target_per_lang = {
        la: int(n * NEGATIVES_FRACTION) for la, n in pos_per_lang.items()
    }

    admitted_neg: set[str] = set()
    n_curated = 0
    neg_pool: dict[str, list[_ChunkMeta]] = defaultdict(list)
    for c in chunks:
        if c.id not in kept_a:
            continue  # Phase A dropped this; cannot bypass via negative
        if c.id in admitted_positives:
            continue
        surv = surviving.get(c.id, set())
        if surv:
            continue  # had surviving spans but Phase C rejected — drop

        # Tier 1: always admit hand-curated adversarial negatives.
        if c.subset == "synthetic_adversarial":
            admitted_neg.add(c.id)
            n_curated += 1
            continue
        # Tier 2: candidate for random sampling.
        neg_pool[c.language].append(c)

    n_sampled = 0
    for la, target in target_per_lang.items():
        pool = neg_pool.get(la, [])
        if not pool:
            continue
        sampled = rng.sample(pool, min(target, len(pool)))
        admitted_neg.update(c.id for c in sampled)
        n_sampled += len(sampled)
    total_target = sum(min(target_per_lang[la], len(neg_pool.get(la, [])))
                       for la in target_per_lang)
    print(f"  Phase D: admitted {len(admitted_neg):,} negative chunks "
          f"({n_curated:,} curated adversarial + "
          f"{n_sampled:,} sampled real-text @ "
          f"{int(NEGATIVES_FRACTION * 100)}%/lang)")
    return admitted_neg


# ----------------------------------------------------- pass 2: serialise


def _stream_assembled_records() -> Iterator[dict]:
    with ASSEMBLED_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _emit_real_chunk(d: dict,
                     surviving: set[tuple[int, int]]) -> LabelledExample:
    """Reconstruct a LabelledExample from an assembled record, keeping only
    spans whose (start, end) is in ``surviving``. The signals field
    carries the labeller agreement set through verbatim, per the pipeline design.
    """
    spans = []
    for sp in d.get("spans", []):
        if (sp["start"], sp["end"]) not in surviving:
            continue
        spans.append(LabelledSpan(
            start=sp["start"], end=sp["end"], label=sp["label"],
            value=sp["value"], signals=tuple(sp["signals"]),
            confidence=sp["confidence"],
            regex_subtype=sp.get("regex_subtype"),
        ))
    return LabelledExample(
        id=d["id"], text=d["text"], language=d["language"],
        subset=d["subset"], doc_id=d.get("doc_id") or d["id"],
        spans=spans,
        labelers=d.get("labelers", []),
        meta=d.get("meta", {}),
    )


def _emit_synthetic_chunks(path: Path, subset: str,
                           admitted: set[str],
                           surviving: dict[str, set[tuple[int, int]]],
                           ) -> Iterator[LabelledExample]:
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = f"{subset}__{i}"
            if cid not in admitted:
                continue
            text = d["text"]
            surv = surviving.get(cid, set())
            spans = []
            for sp in d.get("spans", []):
                if (sp["start"], sp["end"]) not in surv:
                    continue
                val = text[sp["start"]:sp["end"]]
                spans.append(LabelledSpan(
                    start=sp["start"], end=sp["end"], label=sp["label"],
                    value=val, signals=("synthetic",), confidence=1.0,
                ))
            yield LabelledExample(
                id=cid, text=text, language=d["language"],
                subset=subset,
                doc_id=d.get("template_id") or cid,
                spans=spans,
                labelers=["synthetic:" + subset],
                meta={"template_id": d.get("template_id"),
                      "source": d.get("source")},
            )


# ------------------------------------------------------------------ main


def main() -> None:
    rng = random.Random(SEED)
    print(f"the pipeline balance — seed={SEED}, floor={CONFIDENCE_FLOOR}, "
          f"per-doc={PER_DOC_CAP}, per-value={PER_VALUE_CAP}, "
          f"negatives={int(NEGATIVES_FRACTION * 100)}%/lang")
    print()

    print(f"[Pass 1] Loading {ASSEMBLED_PATH} …")
    real = _load_assembled_metadata()
    print(f"[Pass 1] Loading {SYNTH_MULTILANG_DOCS_PATH} …")
    syn_multilang_docs = _load_synthetic_metadata(SYNTH_MULTILANG_DOCS_PATH, "synthetic_multilang_docs")
    print(f"  layer1: {len(syn_multilang_docs):,} chunks")
    print(f"[Pass 1] Loading {SLOT_FILL_PATH} …")
    syn_slot_fill = _load_synthetic_metadata(SLOT_FILL_PATH, "synthetic_slot_fill")
    print(f"  layer9: {len(syn_slot_fill):,} chunks")
    print(f"[Pass 1] Loading {SYNTH_RM_SECRETS_PATH} …")
    syn_rm_secrets = _load_synthetic_metadata(
        SYNTH_RM_SECRETS_PATH, "synthetic_rm_secrets",
    )
    print(f"  rm_secrets: {len(syn_rm_secrets):,} chunks")
    # v3 synthetic layers
    print(f"[Pass 1] Loading {SYNTH_EMAILS_PATH} …")
    syn_emails = _load_synthetic_metadata(
        SYNTH_EMAILS_PATH, "synthetic_emails",
    )
    print(f"  v3 emails: {len(syn_emails):,} chunks")
    print(f"[Pass 1] Loading {SYNTH_DOCS_PATH} …")
    syn_docs = _load_synthetic_metadata(SYNTH_DOCS_PATH, "synthetic_docs")
    print(f"  v3 docs: {len(syn_docs):,} chunks")
    print(f"[Pass 1] Loading {SYNTH_SHORT_FORM_PATH} …")
    syn_short_form = _load_synthetic_metadata(
        SYNTH_SHORT_FORM_PATH, "synthetic_short_form",
    )
    print(f"  v3 short_form: {len(syn_short_form):,} chunks")
    print(f"[Pass 1] Loading {SYNTH_FORMS_PATH} …")
    syn_forms = _load_synthetic_metadata(SYNTH_FORMS_PATH, "synthetic_forms")
    print(f"  v3 forms: {len(syn_forms):,} chunks")
    print(f"[Pass 1] Loading {SYNTH_COMMON_WORD_PATH} …")
    syn_common_word = _load_synthetic_metadata(
        SYNTH_COMMON_WORD_PATH, "synthetic_common_word",
    )
    print(f"  v3 common_word: {len(syn_common_word):,} chunks")
    print(f"[Pass 1] Loading {SYNTH_ADVERSARIAL_PATH} …")
    syn_adversarial = _load_synthetic_metadata(
        SYNTH_ADVERSARIAL_PATH, "synthetic_adversarial",
    )
    print(f"  v3 adversarial: {len(syn_adversarial):,} chunks")
    print()

    all_chunks = (
        real + syn_multilang_docs + syn_slot_fill + syn_rm_secrets
        + syn_emails + syn_docs + syn_short_form
        + syn_forms + syn_common_word + syn_adversarial
    )
    print(f"Combined pool: {len(all_chunks):,} chunks")
    print()

    print("Phase A (per-doc cap):")
    kept_a = _phase_a_per_doc_cap(all_chunks, rng)
    print()

    print("Phase B (per-value cap):")
    surviving = _phase_b_per_value_cap(all_chunks, kept_a)
    print()

    print("Phase C (per-(lang, source, cat) cell caps):")
    admitted_pos, cell_count = _phase_c_cell_caps(
        all_chunks, kept_a, surviving, rng,
    )
    print()

    print("Phase D (negatives):")
    admitted_neg = _phase_d_negatives(
        all_chunks, kept_a, admitted_pos, surviving, rng,
    )
    print()

    # ---- Pass 2: write output ----
    admitted_all = admitted_pos | admitted_neg
    # Split admitted ids by source for routing in pass 2.
    real_ids = {c.id for c in real if c.id in admitted_all}
    syn_multilang_docs_ids = {c.id for c in syn_multilang_docs if c.id in admitted_all}
    syn_slot_fill_ids = {c.id for c in syn_slot_fill if c.id in admitted_all}
    syn_rm_secrets_ids = {c.id for c in syn_rm_secrets if c.id in admitted_all}
    syn_ids = {
        "synthetic_emails": {c.id for c in syn_emails if c.id in admitted_all},
        "synthetic_docs": {c.id for c in syn_docs if c.id in admitted_all},
        "synthetic_short_form": {c.id for c in syn_short_form if c.id in admitted_all},
        "synthetic_forms": {c.id for c in syn_forms if c.id in admitted_all},
        "synthetic_common_word": {c.id for c in syn_common_word if c.id in admitted_all},
        "synthetic_adversarial": {c.id for c in syn_adversarial if c.id in admitted_all},
    }
    print(f"[Pass 2] Writing {OUT_PATH} …")
    print(f"  real: {len(real_ids):,}  l1: {len(syn_multilang_docs_ids):,}  "
          f"l9: {len(syn_slot_fill_ids):,}  rm_secrets: {len(syn_rm_secrets_ids):,}  "
          f"")
    print(f"  v3 emails: {len(syn_ids['synthetic_emails']):,}  "
          f"docs: {len(syn_ids['synthetic_docs']):,}  "
          f"short_form: {len(syn_ids['synthetic_short_form']):,}  "
          f"forms: {len(syn_ids['synthetic_forms']):,}  "
          f"common_word: {len(syn_ids['synthetic_common_word']):,}  "
          f"adversarial: {len(syn_ids['synthetic_adversarial']):,}")

    def _generate() -> Iterator[LabelledExample]:
        for d in _stream_assembled_records():
            cid = d["id"]
            if cid not in real_ids:
                continue
            yield _emit_real_chunk(d, surviving.get(cid, set()))
        yield from _emit_synthetic_chunks(
            SYNTH_MULTILANG_DOCS_PATH, "synthetic_multilang_docs", syn_multilang_docs_ids, surviving,
        )
        yield from _emit_synthetic_chunks(
            SLOT_FILL_PATH, "synthetic_slot_fill", syn_slot_fill_ids, surviving,
        )
        yield from _emit_synthetic_chunks(
            SYNTH_RM_SECRETS_PATH, "synthetic_rm_secrets", syn_rm_secrets_ids,
            surviving,
        )
        # v3 synthetic layers
        v3_paths = {
            "synthetic_emails": SYNTH_EMAILS_PATH,
            "synthetic_docs": SYNTH_DOCS_PATH,
            "synthetic_short_form": SYNTH_SHORT_FORM_PATH,
            "synthetic_forms": SYNTH_FORMS_PATH,
            "synthetic_common_word": SYNTH_COMMON_WORD_PATH,
            "synthetic_adversarial": SYNTH_ADVERSARIAL_PATH,
        }
        for source_name, path in v3_paths.items():
            yield from _emit_synthetic_chunks(
                path, source_name, syn_ids[source_name], surviving,
            )

    n = write_jsonl(OUT_PATH, _generate())
    print(f"  wrote {n:,} examples")
    print()

    # ---- Final report ----
    print("=" * 60)
    print("the pipeline balance summary")
    print("=" * 60)
    print(f"Output: {OUT_PATH}  ({n:,} chunks)")
    print()
    print("Per (lang × source) chunk counts:")
    by_ls: Counter[tuple[str, str]] = Counter()
    by_lang: Counter[str] = Counter()
    n_pos = n_neg = 0
    n_span_signals_dist: Counter[int] = Counter()
    for d in _stream_assembled_records():
        if d["id"] not in real_ids:
            continue
        by_ls[(d["language"], d["subset"])] += 1
        by_lang[d["language"]] += 1
        surv = surviving.get(d["id"], set())
        kept_spans = [sp for sp in d.get("spans", [])
                      if (sp["start"], sp["end"]) in surv]
        if kept_spans:
            n_pos += 1
            for sp in kept_spans:
                n_sig = len([s for s in sp["signals"] if s != "audit"])
                n_span_signals_dist[n_sig] += 1
        else:
            n_neg += 1
    # add synthetic to per-(lang,source) counts
    for synth_pool in (
        syn_multilang_docs, syn_slot_fill, syn_rm_secrets,
        syn_emails, syn_docs, syn_short_form,
        syn_forms, syn_common_word, syn_adversarial,
    ):
        for c in synth_pool:
            if c.id in admitted_all:
                by_ls[(c.language, c.subset)] += 1
                by_lang[c.language] += 1
                n_pos += 1 if surviving.get(c.id) else 0
                n_neg += 0 if surviving.get(c.id) else 1
                for sp in c.spans:
                    if (sp.start, sp.end) in surviving.get(c.id, set()):
                        n_span_signals_dist[1] += 1  # synthetic = 1 signal

    print(f"  {'lang':<8} {'source':<18} {'chunks':>10}")
    for (la, src), k in sorted(by_ls.items(), key=lambda x: -x[1]):
        print(f"  {la:<8} {src:<18} {k:>10,}")
    print()
    print(f"Totals: positives={n_pos:,}  negatives={n_neg:,}  total={n:,}")
    print()
    print(f"Per-language: {dict(by_lang)}")
    print()
    print("Signal-strength of surviving spans (non-audit signals):")
    total = sum(n_span_signals_dist.values())
    for k in sorted(n_span_signals_dist):
        v = n_span_signals_dist[k]
        print(f"  {k} signals: {v:>9,} ({100*v/total:5.1f}%)")

    # Persist a small summary JSON so the paper / model card has a
    # single source of truth for the v2 balance.
    summary = {
        "_meta": {
            "seed": SEED,
            "confidence_floor": CONFIDENCE_FLOOR,
            "per_doc_cap": PER_DOC_CAP,
            "per_value_cap": PER_VALUE_CAP,
            "negatives_fraction": NEGATIVES_FRACTION,
        },
        "n_chunks": n,
        "n_positives": n_pos,
        "n_negatives": n_neg,
        "by_language": dict(by_lang),
        "by_lang_source": {f"{la}__{src}": k for (la, src), k in by_ls.items()},
        "signal_strength_dist": {str(k): v for k, v in n_span_signals_dist.items()},
        "total_surviving_spans": total,
    }
    summary_path = OUT_PATH.with_name("balanced_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
