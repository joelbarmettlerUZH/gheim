"""V2-10: stratified document-level train/val/test split of v2_balanced.

Reads ``data/balanced.jsonl`` and writes three split files plus a
summary JSON. Targets 80/10/10 by chunk count, stratified by
``(language × source)`` (20 strata), with two grouping rules:

- **Real-text** chunks group by ``doc_id``. All chunks from the same
  source document end up in the same split. This is non-negotiable:
  chunks from the same court ruling share register, named entities, and
  writing style — splitting at chunk-level would inflate F1 via shared-
  doc leakage.

- **Synthetic** chunks group by ``id`` (chunk-level). For synthetic the
  ``doc_id`` is a template_id, but two chunks sharing a template are
  independent slot-fill samples (different Faker values), so chunk-
  level splitting is correct and gives the test set proportional
  template coverage.

After splitting, a coverage report flags any (lang × cat) cell that
ends up empty in val or test — these would yield ``n/a`` F1 in the
eval matrix and need separate attention if they appear.

Outputs
-------
- ``data/balanced_train.jsonl``  (~80% of chunks)
- ``data/balanced_val.jsonl``    (~10%)
- ``data/balanced_test.jsonl``   (~10%)
- ``data/split_summary.json``    per-split + per-(lang × cat) counts

Run
---
    uv run python -m gheim_training.data.split
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .balance import SEED, SYNTHETIC_SOURCES
from .schema import V2Example, read_jsonl, write_jsonl

IN_PATH = Path("data/balanced.jsonl")  # was v2_balanced; now reads v3 output
OUT_TRAIN = Path("data/balanced_train.jsonl")
OUT_VAL = Path("data/balanced_val.jsonl")
OUT_TEST = Path("data/balanced_test.jsonl")
SUMMARY = Path("data/split_summary.json")

TARGET_VAL = 0.10
TARGET_TEST = 0.10

LANGS = ("de_ch", "fr_ch", "it_ch", "rm", "en")
CATS = (
    "account_number", "private_address", "private_date", "private_email",
    "private_person", "private_phone", "private_url", "secret",
)


def _group_key(ex: V2Example) -> tuple[str, str, str]:
    """Return the (lang, source, group_id) tuple used as a split unit.

    For synthetic sources the group is the chunk itself; for real text
    the group is the source document so all chunks from it travel
    together.
    """
    if ex.subset in SYNTHETIC_SOURCES:
        return (ex.language, ex.subset, ex.id)
    return (ex.language, ex.subset, ex.doc_id)


def _assign_groups_to_splits(
    stratum_groups: list[tuple[str, list[V2Example]]],
    rng: random.Random,
) -> tuple[set[str], set[str], set[str]]:
    """Greedily assign groups within one (lang, source) stratum to
    train/val/test by chunk count, targeting 80/10/10.

    Groups are shuffled then walked in order; each is assigned to
    whichever split is furthest from its target ratio so far. This keeps
    splits within ~1-2 chunks of target without depending on group-size
    homogeneity.
    """
    rng.shuffle(stratum_groups)
    total = sum(len(chunks) for _, chunks in stratum_groups)
    target = {
        "train": total * (1 - TARGET_VAL - TARGET_TEST),
        "val": total * TARGET_VAL,
        "test": total * TARGET_TEST,
    }
    assigned: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    count: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    for gid, chunks in stratum_groups:
        # Pick split with the largest *deficit* relative to its target.
        deficits = {s: target[s] - count[s] for s in ("train", "val", "test")}
        pick = max(deficits, key=deficits.get)
        assigned[pick].append(gid)
        count[pick] += len(chunks)
    return (set(assigned["train"]), set(assigned["val"]), set(assigned["test"]))


def main() -> None:
    rng = random.Random(SEED)

    print(f"V2-10 split — seed={SEED}, target=80/10/10 by chunk count, "
          f"stratify=(language × source)")
    print()

    print(f"Loading {IN_PATH} …")
    # Group chunks by (lang, source, group_id) where group_id is doc_id
    # for real-text and chunk_id for synthetic.
    groups: dict[tuple[str, str], dict[str, list[V2Example]]] = defaultdict(
        lambda: defaultdict(list),
    )
    n_chunks = 0
    for ex in read_jsonl(IN_PATH):
        lang, src, gid = _group_key(ex)
        groups[(lang, src)][gid].append(ex)
        n_chunks += 1
    print(f"  {n_chunks:,} chunks loaded across "
          f"{len(groups)} (language × source) strata")
    print()

    # Per-stratum split decisions.
    train_gids: dict[tuple[str, str], set[str]] = {}
    val_gids: dict[tuple[str, str], set[str]] = {}
    test_gids: dict[tuple[str, str], set[str]] = {}

    print("Per-stratum split sizes:")
    print(f"  {'lang':<8} {'source':<18} {'groups':>8} {'chunks':>8} "
          f"{'train':>7} {'val':>6} {'test':>6}")
    for (lang, src), gdict in sorted(groups.items()):
        stratum_list = list(gdict.items())
        tr, va, te = _assign_groups_to_splits(stratum_list, rng)
        train_gids[(lang, src)] = tr
        val_gids[(lang, src)] = va
        test_gids[(lang, src)] = te
        n_tr = sum(len(gdict[g]) for g in tr)
        n_va = sum(len(gdict[g]) for g in va)
        n_te = sum(len(gdict[g]) for g in te)
        n_g = len(stratum_list)
        n_c = sum(len(c) for _, c in stratum_list)
        print(f"  {lang:<8} {src:<18} {n_g:>8,} {n_c:>8,} "
              f"{n_tr:>7,} {n_va:>6,} {n_te:>6,}")
    print()

    # Emit the three files in one pass over the input.
    train_buf: list[V2Example] = []
    val_buf: list[V2Example] = []
    test_buf: list[V2Example] = []
    for ex in read_jsonl(IN_PATH):
        lang, src, gid = _group_key(ex)
        if gid in train_gids[(lang, src)]:
            train_buf.append(ex)
        elif gid in val_gids[(lang, src)]:
            val_buf.append(ex)
        elif gid in test_gids[(lang, src)]:
            test_buf.append(ex)

    n_train = write_jsonl(OUT_TRAIN, train_buf)
    n_val = write_jsonl(OUT_VAL, val_buf)
    n_test = write_jsonl(OUT_TEST, test_buf)
    print(f"Wrote: train={n_train:,}  val={n_val:,}  test={n_test:,}  "
          f"total={n_train + n_val + n_test:,}")
    print()

    # ---- Coverage sanity check ----
    def _per_lang_cat(buf: list[V2Example]) -> Counter[tuple[str, str]]:
        out: Counter[tuple[str, str]] = Counter()
        for ex in buf:
            for sp in ex.spans:
                out[(ex.language, sp.label)] += 1
        return out

    cov_train = _per_lang_cat(train_buf)
    cov_val = _per_lang_cat(val_buf)
    cov_test = _per_lang_cat(test_buf)

    print("Per (lang × cat) span coverage (train / val / test):")
    print(f"  {'lang':<8} {'cat':<20} {'train':>8} {'val':>6} {'test':>6}")
    empty_val: list[tuple[str, str]] = []
    empty_test: list[tuple[str, str]] = []
    for la in LANGS:
        for cat in CATS:
            t = cov_train[(la, cat)]
            v = cov_val[(la, cat)]
            te = cov_test[(la, cat)]
            if t + v + te == 0:
                continue
            print(f"  {la:<8} {cat:<20} {t:>8,} {v:>6,} {te:>6,}")
            if v == 0:
                empty_val.append((la, cat))
            if te == 0:
                empty_test.append((la, cat))
    print()
    if empty_val:
        print(f"WARNING: {len(empty_val)} (lang × cat) cells empty in val: "
              f"{empty_val}")
    if empty_test:
        print(f"WARNING: {len(empty_test)} (lang × cat) cells empty in test: "
              f"{empty_test}")
    if not empty_val and not empty_test:
        print("All (lang × cat) cells have at least one span in val + test ✓")
    print()

    # ---- Per-split language + source distributions ----
    def _per_lang(buf: list[V2Example]) -> Counter[str]:
        return Counter(ex.language for ex in buf)

    def _per_src(buf: list[V2Example]) -> Counter[str]:
        return Counter(ex.subset for ex in buf)

    summary = {
        "_meta": {
            "seed": SEED,
            "target_val": TARGET_VAL,
            "target_test": TARGET_TEST,
            "stratify": "(language, source)",
            "grouping": {
                "real_text": "by doc_id (all chunks from a source doc "
                             "stay together)",
                "synthetic": "by chunk_id (template-only `doc_id` is not "
                             "a real grouping signal)",
            },
        },
        "splits": {
            "train": {
                "n_chunks": n_train,
                "by_language": dict(_per_lang(train_buf)),
                "by_source": dict(_per_src(train_buf)),
                "by_lang_cat_spans": {
                    f"{la}__{cat}": cov_train[(la, cat)]
                    for la in LANGS for cat in CATS
                    if cov_train[(la, cat)] > 0
                },
            },
            "val": {
                "n_chunks": n_val,
                "by_language": dict(_per_lang(val_buf)),
                "by_source": dict(_per_src(val_buf)),
                "by_lang_cat_spans": {
                    f"{la}__{cat}": cov_val[(la, cat)]
                    for la in LANGS for cat in CATS
                    if cov_val[(la, cat)] > 0
                },
            },
            "test": {
                "n_chunks": n_test,
                "by_language": dict(_per_lang(test_buf)),
                "by_source": dict(_per_src(test_buf)),
                "by_lang_cat_spans": {
                    f"{la}__{cat}": cov_test[(la, cat)]
                    for la in LANGS for cat in CATS
                    if cov_test[(la, cat)] > 0
                },
            },
        },
        "coverage_warnings": {
            "val_empty_cells": [f"{la}__{cat}" for la, cat in empty_val],
            "test_empty_cells": [f"{la}__{cat}" for la, cat in empty_test],
        },
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {SUMMARY}")


if __name__ == "__main__":
    main()
