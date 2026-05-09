"""LLRD smoke test: load each base model and verify _build_llrd_param_groups
detects the layer structure, assigns LRs sensibly, and covers all trainable params.

For each model, prints:
  - n_layers detected
  - LR at embeddings / layer 0 / layer N-1 / head
  - Param counts in each bucket (embed/head/layer/other)
  - Confirmation that # of grouped params == # of trainable params

Run:
  uv run --project training python -m gheim_training.scripts.smoke_llrd
"""
from __future__ import annotations

import sys

from gheim_training.data.label_space import ID2LABEL, LABEL2ID, NUM_LABELS
from gheim_training.train import _build_llrd_param_groups, _LAYER_RE


MODELS_TO_TEST = [
    ("ZurichNLP/swissbert",         "swissbert (270M)"),
    ("FacebookAI/xlm-roberta-large", "xlm-r-large (550M)"),
    ("openai/privacy-filter",        "privacy-filter (1.4B MoE)"),
]


def _check(model_id: str, label: str) -> bool:
    print()
    print("=" * 80)
    print(f"MODEL: {label}  ({model_id})")
    print("=" * 80)

    from transformers import AutoModelForTokenClassification

    print(f"Loading {model_id} ...", flush=True)
    try:
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
    except Exception as e:
        print(f"  FAILED to load: {e}")
        return False

    # Show a sample of parameter names so we can debug regex misses
    all_names = [n for n, _ in model.named_parameters()]
    print(f"  total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  total trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  named_parameters count: {len(all_names)}")
    layer_names = [n for n in all_names if _LAYER_RE.search(n)]
    embed_names = [n for n in all_names if "embed" in n.lower() and not _LAYER_RE.search(n)]
    head_names = [n for n in all_names
                  if any(k in n for k in ("classifier", "cls.", "lm_head", "predictions", "pooler"))
                  and not _LAYER_RE.search(n)]
    print(f"  param-name buckets: layer-matched={len(layer_names)}, "
          f"embedding={len(embed_names)}, head={len(head_names)}")
    print(f"  sample names (first 6):")
    for n in all_names[:6]:
        print(f"    {n}")
    print(f"  sample layer-matched names:")
    for n in layer_names[:3]:
        print(f"    {n}")
    print(f"  sample head names:")
    for n in head_names[:3]:
        print(f"    {n}")

    # Build LLRD groups at llrd=0.95
    print()
    print(f"Building LLRD param groups (base_lr=2e-5, llrd=0.95, weight_decay=0.01) ...")
    groups = _build_llrd_param_groups(model, base_lr=2e-5, llrd=0.95, weight_decay=0.01)

    if groups is None:
        print(f"  FAILED: _build_llrd_param_groups returned None — no layer structure detected.")
        return False

    n_grouped = sum(len(g["params"]) for g in groups)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"  groups: {len(groups)} (expected ≈ #trainable params = {n_trainable})")
    print(f"  total grouped params: {n_grouped} (expected = {n_trainable})")
    if n_grouped != n_trainable:
        print(f"  WARNING: param-group coverage mismatch")
        return False

    lrs = [g["lr"] for g in groups]
    print(f"  LR range: min={min(lrs):.2e}  max={max(lrs):.2e}")
    print(f"  unique LR values ({len(set(lrs))}):")
    for lr in sorted(set(lrs))[:5]:
        n = sum(1 for g in groups if g["lr"] == lr)
        print(f"    {lr:.2e}  ({n} param groups)")
    if len(set(lrs)) > 5:
        print(f"    ... and {len(set(lrs)) - 5} more")

    # Sanity: max LR should equal base_lr (head)
    if abs(max(lrs) - 2e-5) > 1e-9:
        print(f"  WARNING: max LR is {max(lrs):.2e}, expected 2.0e-5 (head LR)")
    print(f"  PASS")
    return True


def main() -> None:
    results = []
    for model_id, label in MODELS_TO_TEST:
        ok = _check(model_id, label)
        results.append((label, ok))

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for label, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not all(ok for _, ok in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
