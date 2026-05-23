"""ROUND 17 — full-scale bias sweep.

For each candidate bias ∈ {0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5}:
  - Run on ALL 237 pathology cases → category coverage + overlap-F1
  - Run on ALL 15,861 test_v1 chunks → strict-span F1 + char F1

Goal: find the bias that maximises pathology recovery without
regressing test_v1 (the headline 0.916 metric).

The calibrated inference path is a from-scratch reimplementation of
aggregation_strategy="simple" — apply bias to O logit, argmax, then
BIO-aggregate contiguous same-label tokens into spans. Length-preserving
char offsets via the tokenizer's offset_mapping.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification, AutoTokenizer

REPO = "joelbarmettler/gheim-ch-560m"
PATHOLOGY_PATH = Path("/home/joelbarmettler/projects/gheim/data/test_pathology.jsonl")
TEST_V1_DIR = Path("/home/joelbarmettler/projects/gheim/data/built_v2")  # has 'test' split

print(f"loading {REPO} on CUDA …", flush=True)
tok = AutoTokenizer.from_pretrained(REPO)
mdl = AutoModelForTokenClassification.from_pretrained(REPO).cuda().eval()
id2label = mdl.config.id2label
label2id = {v: k for k, v in id2label.items()}
O_ID = label2id["O"]
print(f"  {len(id2label)} labels, O at id {O_ID}", flush=True)


@torch.no_grad()
def detect_batch(texts: list[str], bias: float, batch_size: int = 64
                 ) -> list[list[tuple[int, int, str]]]:
    """Run model on a batch of texts with O-logit bias, return per-text spans."""
    all_spans: list[list[tuple[int, int, str]]] = []
    for batch_start in range(0, len(texts), batch_size):
        batch = texts[batch_start:batch_start + batch_size]
        # Per-input tokenisation (we need offsets per item)
        encs = [tok(t, return_tensors="pt", return_offsets_mapping=True,
                    truncation=True, max_length=512) for t in batch]
        # Pad manually to max length in batch
        max_len = max(e["input_ids"].shape[1] for e in encs)
        input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
        attn_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
        offsets_list = []
        for i, e in enumerate(encs):
            n = e["input_ids"].shape[1]
            input_ids[i, :n] = e["input_ids"][0]
            attn_mask[i, :n] = e["attention_mask"][0]
            offsets_list.append(e["offset_mapping"][0].tolist() + [(0, 0)] * (max_len - n))
        logits = mdl(input_ids=input_ids.cuda(),
                     attention_mask=attn_mask.cuda()).logits.cpu()
        logits[:, :, O_ID] -= bias  # the calibration knob
        pred_ids = logits.argmax(dim=-1).tolist()
        for i, text in enumerate(batch):
            spans = []
            cur_start, cur_end, cur_cat = None, None, None
            for (s, e), pid in zip(offsets_list[i], pred_ids[i]):
                if s == e:
                    continue
                lab = id2label[pid]
                cat = None if lab == "O" else lab.split("-", 1)[1]
                if cat == cur_cat and cat is not None:
                    cur_end = e
                else:
                    if cur_cat is not None:
                        spans.append((cur_start, cur_end, cur_cat))
                    cur_start, cur_end, cur_cat = s, e, cat
            if cur_cat is not None:
                spans.append((cur_start, cur_end, cur_cat))
            all_spans.append(spans)
    return all_spans


# ─── Pathology eval ────────────────────────────────────────────────────
print("\nloading pathology …", flush=True)
pathology = []
with PATHOLOGY_PATH.open() as f:
    for line in f:
        if line.strip():
            pathology.append(json.loads(line))
print(f"  {len(pathology)} pathology cases", flush=True)


def overlap(a, b, c, d):
    return a < d and c < b


def eval_pathology(bias: float) -> dict:
    texts = [r["text"] for r in pathology]
    preds = detect_batch(texts, bias, batch_size=32)
    full_cov, n_examples = 0, len(pathology)
    o_tp = defaultdict(int); o_fp = defaultdict(int); o_fn = defaultdict(int)
    for r, pred in zip(pathology, preds):
        gold = [(int(s["start"]), int(s["end"]), s["label"]) for s in r["spans"]]
        gold_cats = {g[2] for g in gold}
        pred_cats = {p[2] for p in pred}
        if gold_cats <= pred_cats:
            full_cov += 1
        # overlap-F1 per category
        gold_matched = [False] * len(gold)
        pred_matched = [False] * len(pred)
        for gi, (gs, ge, gl) in enumerate(gold):
            for pi, (ps, pe, pl) in enumerate(pred):
                if gl == pl and overlap(gs, ge, ps, pe):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
        for gi, m in enumerate(gold_matched):
            if m: o_tp[gold[gi][2]] += 1
            else: o_fn[gold[gi][2]] += 1
        for pi, m in enumerate(pred_matched):
            if not m: o_fp[pred[pi][2]] += 1
    tp = sum(o_tp.values()); fp = sum(o_fp.values()); fn = sum(o_fn.values())
    p = tp / max(1, tp + fp)
    rr = tp / max(1, tp + fn)
    f1 = 2 * p * rr / max(1e-12, p + rr) if (p + rr) else 0.0
    return {"bias": bias, "full_cov": full_cov / n_examples,
            "overlap_f1": f1, "overlap_p": p, "overlap_r": rr,
            "tp": tp, "fp": fp, "fn": fn}


# ─── test_v1 eval ──────────────────────────────────────────────────────
print("\nloading test_v1 …", flush=True)
dd = load_from_disk(str(TEST_V1_DIR))
test_v1 = list(dd["test"])
print(f"  {len(test_v1)} test_v1 chunks", flush=True)


def eval_test_v1(bias: float) -> dict:
    """Strict-span F1 (any span where (start, end, label) exactly match)."""
    texts = [r["text"] for r in test_v1]
    preds = detect_batch(texts, bias, batch_size=64)
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    o_tp = defaultdict(int); o_fp = defaultdict(int); o_fn = defaultdict(int)
    for r, pred in zip(test_v1, preds):
        gold = {(int(s["start"]), int(s["end"]), s["label"]) for s in r["spans"]}
        pred_set = set(pred)
        for s in gold & pred_set:
            tp[s[2]] += 1
        for s in pred_set - gold:
            fp[s[2]] += 1
        for s in gold - pred_set:
            fn[s[2]] += 1
        # Also overlap-F1 since strict matches are noisy under BPE merges
        gold_matched = [False] * len(gold)
        pred_matched = [False] * len(pred)
        gold_list = list(gold)
        for gi, (gs, ge, gl) in enumerate(gold_list):
            for pi, (ps, pe, pl) in enumerate(pred):
                if gl == pl and overlap(gs, ge, ps, pe):
                    gold_matched[gi] = True
                    pred_matched[pi] = True
        for gi, m in enumerate(gold_matched):
            if m: o_tp[gold_list[gi][2]] += 1
            else: o_fn[gold_list[gi][2]] += 1
        for pi, m in enumerate(pred_matched):
            if not m: o_fp[pred[pi][2]] += 1
    tp_t = sum(tp.values()); fp_t = sum(fp.values()); fn_t = sum(fn.values())
    p_strict = tp_t / max(1, tp_t + fp_t)
    r_strict = tp_t / max(1, tp_t + fn_t)
    f1_strict = 2 * p_strict * r_strict / max(1e-12, p_strict + r_strict) if (p_strict + r_strict) else 0.0
    o_tp_t = sum(o_tp.values()); o_fp_t = sum(o_fp.values()); o_fn_t = sum(o_fn.values())
    p_o = o_tp_t / max(1, o_tp_t + o_fp_t)
    r_o = o_tp_t / max(1, o_tp_t + o_fn_t)
    f1_o = 2 * p_o * r_o / max(1e-12, p_o + r_o) if (p_o + r_o) else 0.0
    return {"bias": bias,
            "strict_f1": f1_strict, "strict_p": p_strict, "strict_r": r_strict,
            "overlap_f1": f1_o, "overlap_p": p_o, "overlap_r": r_o,
            "tp_strict": tp_t, "fp_strict": fp_t, "fn_strict": fn_t}


# ─── Sweep ─────────────────────────────────────────────────────────────
BIASES = [0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

print("\n=== PATHOLOGY (237 cases) — sweep across all 8 biases ===")
path_results = []
for b in BIASES:
    t0 = time.time()
    r = eval_pathology(b)
    print(f"  bias={b:>4.2f}  full_cov={r['full_cov']*100:>5.1f}%  "
          f"overlap_f1={r['overlap_f1']:.4f}  "
          f"P={r['overlap_p']:.4f} R={r['overlap_r']:.4f}  "
          f"TP={r['tp']:>4} FP={r['fp']:>4} FN={r['fn']:>4}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    path_results.append(r)

print("\n=== test_v1 (15,861 chunks) — sweep across all 8 biases ===")
t1_results = []
for b in BIASES:
    t0 = time.time()
    r = eval_test_v1(b)
    print(f"  bias={b:>4.2f}  strict_f1={r['strict_f1']:.4f}  "
          f"strict_P={r['strict_p']:.4f} R={r['strict_r']:.4f}  "
          f"overlap_f1={r['overlap_f1']:.4f}  "
          f"FP={r['fp_strict']:>5} FN={r['fn_strict']:>5}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    t1_results.append(r)

# Save full report
out_path = Path("/home/joelbarmettler/projects/gheim/eval/calibration_sweep.json")
out_path.write_text(json.dumps(
    {"pathology": path_results, "test_v1": t1_results, "model": REPO},
    indent=2,
))
print(f"\nfull report → {out_path}")
