"""CPU benchmark for openai/privacy-filter."""
import os
# Force CPU before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import time
import statistics
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

MODEL_ID = "openai/privacy-filter"
DEVICE = "cpu"

print(f"torch={torch.__version__} threads={torch.get_num_threads()}")

print("Loading tokenizer + model on CPU...")
t0 = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_ID, dtype=torch.float32
).to(DEVICE)
model.eval()
load_time = time.perf_counter() - t0
n_params = sum(p.numel() for p in model.parameters())
print(f"Loaded in {load_time:.2f}s | params={n_params/1e9:.3f}B | dtype={next(model.parameters()).dtype}")

# Sample inputs of varied length
SHORT = "My name is Alice Smith and my email is alice@example.com."
MEDIUM = (
    "Hi, my name is Harry Potter, I live at 4 Privet Drive in Little Whinging. "
    "You can reach me at harry.potter@hogwarts.edu or call +44 20 7946 0958. "
    "My account number is 1234-5678-9012 and my SSN is 123-45-6789. "
    "Please send the package to Hermione Granger at the same address by 2024-09-01."
) * 1
LONG = MEDIUM * 8  # ~ a few hundred tokens
HUGE = MEDIUM * 64  # very long context

samples = {
    "short": SHORT,
    "medium": MEDIUM,
    "long": LONG,
    "huge": HUGE,
}

def bench_raw(text: str, n_warmup=2, n_runs=5):
    inputs = tokenizer(text, return_tensors="pt", truncation=False).to(DEVICE)
    n_tokens = inputs["input_ids"].shape[1]
    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**inputs)
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = model(**inputs)
            times.append(time.perf_counter() - t0)
    return n_tokens, times

print("\n=== Raw forward pass (no post-processing) ===")
print(f"{'name':<8} {'tokens':>7} {'mean(ms)':>10} {'p50(ms)':>10} {'p95(ms)':>10} {'tok/s':>10}")
for name, text in samples.items():
    n_tok, times = bench_raw(text)
    mean_s = statistics.mean(times)
    p50 = statistics.median(times)
    p95 = sorted(times)[max(0, int(len(times)*0.95)-1)]
    tps = n_tok / mean_s
    print(f"{name:<8} {n_tok:>7d} {mean_s*1000:>10.2f} {p50*1000:>10.2f} {p95*1000:>10.2f} {tps:>10.1f}")

# Pipeline benchmark (includes tokenization + post-processing aggregation)
print("\n=== Pipeline (tokenize + forward + aggregate) ===")
clf = pipeline(
    task="token-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1,  # CPU
    aggregation_strategy="simple",
)

def bench_pipe(text: str, n_warmup=2, n_runs=5):
    n_tokens = len(tokenizer(text)["input_ids"])
    for _ in range(n_warmup):
        clf(text)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        out = clf(text)
        times.append(time.perf_counter() - t0)
    return n_tokens, times, out

print(f"{'name':<8} {'tokens':>7} {'mean(ms)':>10} {'p50(ms)':>10} {'tok/s':>10}")
last_outputs = {}
for name, text in samples.items():
    n_tok, times, out = bench_pipe(text)
    mean_s = statistics.mean(times)
    p50 = statistics.median(times)
    tps = n_tok / mean_s
    print(f"{name:<8} {n_tok:>7d} {mean_s*1000:>10.2f} {p50*1000:>10.2f} {tps:>10.1f}")
    last_outputs[name] = out

print("\n=== Sample output (medium) ===")
for ent in last_outputs["medium"][:10]:
    print(f"  {ent['entity_group']:<18} score={ent['score']:.4f} word={ent['word']!r}")
print(f"  ... ({len(last_outputs['medium'])} entities total)")
