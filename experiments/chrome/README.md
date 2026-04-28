# transformers.js browser smoke

Manual smoke test that `@huggingface/transformers` can load, download a model,
and run token-classification inference in a real Chrome tab — with the same
output shape our JS `LocalDetector` depends on.

## Run

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory experiments/chrome
# → http://127.0.0.1:8765/transformers-smoke.html
```

## What it confirms

Running against `@huggingface/transformers@4.2.0` + `Xenova/bert-base-NER`:

| property | result |
| --- | --- |
| library loads from jsdelivr | ✅ |
| model downloads from HuggingFace | ✅ |
| `token-classification` pipeline runs | ✅ |
| `aggregation_strategy: 'simple'` produces `entity_group` | ✅ |
| output contains `start` / `end` char offsets | ❌ |

transformers.js (v4.2) does **not** emit character offsets for aggregated
entities. `gheim.LocalDetector` (JS) compensates by falling back to substring
search (`text.indexOf(word, cursor)`), preserving left-to-right ordering for
repeated surfaces. Unit tests cover the fallback path in
`tests/local-detector.test.ts`.

## Loaded result

`window.__GHEIM_SMOKE` is exposed for scripted inspection:

```json
{
  "status": "ok",
  "fields": ["entity_group", "score", "word"],
  "hasEntityGroup": true,
  "hasCharOffsets": false,
  "spans": [
    {"entity_group": "PER", "score": 0.965, "word": "Alice"},
    {"entity_group": "ORG", "score": 0.998, "word": "Microsoft"},
    {"entity_group": "LOC", "score": 0.995, "word": "Seattle"}
  ]
}
```
