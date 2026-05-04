# gheim training

Swiss-market fine-tuning pipeline for `openai/privacy-filter`. Produces
`joelbarmettler/gheim-1`, a drop-in token-classification checkpoint compatible
with `gheim.detectors.LocalDetector` and the gheim detection server.

## Layout

```
training/
├── configs/                        # hyperparams + accelerate config
├── src/gheim_training/
│   ├── data/
│   │   ├── label_space.py          # 33 BIOES tags — single source of truth
│   │   ├── schema.py               # canonical example dataclass
│   │   ├── bioes.py                # char-spans → token BIOES via tokenizer offsets
│   │   ├── synthetic/              # Layer 1: Faker-CH + handwritten templates
│   │   ├── apertus/                # Layer 3: Apertus-8B-Instruct slot-filled generation
│   │   ├── ai4privacy.py           # Layer 2: pii-masking-300k filter + remap
│   │   ├── english_anchor.py       # Layer 4: ~25% English to prevent forgetting
│   │   └── build.py                # combine, dedupe, stratified split
│   ├── scripts/
│   │   ├── tokenizer_audit.py      # fragmentation report on Swiss surface forms
│   │   └── push_to_hub.py          # publish joelbarmettler/gheim-1
│   ├── train.py                    # HF Trainer entrypoint
│   └── eval/
│       ├── harness.py              # F1 per entity × per language × per canton
│       ├── baselines/              # zero-shot, presidio_ch, swissbert_ner
│       ├── swissner.py             # ZurichNLP SwissNER loader
│       └── handcrafted/            # frozen tricky-case eval set
└── pyproject.toml
```

## Sequence

1. Tokenizer audit (cheap, run first)
2. Build datasets (Layer 1 + 2 + 3 + 4)
3. Train v1 (~1 day on 2x4090)
4. Eval against baselines
5. Push to hub
