"""Fine-tune ``openai/privacy-filter`` for the Swiss market.

Reads a config YAML (see ``configs/v1.yaml``) and a built dataset directory
(produced by ``data/build.py``), then trains via HF Trainer with seqeval BIOES
F1 as the primary metric.

Launch via accelerate for multi-GPU DDP:
    uv run accelerate launch \\
        --config_file configs/accelerate_ddp.yaml \\
        -m gheim_training.train \\
        --config configs/v1.yaml
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data.bioes import IGNORE_INDEX, encode_example
from .data.label_space import ID2LABEL, LABEL2ID, NUM_LABELS
from .data.schema import Example, Span

_LAYER_RE = re.compile(r"\.(?:layer|layers|h)\.(\d+)\.")
_NO_DECAY = ("bias", "LayerNorm.weight", "layer_norm.weight", "ln_f.weight")
_HEAD_KEYS = ("classifier", "cls.", "lm_head", "predictions", "pooler", "score.")


def _build_llrd_param_groups(
    model, base_lr: float, llrd: float, weight_decay: float,
) -> list[dict] | None:
    """Layer-wise LR decay grouping.

    Each transformer layer N (0 = closest to embeddings) gets:
        lr_N = base_lr * llrd ** (n_layers - 1 - N)

    Embeddings get the lowest LR (lr * llrd^n_layers); the classifier
    head keeps base_lr (no decay applied to the freshly-initialised head).

    Returns a list of param_groups for AdamW, or None if no layer
    structure is detected (caller should fall back to default optimizer).
    """
    max_layer = -1
    for name, _ in model.named_parameters():
        m = _LAYER_RE.search(name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    if max_layer < 0:
        return None
    n_layers = max_layer + 1

    groups: list[dict] = []
    seen: set[int] = set()
    counts = {"embed": 0, "head": 0, "layer": 0, "other": 0}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))

        m = _LAYER_RE.search(name)
        if m:
            layer_idx = int(m.group(1))
            depth_from_top = n_layers - 1 - layer_idx
            lr = base_lr * (llrd ** depth_from_top)
            counts["layer"] += 1
        elif "embed" in name.lower():
            lr = base_lr * (llrd ** n_layers)
            counts["embed"] += 1
        elif any(k in name for k in _HEAD_KEYS):
            lr = base_lr
            counts["head"] += 1
        else:
            lr = base_lr
            counts["other"] += 1

        wd = 0.0 if any(nd in name for nd in _NO_DECAY) else weight_decay
        groups.append({"params": [param], "lr": lr, "weight_decay": wd})

    print(
        f"LLRD={llrd}: n_layers={n_layers}, "
        f"lr@embed={base_lr * (llrd ** n_layers):.2e}, "
        f"lr@layer0={base_lr * (llrd ** (n_layers - 1)):.2e}, "
        f"lr@layer{n_layers-1}={base_lr:.2e}, "
        f"params: embed={counts['embed']} head={counts['head']} "
        f"layer={counts['layer']} other={counts['other']}",
        flush=True,
    )
    return groups


def _example_from_hf_record(rec: dict) -> Example:
    return Example(
        text=rec["text"],
        spans=[Span(**s) for s in rec["spans"]],
        language=rec["language"],
        source=rec["source"],
        template_id=rec.get("template_id") or None,
    )


def _build_compute_metrics():
    """seqeval-based F1 with overall + per-entity breakdowns."""
    from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

    def compute(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        true_labels: list[list[str]] = []
        pred_labels: list[list[str]] = []
        for p_row, l_row in zip(preds, labels, strict=True):
            t: list[str] = []
            p: list[str] = []
            for p_id, l_id in zip(p_row, l_row, strict=True):
                if l_id == IGNORE_INDEX:
                    continue
                t.append(ID2LABEL[int(l_id)])
                p.append(ID2LABEL[int(p_id)])
            true_labels.append(t)
            pred_labels.append(p)

        out: dict[str, float] = {
            "overall_f1": float(f1_score(true_labels, pred_labels)),
            "overall_precision": float(precision_score(true_labels, pred_labels)),
            "overall_recall": float(recall_score(true_labels, pred_labels)),
        }
        report = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0)
        for entity, scores in report.items():
            if not isinstance(scores, dict):
                continue
            if entity in {"micro avg", "macro avg", "weighted avg"}:
                continue
            out[f"f1/{entity}"] = float(scores.get("f1-score", 0.0))
        return out

    return compute


def _parse_override(spec: str) -> tuple[str, Any]:
    """Parse a "key=value" override into (key, typed_value).

    Values are interpreted as YAML so booleans, ints, floats, and lists work
    naturally (e.g. ``--override learning_rate=5e-6`` parses 5e-6 as float;
    ``--override freeze_embeddings=true`` parses to bool). Used by the
    wandb sweep wrapper to inject per-variant hyperparams without needing
    one YAML config per variant.
    """
    if "=" not in spec:
        raise ValueError(f"override must be 'key=value', got {spec!r}")
    key, value = spec.split("=", 1)
    return key, yaml.safe_load(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument(
        "--override", action="append", default=[],
        help="Override a YAML cfg field, e.g. --override learning_rate=5e-6. "
             "Repeatable. Parsed as YAML so types are preserved.",
    )
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    for spec in args.override:
        k, v = _parse_override(spec)
        cfg[k] = v
        print(f"override: {k} = {v!r}", flush=True)

    from datasets import load_from_disk
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        Trainer,
        TrainingArguments,
    )

    base_model = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)
    if tokenizer is None:
        raise RuntimeError(f"AutoTokenizer.from_pretrained({base_model!r}) returned None")
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{base_model} did not provide a fast tokenizer; offset mapping is required."
        )

    # Label space matches the base model exactly (see label_space.py), so the
    # pretrained classifier head weights are reused — no ignore_mismatched_sizes.
    # Bases with a different head shape (xlm-roberta, swissbert) get a
    # randomly-initialised head, which is the intended Phase 3 design.
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    # Optional: freeze embedding layer (sweep variant). Reduces trainable
    # parameter count, speeds training, and acts as a regulariser by
    # forcing the model to use the pretrained embedding semantics as-is.
    if cfg.get("freeze_embeddings"):
        n_frozen = 0
        for name, param in model.named_parameters():
            if "embedding" in name.lower() or "embed_tokens" in name.lower():
                param.requires_grad = False
                n_frozen += param.numel()
        print(f"freeze_embeddings=true: froze {n_frozen:,} embedding params",
              flush=True)

    # XMOD support (SwissBERT and similar): the model has language-specific
    # adapter sets and refuses to forward without a default language being
    # set. Honest limitation: HF Trainer feeds mixed-language batches without
    # the model knowing which adapter to use, so we pin a single default for
    # the whole run. This means the chosen adapter gets ALL the gradient
    # updates; other-language adapters stay frozen at their pretrained values.
    # Effectively makes swissbert a single-language fine-tune. Documented as
    # a known limitation for the bake-off.
    xmod_default = cfg.get("xmod_default_language")
    if xmod_default is not None and hasattr(model, "set_default_language"):
        model.set_default_language(xmod_default)
        print(f"XMOD default language set to {xmod_default!r}", flush=True)

    from datasets import DatasetDict
    raw_dd = load_from_disk(str(cfg["dataset_dir"]))
    if not isinstance(raw_dd, DatasetDict):
        raise RuntimeError(
            f"{cfg['dataset_dir']} loaded as {type(raw_dd).__name__}, expected DatasetDict"
        )
    max_len = int(cfg.get("max_seq_length", 1024))

    def _encode_batch(records: dict[str, list]) -> dict[str, list]:
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(records["text"])):
            ex = _example_from_hf_record({k: records[k][i] for k in records})
            enc = encode_example(ex, tokenizer, max_length=max_len)
            out["input_ids"].append(enc["input_ids"])
            out["attention_mask"].append(enc["attention_mask"])
            out["labels"].append(enc["labels"])
        return out

    encoded = raw_dd.map(
        _encode_batch,
        batched=True,
        batch_size=64,
        remove_columns=raw_dd["train"].column_names,
        desc="tokenize+align",
    )

    # Pick which split to use as the trainer's primary eval (the one used for
    # `metric_for_best_model` and early stopping). Defaults to "validation"
    # (in-distribution random split) but can point at "holdout" for OOD.
    eval_split_name = cfg.get("eval_split", "validation")
    if eval_split_name not in encoded:
        raise ValueError(
            f"eval_split={eval_split_name!r} not in dataset; available: {list(encoded)}"
        )
    eval_dataset = encoded[eval_split_name]

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=IGNORE_INDEX)

    # Initialize wandb (only on rank 0 in DDP). Fail-soft: if wandb's
    # network init hangs (we've seen 90s timeouts on a flaky upstream),
    # fall back to offline mode so a 5h training run isn't lost to a
    # transient connectivity issue. The bakeoff_xlmr run on 2026-05-04
    # died after 4 minutes for exactly this reason.
    report_to = cfg.get("report_to", []) or []
    if "wandb" in report_to:
        import os
        if int(os.environ.get("RANK", "0")) == 0:
            import wandb
            try:
                wandb.init(
                    project=cfg.get("wandb_project", "gheim-1"),
                    name=cfg.get("wandb_run_name", Path(cfg["output_dir"]).name),
                    entity=cfg.get("wandb_entity"),
                    config=cfg,
                    settings=wandb.Settings(init_timeout=180),
                )
            except Exception as exc:
                print(f"WARN: wandb online init failed ({exc!r}); falling back "
                      f"to WANDB_MODE=offline for this run.", flush=True)
                os.environ["WANDB_MODE"] = "offline"
                wandb.init(
                    project=cfg.get("wandb_project", "gheim-1"),
                    name=cfg.get("wandb_run_name", Path(cfg["output_dir"]).name),
                    entity=cfg.get("wandb_entity"),
                    config=cfg,
                )

    # ta_kwargs is built from a YAML cfg; values are heterogeneous and
    # TrainingArguments has narrowly-typed parameters per field. The runtime
    # validation below catches type errors; ty's static checking can't track
    # the field-by-field types from a dict-of-Any spread, so we use Any.
    ta_kwargs: dict[str, Any] = dict(
        output_dir=cfg["output_dir"],
        learning_rate=float(cfg["learning_rate"]),
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=float(cfg["warmup_ratio"]),
        num_train_epochs=float(cfg["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(cfg["gradient_accumulation_steps"]),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=int(cfg["logging_steps"]),
        eval_strategy=cfg["eval_strategy"],
        eval_steps=int(cfg["eval_steps"]),
        save_strategy=cfg["save_strategy"],
        save_total_limit=int(cfg["save_total_limit"]),
        load_best_model_at_end=bool(cfg["load_best_model_at_end"]),
        metric_for_best_model=cfg["metric_for_best_model"],
        greater_is_better=bool(cfg["greater_is_better"]),
        run_name=cfg.get("wandb_run_name", Path(cfg["output_dir"]).name),
        seed=int(cfg["seed"]),
        report_to=cfg.get("report_to", []),
        ddp_find_unused_parameters=False,
    )
    # Conditional kwargs (only present when relevant)
    if cfg.get("save_strategy") not in (None, "no") and "save_steps" in cfg:
        ta_kwargs["save_steps"] = int(cfg["save_steps"])
    if "max_steps" in cfg and cfg["max_steps"]:
        ta_kwargs["max_steps"] = int(cfg["max_steps"])

    training_args = TrainingArguments(**ta_kwargs)

    # Optional layer-wise LR decay: when set in cfg (e.g. llrd: 0.95),
    # build a custom AdamW optimizer with per-layer LRs that decay from
    # the head (full base_lr) down to the embeddings (base_lr * llrd^N).
    llrd = cfg.get("llrd")
    custom_optim = None
    if llrd is not None and float(llrd) < 1.0:
        from torch.optim import AdamW
        param_groups = _build_llrd_param_groups(
            model,
            base_lr=float(cfg["learning_rate"]),
            llrd=float(llrd),
            weight_decay=float(training_args.weight_decay),
        )
        if param_groups is not None:
            custom_optim = AdamW(
                param_groups,
                lr=float(cfg["learning_rate"]),
                betas=(training_args.adam_beta1, training_args.adam_beta2),
                eps=training_args.adam_epsilon,
            )
        else:
            print(f"WARN: llrd={llrd} requested but no layer structure detected; "
                  f"falling back to default optimizer", flush=True)

    trainer_kwargs: dict[str, Any] = dict(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_build_compute_metrics(),
    )
    if custom_optim is not None:
        trainer_kwargs["optimizers"] = (custom_optim, None)  # HF builds the scheduler

    trainer = Trainer(**trainer_kwargs)

    # Early stopping (only if patience configured)
    if cfg.get("early_stopping_patience"):
        from transformers import EarlyStoppingCallback
        trainer.add_callback(EarlyStoppingCallback(
            early_stopping_patience=int(cfg["early_stopping_patience"]),
            early_stopping_threshold=float(cfg.get("early_stopping_threshold", 0.0)),
        ))

    trainer.train()
    if cfg.get("save_strategy") not in (None, "no"):
        trainer.save_model(cfg["output_dir"])
        tokenizer.save_pretrained(cfg["output_dir"])
        with (Path(cfg["output_dir"]) / "label_space.json").open("w", encoding="utf-8") as f:
            import json
            json.dump({"id2label": ID2LABEL, "label2id": LABEL2ID, "num_labels": NUM_LABELS}, f, indent=2)


if __name__ == "__main__":
    # Sanity warning when launched without accelerate but more than one GPU is visible.
    if "WORLD_SIZE" not in os.environ and "RANK" not in os.environ:
        try:
            import torch
            n = torch.cuda.device_count()
            if n > 1:
                print(f"NOTE: {n} GPUs visible but no accelerate launch detected — running on rank 0 only.")
        except Exception:
            pass
    main()
