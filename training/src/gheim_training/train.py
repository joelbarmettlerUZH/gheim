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
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data.bioes import IGNORE_INDEX, encode_example
from .data.label_space import ID2LABEL, LABEL2ID, NUM_LABELS
from .data.schema import Example, Span


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)

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
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"{base_model} did not provide a fast tokenizer; offset mapping is required."
        )

    # Label space matches the base model exactly (see label_space.py), so the
    # pretrained classifier head weights are reused — no ignore_mismatched_sizes.
    model = AutoModelForTokenClassification.from_pretrained(
        base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        trust_remote_code=True,
    )
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()

    raw_dd = load_from_disk(str(cfg["dataset_dir"]))
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

    # Initialize wandb (only on rank 0 in DDP).
    report_to = cfg.get("report_to", []) or []
    if "wandb" in report_to:
        import os
        if int(os.environ.get("RANK", "0")) == 0:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "gheim-1"),
                name=cfg.get("wandb_run_name", Path(cfg["output_dir"]).name),
                entity=cfg.get("wandb_entity"),
                config=cfg,
            )

    ta_kwargs = dict(
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=_build_compute_metrics(),
    )

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
