"""Standalone fp16 ONNX regenerator (historical / one-off diagnostic).

NOTE: the logic in this file has been folded into the production pipeline at
`training/src/gheim_training/eval/quantize_onnx.py::quantize_fp16`. Use that
for new exports. This file is kept as the standalone script that produced
`checkpoints/gheim-ch_onnx_fp16_v2/` (the export uploaded to the Hub) and is
referenced from `eval/q8_quality_report.md`.

The reason this script existed: the naked `convert_float_to_float16` output
was unloadable in any onnxruntime because XLM-R-large's attention block casts
to fp32 around its sqrt(d_k) for numerical stability, and the converter
doesn't propagate that mix through downstream Div/Mul/MatMul/LayerNormalization
ops. This script (and now `quantize_fp16`) post-processes the converter
output: iteratively inserts fp16->fp32 promotion Casts at type-match-op
boundaries, then strips the redundant trailing Cast wrapping the classifier
output.

Usage (only if you want to reproduce the historical artefact):
    uv run --project training python eval/regenerate_fp16_onnx.py \\
        --src checkpoints/gheim-ch_onnx \\
        --out checkpoints/gheim-ch_onnx_fp16_v2
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import warnings
from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnxconverter_common import float16

warnings.filterwarnings("ignore")

# Ops that require all (or designated subset of) inputs to share a dtype.
# `Where` takes (cond, then, else) — only then/else need to match.
# `LayerNormalization` takes (X, scale, bias?) — first 3 inputs must match.
_TYPE_MATCH_OPS = {
    "Div", "Mul", "Add", "Sub", "Pow", "Where", "Min", "Max",
    "MatMul", "Gemm", "Concat", "LayerNormalization",
}

_SHARED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
)


def _propagate_types(model: onnx.ModelProto) -> dict[str, int]:
    """Compute the data_type of every value in the graph by topological propagation."""
    dtype_of: dict[str, int] = {}
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type:
            dtype_of[inp.name] = inp.type.tensor_type.elem_type
    for init in model.graph.initializer:
        dtype_of[init.name] = init.data_type
    nodes = list(model.graph.node)
    for _ in range(50):
        progress = False
        for n in nodes:
            if any(o in dtype_of for o in n.output):
                continue
            if n.op_type == "Constant":
                for a in n.attribute:
                    if a.name == "value":
                        for o in n.output:
                            dtype_of[o] = a.t.data_type
                        progress = True
                        break
            elif n.op_type == "Cast":
                for a in n.attribute:
                    if a.name == "to":
                        for o in n.output:
                            dtype_of[o] = a.i
                        progress = True
                        break
            elif n.op_type in ("Shape", "Size"):
                for o in n.output:
                    dtype_of[o] = TensorProto.INT64
                progress = True
            elif n.op_type in ("Equal", "Greater", "Less", "GreaterOrEqual",
                                "LessOrEqual", "Not", "IsNaN", "IsInf"):
                for o in n.output:
                    dtype_of[o] = TensorProto.BOOL
                progress = True
            else:
                # For arithmetic ops, prefer fp32 if any input is fp32 (the
                # output of a mixed op promotes via our Cast insertion).
                input_types = [dtype_of.get(i) for i in n.input]
                ft = None
                if TensorProto.FLOAT in input_types:
                    ft = TensorProto.FLOAT
                elif TensorProto.FLOAT16 in input_types:
                    ft = TensorProto.FLOAT16
                else:
                    for it in input_types:
                        if it is not None:
                            ft = it
                            break
                if ft is not None:
                    for o in n.output:
                        dtype_of[o] = ft
                    progress = True
        if not progress:
            break
    return dtype_of


def _insert_promotion_casts(model: onnx.ModelProto, dtype_of: dict[str, int],
                              counter: list[int]) -> int:
    """Find every type-matched op with a fp16/fp32 input mix and Cast fp16 -> fp32.

    Returns the number of casts inserted.
    """
    new_cast_nodes: list = []
    for n in model.graph.node:
        if n.op_type not in _TYPE_MATCH_OPS:
            continue
        if n.op_type == "Where":
            idx_range = [1, 2]
        elif n.op_type == "LayerNormalization":
            idx_range = list(range(min(3, len(n.input))))
        else:
            idx_range = list(range(len(n.input)))
        types = [dtype_of.get(n.input[i]) for i in idx_range]
        has_fp32 = TensorProto.FLOAT in types
        has_fp16 = TensorProto.FLOAT16 in types
        if not (has_fp32 and has_fp16):
            continue
        for idx in idx_range:
            t = dtype_of.get(n.input[idx])
            if t != TensorProto.FLOAT16:
                continue
            cast_name = f"auto_promo_cast_{counter[0]}"
            cast_out = f"{cast_name}_out"
            cast_node = helper.make_node(
                "Cast", inputs=[n.input[idx]], outputs=[cast_out],
                to=int(TensorProto.FLOAT), name=cast_name,
            )
            new_cast_nodes.append(cast_node)
            n.input[idx] = cast_out
            dtype_of[cast_out] = TensorProto.FLOAT
            counter[0] += 1
    for cn in new_cast_nodes:
        # Insert at the start; onnxruntime topologically sorts the graph.
        model.graph.node.insert(0, cn)
    return len(new_cast_nodes)


def _strip_redundant_output_cast(model: onnx.ModelProto) -> bool:
    """convert_float_to_float16 wraps the classifier output in a Cast to
    keep the model output fp32; after our promotion the Cast input is also
    fp32, so the Cast is a no-op type alias that ORT rejects (its declared
    input dtype is fp16 even though the producer now emits fp32). Drop it
    and rename the producer's output to the model's `logits` output."""
    cast_node = None
    for n in model.graph.node:
        if n.op_type == "Cast" and n.name == "/classifier/Add_cast_to_logits":
            cast_node = n
            break
    if cast_node is None:
        return False
    old_name = cast_node.input[0]
    new_name = cast_node.output[0]  # 'logits'
    # Rename the producing node's output
    for n in model.graph.node:
        for i, o in enumerate(list(n.output)):
            if o == old_name:
                n.output[i] = new_name
    # Rewire any other consumers of the old name (should be none)
    for n in model.graph.node:
        if n is cast_node:
            continue
        for i, inp in enumerate(list(n.input)):
            if inp == old_name:
                n.input[i] = new_name
    model.graph.node.remove(cast_node)
    return True


def regenerate_fp16(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    # Clean stale ONNX artifacts so the re-save doesn't append fragmented external data
    for f in ("model.onnx", "model.onnx_data"):
        p = out / f
        if p.exists():
            p.unlink()

    src_onnx = src / "model.onnx"
    print(f"[fp16] loading {src_onnx} ...", flush=True)
    t0 = time.time()
    m_orig = onnx.load(str(src_onnx), load_external_data=True)
    print(f"  loaded in {time.time()-t0:.1f}s")
    print("[fp16] converting weights to float16 ...", flush=True)
    t0 = time.time()
    m = float16.convert_float_to_float16(
        m_orig, keep_io_types=True, disable_shape_infer=True,
    )
    print(f"  converted in {time.time()-t0:.1f}s")

    print("[fp16] inserting fp16->fp32 promotion casts ...", flush=True)
    cast_counter = [0]
    for it in range(10):
        dtype_of = _propagate_types(m)
        n_cast = _insert_promotion_casts(m, dtype_of, cast_counter)
        print(f"  iter {it}: +{n_cast} casts (total {cast_counter[0]})")
        if n_cast == 0:
            break

    print("[fp16] stripping redundant output Cast ...", flush=True)
    stripped = _strip_redundant_output_cast(m)
    print(f"  stripped: {stripped}")

    # Re-route via convert_model_from_external_data so save_as_external_data
    # writes a clean, non-fragmented data file.
    from onnx.external_data_helper import convert_model_from_external_data
    convert_model_from_external_data(m)

    out_onnx = out / "model.onnx"
    print(f"[fp16] saving {out_onnx} (with external data) ...", flush=True)
    onnx.save(m, str(out_onnx), save_as_external_data=True,
              all_tensors_to_one_file=True, location="model.onnx_data",
              size_threshold=1024)

    for f in _SHARED_FILES:
        src_f = src / f
        if src_f.exists():
            shutil.copy(src_f, out / f)

    sz = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"[fp16] done -- {sz / 1024 / 1024:.1f} MB total", flush=True)

    # Smoke-test load + 1 inference to catch any remaining type errors early.
    print("[fp16] smoke-testing load via optimum ORTModelForTokenClassification ...",
          flush=True)
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoTokenizer
    mod = ORTModelForTokenClassification.from_pretrained(
        str(out), file_name="model.onnx", provider="CPUExecutionProvider",
    )
    tok = AutoTokenizer.from_pretrained(str(out), use_fast=True)
    enc = tok("Hallo Marius, danke für deine Nachricht.", return_tensors="pt")
    out_l = mod(**enc).logits.detach().cpu().numpy()
    print(f"  logits shape={out_l.shape} dtype={out_l.dtype} max={out_l.max():.3f} mean={out_l.mean():.3f}")
    print("[fp16] OK")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True,
                    help="Source fp32 ONNX dir (must contain model.onnx).")
    p.add_argument("--out", type=Path, required=True,
                    help="Output fp16 ONNX dir (created if absent).")
    args = p.parse_args()
    if not (args.src / "model.onnx").exists():
        raise SystemExit(f"no model.onnx in {args.src}")
    regenerate_fp16(args.src, args.out)


if __name__ == "__main__":
    main()
