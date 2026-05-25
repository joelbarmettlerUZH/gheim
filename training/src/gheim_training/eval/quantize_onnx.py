"""Quantize the fp32 ONNX export into the variants we ship to consumers.

Produces two new directories alongside the input:
  <onnx_dir>_q8      dynamic int8 (CPU-friendly, browser default)
  <onnx_dir>_fp16    half precision (GPU / WebGPU fp16)

Each output dir is self-contained: model.onnx (+ optional .onnx_data),
plus the original tokenizer / config files copied over so the dir loads
directly via ORTModelForTokenClassification.from_pretrained.

Run:
    uv run --project training python -m gheim_training.eval.quantize_onnx \\
        --src checkpoints/stage2_xlmr_onnx
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Tokenizer + config carry over verbatim — they describe the I/O contract,
# which quantization preserves; only the weight tensor dtype changes.
_SHARED_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
)


def _copy_shared(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in _SHARED_FILES:
        if (src / f).exists():
            shutil.copy(src / f, dst / f)


def quantize_int8(src: Path) -> Path:
    """Dynamic int8 (UINT8 weights). The browser default."""
    from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    out = src.parent / f"{src.name}_q8"
    print(f"\n[q8] {src} → {out}", flush=True)

    model = ORTModelForTokenClassification.from_pretrained(
        str(src), file_name="model.onnx"
    )
    qcfg = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=True)
    quantizer = ORTQuantizer.from_pretrained(model)
    quantizer.quantize(save_dir=str(out), quantization_config=qcfg)

    # Optimum names the output `model_quantized.onnx` — that is the exact
    # filename transformers.js looks for under dtype: "q8", so do not rename.
    _copy_shared(src, out)
    sz = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"[q8] done — {sz / 1024 / 1024:.1f} MB total", flush=True)
    return out


# Ops that require all (or a designated subset of) inputs to share a dtype.
# `Where` takes (cond, then, else) -- only then/else need to match.
# `LayerNormalization` takes (X, scale, bias?) -- first 3 inputs must match.
_TYPE_MATCH_OPS = {
    "Div", "Mul", "Add", "Sub", "Pow", "Where", "Min", "Max",
    "MatMul", "Gemm", "Concat", "LayerNormalization",
}


def _propagate_dtypes(model) -> dict[str, int]:
    """Topologically infer the data_type of every tensor in the graph.

    The fp16 converter leaves Cast nodes around the model's sqrt(d_k)
    fallback-to-fp32 in attention, but downstream binary ops still see
    fp16/fp32 mixes that onnxruntime rejects at load. We need a dtype map
    so we know where to insert promotion Casts.
    """
    from onnx import TensorProto
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


def _insert_fp16_to_fp32_promotion_casts(model, dtype_of: dict[str, int],
                                          counter: list[int]) -> int:
    """For every type-match op with a fp16/fp32 input mix, Cast fp16 -> fp32.

    Returns the number of Cast nodes inserted this pass.
    """
    from onnx import TensorProto, helper
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
        # Insert at the start; onnxruntime topo-sorts the graph at load.
        model.graph.node.insert(0, cn)
    return len(new_cast_nodes)


def _strip_redundant_classifier_output_cast(model) -> bool:
    """Remove the redundant Cast wrapper around the classifier output.

    `convert_float_to_float16(keep_io_types=True)` wraps the classifier
    output in a `Cast -> fp32` so the model output stays fp32. After our
    promotion passes, the Cast's input is already fp32, leaving a no-op
    type alias whose declared input dtype (fp16) no longer matches the
    producer's emitted dtype (fp32) — onnxruntime rejects this.
    """
    cast_node = None
    for n in model.graph.node:
        if n.op_type == "Cast" and n.name == "/classifier/Add_cast_to_logits":
            cast_node = n
            break
    if cast_node is None:
        return False
    old_name = cast_node.input[0]
    new_name = cast_node.output[0]  # 'logits'
    for n in model.graph.node:
        for i, o in enumerate(list(n.output)):
            if o == old_name:
                n.output[i] = new_name
    for n in model.graph.node:
        if n is cast_node:
            continue
        for i, inp in enumerate(list(n.input)):
            if inp == old_name:
                n.input[i] = new_name
    model.graph.node.remove(cast_node)
    return True


def quantize_fp16(src: Path) -> Path:
    """fp16 conversion via onnxconverter_common's float16 converter, with
    post-fixups so the result actually loads in onnxruntime.

    The naked converter output is unloadable on this model because the
    XLM-R attention block casts to fp32 around its sqrt(d_k) for numerical
    stability, and the converter doesn't propagate that mix through the
    downstream binary / MatMul / LayerNormalization ops. We:

      1. Run convert_float_to_float16(keep_io_types=True, disable_shape_infer=True)
         — keep_io_types preserves the fp32 inputs/outputs the inference
         pipeline expects.
      2. Iteratively walk the graph propagating dtypes, and insert
         fp16->fp32 Cast nodes wherever a type-match op (Div/Mul/MatMul/
         LayerNorm/...) has a mix of fp16 and fp32 inputs.
      3. Strip the redundant trailing Cast around the classifier output
         that the converter inserted but our promotion made a no-op.

    The result loads via ORTModelForTokenClassification on
    CPUExecutionProvider and produces logits within 1e-3 mean-abs of fp32
    (see eval/q8_quality_report.md). Saves with external-data on so the
    >2GB protobuf limit is not an issue.
    """
    import sys

    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    out = src.parent / f"{src.name}_fp16"
    print(f"\n[fp16] {src} -> {out}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    # Clean stale ONNX artifacts so the re-save doesn't append fragmented data.
    for f in ("model.onnx", "model.onnx_data"):
        p = out / f
        if p.exists():
            p.unlink()

    src_onnx = src / "model.onnx"
    print(f"[fp16] loading {src_onnx} ...", flush=True)
    model = onnx.load(str(src_onnx), load_external_data=True)

    print("[fp16] converting weights to float16 ...", flush=True)
    sys.stdout.flush()
    from onnxconverter_common import float16
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        disable_shape_infer=True,
    )

    print("[fp16] inserting fp16->fp32 promotion casts ...", flush=True)
    cast_counter = [0]
    for it in range(10):
        dtype_of = _propagate_dtypes(fp16_model)
        n_cast = _insert_fp16_to_fp32_promotion_casts(
            fp16_model, dtype_of, cast_counter,
        )
        print(f"  iter {it}: +{n_cast} casts (total {cast_counter[0]})", flush=True)
        if n_cast == 0:
            break

    print("[fp16] stripping redundant output Cast ...", flush=True)
    stripped = _strip_redundant_classifier_output_cast(fp16_model)
    print(f"  stripped: {stripped}", flush=True)

    # Convert from external data so the re-save writes one clean .onnx_data file
    # instead of fragmenting across the previous external locations.
    convert_model_from_external_data(fp16_model)

    out_onnx = out / "model.onnx"
    print(f"[fp16] saving {out_onnx} (with external data) ...", flush=True)
    onnx.save(
        fp16_model,
        str(out_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx_data",
        size_threshold=1024,
    )

    _copy_shared(src, out)
    sz = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"[fp16] done -- {sz / 1024 / 1024:.1f} MB total", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True,
                        help="Source ONNX dir (must contain model.onnx)")
    parser.add_argument("--variants", type=str, default="q8,fp16",
                        help="Comma-separated: q8, fp16")
    args = parser.parse_args()

    if not (args.src / "model.onnx").exists():
        raise SystemExit(f"no model.onnx in {args.src}")

    variants = {v.strip() for v in args.variants.split(",")}
    out_dirs: list[Path] = []
    if "q8" in variants:
        out_dirs.append(quantize_int8(args.src))
    if "fp16" in variants:
        out_dirs.append(quantize_fp16(args.src))

    print("\n=== Done ===", flush=True)
    for d in out_dirs:
        files = sorted(p.name for p in d.iterdir() if p.is_file())
        print(f"  {d}", flush=True)
        for f in files:
            sz = (d / f).stat().st_size
            print(f"    {f:<28} {sz / 1024 / 1024:>9.1f} MB", flush=True)


if __name__ == "__main__":
    main()
