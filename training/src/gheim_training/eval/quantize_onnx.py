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


def quantize_fp16(src: Path) -> Path:
    """fp16 conversion via ORT's float16 converter.

    Saves with external-data on so the >2GB protobuf limit is not an issue
    even on intermediate steps.
    """
    import sys

    import onnx
    from onnxconverter_common import float16

    out = src.parent / f"{src.name}_fp16"
    print(f"\n[fp16] {src} -> {out}", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    src_onnx = src / "model.onnx"
    print(f"[fp16] loading {src_onnx} ...", flush=True)
    model = onnx.load(str(src_onnx), load_external_data=True)

    print("[fp16] converting weights to float16 ...", flush=True)
    sys.stdout.flush()  # convert_float_to_float16 can take minutes on a 560M model.
    fp16_model = float16.convert_float_to_float16(
        model,
        keep_io_types=False,
        disable_shape_infer=True,
    )

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
