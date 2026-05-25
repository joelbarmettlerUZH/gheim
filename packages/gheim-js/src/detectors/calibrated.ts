/**
 * Calibrated detector — subtracts a constant `oBias` from the O-class
 * logit before argmax. Originally introduced to recover the v1 model's
 * "single-category mode" pathology (Müller + IBAN co-occurrence flipped
 * the person prediction to O); the v3 / gheim-ch-560m training pipeline
 * largely resolved that pathology, but the calibration still buys a
 * small precision/recall improvement and remains the default.
 *
 * Sweep on gheim-ch-560m (see eval/calibration_sweep.json):
 * `oBias=0.5` is the Pareto-clean default — probe perfect-rate is flat
 * at 91.5% across bias 0.0 and 0.5, and test char-F1 lifts by 0.27pp.
 * Higher biases (1.0–1.5) trade test F1 for marginal probe gains.
 *
 * A separate sweep on the int8 ONNX path via `transformers.js`
 * (eval/calibration_sweep_q8.json) showed a much larger calibration
 * effect (+1.4pp probe, +0.43pp char-F1 at 0.5). That gap is not a
 * property of the int8 quantisation — Python `onnxruntime` on the same
 * int8 file gives 91.0% probe perfect-rate, essentially fp32-equivalent
 * — but of a runtime divergence inside `onnxruntime-web` / WASM that
 * the calibration partially papers over. The reliable fix is to use the
 * `model_fp16.onnx` variant when WebGPU is available (which is what
 * LocalDetector's default `dtype: "auto"` does); int8 should be treated
 * as a low-RAM fallback. See eval/q8_quality_report.md for the
 * diagnostic.
 *
 * `defaultDetector()` returns this with oBias=0.5, so users get the
 * better behavior automatically.
 */
import type { Detector, Span } from "./base.ts";
import type { LocalDetectorLoadEvent } from "./local.ts";

const DEFAULT_MODEL = "joelbarmettler/gheim-ch-560m";
const DEFAULT_CHUNK_TOKENS = 510;
const DEFAULT_OVERLAP_TOKENS = 32;

/** Constructor options for {@link CalibratedDetector}. */
export interface CalibratedDetectorOptions {
  /** HuggingFace model id (must be a token-classification head). */
  model?: string;
  /**
   * Constant subtracted from the O-class logit before argmax. Default
   * `0.5` is the Pareto-clean point on the reference (fp32/fp16) path:
   *
   *   - oBias=0.0  uncalibrated (matches the raw model)
   *   - oBias=0.5  +0.27pp test char-F1, no probe cost (91.5% perfect)
   *   - oBias=1.0+ marginal probe gains, test F1 starts to fall off
   *
   * See eval/calibration_sweep.json.
   */
  oBias?: number;
  /** Backend selector. See {@link LocalDetectorOptions.device}. */
  device?: string;
  /** Quantization dtype: "fp32" | "fp16" | "q8" | "q4" | "auto". */
  dtype?: string;
  /** Sliding-window size in model tokens (default 510). */
  chunkTokens?: number;
  /** Overlap between consecutive windows (default 32). */
  chunkOverlapTokens?: number;
  /** Strip leading/trailing whitespace from each emitted span. Default true. */
  trimWhitespace?: boolean;
  /** Optional progress callback during model load (mirrors LocalDetector). */
  onProgress?: (event: LocalDetectorLoadEvent) => void;
}

interface TransformersJsModule {
  AutoTokenizer: { from_pretrained: (id: string, opts?: unknown) => Promise<unknown> };
  AutoModelForTokenClassification: {
    from_pretrained: (id: string, opts?: unknown) => Promise<unknown>;
  };
}

interface TokenizerCallable {
  (text: string, opts: {
    return_tensors?: string;
    truncation?: boolean;
    max_length?: number;
    return_offsets_mapping?: boolean;
    add_special_tokens?: boolean;
  }): Promise<TokenizerOutput> | TokenizerOutput;
}

interface TokenizerOutput {
  input_ids?: Tensor;
  attention_mask?: Tensor;
  // transformers.js v4 returns plain objects with these fields. Note:
  // offset_mapping is NOT supported (TODO in transformers.js source).
}

interface Tensor {
  data: ArrayLike<number> | Int32Array | Float32Array | BigInt64Array;
  dims: number[];
  tolist?: () => unknown;
}

interface ModelCallable {
  (inputs: Record<string, Tensor>): Promise<{ logits: Tensor }>;
  config: { id2label: Record<string, string>; label2id: Record<string, number> };
}

/** Trim leading/trailing whitespace from a [start, end) slice of `text`. */
function trimSpan(text: string, start: number, end: number): [number, number] {
  while (start < end && /\s/.test(text[start] ?? "")) start++;
  while (end > start && /\s/.test(text[end - 1] ?? "")) end--;
  return [start, end];
}

/** CalibratedDetector — local model + O-logit bias. */
export class CalibratedDetector implements Detector {
  readonly model: string;
  readonly oBias: number;
  readonly device: string;
  readonly dtype: string;
  readonly chunkTokens: number;
  readonly chunkOverlapTokens: number;
  readonly trimWhitespace: boolean;
  /** Backend actually used after `load()` resolves; `null` before. */
  actualDevice: string | null = null;
  /** Dtype actually used after `load()` resolves; `null` before. Differs
   *  from {@link dtype} only when `dtype: "auto"` was requested. */
  actualDtype: string | null = null;
  private _model: ModelCallable | null = null;
  private _tokenizer: TokenizerCallable | null = null;
  private _id2label: Record<number, string> = {};
  private _oId = 0;
  private _loadPromise: Promise<void> | null = null;
  private readonly _onProgress?: (event: LocalDetectorLoadEvent) => void;

  constructor(opts: CalibratedDetectorOptions = {}) {
    this.model = opts.model ?? DEFAULT_MODEL;
    this.oBias = opts.oBias ?? 0.5;
    this.device = opts.device ?? "auto";
    this.dtype = opts.dtype ?? "auto";
    this.chunkTokens = opts.chunkTokens ?? DEFAULT_CHUNK_TOKENS;
    this.chunkOverlapTokens = opts.chunkOverlapTokens ?? DEFAULT_OVERLAP_TOKENS;
    this.trimWhitespace = opts.trimWhitespace ?? true;
    this._onProgress = opts.onProgress;
    if (this.chunkOverlapTokens >= this.chunkTokens) {
      throw new Error(
        `chunkOverlapTokens (${this.chunkOverlapTokens}) must be smaller than ` +
          `chunkTokens (${this.chunkTokens})`,
      );
    }
  }

  async load(): Promise<void> {
    if (this._model) return;
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = (async () => {
      const mod = await loadTransformersJs();
      // Pick a backend (auto-fallback identical to LocalDetector). Node
      // transformers.js uses onnxruntime-node which doesn't speak "wasm" —
      // fall back to "cpu" instead. We discriminate via `typeof window`
      // (Node 21+ ships a global `navigator`, so that no longer works).
      let preferredDevice: string = this.device;
      if (this.device === "auto") {
        const isBrowser =
          typeof (globalThis as { window?: unknown }).window !== "undefined";
        const nav = typeof navigator !== "undefined"
          ? (navigator as { gpu?: { requestAdapter: () => Promise<unknown> } })
          : null;
        preferredDevice = isBrowser ? "wasm" : "cpu";
        if (nav?.gpu?.requestAdapter) {
          try {
            const adapter = await nav.gpu.requestAdapter();
            if (adapter) preferredDevice = "webgpu";
          } catch {
            /* keep the platform fallback */
          }
        }
      }
      // Resolve `dtype: "auto"` to a concrete file: fp16 on WebGPU (byte-
      // equivalent to fp32 on the forensic probe), q8 elsewhere (small,
      // safe fallback, but degrades on Swiss edge cases via the
      // onnxruntime-web WASM kernel — see eval/q8_quality_report.md).
      const resolvedDtype = this.dtype === "auto"
        ? (preferredDevice === "webgpu" ? "fp16" : "q8")
        : this.dtype;
      const tokenizer = (await mod.AutoTokenizer.from_pretrained(this.model)) as TokenizerCallable;
      const model = (await mod.AutoModelForTokenClassification.from_pretrained(this.model, {
        device: preferredDevice,
        dtype: resolvedDtype,
        progress_callback: this._onProgress
          ? (raw: RawProgress) => this._emitProgress(raw, preferredDevice)
          : undefined,
      })) as ModelCallable;
      this._tokenizer = tokenizer;
      this._model = model;
      this.actualDtype = resolvedDtype;
      this._id2label = model.config.id2label as unknown as Record<number, string>;
      const label2id = model.config.label2id;
      this._oId = label2id?.["O"] ?? 0;
      this.actualDevice = preferredDevice;
    })();
    return this._loadPromise;
  }

  async detect(text: string, opts?: { model?: string }): Promise<Span[]> {
    if (opts?.model && opts.model !== this.model) {
      throw new Error(
        `CalibratedDetector was constructed with model='${this.model}'; ` +
          `per-call model override ('${opts.model}') not supported.`,
      );
    }
    if (!text) return [];
    await this.load();
    return this._detectChunked(text);
  }

  private async _detectChunked(text: string): Promise<Span[]> {
    // transformers.js's tokenizer doesn't return offset_mapping, so we
    // can't slice token-windows precisely. Fall back to char-window
    // chunking with overlap (mirrors LocalDetector's character path).
    if (text.length <= 1500) {
      return await this._detectWindow(text, 0);
    }
    const stride = 1500 - 200; // ~chunkTokens × ~3 chars/tok with overlap
    const seen = new Set<string>();
    const out: Span[] = [];
    for (let s = 0; s < text.length; s += stride) {
      const e = Math.min(s + 1500, text.length);
      const sub = text.slice(s, e);
      const subSpans = await this._detectWindow(sub, s);
      for (const sp of subSpans) {
        const key = `${sp.start}|${sp.end}|${sp.label}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(sp);
      }
      if (e >= text.length) break;
    }
    out.sort((a, b) => a.start - b.start || a.end - b.end);
    return out;
  }

  private async _detectWindow(text: string, charOffset: number): Promise<Span[]> {
    const tokenizer = this._tokenizer! as unknown as {
      (texts: string[] | string, opts: unknown): Promise<TokenizerOutput>;
      decode: (ids: number[], opts: { skip_special_tokens?: boolean }) => string;
    };
    const model = this._model!;
    const enc = await tokenizer([text], {
      padding: true,
      truncation: true,
      max_length: this.chunkTokens + 2,
    });
    const inputs: Record<string, Tensor> = {};
    if (enc.input_ids) inputs["input_ids"] = enc.input_ids;
    if (enc.attention_mask) inputs["attention_mask"] = enc.attention_mask;
    const out = await model(inputs);
    const logits = readLogits(out.logits); // [seq, num_labels]
    // Apply the O-bias before argmax.
    if (this.oBias !== 0) {
      for (let t = 0; t < logits.length; t++) {
        const row = logits[t];
        if (row === undefined) continue;
        const cur = row[this._oId];
        if (cur !== undefined) {
          row[this._oId] = cur - this.oBias;
        }
      }
    }
    // Argmax per token; build (token_id, label) list.
    const inputIds = readInputIds(enc.input_ids);
    const tokenLabels: Array<{ id: number; label: string }> = [];
    for (let t = 0; t < logits.length; t++) {
      const row = logits[t];
      const id = inputIds[t];
      if (!row || id === undefined) continue;
      let maxIdx = 0;
      let maxVal = row[0]!;
      for (let i = 1; i < row.length; i++) {
        if (row[i]! > maxVal) {
          maxVal = row[i]!;
          maxIdx = i;
        }
      }
      tokenLabels.push({ id, label: this._id2label[maxIdx] ?? "O" });
    }
    // BIO aggregation: collapse consecutive same-category tokens. For each
    // collapsed group, decode its tokens to a surface form, then locate
    // that surface in the original `text` via cursor-tracked substring
    // search (transformers.js doesn't expose char offsets — same trick
    // LocalDetector uses for its substring recovery).
    const spans: Span[] = [];
    let cursor = 0;
    let groupIds: number[] = [];
    let groupCat: string | null = null;
    const flush = () => {
      if (groupCat === null || groupIds.length === 0) return;
      const surface = normalizeDecoded(
        tokenizer.decode(groupIds, { skip_special_tokens: true }),
      );
      groupIds = [];
      const cat = groupCat;
      groupCat = null;
      if (!surface) return;
      const idx = text.indexOf(surface, cursor);
      if (idx === -1) return;
      let s = idx;
      let e = idx + surface.length;
      cursor = e;
      if (this.trimWhitespace) [s, e] = trimSpan(text, s, e);
      if (s >= e) return;
      spans.push({
        start: s + charOffset,
        end: e + charOffset,
        label: cat,
        text: text.slice(s, e),
        score: 1.0,
      });
    };
    for (const { id, label } of tokenLabels) {
      const cat = label === "O" ? null : label.split("-", 2)[1] ?? null;
      if (cat === groupCat && cat !== null) {
        groupIds.push(id);
      } else {
        flush();
        if (cat !== null) {
          groupIds.push(id);
          groupCat = cat;
        }
      }
    }
    flush();
    return spans;
  }

  private _emitProgress(raw: RawProgress, device: string): void {
    if (!this._onProgress) return;
    const event: LocalDetectorLoadEvent = {
      phase: (raw.status as LocalDetectorLoadEvent["phase"]) ?? "progress",
      device,
    };
    if (raw.file) event.file = raw.file;
    if (raw.status === "progress") {
      if (raw.loaded != null) event.loadedBytes = raw.loaded;
      if (raw.total != null) event.totalBytes = raw.total;
      if (raw.progress != null) {
        const norm = raw.progress > 1 ? raw.progress / 100 : raw.progress;
        event.fraction = Math.max(0, Math.min(1, norm));
      }
    }
    this._onProgress(event);
  }
}

interface RawProgress {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

async function loadTransformersJs(): Promise<TransformersJsModule> {
  try {
    return (await import("@huggingface/transformers")) as unknown as TransformersJsModule;
  } catch {
    throw new Error(
      "CalibratedDetector requires `@huggingface/transformers`. " +
        "Install with: bun add @huggingface/transformers",
    );
  }
}

/** Read a flat input_ids array from a [batch=1, seq] tensor. */
function readInputIds(tensor: Tensor | undefined): number[] {
  if (!tensor) return [];
  if (typeof tensor.tolist === "function") {
    const list = tensor.tolist();
    if (Array.isArray(list) && Array.isArray(list[0])) {
      return (list[0] as unknown[]).map((x) => Number(x));
    }
    if (Array.isArray(list)) return (list as unknown[]).map((x) => Number(x));
  }
  const dims = tensor.dims;
  const seq = dims[dims.length - 1] ?? 0;
  const data = tensor.data;
  const out: number[] = [];
  for (let i = 0; i < seq; i++) out.push(Number(data[i]));
  return out;
}

/** Strip BPE wordpiece markers and outer whitespace from a decoded surface. */
function normalizeDecoded(s: string): string {
  return s.replace(/##/g, "").trim();
}

/** Read a [seq, num_labels] logits tensor as a number[][]. */
function readLogits(tensor: Tensor): number[][] {
  if (typeof tensor.tolist === "function") {
    const list = tensor.tolist();
    if (Array.isArray(list) && Array.isArray(list[0]) && Array.isArray((list[0] as unknown[])[0])) {
      // [batch, seq, num_labels] — take first batch.
      return (list as unknown as number[][][])[0]!;
    }
    if (Array.isArray(list) && Array.isArray(list[0])) {
      return list as number[][];
    }
  }
  // Fallback: read raw data + dims.
  const dims = tensor.dims;
  const seq = dims[dims.length - 2] ?? 0;
  const cls = dims[dims.length - 1] ?? 0;
  const data = tensor.data;
  const out: number[][] = [];
  for (let t = 0; t < seq; t++) {
    const row: number[] = [];
    for (let c = 0; c < cls; c++) {
      row.push(Number(data[t * cls + c]));
    }
    out.push(row);
  }
  return out;
}
