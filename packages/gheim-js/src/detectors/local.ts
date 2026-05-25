/**
 * Local detector — runs the privacy-filter model in-process via @huggingface/transformers.
 *
 * Works in Node and the browser (with `device: 'webgpu'` for the latter).
 * `@huggingface/transformers` is an *optional peer dependency* — only needed
 * when this detector is actually instantiated.
 *
 * Long inputs (more tokens than the model's positional limit, typically
 * 512 for BERT/XLM-R/roberta encoders) are split into overlapping sliding
 * windows; spans from each window are reprojected to the original character
 * offsets and deduplicated. Without this, anything past ~512 tokens would
 * be silently truncated by the underlying pipeline. Multiple inputs and
 * within-input chunks are batched together for one forward pass.
 */
import type { Detector, Span } from "./base.ts";

export const DEFAULT_MODEL = "joelbarmettler/gheim-ch-560m";

/** Default sliding-window chunking parameters; see LocalDetector docstring. */
const DEFAULT_CHUNK_TOKENS = 510;
const DEFAULT_OVERLAP_TOKENS = 32;
/** Default forward-pass batch size; CPU-safe, raise on GPU. */
const DEFAULT_BATCH_SIZE = 8;

/** transformers.js token-classification options used by this detector. */
export interface LocalDetectorOptions {
  /** HuggingFace model id (must be a token-classification model). */
  model?: string;
  /**
   * Backend selector. Defaults to `"auto"`, which:
   *   - in the browser, probes `navigator.gpu.requestAdapter()` and uses
   *     WebGPU if a real adapter is available, otherwise WASM. If WebGPU
   *     loading throws (shader compile, OOM, driver bug) it falls back
   *     to WASM transparently.
   *   - in Node, just uses WASM (no WebGPU bindings).
   * Pass `"webgpu"`, `"wasm"`, `"cpu"`, etc. to force a specific backend
   * (no probe, no fallback — your choice is respected and may throw).
   */
  device?: string;
  /**
   * Quantization dtype: `"fp32" | "fp16" | "q8" | "q4" | "auto"`. Defaults
   * to `"auto"`, which picks based on the resolved backend: `"fp16"` when
   * WebGPU was selected (large but byte-equivalent to fp32 on the forensic
   * probe), `"q8"` otherwise (~550 MB; safe everywhere, but degrades on
   * the `onnxruntime-web` WASM kernel on Swiss edge cases — see
   * eval/q8_quality_report.md). Set explicitly to opt out of this
   * trade-off.
   */
  dtype?: string;
  /** Aggregation strategy passed to the pipeline call. */
  aggregationStrategy?: string;
  /** Sliding-window size in model tokens (default 510 for a 512-token model). */
  chunkTokens?: number;
  /**
   * Token overlap between consecutive windows. Must exceed the longest PII
   * span you expect to detect; default 32 covers all realistic cases.
   */
  chunkOverlapTokens?: number;
  /**
   * Forward-pass batch size used both for within-input chunk batching and
   * for `detectBatch` cross-input batching.
   */
  batchSize?: number;
  /** Optional pre-built pipeline instance (e.g. for tests). */
  pipelineInstance?: unknown;
  /**
   * Optional callback fired during model loading. Receives a normalized
   * event so consumers don't need to reconcile transformers.js's
   * 0-1-vs-0-100 inconsistency or its raw status strings. Useful for
   * driving a download progress bar.
   */
  onProgress?: (event: LocalDetectorLoadEvent) => void;
}

/** Normalized model-load event surfaced to {@link LocalDetectorOptions.onProgress}. */
export interface LocalDetectorLoadEvent {
  /** Phase of the load. */
  phase: "initiate" | "download" | "progress" | "ready" | "done";
  /** File being processed, when applicable. */
  file?: string;
  /** Bytes downloaded so far (only on `phase: "progress"`). */
  loadedBytes?: number;
  /** Total bytes for `file` (only on `phase: "progress"`). */
  totalBytes?: number;
  /** Fraction in [0, 1] (only on `phase: "progress"`). Normalized so
   *  consumers can use the value directly without bounds-clamping. */
  fraction?: number;
  /** Backend currently being attempted (e.g. `"webgpu"`, `"wasm"`).
   *  When auto-fallback fires, two phase-streams may be observed: one
   *  for the failed `"webgpu"` attempt, then a fresh stream for `"wasm"`. */
  device?: string;
}

interface DetectionItem {
  start?: number;
  end?: number;
  word?: string;
  entity?: string;
  entity_group?: string;
  score?: number;
}

type PipelineCallable = (
  text: string | string[],
  opts?: { aggregation_strategy?: string; batch_size?: number },
) => Promise<DetectionItem[] | DetectionItem[][]>;

interface PipelineWithTokenizer {
  tokenizer?: TokenizerCallable;
}

type TokenizerCallable = (
  text: string,
  opts: {
    return_offsets_mapping?: boolean;
    add_special_tokens?: boolean;
    truncation?: boolean;
  },
) => Promise<TokenizerOutput> | TokenizerOutput;

interface TokenizerOutput {
  input_ids?: { dims?: number[]; data?: ArrayLike<number>; tolist?: () => number[] | number[][] };
  offset_mapping?: {
    dims?: number[];
    data?: ArrayLike<number>;
    tolist?: () => number[][] | number[][][];
  };
}

export class LocalDetector implements Detector {
  readonly model: string;
  /** Device preference as passed to the constructor (`"auto"`, `"webgpu"`, ...). */
  readonly device: string;
  readonly dtype: string;
  readonly aggregationStrategy: string;
  readonly chunkTokens: number;
  readonly chunkOverlapTokens: number;
  readonly batchSize: number;
  /** Backend that ended up being used after `load()` resolves. `null` until
   *  the model is loaded. Different from {@link device} only when
   *  `device: "auto"` was requested. */
  actualDevice: string | null = null;
  /** Dtype that ended up being used after `load()` resolves. `null` until
   *  the model is loaded. Different from {@link dtype} only when
   *  `dtype: "auto"` was requested — in that case it resolves to `"fp16"`
   *  on WebGPU and `"q8"` everywhere else. */
  actualDtype: string | null = null;
  private _pipeline: PipelineCallable | null = null;
  private _loadPromise: Promise<PipelineCallable> | null = null;
  private readonly _onProgress?: (event: LocalDetectorLoadEvent) => void;

  constructor(opts: LocalDetectorOptions = {}) {
    this.model = opts.model ?? DEFAULT_MODEL;
    this.device = opts.device ?? "auto";
    this.dtype = opts.dtype ?? "auto";
    this.aggregationStrategy = opts.aggregationStrategy ?? "simple";
    this.chunkTokens = opts.chunkTokens ?? DEFAULT_CHUNK_TOKENS;
    this.chunkOverlapTokens = opts.chunkOverlapTokens ?? DEFAULT_OVERLAP_TOKENS;
    this.batchSize = opts.batchSize ?? DEFAULT_BATCH_SIZE;
    this._onProgress = opts.onProgress;
    if (this.chunkOverlapTokens >= this.chunkTokens) {
      throw new Error(
        `chunkOverlapTokens (${this.chunkOverlapTokens}) must be smaller than chunkTokens (${this.chunkTokens})`,
      );
    }
    if (opts.pipelineInstance) {
      this._pipeline = opts.pipelineInstance as PipelineCallable;
      // A pre-built pipeline bypasses `load()` entirely; we don't know which
      // backend it was built with, but the consumer almost certainly does.
      this.actualDevice = this.device === "auto" ? null : this.device;
      this.actualDtype = this.dtype === "auto" ? null : this.dtype;
    }
  }

  /** Resolve `dtype: "auto"` for a concrete device: fp16 on WebGPU (byte-
   *  equivalent to fp32 on the forensic probe), q8 on WASM/CPU. The
   *  WASM-q8 path is known to degrade on Swiss edge cases via
   *  `onnxruntime-web`, but it's the small-file fallback for low-RAM
   *  contexts where WebGPU isn't available. Explicit dtype passes through. */
  private _resolveDtype(device: string): string {
    if (this.dtype !== "auto") return this.dtype;
    return device === "webgpu" ? "fp16" : "q8";
  }

  /**
   * Load the model. Idempotent; concurrent calls share a single load.
   * With `device: "auto"`, probes `navigator.gpu.requestAdapter()` and
   * walks the candidate-backend list, falling back from WebGPU to WASM
   * if construction throws (and reporting both attempts via `onProgress`).
   */
  async load(): Promise<PipelineCallable> {
    if (this._pipeline) return this._pipeline;
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = (async () => {
      const mod = await loadTransformersJs();
      const sequence = await this._planDeviceSequence();
      let lastErr: unknown;
      for (let i = 0; i < sequence.length; i++) {
        const dev = sequence[i] ?? "wasm";
        const resolvedDtype = this._resolveDtype(dev);
        try {
          const pipe = await mod.pipeline("token-classification", this.model, {
            device: dev,
            dtype: resolvedDtype,
            progress_callback: this._onProgress
              ? (raw: RawProgress) => this._emitProgress(raw, dev)
              : undefined,
          });
          this._pipeline = pipe;
          this.actualDevice = dev;
          this.actualDtype = resolvedDtype;
          return pipe;
        } catch (e) {
          lastErr = e;
          // Only fall through to the next candidate when "auto" requested
          // multiple. Explicit device choices propagate the failure so the
          // caller knows their constraint couldn't be met.
          if (i < sequence.length - 1) continue;
          throw e;
        }
      }
      throw lastErr ?? new Error("LocalDetector.load(): no candidate backends");
    })();
    return this._loadPromise;
  }

  /** Build the ordered list of backends to attempt.
   *
   * In a browser we try WebGPU (if a real adapter is reachable) and fall
   * back to WASM. In Node, transformers.js uses onnxruntime-node, which
   * speaks "cpu" / "cuda" / "webgpu" — never "wasm" — so the fallback
   * has to be "cpu" instead. We detect "browser-ness" via `typeof window`;
   * Node 21+ now ships a global `navigator`, so that's no longer a
   * reliable discriminator on its own.
   */
  private async _planDeviceSequence(): Promise<string[]> {
    if (this.device !== "auto") return [this.device];
    // `typeof window` is the discriminator; declared in DOM lib but not in
    // our tsconfig, so go via globalThis to keep `tsc --noEmit` happy.
    const isBrowser = typeof (globalThis as { window?: unknown }).window !== "undefined";
    const seq: string[] = [];
    const nav =
      typeof navigator !== "undefined"
        ? (navigator as { gpu?: { requestAdapter: () => Promise<unknown> } })
        : null;
    if (nav?.gpu?.requestAdapter) {
      try {
        const adapter = await nav.gpu.requestAdapter();
        if (adapter) seq.push("webgpu");
      } catch {
        // Adapter request itself threw — skip WebGPU.
      }
    }
    seq.push(isBrowser ? "wasm" : "cpu");
    return seq;
  }

  /** Translate a raw transformers.js progress callback into our normalized event. */
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
        // transformers.js reports `progress` as 0-100 for some files and 0-1
        // for others, and occasionally overshoots if `loaded > total`
        // mid-stream. Normalize to [0, 1] so consumers can use it directly.
        const norm = raw.progress > 1 ? raw.progress / 100 : raw.progress;
        event.fraction = Math.max(0, Math.min(1, norm));
      }
    }
    this._onProgress(event);
  }

  async detect(text: string, opts?: { model?: string }): Promise<Span[]> {
    if (opts?.model && opts.model !== this.model) {
      throw new Error(
        `LocalDetector was constructed with model='${this.model}'; ` +
          `per-call model override ('${opts.model}') not supported.`,
      );
    }
    if (!text) return [];
    const results = await this.detectBatch([text]);
    return results[0] ?? [];
  }

  /**
   * Detect PII in many inputs in a single batched forward pass. Inputs that
   * exceed the model's token limit are split into overlapping windows; all
   * chunks across all inputs are batched into one pipeline call (subject to
   * `batchSize`). Empty texts produce empty `[]` at the matching position.
   */
  async detectBatch(texts: string[], opts?: { model?: string }): Promise<Span[][]> {
    if (opts?.model && opts.model !== this.model) {
      throw new Error(
        `LocalDetector was constructed with model='${this.model}'; ` +
          `per-call model override ('${opts.model}') not supported.`,
      );
    }
    if (texts.length === 0) return [];
    const pipe = await this.load();

    // Build the flat plan of (inputIndex, charOffset, chunkText). Token-aware
    // chunking is preferred (uses the tokenizer's offset_mapping for exact
    // token-bounded windows); we fall back to character-based chunking if the
    // tokenizer is not introspectable in this transformers.js version.
    const tokenizer = (pipe as unknown as PipelineWithTokenizer).tokenizer;
    const plan: Array<{ inputIdx: number; charOffset: number; text: string }> = [];
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i] ?? "";
      if (!text) continue;
      const chunks = await this.chunkText(text, tokenizer);
      for (const c of chunks) plan.push({ inputIdx: i, charOffset: c.offset, text: c.text });
    }

    const out: Span[][] = texts.map(() => []);
    if (plan.length === 0) return out;

    // Single batched call. transformers.js returns DetectionItem[][] when the
    // input is an array; the order matches the input order. batch_size limits
    // how many are tokenized + run together in one tensor.
    const batchInput = plan.map((p) => p.text);
    const raw = (await pipe(batchInput, {
      aggregation_strategy: this.aggregationStrategy,
      batch_size: this.batchSize,
    })) as DetectionItem[][];

    // Per-input dedup keyed on `${start}|${end}|${label}`.
    const seen: Array<Set<string>> = texts.map(() => new Set());
    for (let p = 0; p < plan.length; p++) {
      const planEntry = plan[p];
      if (!planEntry) continue;
      const { inputIdx, charOffset, text: chunkText } = planEntry;
      const fullText = texts[inputIdx] ?? "";
      const inputSeen = seen[inputIdx];
      const inputOut = out[inputIdx];
      if (!inputSeen || !inputOut) continue;
      const items = raw[p] ?? [];
      let cursor = 0; // chunk-local cursor for fallback substring search
      for (const item of items) {
        const label = String(item.entity_group ?? item.entity ?? "unknown");
        const score = typeof item.score === "number" ? item.score : 1.0;

        let localStart: number;
        let localEnd: number;
        if (item.start === undefined || item.end === undefined) {
          // transformers.js sometimes drops char offsets for aggregated
          // entities; recover by searching the chunk text. Cursor is
          // chunk-local so repeated surface forms inside the same chunk
          // are still ordered left-to-right.
          const surface = normalizeWord(item.word ?? "");
          if (!surface) continue;
          const idx = chunkText.indexOf(surface, cursor);
          if (idx === -1) continue;
          localStart = idx;
          localEnd = idx + surface.length;
          cursor = localEnd;
        } else {
          localStart = item.start;
          localEnd = item.end;
        }
        const start = localStart + charOffset;
        const end = localEnd + charOffset;
        const key = `${start}|${end}|${label}`;
        if (inputSeen.has(key)) continue;
        inputSeen.add(key);
        inputOut.push({
          label,
          start,
          end,
          text: fullText.slice(start, end),
          score,
        });
      }
    }
    for (const spans of out) {
      spans.sort((a, b) => a.start - b.start || a.end - b.end);
    }
    return out;
  }

  private async chunkText(
    text: string,
    tokenizer: TokenizerCallable | undefined,
  ): Promise<Array<{ offset: number; text: string }>> {
    if (tokenizer) {
      const tokenChunks = await chunkByTokens(
        text,
        tokenizer,
        this.chunkTokens,
        this.chunkOverlapTokens,
      );
      if (tokenChunks) return tokenChunks;
    }
    // Conservative character-based fallback: 1 token ≈ 4 chars worst-case
    // for Latin scripts, ≈ 2 chars for some CJK; sizing 1500 chars per
    // window stays well under 512 tokens on the typical input mix.
    return chunkByChars(text, 1500, 200);
  }
}

/**
 * Token-aware sliding window. Returns null if the tokenizer call doesn't
 * return a usable offset_mapping (older transformers.js versions); the
 * caller falls back to character-based chunking.
 */
async function chunkByTokens(
  text: string,
  tokenizer: TokenizerCallable,
  chunkTokens: number,
  overlapTokens: number,
): Promise<Array<{ offset: number; text: string }> | null> {
  let enc: TokenizerOutput;
  try {
    enc = await tokenizer(text, {
      return_offsets_mapping: true,
      add_special_tokens: false,
      truncation: false,
    });
  } catch {
    return null;
  }
  const offsets = readOffsetMapping(enc);
  if (!offsets) return null;
  // Skip (0, 0) entries — fast tokenizers emit them for control characters
  // that produce no surface; including them would corrupt window slicing.
  const real = offsets.filter(([a, b]) => !(a === 0 && b === 0));
  if (real.length === 0) return [{ offset: 0, text }];
  if (real.length <= chunkTokens) return [{ offset: 0, text }];
  const stride = Math.max(1, chunkTokens - overlapTokens);
  const out: Array<{ offset: number; text: string }> = [];
  for (let s = 0; s < real.length; s += stride) {
    const e = Math.min(s + chunkTokens, real.length);
    const startPair = real[s];
    const endPair = real[e - 1];
    if (!startPair || !endPair) break;
    const charStart = startPair[0];
    const charEnd = endPair[1];
    out.push({ offset: charStart, text: text.slice(charStart, charEnd) });
    if (e >= real.length) break;
  }
  return out;
}

/** Pull a flat (n, 2) list of [start, end] from a TokenizerOutput. */
function readOffsetMapping(enc: TokenizerOutput): Array<[number, number]> | null {
  const om = enc.offset_mapping;
  if (!om) return null;
  if (typeof om.tolist === "function") {
    const list = om.tolist();
    if (!Array.isArray(list) || list.length === 0) return [];
    const inner = list[0];
    if (Array.isArray(inner) && Array.isArray(inner[0])) {
      // Shape [batch, n, 2] — return the first batch row.
      return inner as Array<[number, number]>;
    }
    return list as Array<[number, number]>;
  }
  if (om.data && om.dims && om.dims.length >= 2) {
    const dims = om.dims;
    const pairs = dims[dims.length - 2] ?? 0;
    const data = om.data;
    const out: Array<[number, number]> = [];
    for (let i = 0; i < pairs; i++) {
      out.push([Number(data[i * 2]), Number(data[i * 2 + 1])]);
    }
    return out;
  }
  return null;
}

function chunkByChars(
  text: string,
  chunkChars: number,
  overlapChars: number,
): Array<{ offset: number; text: string }> {
  if (text.length <= chunkChars) return [{ offset: 0, text }];
  const stride = Math.max(1, chunkChars - overlapChars);
  const out: Array<{ offset: number; text: string }> = [];
  for (let s = 0; s < text.length; s += stride) {
    const e = Math.min(s + chunkChars, text.length);
    out.push({ offset: s, text: text.slice(s, e) });
    if (e >= text.length) break;
  }
  return out;
}

/**
 * transformers.js prefixes wordpiece-merged words with a leading space and may
 * include the BPE marker '##' inside merged tokens. Strip both so the surface
 * matches the original input.
 */
function normalizeWord(word: string): string {
  return word.replace(/##/g, "").trim();
}

/** Raw shape transformers.js feeds to `progress_callback`. */
interface RawProgress {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

interface TransformersModule {
  pipeline: (task: string, model: string, opts: unknown) => Promise<PipelineCallable>;
}

async function loadTransformersJs(): Promise<TransformersModule> {
  try {
    return (await import("@huggingface/transformers")) as unknown as TransformersModule;
  } catch {
    throw new Error(
      "LocalDetector requires `@huggingface/transformers`. " +
        "Install with: bun add @huggingface/transformers",
    );
  }
}
