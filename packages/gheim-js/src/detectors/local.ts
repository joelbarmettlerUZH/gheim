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

export const DEFAULT_MODEL = "openai/privacy-filter";

/** Default sliding-window chunking parameters; see LocalDetector docstring. */
const DEFAULT_CHUNK_TOKENS = 510;
const DEFAULT_OVERLAP_TOKENS = 32;
/** Default forward-pass batch size; CPU-safe, raise on GPU. */
const DEFAULT_BATCH_SIZE = 8;

/** transformers.js token-classification options used by this detector. */
export interface LocalDetectorOptions {
  /** HuggingFace model id (must be a token-classification model). */
  model?: string;
  /** transformers.js device: "auto" | "cpu" | "wasm" | "webgpu" | "gpu" | ... */
  device?: string;
  /** Quantization dtype: "fp32" | "fp16" | "q8" | "q4" | "auto" */
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
  readonly device: string;
  readonly dtype: string;
  readonly aggregationStrategy: string;
  readonly chunkTokens: number;
  readonly chunkOverlapTokens: number;
  readonly batchSize: number;
  private _pipeline: PipelineCallable | null = null;
  private _loadPromise: Promise<PipelineCallable> | null = null;

  constructor(opts: LocalDetectorOptions = {}) {
    this.model = opts.model ?? DEFAULT_MODEL;
    this.device = opts.device ?? "auto";
    this.dtype = opts.dtype ?? "fp32";
    this.aggregationStrategy = opts.aggregationStrategy ?? "simple";
    this.chunkTokens = opts.chunkTokens ?? DEFAULT_CHUNK_TOKENS;
    this.chunkOverlapTokens = opts.chunkOverlapTokens ?? DEFAULT_OVERLAP_TOKENS;
    this.batchSize = opts.batchSize ?? DEFAULT_BATCH_SIZE;
    if (this.chunkOverlapTokens >= this.chunkTokens) {
      throw new Error(
        `chunkOverlapTokens (${this.chunkOverlapTokens}) must be smaller than chunkTokens (${this.chunkTokens})`,
      );
    }
    if (opts.pipelineInstance) {
      this._pipeline = opts.pipelineInstance as PipelineCallable;
    }
  }

  private async load(): Promise<PipelineCallable> {
    if (this._pipeline) return this._pipeline;
    if (this._loadPromise) return this._loadPromise;
    this._loadPromise = (async () => {
      let mod: { pipeline: (task: string, model: string, opts: unknown) => Promise<PipelineCallable> };
      try {
        mod = (await import("@huggingface/transformers")) as unknown as typeof mod;
      } catch (e) {
        throw new Error(
          "LocalDetector requires `@huggingface/transformers`. " +
            "Install with: bun add @huggingface/transformers",
        );
      }
      const pipe = await mod.pipeline("token-classification", this.model, {
        device: this.device,
        dtype: this.dtype,
      });
      this._pipeline = pipe;
      return pipe;
    })();
    return this._loadPromise;
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
