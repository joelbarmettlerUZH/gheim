/**
 * Local detector — runs the privacy-filter model in-process via @huggingface/transformers.
 *
 * Works in Node and the browser (with `device: 'webgpu'` for the latter).
 * `@huggingface/transformers` is an *optional peer dependency* — only needed
 * when this detector is actually instantiated.
 */
import type { Detector, Span } from "./base.ts";

export const DEFAULT_MODEL = "openai/privacy-filter";

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
  text: string,
  opts?: { aggregation_strategy?: string },
) => Promise<DetectionItem[]>;

export class LocalDetector implements Detector {
  readonly model: string;
  readonly device: string;
  readonly dtype: string;
  readonly aggregationStrategy: string;
  private _pipeline: PipelineCallable | null = null;
  private _loadPromise: Promise<PipelineCallable> | null = null;

  constructor(opts: LocalDetectorOptions = {}) {
    this.model = opts.model ?? DEFAULT_MODEL;
    this.device = opts.device ?? "auto";
    this.dtype = opts.dtype ?? "fp32";
    this.aggregationStrategy = opts.aggregationStrategy ?? "simple";
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
    const pipe = await this.load();
    const raw = await pipe(text, { aggregation_strategy: this.aggregationStrategy });

    const out: Span[] = [];
    let cursor = 0; // for left-to-right surface re-discovery
    for (const item of raw) {
      const label = String(item.entity_group ?? item.entity ?? "unknown");
      const score = typeof item.score === "number" ? item.score : 1.0;

      let start = item.start;
      let end = item.end;

      if (start === undefined || end === undefined) {
        // transformers.js (as of v4.2) does not return char offsets for
        // aggregated token-classification entities. Fall back to substring
        // search starting at the cursor, which preserves left-to-right ordering
        // and copes with repeated surface forms.
        const surface = normalizeWord(item.word ?? "");
        if (!surface) continue;
        const idx = text.indexOf(surface, cursor);
        if (idx === -1) {
          // Couldn't recover an offset; skip this entity rather than corrupt the round-trip.
          continue;
        }
        start = idx;
        end = idx + surface.length;
      }
      cursor = end;
      out.push({ label, start, end, text: text.slice(start, end), score });
    }
    return out;
  }
}

/**
 * transformers.js prefixes wordpiece-merged words with a leading space and may
 * include the BPE marker '##' inside merged tokens. Strip both so the surface
 * matches the original input.
 */
function normalizeWord(word: string): string {
  return word.replace(/##/g, "").trim();
}
