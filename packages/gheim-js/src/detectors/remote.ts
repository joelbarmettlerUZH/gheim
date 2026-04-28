/** Remote detector — calls a gheim-server (or compatible) over HTTP. */
import { type Detector, type Span, trimSpan } from "./base.ts";

export const DEFAULT_BASE_URL = "https://api.gheim.ch";
export const DEFAULT_MODEL = "openai/privacy-filter";

export interface RemoteDetectorOptions {
  baseUrl?: string;
  apiKey?: string;
  model?: string;
  /** Strip leading/trailing whitespace from each span (default: true). */
  trimWhitespace?: boolean;
  fetch?: typeof fetch;
}

export class RemoteDetector implements Detector {
  private readonly baseUrl: string;
  private readonly apiKey?: string;
  private readonly model: string;
  private readonly trimWhitespace: boolean;
  private readonly fetchImpl: typeof fetch;

  constructor(opts: RemoteDetectorOptions = {}) {
    const envBase = typeof process !== "undefined" ? process.env?.GHEIM_BASE_URL : undefined;
    const envKey = typeof process !== "undefined" ? process.env?.GHEIM_API_KEY : undefined;
    this.baseUrl = (opts.baseUrl ?? envBase ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
    this.apiKey = opts.apiKey ?? envKey;
    this.model = opts.model ?? DEFAULT_MODEL;
    this.trimWhitespace = opts.trimWhitespace ?? true;
    this.fetchImpl = opts.fetch ?? fetch;
  }

  async detect(text: string, opts?: { model?: string }): Promise<Span[]> {
    if (!text) return [];
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (this.apiKey) headers.Authorization = `Bearer ${this.apiKey}`;
    const resp = await this.fetchImpl(`${this.baseUrl}/v1/detect`, {
      method: "POST",
      headers,
      body: JSON.stringify({ model: opts?.model ?? this.model, text }),
    });
    if (!resp.ok) {
      throw new Error(`gheim-server ${resp.status}: ${await resp.text()}`);
    }
    const data = (await resp.json()) as { spans?: Array<Omit<Span, "text">> };
    const out: Span[] = [];
    for (const item of data.spans ?? []) {
      let { start, end } = item;
      if (this.trimWhitespace) {
        const trimmed = trimSpan(text, start, end);
        if (!trimmed) continue;
        ({ start, end } = trimmed);
      }
      out.push({
        label: item.label,
        start,
        end,
        text: text.slice(start, end),
        score: item.score,
      });
    }
    return out;
  }
}
