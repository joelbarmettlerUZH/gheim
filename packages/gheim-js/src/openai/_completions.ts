/** Legacy completions API — text-in / text-out. */
import type OpenAIClass from "openai";
import { Session } from "../core/session.ts";
import { StreamDeanonymizer } from "../core/stream.ts";
import type { Detector } from "../detectors/base.ts";
import { StreamingResponse, anonymizeValue, type CallExtras, resolveCallSession } from "./_base.ts";

interface RawCompletions {
  create(body: unknown): Promise<unknown>;
}

interface LegacyChunk {
  choices: Array<{ text?: string | null; index: number; finish_reason?: string | null }>;
  id: string;
  model: string;
  object?: string;
  created?: number;
}

function patchLegacyChunk(chunk: LegacyChunk, buf: StreamDeanonymizer): void {
  for (const choice of chunk.choices ?? []) {
    if (typeof choice.text === "string") {
      choice.text = buf.feed(choice.text);
    }
  }
}

function makeLegacyTail(template: LegacyChunk, tail: string): LegacyChunk {
  return {
    id: template.id,
    object: template.object ?? "text_completion.chunk",
    created: template.created ?? 0,
    model: template.model,
    choices: [{ index: 0, text: tail, finish_reason: null }],
  };
}

export class GheimCompletions {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { session, detector, rest } = resolveCallSession(body, this.defaultDet);
    if ("prompt" in rest) {
      rest.prompt = await anonymizeValue(rest.prompt, session, detector);
    }
    const client = await this.clientPromise;
    const response = await (client as unknown as { completions: RawCompletions }).completions.create(rest);
    if (rest.stream) {
      return new StreamingResponse(
        response as AsyncIterable<LegacyChunk>,
        session,
        patchLegacyChunk,
        makeLegacyTail,
      );
    }
    const r = response as { choices?: Array<{ text?: string | null }> };
    for (const choice of r.choices ?? []) {
      if (typeof choice.text === "string") {
        choice.text = session.restore(choice.text);
      }
    }
    return response;
  }
}
