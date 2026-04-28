/** Audio endpoints — speech (anonymize-only); transcriptions/translations (round-trip). */
import type OpenAIClass from "openai";
import { Session } from "../core/session.ts";
import { StreamDeanonymizer } from "../core/stream.ts";
import type { Detector } from "../detectors/base.ts";
import { StreamingResponse, anonymizeValue, type CallExtras, resolveCallSession } from "./_base.ts";

interface AudioInner {
  speech: { create(body: unknown): Promise<unknown> };
  transcriptions: { create(body: unknown): Promise<unknown> };
  translations: { create(body: unknown): Promise<unknown> };
}

interface TranscriptionResp {
  text?: string;
}

interface TranscriptionChunk {
  delta?: string;
  text?: string;
}

function patchTranscriptionChunk(chunk: TranscriptionChunk, buf: StreamDeanonymizer): void {
  if (typeof chunk.delta === "string") {
    chunk.delta = buf.feed(chunk.delta);
    return;
  }
  if (typeof chunk.text === "string") {
    chunk.text = buf.feed(chunk.text);
  }
}

export class GheimSpeech {
  constructor(
    private readonly innerP: Promise<AudioInner>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { session, detector, rest } = resolveCallSession(body, this.defaultDet);
    for (const k of ["input", "instructions"]) {
      if (k in rest) {
        rest[k] = await anonymizeValue(rest[k], session, detector);
      }
    }
    const inner = await this.innerP;
    return inner.speech.create(rest);
  }
}

export class GheimTranscriptions {
  constructor(
    private readonly innerP: Promise<AudioInner>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { session, detector, rest } = resolveCallSession(body, this.defaultDet);
    if (typeof rest.prompt === "string") {
      rest.prompt = await anonymizeValue(rest.prompt, session, detector);
    }
    const inner = await this.innerP;
    const response = await inner.transcriptions.create(rest);
    if (rest.stream) {
      return new StreamingResponse(
        response as AsyncIterable<TranscriptionChunk>,
        session,
        patchTranscriptionChunk,
      );
    }
    const r = response as TranscriptionResp;
    if (typeof r.text === "string") r.text = session.restore(r.text);
    return response;
  }
}

export class GheimTranslations {
  constructor(
    private readonly innerP: Promise<AudioInner>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { session, detector, rest } = resolveCallSession(body, this.defaultDet);
    if (typeof rest.prompt === "string") {
      rest.prompt = await anonymizeValue(rest.prompt, session, detector);
    }
    const inner = await this.innerP;
    const response = await inner.translations.create(rest);
    const r = response as TranscriptionResp;
    if (typeof r.text === "string") r.text = session.restore(r.text);
    return response;
  }
}

export class GheimAudio {
  readonly speech: GheimSpeech;
  readonly transcriptions: GheimTranscriptions;
  readonly translations: GheimTranslations;

  constructor(clientPromise: Promise<OpenAIClass>, detector: Detector) {
    const innerP = clientPromise.then((c) => c.audio as unknown as AudioInner);
    this.speech = new GheimSpeech(innerP, detector);
    this.transcriptions = new GheimTranscriptions(innerP, detector);
    this.translations = new GheimTranslations(innerP, detector);
  }
}
