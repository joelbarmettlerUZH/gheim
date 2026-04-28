/** Anonymize-input-only endpoints: embeddings, moderations, images. */
import type OpenAIClass from "openai";
import type { Detector } from "../detectors/base.ts";
import { anonymizeValue, type CallExtras, resolveCallSession } from "./_base.ts";

interface RawCreate {
  create(body: unknown): Promise<unknown>;
}

interface RawImages {
  generate(body: unknown): Promise<unknown>;
  edit(body: unknown): Promise<unknown>;
  createVariation?: (body: unknown) => Promise<unknown>;
}

async function anonymizeFields(
  body: Record<string, unknown> & CallExtras,
  defaultDet: Detector,
  fields: string[],
) {
  const { session, detector, rest } = resolveCallSession(body, defaultDet);
  for (const f of fields) {
    if (f in rest) {
      rest[f] = await anonymizeValue(rest[f], session, detector);
    }
  }
  return { rest, session };
}

export class GheimEmbeddings {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { rest } = await anonymizeFields(body, this.defaultDet, ["input"]);
    const client = await this.clientPromise;
    return (client as unknown as { embeddings: RawCreate }).embeddings.create(rest);
  }
}

export class GheimModerations {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async create(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { rest } = await anonymizeFields(body, this.defaultDet, ["input"]);
    const client = await this.clientPromise;
    return (client as unknown as { moderations: RawCreate }).moderations.create(rest);
  }
}

export class GheimImages {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async generate(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { rest } = await anonymizeFields(body, this.defaultDet, ["prompt"]);
    const client = await this.clientPromise;
    return (client as unknown as { images: RawImages }).images.generate(rest);
  }

  async edit(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const { rest } = await anonymizeFields(body, this.defaultDet, ["prompt"]);
    const client = await this.clientPromise;
    return (client as unknown as { images: RawImages }).images.edit(rest);
  }

  async createVariation(body: Record<string, unknown> & CallExtras): Promise<unknown> {
    const client = await this.clientPromise;
    const inner = (client as unknown as { images: RawImages }).images;
    if (!inner.createVariation) {
      throw new Error("openai SDK has no images.createVariation in this version");
    }
    return inner.createVariation(body as unknown);
  }
}
