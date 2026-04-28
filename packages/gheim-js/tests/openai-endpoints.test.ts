/**
 * Tests for the new OpenAI endpoint proxies (Responses, Completions,
 * Embeddings, Moderations, Audio, Images) and strict mode (JS).
 *
 * Mirrors test_openai_endpoints.py.
 */
import { describe, expect, test } from "bun:test";
import type OpenAIClass from "openai";
import { OpenAI } from "../src/openai/index.ts";
import { Session } from "../src/core/session.ts";
import { GheimStrictError, _resetWarnings } from "../src/openai/_strict.ts";
import type { Detector, Span } from "../src/detectors/base.ts";

class FakeDetector implements Detector {
  constructor(private surfaces: Record<string, string>) {}
  async detect(text: string): Promise<Span[]> {
    const out: Span[] = [];
    for (const [surface, label] of Object.entries(this.surfaces)) {
      let i = 0;
      while (true) {
        const idx = text.indexOf(surface, i);
        if (idx === -1) break;
        out.push({ label, start: idx, end: idx + surface.length, text: surface });
        i = idx + surface.length;
      }
    }
    return out;
  }
}

interface Capture {
  calls: Array<Record<string, unknown>>;
  response: unknown;
  create: (body: Record<string, unknown>) => Promise<unknown>;
}

function makeCapture(): Capture {
  const c: Capture = {
    calls: [],
    response: null,
    create: async (body) => {
      c.calls.push(body);
      return c.response;
    },
  };
  return c;
}

interface ImagesCapture {
  generateCalls: Array<Record<string, unknown>>;
  editCalls: Array<Record<string, unknown>>;
  generate: (body: Record<string, unknown>) => Promise<unknown>;
  edit: (body: Record<string, unknown>) => Promise<unknown>;
}

function makeImages(): ImagesCapture {
  const ic: ImagesCapture = {
    generateCalls: [],
    editCalls: [],
    generate: async (body) => {
      ic.generateCalls.push(body);
      return { data: [] };
    },
    edit: async (body) => {
      ic.editCalls.push(body);
      return { data: [] };
    },
  };
  return ic;
}

interface FakeFullClient {
  chat: { completions: Capture };
  completions: Capture;
  responses: Capture;
  embeddings: Capture;
  moderations: Capture;
  audio: { speech: Capture; transcriptions: Capture; translations: Capture };
  images: ImagesCapture;
  beta: object;
  batches: object;
  files: object;
}

function makeFakeClient(): FakeFullClient {
  return {
    chat: { completions: makeCapture() },
    completions: makeCapture(),
    responses: makeCapture(),
    embeddings: makeCapture(),
    moderations: makeCapture(),
    audio: {
      speech: makeCapture(),
      transcriptions: makeCapture(),
      translations: makeCapture(),
    },
    images: makeImages(),
    beta: {},
    batches: {},
    files: {},
  };
}

function buildClient(opts?: { strict?: boolean }): { client: OpenAI; fake: FakeFullClient } {
  const fake = makeFakeClient();
  const client = new OpenAI({
    openaiClient: fake as unknown as OpenAIClass,
    gheimDetector: new FakeDetector({ Joel: "private_person" }),
    gheimStrict: opts?.strict ?? true,
  });
  return { client, fake };
}

// ============================================================
// Embeddings
// ============================================================

describe("Embeddings (anonymize input)", () => {
  test("string input is anonymized", async () => {
    const { client, fake } = buildClient();
    fake.embeddings.response = { data: [{ embedding: [0.1, 0.2] }] };
    await client.ready();
    await client.embeddings.create({ model: "text-embedding-3-small", input: "Hi from Joel" });
    expect(fake.embeddings.calls[0]!.input).toBe("Hi from <PERSON_1>");
  });

  test("list[str] input is anonymized element-wise", async () => {
    const { client, fake } = buildClient();
    fake.embeddings.response = { data: [] };
    await client.ready();
    await client.embeddings.create({
      model: "text-embedding-3-small",
      input: ["Joel", "Alice and Joel"],
    });
    expect(fake.embeddings.calls[0]!.input).toEqual(["<PERSON_1>", "Alice and <PERSON_1>"]);
  });

  test("list[int] (pre-tokenized) input passes through", async () => {
    const { client, fake } = buildClient();
    fake.embeddings.response = { data: [] };
    await client.ready();
    await client.embeddings.create({
      model: "text-embedding-3-small",
      input: [1234, 5678, 9012],
    });
    expect(fake.embeddings.calls[0]!.input).toEqual([1234, 5678, 9012]);
  });
});

// ============================================================
// Moderations
// ============================================================

describe("Moderations", () => {
  test("input is anonymized", async () => {
    const { client, fake } = buildClient();
    fake.moderations.response = { results: [] };
    await client.ready();
    await client.moderations.create({ model: "omni-moderation-latest", input: "Joel said x" });
    expect(fake.moderations.calls[0]!.input).toBe("<PERSON_1> said x");
  });
});

// ============================================================
// Legacy completions
// ============================================================

describe("Legacy completions", () => {
  test("non-streaming round-trip", async () => {
    const { client, fake } = buildClient();
    fake.completions.response = {
      choices: [{ text: "Hello <PERSON_1>!", index: 0, finish_reason: "stop" }],
    };
    await client.ready();
    const r = (await client.completions.create({
      model: "gpt-3.5-turbo-instruct",
      prompt: "I am Joel",
    })) as { choices: Array<{ text: string }> };
    expect(fake.completions.calls[0]!.prompt).toBe("I am <PERSON_1>");
    expect(r.choices[0]!.text).toBe("Hello Joel!");
  });

  test("streaming round-trip with split sentinel", async () => {
    const { client, fake } = buildClient();
    const chunks = [
      { id: "c", model: "x", choices: [{ index: 0, text: "Hi ", finish_reason: null }] },
      { id: "c", model: "x", choices: [{ index: 0, text: "<PER", finish_reason: null }] },
      { id: "c", model: "x", choices: [{ index: 0, text: "SON_1>", finish_reason: null }] },
      { id: "c", model: "x", choices: [{ index: 0, text: "!", finish_reason: null }] },
    ];
    fake.completions.response = {
      async *[Symbol.asyncIterator]() {
        for (const c of chunks) yield c;
      },
    };
    await client.ready();
    const stream = (await client.completions.create({
      model: "gpt-3.5-turbo-instruct",
      prompt: "I am Joel",
      stream: true,
    })) as AsyncIterable<{ choices: Array<{ text: string }> }>;
    let out = "";
    for await (const c of stream) out += c.choices[0]!.text ?? "";
    expect(out).toBe("Hi Joel!");
  });
});

// ============================================================
// Responses API
// ============================================================

describe("Responses API", () => {
  test("non-streaming round-trip", async () => {
    const { client, fake } = buildClient();
    fake.responses.response = {
      output: [{ content: [{ text: "Hi <PERSON_1>!", type: "output_text" }] }],
    };
    await client.ready();
    const r = (await client.responses.create({
      model: "gpt-5",
      input: "I am Joel",
    })) as { output: Array<{ content: Array<{ text: string }> }> };
    expect(fake.responses.calls[0]!.input).toBe("I am <PERSON_1>");
    expect(r.output[0]!.content[0]!.text).toBe("Hi Joel!");
  });

  test("anonymizes instructions", async () => {
    const { client, fake } = buildClient();
    fake.responses.response = { output: [] };
    await client.ready();
    await client.responses.create({
      model: "gpt-5",
      input: "hi",
      instructions: "Write to Joel",
    });
    expect(fake.responses.calls[0]!.instructions).toBe("Write to <PERSON_1>");
  });

  test("event-based streaming reassembles split sentinel", async () => {
    const { client, fake } = buildClient();
    const events = [
      { type: "response.output_text.delta", delta: "Hi " },
      { type: "response.output_text.delta", delta: "<PER" },
      { type: "response.output_text.delta", delta: "SON_1>" },
      { type: "response.output_text.delta", delta: "!" },
      { type: "response.completed" },
    ];
    fake.responses.response = {
      async *[Symbol.asyncIterator]() {
        for (const e of events) yield e;
      },
    };
    await client.ready();
    const stream = (await client.responses.create({
      model: "gpt-5",
      input: "I am Joel",
      stream: true,
    })) as AsyncIterable<{ type: string; delta?: string }>;
    let text = "";
    let sawCompleted = false;
    for await (const e of stream) {
      if (e.type === "response.output_text.delta" && e.delta) text += e.delta;
      if (e.type === "response.completed") sawCompleted = true;
    }
    expect(text).toBe("Hi Joel!");
    expect(sawCompleted).toBe(true);
  });

  test("unknown event types pass through unchanged", async () => {
    const { client, fake } = buildClient();
    fake.responses.response = {
      async *[Symbol.asyncIterator]() {
        yield { type: "response.something.weird" };
      },
    };
    await client.ready();
    const stream = (await client.responses.create({
      model: "gpt-5",
      input: "x",
      stream: true,
    })) as AsyncIterable<{ type: string }>;
    const types: string[] = [];
    for await (const e of stream) types.push(e.type);
    expect(types).toContain("response.something.weird");
  });
});

// ============================================================
// Audio: speech / transcriptions / translations
// ============================================================

describe("Audio", () => {
  test("speech anonymizes input + instructions", async () => {
    const { client, fake } = buildClient();
    fake.audio.speech.response = new Uint8Array();
    await client.ready();
    await client.audio.speech.create({
      model: "tts-1",
      voice: "alloy",
      input: "Hello Joel",
      instructions: "speak like Joel",
    });
    expect(fake.audio.speech.calls[0]!.input).toBe("Hello <PERSON_1>");
    expect(fake.audio.speech.calls[0]!.instructions).toBe("speak like <PERSON_1>");
  });

  test("transcriptions restore returned text", async () => {
    const { client, fake } = buildClient();
    fake.audio.transcriptions.response = { text: "Hello <PERSON_1>!" };
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    await client.ready();
    const r = (await client.audio.transcriptions.create({
      file: new Uint8Array(),
      model: "whisper-1",
      gheimSession: sess,
    })) as { text: string };
    expect(r.text).toBe("Hello Joel!");
  });

  test("transcriptions anonymize the prompt hint", async () => {
    const { client, fake } = buildClient();
    fake.audio.transcriptions.response = { text: "ok" };
    await client.ready();
    await client.audio.transcriptions.create({
      file: new Uint8Array(),
      model: "whisper-1",
      prompt: "Joel will be speaking",
    });
    expect(fake.audio.transcriptions.calls[0]!.prompt).toBe("<PERSON_1> will be speaking");
  });

  test("translations restore returned text", async () => {
    const { client, fake } = buildClient();
    fake.audio.translations.response = { text: "Bonjour <PERSON_1>" };
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    await client.ready();
    const r = (await client.audio.translations.create({
      file: new Uint8Array(),
      model: "whisper-1",
      gheimSession: sess,
    })) as { text: string };
    expect(r.text).toBe("Bonjour Joel");
  });
});

// ============================================================
// Images
// ============================================================

describe("Images", () => {
  test("generate anonymizes prompt", async () => {
    const { client, fake } = buildClient();
    await client.ready();
    await client.images.generate({ model: "dall-e-3", prompt: "A portrait of Joel" });
    expect(fake.images.generateCalls[0]!.prompt).toBe("A portrait of <PERSON_1>");
  });

  test("edit anonymizes prompt", async () => {
    const { client, fake } = buildClient();
    await client.ready();
    await client.images.edit({ image: new Uint8Array(), prompt: "Make Joel smile" });
    expect(fake.images.editCalls[0]!.prompt).toBe("Make <PERSON_1> smile");
  });
});

// ============================================================
// Strict mode
// ============================================================

describe("Strict mode", () => {
  test("client.beta.assistants.create throws GheimStrictError in strict mode", async () => {
    const { client } = buildClient({ strict: true });
    await client.ready();
    expect(() => {
      // The strict proxy throws on .apply (call), so we have to invoke.
      const beta = client.beta as unknown as { assistants: { create(b: unknown): unknown } };
      beta.assistants.create({ name: "x" });
    }).toThrow(GheimStrictError);
  });

  test("strict error message points to client.raw and gheimStrict=false", () => {
    const { client } = buildClient({ strict: true });
    let caught: Error | null = null;
    try {
      const beta = client.beta as unknown as { assistants: { create(b: unknown): unknown } };
      beta.assistants.create({ name: "x" });
    } catch (e) {
      caught = e as Error;
    }
    expect(caught).not.toBeNull();
    expect(caught!.message).toContain("client.raw");
    expect(caught!.message).toContain("gheimStrict: false");
  });

  test("client.batches.create raises in strict mode", () => {
    const { client } = buildClient({ strict: true });
    expect(() => {
      const b = client.batches as unknown as { create(body: unknown): unknown };
      b.create({});
    }).toThrow(GheimStrictError);
  });

  test("client.files.create raises in strict mode", () => {
    const { client } = buildClient({ strict: true });
    expect(() => {
      const f = client.files as unknown as { create(body: unknown): unknown };
      f.create({});
    }).toThrow(GheimStrictError);
  });

  test("loose mode (gheimStrict=false) warns once then forwards", async () => {
    _resetWarnings();
    const { client } = buildClient({ strict: false });
    await client.ready();
    const warns: string[] = [];
    const origWarn = console.warn;
    console.warn = (...args: unknown[]) => warns.push(args.map(String).join(" "));
    try {
      const beta = client.beta as unknown as { someProp?: unknown };
      void beta.someProp; // triggers warn + forwards
      void beta.someProp; // second access — no extra warning
    } finally {
      console.warn = origWarn;
    }
    expect(warns.length).toBe(1);
    expect(warns[0]!).toContain("WITHOUT gheim PII redaction");
  });

  test("client.raw bypasses gheim entirely", async () => {
    const { client, fake } = buildClient({ strict: true });
    await client.ready();
    expect(client.raw).toBe(fake as unknown as OpenAIClass);
  });
});
