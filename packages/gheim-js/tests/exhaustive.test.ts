/**
 * Exhaustive cross-cutting JS coverage. Mirrors test_exhaustive.py.
 */
import { describe, expect, test } from "bun:test";
import type OpenAIClass from "openai";
import { OpenAI } from "../src/openai/index.ts";
import { Session } from "../src/core/session.ts";
import { _resetWarnings } from "../src/openai/_strict.ts";
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

function cap(): Capture {
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

interface Imgs {
  generateCalls: Array<Record<string, unknown>>;
  editCalls: Array<Record<string, unknown>>;
  generate: (b: Record<string, unknown>) => Promise<unknown>;
  edit: (b: Record<string, unknown>) => Promise<unknown>;
}
function imgs(): Imgs {
  const i: Imgs = {
    generateCalls: [],
    editCalls: [],
    generate: async (b) => {
      i.generateCalls.push(b);
      return { data: [] };
    },
    edit: async (b) => {
      i.editCalls.push(b);
      return { data: [] };
    },
  };
  return i;
}

interface FakeFull {
  chat: { completions: Capture };
  completions: Capture;
  responses: Capture;
  embeddings: Capture;
  moderations: Capture;
  audio: { speech: Capture; transcriptions: Capture; translations: Capture };
  images: Imgs;
  beta: object;
  batches: object;
  files: object;
}

function makeFake(): FakeFull {
  return {
    chat: { completions: cap() },
    completions: cap(),
    responses: cap(),
    embeddings: cap(),
    moderations: cap(),
    audio: { speech: cap(), transcriptions: cap(), translations: cap() },
    images: imgs(),
    beta: {},
    batches: {},
    files: {},
  };
}

function build(opts?: { strict?: boolean }): { client: OpenAI; fake: FakeFull } {
  const fake = makeFake();
  const client = new OpenAI({
    openaiClient: fake as unknown as OpenAIClass,
    gheimDetector: new FakeDetector({ Joel: "private_person", Alice: "private_person" }),
    gheimStrict: opts?.strict ?? true,
  });
  return { client, fake };
}

// Async-iterable helper.
function asyncIter<T>(items: T[]): AsyncIterable<T> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const it of items) yield it;
    },
  };
}

// ============================================================
// Cross-endpoint session reuse
// ============================================================

describe("Cross-endpoint session reuse", () => {
  test("session reused across chat → embeddings keeps same sentinel", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    fake.embeddings.response = { data: [] };
    await client.ready();
    const sess = new Session();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "I am Joel" }],
      gheimSession: sess,
    });
    await client.embeddings.create({
      model: "x",
      input: "Joel works at Acme",
      gheimSession: sess,
    });
    expect(fake.chat.completions.calls[0]!.messages).toMatchObject([{ content: "I am <PERSON_1>" }]);
    expect(fake.embeddings.calls[0]!.input).toBe("<PERSON_1> works at Acme");
    expect(sess.mapping).toEqual({ "<PERSON_1>": "Joel" });
  });

  test("one Session through every wrapped endpoint allocates ONE sentinel", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    fake.responses.response = { output: [] };
    fake.embeddings.response = { data: [] };
    fake.moderations.response = { results: [] };
    fake.audio.speech.response = new Uint8Array();
    fake.audio.transcriptions.response = { text: "ok" };
    fake.audio.translations.response = { text: "ok" };
    fake.completions.response = { choices: [{ text: "ok", index: 0, finish_reason: "stop" }] };
    await client.ready();

    const sess = new Session();
    const text = "Joel here";

    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: text }],
      gheimSession: sess,
    });
    await client.responses.create({ model: "gpt-5", input: text, gheimSession: sess });
    await client.completions.create({ model: "x", prompt: text, gheimSession: sess });
    await client.embeddings.create({ model: "x", input: text, gheimSession: sess });
    await client.moderations.create({ model: "x", input: text, gheimSession: sess });
    await client.audio.speech.create({ model: "tts", voice: "alloy", input: text, gheimSession: sess });
    await client.audio.transcriptions.create({
      file: new Uint8Array(), model: "whisper-1", prompt: text, gheimSession: sess,
    });
    await client.audio.translations.create({
      file: new Uint8Array(), model: "whisper-1", prompt: text, gheimSession: sess,
    });
    await client.images.generate({ model: "dall-e-3", prompt: text, gheimSession: sess });
    await client.images.edit({ image: new Uint8Array(), prompt: text, gheimSession: sess });

    expect(sess.mapping).toEqual({ "<PERSON_1>": "Joel" });

    // Spot-check actual outbound payloads.
    expect((fake.chat.completions.calls[0]!.messages as Array<{ content: string }>)[0]!.content)
      .toBe("<PERSON_1> here");
    expect(fake.responses.calls[0]!.input).toBe("<PERSON_1> here");
    expect(fake.completions.calls[0]!.prompt).toBe("<PERSON_1> here");
    expect(fake.embeddings.calls[0]!.input).toBe("<PERSON_1> here");
    expect(fake.moderations.calls[0]!.input).toBe("<PERSON_1> here");
    expect(fake.audio.speech.calls[0]!.input).toBe("<PERSON_1> here");
    expect(fake.audio.transcriptions.calls[0]!.prompt).toBe("<PERSON_1> here");
    expect(fake.audio.translations.calls[0]!.prompt).toBe("<PERSON_1> here");
    expect(fake.images.generateCalls[0]!.prompt).toBe("<PERSON_1> here");
    expect(fake.images.editCalls[0]!.prompt).toBe("<PERSON_1> here");
  });
});

// ============================================================
// Multi-tool / multi-choice streaming
// ============================================================

describe("Multi-tool, multi-choice", () => {
  test("streaming chunk with multiple choices restores each delta", async () => {
    const { client, fake } = build();
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    fake.chat.completions.response = asyncIter([
      {
        id: "x",
        object: "chat.completion.chunk",
        created: 1,
        model: "gpt-4o",
        choices: [
          { index: 0, delta: { content: "A: hi <PERSON_1>" }, finish_reason: null },
          { index: 1, delta: { content: "B: hello <PERSON_1>" }, finish_reason: null },
        ],
      },
    ]);
    await client.ready();
    const stream = (await client.chat.completions.create({
      model: "gpt-4o",
      n: 2,
      messages: [{ role: "user", content: "Joel" }],
      stream: true,
      gheimSession: sess,
    })) as AsyncIterable<{ choices: Array<{ delta: { content: string } }> }>;
    const chunks: Array<{ choices: Array<{ delta: { content: string } }> }> = [];
    for await (const c of stream) chunks.push(c);
    expect(chunks[0]!.choices[0]!.delta.content).toBe("A: hi Joel");
    expect(chunks[0]!.choices[1]!.delta.content).toBe("B: hello Joel");
  });

  test("non-streaming response with multiple choices restores all", async () => {
    const { client, fake } = build();
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    fake.chat.completions.response = {
      choices: [
        { index: 0, message: { role: "assistant", content: "A: <PERSON_1>" }, finish_reason: "stop" },
        { index: 1, message: { role: "assistant", content: "B: <PERSON_1>" }, finish_reason: "stop" },
      ],
    };
    await client.ready();
    const r = (await client.chat.completions.create({
      model: "gpt-4o",
      n: 2,
      messages: [{ role: "user", content: "Joel" }],
      gheimSession: sess,
    })) as { choices: Array<{ message: { content: string } }> };
    expect(r.choices[0]!.message.content).toBe("A: Joel");
    expect(r.choices[1]!.message.content).toBe("B: Joel");
  });

  test("response with multiple tool_calls restores all argument blobs", async () => {
    const { client, fake } = build();
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    fake.chat.completions.response = {
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              { id: "1", type: "function", function: { name: "lookup", arguments: '{"name": "<PERSON_1>"}' } },
              { id: "2", type: "function", function: { name: "audit", arguments: '{"who": "<PERSON_1>"}' } },
            ],
          },
          finish_reason: "tool_calls",
        },
      ],
    };
    await client.ready();
    const r = (await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "lookup Joel" }],
      gheimSession: sess,
    })) as { choices: Array<{ message: { tool_calls: Array<{ function: { arguments: string } }> } }> };
    expect(r.choices[0]!.message.tool_calls[0]!.function.arguments).toBe('{"name": "Joel"}');
    expect(r.choices[0]!.message.tool_calls[1]!.function.arguments).toBe('{"who": "Joel"}');
  });
});

// ============================================================
// Tool definitions + name field
// ============================================================

describe("Tool defs + name", () => {
  test("tools[*].function.description anonymized; parameters left alone", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      tools: [
        {
          type: "function",
          function: {
            name: "lookup",
            description: "Look up Joel's records",
            parameters: { type: "object", properties: {} },
          },
        },
      ],
    });
    const sent = fake.chat.completions.calls[0]!;
    const tools = sent.tools as Array<{ function: { description: string; parameters: unknown } }>;
    expect(tools[0]!.function.description).toBe("Look up <PERSON_1>'s records");
    expect(tools[0]!.function.parameters).toEqual({ type: "object", properties: {} });
  });

  test("tool without description passes through unchanged", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "hi" }],
      tools: [{ type: "function", function: { name: "x" } }],
    });
    expect(fake.chat.completions.calls[0]!.tools).toEqual([
      { type: "function", function: { name: "x" } },
    ]);
  });

  test("messages[*].name field is anonymized", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi", name: "Joel" }],
    });
    const sent = (fake.chat.completions.calls[0]!.messages as Array<{ name: string }>)[0];
    expect(sent!.name).toBe("<PERSON_1>");
  });
});

// ============================================================
// Multimodal content
// ============================================================

describe("Multimodal content", () => {
  test("text parts redacted, image parts pass through", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "This is Joel" },
            { type: "image_url", image_url: { url: "https://x.png" } },
            { type: "text", text: "with another mention of Joel" },
          ],
        },
      ],
    });
    const parts = (fake.chat.completions.calls[0]!.messages as Array<{ content: unknown[] }>)[0]!.content;
    expect(parts[0]).toEqual({ type: "text", text: "This is <PERSON_1>" });
    expect(parts[1]).toEqual({ type: "image_url", image_url: { url: "https://x.png" } });
    expect(parts[2]).toEqual({ type: "text", text: "with another mention of <PERSON_1>" });
  });
});

// ============================================================
// Empty / null edges
// ============================================================

describe("Empty / null edges", () => {
  test("empty messages array", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await client.chat.completions.create({ model: "gpt-4o", messages: [] });
    expect(fake.chat.completions.calls[0]!.messages).toEqual([]);
  });

  test("null content alongside other messages", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [
        { role: "assistant", content: null, tool_calls: [] },
        { role: "user", content: "Joel" },
      ],
    });
    const sent = fake.chat.completions.calls[0]!.messages as Array<{ content: unknown }>;
    expect(sent[0]!.content).toBeNull();
    expect(sent[1]!.content).toBe("<PERSON_1>");
  });

  test("empty embeddings input produces no sentinels", async () => {
    const { client, fake } = build();
    fake.embeddings.response = { data: [] };
    await client.ready();
    const sess = new Session();
    await client.embeddings.create({ model: "x", input: "", gheimSession: sess });
    expect(sess.mapping).toEqual({});
  });

  test("responses.create without instructions works", async () => {
    const { client, fake } = build();
    fake.responses.response = { output: [] };
    await client.ready();
    await client.responses.create({ model: "gpt-5", input: "Joel here" });
    expect(fake.responses.calls[0]!.input).toBe("<PERSON_1> here");
    expect(fake.responses.calls[0]!.instructions).toBeUndefined();
  });
});

// ============================================================
// Concurrency
// ============================================================

describe("Concurrency", () => {
  test("Promise.all of 20 chat calls all complete", async () => {
    const { client, fake } = build();
    fake.chat.completions.response = {
      choices: [{ index: 0, message: { role: "assistant", content: "ok" }, finish_reason: "stop" }],
    };
    await client.ready();
    await Promise.all(
      Array.from({ length: 20 }, () =>
        client.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: "Joel" }],
        }),
      ),
    );
    expect(fake.chat.completions.calls.length).toBe(20);
  });
});

// ============================================================
// Strict + raw + loose
// ============================================================

describe("Strict + raw + loose", () => {
  test("chained access still raises in strict mode", () => {
    const { client } = build({ strict: true });
    expect(() => {
      const beta = client.beta as unknown as { assistants: { create(b: unknown): unknown } };
      beta.assistants.create({ name: "x" });
    }).toThrow();
  });

  test("raw works after a strict raise", async () => {
    const { client, fake } = build({ strict: true });
    await client.ready();
    try {
      const beta = client.beta as unknown as { assistants: { create(b: unknown): unknown } };
      beta.assistants.create({ name: "x" });
    } catch {
      // expected
    }
    expect(client.raw).toBe(fake as unknown as OpenAIClass);
  });

  test("loose mode does NOT warn for wrapped endpoints", async () => {
    _resetWarnings();
    const { client, fake } = build({ strict: false });
    fake.embeddings.response = { data: [] };
    await client.ready();
    const warns: string[] = [];
    const orig = console.warn;
    console.warn = (...a: unknown[]) => warns.push(a.map(String).join(" "));
    try {
      await client.embeddings.create({ model: "x", input: "Joel" });
    } finally {
      console.warn = orig;
    }
    expect(warns).toEqual([]);
  });
});

// ============================================================
// Session JSON shape stability
// ============================================================

describe("Session JSON shape stability", () => {
  test("toJSON shape is just { mapping }", () => {
    const s = new Session();
    s.allocate("private_person", "Joel");
    s.allocate("private_email", "joel@x");
    expect(Object.keys(s.toJSON())).toEqual(["mapping"]);
  });

  test("fromJSON ignores unknown top-level fields", () => {
    const sess = Session.fromJSON({
      mapping: { "<PERSON_1>": "Joel" },
      // @ts-expect-error - extra field deliberate
      schema_version: "0.2",
      // @ts-expect-error - extra field deliberate
      future: true,
    } as never);
    expect(sess.mapping).toEqual({ "<PERSON_1>": "Joel" });
  });
});

// ============================================================
// No explicit session = fresh per call
// ============================================================

describe("Default session per call", () => {
  test("two embeddings calls without session each get <PERSON_1>", async () => {
    const { client, fake } = build();
    fake.embeddings.response = { data: [] };
    await client.ready();
    await client.embeddings.create({ model: "x", input: "Joel here" });
    await client.embeddings.create({ model: "x", input: "Joel again" });
    expect(fake.embeddings.calls[0]!.input).toBe("<PERSON_1> here");
    expect(fake.embeddings.calls[1]!.input).toBe("<PERSON_1> again");
  });
});

// ============================================================
// Per-call detector override
// ============================================================

describe("Per-call detector override", () => {
  test("override applies on embeddings", async () => {
    const { client, fake } = build();
    fake.embeddings.response = { data: [] };
    await client.ready();
    await client.embeddings.create({
      model: "x",
      input: "Joel and Alice",
      gheimDetector: new FakeDetector({ Alice: "private_person" }),
    });
    expect(fake.embeddings.calls[0]!.input).toBe("Joel and <PERSON_1>");
  });

  test("override applies on responses", async () => {
    const { client, fake } = build();
    fake.responses.response = { output: [] };
    await client.ready();
    await client.responses.create({
      model: "gpt-5",
      input: "Joel and Alice",
      gheimDetector: new FakeDetector({ Alice: "private_person" }),
    });
    expect(fake.responses.calls[0]!.input).toBe("Joel and <PERSON_1>");
  });
});
