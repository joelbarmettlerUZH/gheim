/**
 * Edge-case tests for the gheim/openai drop-in client (Batch A).
 *
 * Mirrors test_openai_edges.py — same scenarios, same fakes.
 */
import { describe, expect, test } from "bun:test";
import type OpenAIClass from "openai";
import { OpenAI } from "../src/openai.ts";
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

class FakeOpenAI {
  createCalls: Array<{ messages: unknown[]; stream: boolean; [k: string]: unknown }> = [];
  response: unknown = null;

  chat = {
    completions: {
      create: async (body: { messages: unknown[]; stream?: boolean; [k: string]: unknown }) => {
        this.createCalls.push({ ...body, stream: !!body.stream });
        if (body.stream) return this.response;
        return this.response;
      },
    },
  };
}

function chunkAsyncIter(
  items: Array<OpenAIClass.Chat.Completions.ChatCompletionChunk | Error>,
): AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const it of items) {
        if (it instanceof Error) throw it;
        yield it;
      }
    },
  };
}

const ck = (
  delta: { content?: string | null; role?: string; tool_calls?: unknown },
  finish_reason: string | null = null,
): OpenAIClass.Chat.Completions.ChatCompletionChunk => ({
  id: "x",
  object: "chat.completion.chunk",
  created: 1,
  model: "gpt-4o",
  choices: [{ index: 0, delta: delta as never, finish_reason: finish_reason as never }],
}) as OpenAIClass.Chat.Completions.ChatCompletionChunk;

function clientWith(detector: Detector, fake: FakeOpenAI): OpenAI {
  return new OpenAI({
    openaiClient: fake as unknown as OpenAIClass,
    gheimDetector: detector,
  });
}

// ---------- A.1 tool-call argument restoration ----------

describe("Batch A: gheim/openai edge cases", () => {
  test("non-streaming restores sentinels in tool_call.function.arguments", async () => {
    const fake = new FakeOpenAI();
    fake.response = {
      id: "x",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          finish_reason: "tool_calls",
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "call_1",
                type: "function",
                function: {
                  name: "lookup_user",
                  arguments: '{"name": "<PERSON_1>"}',
                },
              },
            ],
            refusal: null,
          },
        },
      ],
    };
    const { Session } = await import("../src/core/session.ts");
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    const client = clientWith(new FakeDetector({ Joel: "private_person" }), fake);
    await client.ready();
    const r = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Joel" }],
      gheimSession: sess,
    });
    const args = (r.choices[0]!.message.tool_calls ?? [])[0]?.function?.arguments;
    expect(args).toBe('{"name": "Joel"}');
  });

  // ---------- A.2 session reuse across calls ----------

  test("gheimSession reuse across calls keeps sentinels stable", async () => {
    const fake = new FakeOpenAI();
    fake.response = {
      id: "x",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "<PERSON_1> noted", refusal: null },
          finish_reason: "stop",
        },
      ],
    };
    const { Session } = await import("../src/core/session.ts");
    const sess = new Session();
    const client = clientWith(new FakeDetector({ Joel: "private_person" }), fake);
    await client.ready();

    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "I am Joel" }],
      gheimSession: sess,
    });
    expect((fake.createCalls[0]!.messages as Array<{ content: string }>)[0]!.content)
      .toBe("I am <PERSON_1>");

    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Joel asked again" }],
      gheimSession: sess,
    });
    expect((fake.createCalls[1]!.messages as Array<{ content: string }>)[0]!.content)
      .toBe("<PERSON_1> asked again");
    expect(sess.mapping).toEqual({ "<PERSON_1>": "Joel" });
  });

  // ---------- A.3 chunks with delta.content == null pass through ----------

  test("streaming passes role-only and finish-only chunks through unmodified", async () => {
    const fake = new FakeOpenAI();
    fake.response = chunkAsyncIter([
      ck({ role: "assistant", content: null }),
      ck({ content: "Hi " }),
      ck({ content: "<PERSON_1>" }),
      ck({ content: null }, "stop"),
    ]);
    const { Session } = await import("../src/core/session.ts");
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    const client = clientWith(new FakeDetector({ Joel: "private_person" }), fake);
    await client.ready();
    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Joel" }],
      stream: true,
      gheimSession: sess,
    });
    const seen: Array<{
      role: string | null;
      content: string | null;
      finish: string | null;
    }> = [];
    for await (const c of stream) {
      const choice = c.choices[0];
      const delta = choice?.delta as { role?: string; content?: string | null };
      seen.push({
        role: delta?.role ?? null,
        content: delta?.content ?? null,
        finish: choice?.finish_reason ?? null,
      });
    }
    expect(seen.length).toBe(4);
    expect(seen[0]!.role).toBe("assistant");
    expect(seen[0]!.content).toBeNull();
    expect(seen[1]!.content).toBe("Hi ");
    expect(seen[2]!.content).toBe("Joel");
    expect(seen[3]!.finish).toBe("stop");
  });

  // ---------- A.4 early break ----------

  test("streaming early break does not throw or hang", async () => {
    const fake = new FakeOpenAI();
    fake.response = chunkAsyncIter([
      ck({ content: "Hi " }),
      ck({ content: "<PERSON_1>" }),
      ck({ content: ", how can I help?" }),
    ]);
    const { Session } = await import("../src/core/session.ts");
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    const client = clientWith(new FakeDetector({ Joel: "private_person" }), fake);
    await client.ready();
    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Joel" }],
      stream: true,
      gheimSession: sess,
    });
    const seen: string[] = [];
    for await (const c of stream) {
      seen.push(c.choices[0]?.delta.content ?? "");
      if (seen.length === 1) break;
    }
    expect(seen).toEqual(["Hi "]);
  });

  // ---------- A.5 exception during iteration propagates ----------

  test("upstream error during streaming propagates", async () => {
    const fake = new FakeOpenAI();
    fake.response = chunkAsyncIter([
      ck({ content: "Hi <" }),
      new Error("upstream blew up"),
    ]);
    const { Session } = await import("../src/core/session.ts");
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    const client = clientWith(new FakeDetector({ Joel: "private_person" }), fake);
    await client.ready();
    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Joel" }],
      stream: true,
      gheimSession: sess,
    });
    const collected: string[] = [];
    let caught: unknown;
    try {
      for await (const c of stream) {
        collected.push(c.choices[0]?.delta.content ?? "");
      }
    } catch (e) {
      caught = e;
    }
    expect(caught).toBeInstanceOf(Error);
    expect((caught as Error).message).toBe("upstream blew up");
    // First chunk arrived; the held-back '<' is buried in the buffer.
    expect(collected).toEqual(["Hi "]);
  });

  // ---------- A.6 per-call detector override ----------

  test("per-call gheimDetector overrides the client default", async () => {
    const fake = new FakeOpenAI();
    fake.response = {
      id: "x",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "ok", refusal: null },
          finish_reason: "stop",
        },
      ],
    };
    const client = clientWith(new FakeDetector({ Joel: "private_person" }), fake);
    await client.ready();
    await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Joel and Alice" }],
      gheimDetector: new FakeDetector({ Alice: "private_person" }),
    });
    const sent = (fake.createCalls[0]!.messages as Array<{ content: string }>)[0]!.content;
    expect(sent).toBe("Joel and <PERSON_1>");
  });
});
