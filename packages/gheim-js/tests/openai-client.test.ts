import { describe, expect, test } from "bun:test";
import type OpenAIClass from "openai";
import { OpenAI } from "../src/openai/index.ts";
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

/** Minimal fake of `openai.OpenAI` that captures requests and replays canned responses. */
class FakeOpenAI {
  lastBody: unknown;
  response: unknown = null;

  chat = {
    completions: {
      create: async (body: unknown) => {
        this.lastBody = body;
        return this.response;
      },
    },
  };
}

function makeChunks(deltas: string[]): AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk> {
  return {
    async *[Symbol.asyncIterator]() {
      for (const d of deltas) {
        yield {
          id: "x",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o",
          choices: [{ index: 0, delta: { content: d }, finish_reason: null }],
        } as OpenAIClass.Chat.Completions.ChatCompletionChunk;
      }
    },
  };
}

describe("gheim/openai drop-in OpenAI client", () => {
  test("non-streaming round trip", async () => {
    const fake = new FakeOpenAI();
    fake.response = {
      id: "x",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hello <PERSON_1>!", refusal: null },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
    };

    const client = new OpenAI({
      openaiClient: fake as unknown as OpenAIClass,
      gheimDetector: new FakeDetector({ Joel: "private_person" }),
    });
    await client.ready();

    const r = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "I am Joel" }],
    });

    // OpenAI saw the redacted form
    const sent = (fake.lastBody as { messages: Array<{ content: string }> }).messages;
    expect(sent[0]!.content).toBe("I am <PERSON_1>");

    // End user gets the restored form
    expect(r.choices[0]!.message.content).toBe("Hello Joel!");
  });

  test("streaming round trip", async () => {
    const fake = new FakeOpenAI();
    fake.response = makeChunks(["Hi ", "<PER", "SON_1>", ", how can I help?"]);
    const client = new OpenAI({
      openaiClient: fake as unknown as OpenAIClass,
      gheimDetector: new FakeDetector({ Joel: "private_person" }),
    });

    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "I am Joel" }],
      stream: true,
    });

    let collected = "";
    for await (const chunk of stream) {
      collected += chunk.choices[0]?.delta.content ?? "";
    }
    expect(collected).toBe("Hi Joel, how can I help?");
  });

  test("streaming flushes held-back tail", async () => {
    const fake = new FakeOpenAI();
    fake.response = makeChunks(["done <PERSON_"]);
    const client = new OpenAI({
      openaiClient: fake as unknown as OpenAIClass,
      gheimDetector: new FakeDetector({ Joel: "private_person" }),
    });

    const stream = await client.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "x" }],
      stream: true,
    });

    let collected = "";
    for await (const chunk of stream) {
      collected += chunk.choices[0]?.delta.content ?? "";
    }
    expect(collected).toBe("done <PERSON_");
  });
});
