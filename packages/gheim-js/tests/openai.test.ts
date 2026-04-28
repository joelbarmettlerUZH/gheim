import { describe, expect, test } from "bun:test";
import type OpenAI from "openai";
import { Session } from "../src/core/session.ts";
import {
  anonymizeOpenAIMessages,
  deanonymizeOpenAIResponse,
  deanonymizeOpenAIStream,
} from "../src/openai/index.ts";
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

function sessionWith(d: Detector): Session {
  const s = new Session();
  (s as unknown as { detector: Detector }).detector = d;
  return s;
}

describe("gheim/openai helpers", () => {
  test("anonymizeOpenAIMessages preserves OpenAI type shape", async () => {
    const s = sessionWith(new FakeDetector({ Joel: "private_person" }));
    const msgs: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [
      { role: "system", content: "You speak to Joel." },
      { role: "user", content: "Hi from Joel" },
    ];
    const out = await anonymizeOpenAIMessages(msgs, s);
    expect(out[0]!.content).toBe("You speak to <PERSON_1>.");
    expect(out[1]!.content).toBe("Hi from <PERSON_1>");
  });

  test("deanonymizeOpenAIResponse mutates message content", () => {
    const s = sessionWith(new FakeDetector({ Joel: "private_person" }));
    s.allocate("private_person", "Joel");
    const response: OpenAI.Chat.Completions.ChatCompletion = {
      id: "x",
      object: "chat.completion",
      created: 1,
      model: "gpt-4o",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "Hi <PERSON_1>!", refusal: null },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
    } as OpenAI.Chat.Completions.ChatCompletion;
    const out = deanonymizeOpenAIResponse(response, s);
    expect(out.choices[0]!.message.content).toBe("Hi Joel!");
  });

  test("deanonymizeOpenAIStream restores split sentinels in chunk deltas", async () => {
    const s = sessionWith(new FakeDetector({ Joel: "private_person" }));
    s.allocate("private_person", "Joel");

    const deltas = ["Hi ", "<PER", "SON_1>", ", how are you?"];
    async function* upstream() {
      for (const d of deltas) {
        const c: OpenAI.Chat.Completions.ChatCompletionChunk = {
          id: "x",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4o",
          choices: [{ index: 0, delta: { content: d }, finish_reason: null }],
        } as OpenAI.Chat.Completions.ChatCompletionChunk;
        yield c;
      }
    }

    let collected = "";
    for await (const chunk of deanonymizeOpenAIStream(upstream(), s)) {
      const piece = chunk.choices[0]?.delta.content ?? "";
      collected += piece;
    }
    expect(collected).toBe("Hi Joel, how are you?");
  });
});
