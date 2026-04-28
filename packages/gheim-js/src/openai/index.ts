/**
 * OpenAI SDK drop-in wrapper (gheim/openai).
 *
 * Public surface:
 *   - `OpenAI` / `AsyncOpenAI` — drop-in clients with PII round-trip on every
 *     text-carrying endpoint
 *   - Stateless helpers (kept for backward-compat with v0.1):
 *       anonymizeOpenAIMessages, deanonymizeOpenAIResponse, deanonymizeOpenAIStream
 *
 * The `openai` package is an *optional* peer dependency — only required when
 * the drop-in client is actually instantiated.
 */
import type OpenAIClass from "openai";
import { Session } from "../core/session.ts";
import { StreamDeanonymizer } from "../core/stream.ts";
import type { Detector } from "../detectors/base.ts";
import { anonymizeMessages } from "../plain.ts";

export { OpenAI, AsyncOpenAI } from "./_client.ts";
export type { GheimOpenAIOptions } from "./_client.ts";
export { GheimStrictError, STRICT_REASONS } from "./_strict.ts";

// ---------- backward-compatible stateless helpers ----------

export function anonymizeOpenAIMessages(
  messages: OpenAIClass.Chat.Completions.ChatCompletionMessageParam[],
  session: Session,
  opts?: { detector?: Detector; model?: string },
): Promise<OpenAIClass.Chat.Completions.ChatCompletionMessageParam[]> {
  return anonymizeMessages(messages, session, opts);
}

export function deanonymizeOpenAIResponse<R extends OpenAIClass.Chat.Completions.ChatCompletion>(
  response: R,
  session: Session,
): R {
  for (const choice of response.choices ?? []) {
    const msg = choice.message;
    if (msg && typeof msg.content === "string") {
      msg.content = session.restore(msg.content);
    }
    for (const tc of msg?.tool_calls ?? []) {
      const fn = (tc as { function?: { arguments?: string } }).function;
      if (fn && typeof fn.arguments === "string") {
        fn.arguments = session.restore(fn.arguments);
      }
    }
  }
  return response;
}

export async function* deanonymizeOpenAIStream(
  stream: AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk>,
  session: Session,
): AsyncGenerator<OpenAIClass.Chat.Completions.ChatCompletionChunk> {
  const buf = new StreamDeanonymizer(session.mapping);
  let last: OpenAIClass.Chat.Completions.ChatCompletionChunk | undefined;
  for await (const chunk of stream) {
    last = chunk;
    for (const choice of chunk.choices ?? []) {
      const delta = choice.delta as { content?: string | null };
      if (typeof delta.content === "string") {
        delta.content = buf.feed(delta.content);
      }
    }
    yield chunk;
  }
  const tail = buf.flush();
  if (tail && last) {
    yield {
      id: last.id,
      object: last.object,
      created: last.created,
      model: last.model,
      choices: [
        {
          index: 0,
          delta: { content: tail, role: undefined as never },
          finish_reason: null,
        },
      ],
    } as OpenAIClass.Chat.Completions.ChatCompletionChunk;
  }
}
