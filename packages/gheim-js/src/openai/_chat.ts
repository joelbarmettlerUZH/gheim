/** Chat completions wrapper — adds tools/name redaction on top of the existing surface. */
import type OpenAIClass from "openai";
import { Session } from "../core/session.ts";
import { StreamDeanonymizer } from "../core/stream.ts";
import type { Detector } from "../detectors/base.ts";
import { anonymizeMessages } from "../plain.ts";
import { StreamingResponse, anonymizeValue, type CallExtras, resolveCallSession } from "./_base.ts";

type ChatCreateBody = (
  | OpenAIClass.Chat.Completions.ChatCompletionCreateParamsNonStreaming
  | OpenAIClass.Chat.Completions.ChatCompletionCreateParamsStreaming
) & CallExtras;

async function anonymizeChatBody(
  rest: Record<string, unknown>,
  session: Session,
  detector: Detector,
): Promise<Record<string, unknown>> {
  const messages = rest.messages as
    | OpenAIClass.Chat.Completions.ChatCompletionMessageParam[]
    | undefined;
  if (messages) {
    const cleaned = await anonymizeMessages(messages, session, { detector });
    // Also redact each message's optional `name` field.
    for (const msg of cleaned as Array<{ name?: unknown }>) {
      if (typeof msg.name === "string") {
        msg.name = (await anonymizeValue(msg.name, session, detector)) as string;
      }
    }
    rest.messages = cleaned;
  }
  // tools[*].function.description
  const tools = rest.tools as Array<Record<string, unknown>> | undefined;
  if (tools) {
    const newTools: Array<Record<string, unknown>> = [];
    for (const tool of tools) {
      const fn = tool.function as Record<string, unknown> | undefined;
      if (fn && typeof fn.description === "string") {
        const newFn = {
          ...fn,
          description: await anonymizeValue(fn.description, session, detector),
        };
        newTools.push({ ...tool, function: newFn });
      } else {
        newTools.push(tool);
      }
    }
    rest.tools = newTools;
  }
  return rest;
}

function restoreChatResponse(
  response: OpenAIClass.Chat.Completions.ChatCompletion,
  session: Session,
): void {
  for (const choice of response.choices ?? []) {
    const msg = choice.message as
      | (typeof choice.message & {
          reasoning?: string | null;
          reasoning_details?: Array<{ text?: string | null }> | null;
        })
      | undefined;
    if (msg && typeof msg.content === "string") {
      msg.content = session.restore(msg.content);
    }
    // Reasoning fields are emitted by reasoning models (deepseek, o1) and
    // may carry sentinel residue. Restore them so dashboards / logs that
    // read these fields don't see "<PERSON_1>" etc.
    if (msg && typeof msg.reasoning === "string") {
      msg.reasoning = session.restore(msg.reasoning);
    }
    if (msg && Array.isArray(msg.reasoning_details)) {
      for (const r of msg.reasoning_details) {
        if (r && typeof r.text === "string") {
          r.text = session.restore(r.text);
        }
      }
    }
    for (const tc of msg?.tool_calls ?? []) {
      const fn = (tc as { function?: { arguments?: string } }).function;
      if (fn && typeof fn.arguments === "string") {
        fn.arguments = session.restore(fn.arguments);
      }
    }
  }
}

function patchChatChunk(
  chunk: OpenAIClass.Chat.Completions.ChatCompletionChunk,
  buf: StreamDeanonymizer,
): void {
  for (const choice of chunk.choices ?? []) {
    const delta = choice.delta as {
      content?: string | null;
      reasoning?: string | null;
      reasoning_details?: Array<{ text?: string | null }> | null;
    };
    if (typeof delta.content === "string") {
      delta.content = buf.feed(delta.content);
    }
    // Reasoning streams (deepseek, o1) — restore sentinel residue here too.
    if (typeof delta.reasoning === "string") {
      delta.reasoning = buf.feed(delta.reasoning);
    }
    if (Array.isArray(delta.reasoning_details)) {
      for (const r of delta.reasoning_details) {
        if (r && typeof r.text === "string") {
          r.text = buf.feed(r.text);
        }
      }
    }
  }
}

function makeChatTailChunk(
  template: OpenAIClass.Chat.Completions.ChatCompletionChunk,
  content: string,
): OpenAIClass.Chat.Completions.ChatCompletionChunk {
  return {
    id: template.id,
    object: template.object,
    created: template.created,
    model: template.model,
    choices: [
      {
        index: 0,
        delta: { content, role: undefined as never },
        finish_reason: null,
      },
    ],
  } as OpenAIClass.Chat.Completions.ChatCompletionChunk;
}

interface RawChat {
  completions: { create(body: unknown): Promise<unknown> };
}

export class GheimChatCompletions {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async create(
    body: OpenAIClass.Chat.Completions.ChatCompletionCreateParamsNonStreaming & CallExtras,
  ): Promise<OpenAIClass.Chat.Completions.ChatCompletion>;
  async create(
    body: OpenAIClass.Chat.Completions.ChatCompletionCreateParamsStreaming & CallExtras,
  ): Promise<AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk>>;
  async create(body: ChatCreateBody): Promise<unknown> {
    const { session, detector, rest } = resolveCallSession(
      body as unknown as Record<string, unknown> & CallExtras,
      this.defaultDet,
    );
    const wireBody = await anonymizeChatBody(rest, session, detector);
    const client = await this.clientPromise;
    const response = await ((client.chat as unknown as RawChat).completions.create(wireBody));
    if (rest.stream) {
      return new StreamingResponse(
        response as AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk>,
        session,
        patchChatChunk,
        makeChatTailChunk,
      );
    }
    restoreChatResponse(response as OpenAIClass.Chat.Completions.ChatCompletion, session);
    return response;
  }
}

export class GheimChat {
  readonly completions: GheimChatCompletions;
  constructor(clientPromise: Promise<OpenAIClass>, detector: Detector) {
    this.completions = new GheimChatCompletions(clientPromise, detector);
  }
}
