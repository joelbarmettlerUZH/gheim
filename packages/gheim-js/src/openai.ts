/**
 * OpenAI SDK drop-in wrapper.
 *
 * Two ways to use:
 *
 * 1. **Drop-in client** — same constructor + same `chat.completions.create`,
 *    automatic round-trip:
 *
 *    ```ts
 *    import { OpenAI } from "gheim/openai";
 *    const client = new OpenAI();
 *    const r = await client.chat.completions.create({
 *      model: "gpt-4o",
 *      messages: [{ role: "user", content: "Hi, my name is Joel" }],
 *    });
 *    // r.choices[0].message.content contains "Joel".
 *    // OpenAI's servers only ever saw "<PERSON_1>".
 *    ```
 *
 *    Streaming works the same: `await client.chat.completions.create({ ..., stream: true })`
 *    returns an `AsyncIterable<ChatCompletionChunk>` whose deltas are de-anonymized.
 *
 *    Pass `gheimSession` / `gheimDetector` in the create call to override defaults.
 *    All other resources (embeddings, files, ...) are accessible via `client.raw`.
 *
 * 2. **Stateless helpers** — when you want full control over how the LLM is called:
 *    `anonymizeOpenAIMessages`, `deanonymizeOpenAIResponse`, `deanonymizeOpenAIStream`.
 *
 * `openai` is an *optional peer dependency*. The helpers and types compile
 * without it; the drop-in client requires it at instantiation time.
 */
import type OpenAIClass from "openai";

import { Session } from "./core/session.ts";
import { StreamDeanonymizer } from "./core/stream.ts";
import type { Detector } from "./detectors/base.ts";
import { defaultDetector } from "./detectors/index.ts";
import { anonymizeMessages } from "./plain.ts";

/* ──────────────────────────── Stateless helpers ──────────────────────────── */

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
  if (tail && last) yield makeTailChunk(last, tail);
}

function makeTailChunk(
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

/* ──────────────────────────── Drop-in client ──────────────────────────── */

/** Extra fields the wrapper accepts on top of openai's create body. */
export interface GheimCreateExtras {
  /** Reuse a session across calls (multi-turn coherent sentinels). */
  gheimSession?: Session;
  /** Per-call detector override. */
  gheimDetector?: Detector;
}

export interface GheimOpenAIOptions {
  /** Detector used for all calls unless overridden via `gheimDetector`. */
  gheimDetector?: Detector;
  /** Pre-built openai client to wrap. Useful for tests. If omitted, gheim
   *  constructs one with `clientOptions`. */
  openaiClient?: OpenAIClass;
  /** Forwarded to `new OpenAI(...)` when constructing the underlying client. */
  clientOptions?: ConstructorParameters<typeof OpenAIClass>[0];
}

type CreateParamsNS = OpenAIClass.Chat.Completions.ChatCompletionCreateParamsNonStreaming;
type CreateParamsS = OpenAIClass.Chat.Completions.ChatCompletionCreateParamsStreaming;
type CreateBody = (CreateParamsNS | CreateParamsS) & GheimCreateExtras;

class GheimChatCompletions {
  constructor(
    private readonly clientPromise: Promise<OpenAIClass>,
    private readonly defaultDet: Detector,
  ) {}

  async create(
    body: CreateParamsNS & GheimCreateExtras,
  ): Promise<OpenAIClass.Chat.Completions.ChatCompletion>;
  async create(
    body: CreateParamsS & GheimCreateExtras,
  ): Promise<AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk>>;
  async create(body: CreateBody): Promise<unknown> {
    const { gheimSession, gheimDetector, ...rest } = body;
    const detector = gheimDetector ?? this.defaultDet;
    const session = gheimSession ?? new Session();
    if (!(session as unknown as { detector?: Detector }).detector) {
      (session as unknown as { detector: Detector }).detector = detector;
    }

    const cleanMessages = await anonymizeMessages(
      rest.messages as OpenAIClass.Chat.Completions.ChatCompletionMessageParam[],
      session,
      { detector },
    );
    const wireBody = { ...rest, messages: cleanMessages };

    const client = await this.clientPromise;
    const response = await (
      client.chat.completions as unknown as { create(body: unknown): Promise<unknown> }
    ).create(wireBody);

    if (rest.stream) {
      return new GheimChunkStream(
        response as AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk> & {
          controller?: AbortController;
        },
        session,
      );
    }
    return deanonymizeOpenAIResponse(
      response as OpenAIClass.Chat.Completions.ChatCompletion,
      session,
    );
  }
}

class GheimChat {
  readonly completions: GheimChatCompletions;

  constructor(clientPromise: Promise<OpenAIClass>, detector: Detector) {
    this.completions = new GheimChatCompletions(clientPromise, detector);
  }
}

class GheimChunkStream
  implements AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk>
{
  readonly controller?: AbortController;

  constructor(
    private readonly inner: AsyncIterable<OpenAIClass.Chat.Completions.ChatCompletionChunk> & {
      controller?: AbortController;
    },
    private readonly session: Session,
  ) {
    this.controller = inner.controller;
  }

  [Symbol.asyncIterator](): AsyncIterator<OpenAIClass.Chat.Completions.ChatCompletionChunk> {
    return deanonymizeOpenAIStream(this.inner, this.session)[Symbol.asyncIterator]();
  }
}

/**
 * Drop-in OpenAI client. The constructor returns immediately; the underlying
 * `openai` SDK is loaded on first await. Use `await client.ready()` to force
 * loading early (or to surface the `openai`-not-installed error eagerly).
 */
export class OpenAI {
  readonly chat: GheimChat;
  /** Underlying `openai` client, available once `ready()` resolves. */
  raw: OpenAIClass | undefined;
  private readonly _clientPromise: Promise<OpenAIClass>;

  constructor(opts: GheimOpenAIOptions = {}) {
    const detector = opts.gheimDetector ?? defaultDetector();
    this._clientPromise = opts.openaiClient
      ? Promise.resolve(opts.openaiClient)
      : loadAndConstructOpenAI(opts.clientOptions);
    this._clientPromise.then((c) => {
      this.raw = c;
    });
    this.chat = new GheimChat(this._clientPromise, detector);
  }

  /** Resolves once the underlying openai client is constructed. */
  async ready(): Promise<OpenAIClass> {
    return this._clientPromise;
  }
}

/** Alias for parity with `gheim.openai.AsyncOpenAI` in Python. The official JS
 *  SDK's class is the same for both sync and async usage. */
export const AsyncOpenAI = OpenAI;
export type AsyncOpenAI = OpenAI;

async function loadAndConstructOpenAI(
  clientOpts: GheimOpenAIOptions["clientOptions"],
): Promise<OpenAIClass> {
  let mod: { default?: typeof OpenAIClass; OpenAI?: typeof OpenAIClass };
  try {
    mod = (await import("openai")) as unknown as typeof mod;
  } catch (e) {
    throw new Error(
      "gheim/openai's drop-in client requires the `openai` package. " +
        "Install with: npm install openai (or bun add openai).",
    );
  }
  const Ctor = mod.default ?? mod.OpenAI;
  if (!Ctor) throw new Error("Failed to load openai SDK: no OpenAI export found.");
  return new Ctor(clientOpts as never);
}
