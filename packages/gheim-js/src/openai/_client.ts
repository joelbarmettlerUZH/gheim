/** Client class — wraps the upstream openai client and exposes wrapped endpoints. */
import type OpenAIClass from "openai";
import type { Detector } from "../detectors/base.ts";
import { defaultDetector } from "../detectors/index.ts";
import { GheimAudio } from "./_audio.ts";
import { GheimChat } from "./_chat.ts";
import { GheimCompletions } from "./_completions.ts";
import { GheimResponses } from "./_responses.ts";
import {
  GheimEmbeddings,
  GheimImages,
  GheimModerations,
} from "./_anonymize_only.ts";
import { STRICT_REASONS, makeStrictProxy, makeWarningProxy } from "./_strict.ts";

/** Options for constructing a gheim-wrapped OpenAI client.
 *
 * gheim-specific keys (`gheimDetector`, `gheimStrict`, `openaiClient`,
 * `clientOptions`) configure the wrapper itself. Any other key is
 * forwarded to the underlying `new OpenAI(...)` constructor — so
 * `apiKey`, `baseURL`, `organization` etc. work at the top level, like
 * the openai SDK itself. This mirrors gheim-py's behavior, where
 * everything except `gheim_*` kwargs gets forwarded via `**kwargs`.
 *
 * If both top-level passthrough AND `clientOptions` carry the same key,
 * the top-level value wins.
 */
export interface GheimOpenAIOptions
  extends Omit<NonNullable<ConstructorParameters<typeof OpenAIClass>[0]>, never> {
  gheimDetector?: Detector;
  /** Strict mode: raise on access to unwrapped endpoints. Default true. */
  gheimStrict?: boolean;
  /** Pre-built openai client to wrap (useful for tests). */
  openaiClient?: OpenAIClass;
  /** Explicit shape for the inner `new OpenAI(...)` opts. Any key here is
   *  overridden by the same key passed at the top level. */
  clientOptions?: ConstructorParameters<typeof OpenAIClass>[0];
}

async function loadOpenAI(opts: GheimOpenAIOptions["clientOptions"]): Promise<OpenAIClass> {
  const mod = (await import("openai")) as unknown as {
    default?: typeof OpenAIClass;
    OpenAI?: typeof OpenAIClass;
  };
  const Ctor = mod.default ?? mod.OpenAI;
  if (!Ctor) throw new Error("openai SDK has no OpenAI export");
  return new Ctor(opts as never);
}

/** Build a strict-mode getter chain for a single attribute. */
function strictGuard(attr: string, inner: unknown, strict: boolean): unknown {
  const reason = STRICT_REASONS[attr];
  if (!reason) return inner; // Unknown attr — pass through.
  if (strict) return makeStrictProxy(attr, reason);
  if (inner === undefined || inner === null) return inner;
  return makeWarningProxy(attr, inner as object, reason);
}

export class OpenAI {
  readonly chat: GheimChat;
  readonly completions: GheimCompletions;
  readonly responses: GheimResponses;
  readonly embeddings: GheimEmbeddings;
  readonly moderations: GheimModerations;
  readonly audio: GheimAudio;
  readonly images: GheimImages;
  /** Underlying openai client; available after ``ready()`` resolves. */
  raw: OpenAIClass | undefined;
  /** Whether strict mode is on. */
  readonly gheimStrict: boolean;
  private readonly _clientPromise: Promise<OpenAIClass>;

  constructor(opts: GheimOpenAIOptions = {}) {
    const {
      gheimDetector,
      gheimStrict,
      openaiClient,
      clientOptions,
      ...passthrough
    } = opts;
    const detector = gheimDetector ?? defaultDetector();
    this.gheimStrict = gheimStrict ?? true;

    // Top-level kwargs win on conflict — mirrors how Python's
    // `super().__init__(**kwargs)` would behave.
    const innerOpts = {
      ...(clientOptions ?? {}),
      ...passthrough,
    } as ConstructorParameters<typeof OpenAIClass>[0];

    this._clientPromise = openaiClient
      ? Promise.resolve(openaiClient)
      : loadOpenAI(innerOpts);
    this._clientPromise.then((c) => {
      this.raw = c;
    });

    this.chat = new GheimChat(this._clientPromise, detector);
    this.completions = new GheimCompletions(this._clientPromise, detector);
    this.responses = new GheimResponses(this._clientPromise, detector);
    this.embeddings = new GheimEmbeddings(this._clientPromise, detector);
    this.moderations = new GheimModerations(this._clientPromise, detector);
    this.audio = new GheimAudio(this._clientPromise, detector);
    this.images = new GheimImages(this._clientPromise, detector);

    // Strict guards for unwrapped endpoints. Access via `client.beta` etc.
    return new Proxy(this, {
      get: (target, prop, receiver) => {
        if (typeof prop === "string" && prop in STRICT_REASONS) {
          // Try to fetch the inner attr lazily for warning mode.
          const innerAttr =
            target.raw && (target.raw as unknown as Record<string, unknown>)[prop];
          return strictGuard(prop, innerAttr, target.gheimStrict);
        }
        return Reflect.get(target, prop, receiver);
      },
    });
  }

  async ready(): Promise<OpenAIClass> {
    return this._clientPromise;
  }
}

export const AsyncOpenAI = OpenAI;
export type AsyncOpenAI = OpenAI;
