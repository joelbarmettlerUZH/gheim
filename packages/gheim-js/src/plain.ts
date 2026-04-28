/** Framework-agnostic surface — works with any LLM client. */
import { StreamDeanonymizer } from "./core/stream.ts";
import type { Session } from "./core/session.ts";
import type { Detector } from "./detectors/base.ts";

/**
 * Loose chat-message shape — compatible with OpenAI / Anthropic / Google / etc.
 *
 * For an OpenAI-typed surface, use `gheim/openai` which constrains messages to
 * `OpenAI.Chat.Completions.ChatCompletionMessageParam`.
 */
export interface ChatMessage {
  role: string;
  content: string | Array<{ type: string; [k: string]: unknown }> | null;
  [k: string]: unknown;
}

function resolve(session: Session, detector?: Detector): Detector {
  const det = detector ?? (session as unknown as { detector?: Detector }).detector;
  if (!det) {
    throw new Error("No detector — pass one to anonymizeText or attach to the session.");
  }
  return det;
}

export async function anonymizeText(
  text: string,
  session: Session,
  opts?: { detector?: Detector; model?: string },
): Promise<string> {
  const det = resolve(session, opts?.detector);
  const spans = await det.detect(text, { model: opts?.model });
  return session.applySpans(text, spans);
}

export function deanonymizeText(text: string, session: Session): string {
  return session.restore(text);
}

/**
 * Anonymize the ``content`` field of each message. The caller's exact message
 * type is preserved in the return type — works equally well with the loose
 * ``ChatMessage`` shape and with strict OpenAI / Anthropic / Google types.
 */
export async function anonymizeMessages<M extends { content?: unknown }>(
  messages: M[],
  session: Session,
  opts?: { detector?: Detector; model?: string },
): Promise<M[]> {
  const det = resolve(session, opts?.detector);
  const out: M[] = [];
  for (const msg of messages) {
    const content = (msg as { content?: unknown }).content;
    if (typeof content === "string") {
      const redacted = await anonymizeText(content, session, { detector: det, model: opts?.model });
      out.push({ ...msg, content: redacted } as M);
    } else if (Array.isArray(content)) {
      const parts = await Promise.all(
        content.map(async (p: unknown) => {
          const part = p as { type?: unknown; text?: unknown };
          if (part && part.type === "text" && typeof part.text === "string") {
            const redacted = await anonymizeText(part.text, session, {
              detector: det,
              model: opts?.model,
            });
            return { ...part, text: redacted };
          }
          return p;
        }),
      );
      out.push({ ...msg, content: parts } as M);
    } else {
      out.push({ ...msg });
    }
  }
  return out;
}

export async function* deanonymizeStream(
  chunks: AsyncIterable<string> | Iterable<string>,
  session: Session,
): AsyncGenerator<string> {
  const buf = new StreamDeanonymizer(session.mapping);
  yield* buf.transform(chunks);
}
