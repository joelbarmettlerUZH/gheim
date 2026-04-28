/** Shared helpers for all endpoint proxies (JS). */
import { Session } from "../core/session.ts";
import { StreamDeanonymizer } from "../core/stream.ts";
import type { Detector } from "../detectors/base.ts";

export interface CallExtras {
  gheimSession?: Session;
  gheimDetector?: Detector;
}

export function resolveCallSession(
  body: Record<string, unknown> & CallExtras,
  defaultDet: Detector,
): { session: Session; detector: Detector; rest: Record<string, unknown> } {
  const { gheimSession, gheimDetector, ...rest } = body;
  const detector = gheimDetector ?? defaultDet;
  const session = gheimSession ?? new Session();
  if (!(session as unknown as { detector?: Detector }).detector) {
    (session as unknown as { detector: Detector }).detector = detector;
  }
  return { session, detector, rest };
}

/**
 * Anonymize a single value: strings get redacted, arrays of strings recurse,
 * arrays of pure ints (pre-tokenized inputs) pass through, anything else
 * passes through.
 */
export async function anonymizeValue(
  value: unknown,
  session: Session,
  detector: Detector,
): Promise<unknown> {
  if (typeof value === "string") {
    if (!value) return value;
    const spans = await detector.detect(value);
    return session.applySpans(value, spans);
  }
  if (Array.isArray(value)) {
    if (value.every((x) => typeof x === "number" && Number.isInteger(x))) {
      return value;
    }
    return Promise.all(value.map((v) => anonymizeValue(v, session, detector)));
  }
  return value;
}

export function restoreString(value: unknown, session: Session): unknown {
  if (typeof value === "string") return session.restore(value);
  return value;
}

/** Streaming wrapper — wraps an AsyncIterable, calls mutate per chunk, emits tail. */
export class StreamingResponse<C> implements AsyncIterable<C> {
  readonly buf: StreamDeanonymizer;

  constructor(
    private readonly inner: AsyncIterable<C>,
    private readonly session: Session,
    private readonly mutate: (chunk: C, buf: StreamDeanonymizer) => void,
    private readonly makeTail?: (template: C, tail: string) => C,
  ) {
    this.buf = new StreamDeanonymizer(session.mapping);
  }

  async *[Symbol.asyncIterator](): AsyncIterator<C> {
    let last: C | undefined;
    for await (const chunk of this.inner) {
      last = chunk;
      this.mutate(chunk, this.buf);
      yield chunk;
    }
    const tail = this.buf.flush();
    if (tail && last !== undefined && this.makeTail) {
      yield this.makeTail(last, tail);
    }
  }
}
