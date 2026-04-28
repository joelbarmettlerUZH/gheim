/** Streaming de-anonymizer. Mirrors gheim-py's StreamDeanonymizer. */
import { isPossibleSentinelPrefix } from "./sentinels.ts";

const SENTINEL_AT = /<([A-Z][A-Z0-9_]*)_(\d+)>/y;
const MAX_SENTINEL_LEN = 64;

export class StreamDeanonymizer {
  private hold = "";

  constructor(private readonly mapping: Record<string, string>) {}

  feed(chunk: string): string {
    if (!chunk) return "";
    const buffer = this.hold + chunk;
    const out: string[] = [];
    let i = 0;
    const n = buffer.length;

    while (i < n) {
      if (buffer[i] !== "<") {
        const lt = buffer.indexOf("<", i);
        if (lt === -1) {
          out.push(buffer.slice(i));
          i = n;
          break;
        }
        out.push(buffer.slice(i, lt));
        i = lt;
      }

      // Try a complete sentinel anchored at i.
      SENTINEL_AT.lastIndex = i;
      const m = SENTINEL_AT.exec(buffer);
      if (m && m.index === i) {
        const sentinel = m[0];
        out.push(this.mapping[sentinel] ?? sentinel);
        i = SENTINEL_AT.lastIndex;
        continue;
      }

      const tail = buffer.slice(i);
      if (tail.length < MAX_SENTINEL_LEN && isPossibleSentinelPrefix(tail)) {
        this.hold = tail;
        return out.join("");
      }

      // Not a sentinel — emit '<' and move on.
      out.push("<");
      i += 1;
    }

    this.hold = "";
    return out.join("");
  }

  flush(): string {
    const held = this.hold;
    this.hold = "";
    return held;
  }

  async *transform(chunks: AsyncIterable<string> | Iterable<string>): AsyncGenerator<string> {
    for await (const chunk of chunks as AsyncIterable<string>) {
      const out = this.feed(chunk);
      if (out) yield out;
    }
    const tail = this.flush();
    if (tail) yield tail;
  }
}
