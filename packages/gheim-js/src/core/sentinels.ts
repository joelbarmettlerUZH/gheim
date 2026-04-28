/**
 * Sentinel format, allocation, and regex matching. Mirrors gheim-py.
 *
 * A sentinel is a stable, uniquely-numbered placeholder like `<PERSON_1>` or
 * `<EMAIL_2>`. ASCII-only angle brackets so downstream LLM tokenizers preserve them.
 */

export const LABEL_TO_TAG: Record<string, string> = {
  private_person: "PERSON",
  private_email: "EMAIL",
  private_phone: "PHONE",
  private_address: "ADDRESS",
  private_url: "URL",
  private_date: "DATE",
  account_number: "ACCOUNT",
  secret: "SECRET",
};

const TAG_RE = /[A-Z][A-Z0-9_]*/;
const INDEX_RE = /\d+/;

/** Matches any complete sentinel anywhere in a string. */
export const SENTINEL_RE = new RegExp(
  `<(?<tag>${TAG_RE.source})_(?<index>${INDEX_RE.source})>`,
  "g",
);

/** Same pattern, anchored — for testing a single token. */
export const SENTINEL_FULL_RE = new RegExp(
  `^<(?<tag>${TAG_RE.source})_(?<index>${INDEX_RE.source})>$`,
);

/** Anchored prefix-matcher used by the streaming deanonymizer. */
const PARTIAL_SENTINEL_RE = new RegExp(
  `^<(${TAG_RE.source})?(_(${INDEX_RE.source})?)?>?$`,
);

export function labelTag(label: string): string {
  return LABEL_TO_TAG[label] ?? label.toUpperCase();
}

export interface Sentinel {
  tag: string;
  index: number;
}

export function renderSentinel(s: Sentinel): string {
  return `<${s.tag}_${s.index}>`;
}

export function parseSentinel(s: string): Sentinel | null {
  const m = SENTINEL_FULL_RE.exec(s);
  if (!m || !m.groups) return null;
  return { tag: m.groups.tag!, index: Number.parseInt(m.groups.index!, 10) };
}

/**
 * Could `buffer` still become a sentinel if more characters arrive?
 * True iff buffer is a non-empty *strict* prefix of a valid sentinel.
 */
export function isPossibleSentinelPrefix(buffer: string): boolean {
  if (!buffer || buffer[0] !== "<") return false;
  if (buffer.endsWith(">")) return false; // already complete
  return PARTIAL_SENTINEL_RE.test(buffer);
}
