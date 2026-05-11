/* Tiny copy of the gheim sentinel session for in-page round-trip demo.
 *
 * Mirrors gheim-py / gheim-js Session in two important ways:
 *   1. Same-label spans separated by punctuation or short whitespace are
 *      merged into one span before sentinel allocation. The model emits
 *      one prediction per BPE subword, so a single IBAN like
 *      "CH93 0076 2011 6238 5295 7" comes back as a dozen contiguous
 *      account_number spans — without merging, each becomes its own
 *      <ACCOUNT_n>. Same for emails (split at `.` and `@`), phones
 *      (split at spaces), dates ("März" + "2026").
 *   2. The same (label, surface) coalesces to a stable sentinel so an
 *      LLM that refers back to <PERSON_1> always means the same person.
 */
import type { DetectedSpan } from "../composables/useDetector";

export const LABEL_TO_TAG: Record<string, string> = {
  private_person: "PERSON",
  private_email: "EMAIL",
  private_phone: "PHONE",
  private_address: "ADDRESS",
  private_date: "DATE",
  private_url: "URL",
  account_number: "ACCOUNT",
  secret: "SECRET",
};

interface MergedSpan {
  start: number;
  end: number;
  label: string;
}

/** Merge same-label spans that touch or are separated by ≤ MERGE_GAP characters
 *  of either whitespace or punctuation (`.`, `@`, `-`, `,`, `:`, `/`, `+`).
 *  Models tokenise IBAN / email / phone surfaces into many same-label pieces
 *  joined by exactly these characters; merging recovers the original entity. */
function mergeSpans(text: string, spans: DetectedSpan[]): MergedSpan[] {
  const MERGE_GAP = 2;
  const sorted = [...spans].sort((a, b) => a.start - b.start || a.end - b.end);
  const out: MergedSpan[] = [];
  for (const s of sorted) {
    if (s.start < 0 || s.end > text.length || s.start >= s.end) continue;
    const last = out[out.length - 1];
    if (!last) {
      out.push({ start: s.start, end: s.end, label: s.label });
      continue;
    }
    if (s.start < last.end) continue; // strict overlap; first wins
    const gap = text.slice(last.end, s.start);
    const bridges =
      gap.length <= MERGE_GAP &&
      (gap === "" || /^[\s.@\-,:/+]+$/.test(gap));
    if (s.label === last.label && bridges) {
      last.end = s.end;
    } else {
      out.push({ start: s.start, end: s.end, label: s.label });
    }
  }
  return out;
}

export function substitute(text: string, spans: DetectedSpan[]) {
  /* Returns { redacted, mapping, merged } where mapping is sentinel → original
   * and merged is the post-merge span list (used by the demo to render the
   * Bars view consistently with the Sentinel view). */
  const merged = mergeSpans(text, spans);
  const counts: Record<string, number> = {};
  const seen = new Map<string, string>(); // tag::surface → sentinel
  const mapping: Record<string, string> = {};

  let out = "";
  let cursor = 0;
  for (const s of merged) {
    out += text.slice(cursor, s.start);
    const surface = text.slice(s.start, s.end);
    const tag = LABEL_TO_TAG[s.label] ?? s.label.toUpperCase();
    const key = `${tag}::${surface}`;
    let sentinel = seen.get(key);
    if (!sentinel) {
      counts[tag] = (counts[tag] ?? 0) + 1;
      sentinel = `<${tag}_${counts[tag]}>`;
      seen.set(key, sentinel);
      mapping[sentinel] = surface;
    }
    out += sentinel;
    cursor = s.end;
  }
  out += text.slice(cursor);
  return { redacted: out, mapping, merged };
}

export function restore(text: string, mapping: Record<string, string>) {
  let out = text;
  for (const [sentinel, original] of Object.entries(mapping)) {
    out = out.split(sentinel).join(original);
  }
  return out;
}
