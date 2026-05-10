/* Tiny copy of the gheim sentinel session for in-page round-trip demo. */
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

export function substitute(text: string, spans: DetectedSpan[]) {
  /* Returns { redacted, mapping } where mapping is sentinel → original */
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const counts: Record<string, number> = {};
  const seen = new Map<string, string>(); // original → sentinel (stable per text)
  const mapping: Record<string, string> = {};

  let out = "";
  let cursor = 0;
  for (const s of sorted) {
    if (s.start < cursor) continue; // skip overlapping
    out += text.slice(cursor, s.start);
    const tag = LABEL_TO_TAG[s.label] ?? s.label.toUpperCase();
    const key = `${tag}::${s.text}`;
    let sentinel = seen.get(key);
    if (!sentinel) {
      counts[tag] = (counts[tag] ?? 0) + 1;
      sentinel = `<${tag}_${counts[tag]}>`;
      seen.set(key, sentinel);
      mapping[sentinel] = s.text;
    }
    out += sentinel;
    cursor = s.end;
  }
  out += text.slice(cursor);
  return { redacted: out, mapping };
}

export function restore(text: string, mapping: Record<string, string>) {
  let out = text;
  for (const [sentinel, original] of Object.entries(mapping)) {
    out = out.split(sentinel).join(original);
  }
  return out;
}
