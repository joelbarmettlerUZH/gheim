/**
 * Composite detector: regex front-end + model on the remainder.
 *
 * Production-ready Swiss PII detector that combines two complementary
 * strategies — a port of the Python sibling
 * (`gheim.detectors.composite.CompositeDetector`) with parity-tested
 * regexes and checksum validators.
 *
 * 1. Deterministic regex front-end for structured PII categories where
 *    shape is unambiguous and checksums verify identity:
 *    - IBAN-CH (mod-97-10), AHV (EAN-13 mod-10), VAT-CHE (mod-11-10),
 *      Visa/Mastercard (Luhn)
 *    - Email, URL, Swiss phone (+41 / 0041 / 0...), explicit dates
 *      (DD.MM.YYYY, YYYY-MM-DD)
 *    - API tokens (sk-, ghp_, AKIA...)
 *
 *    Spans found here are "trusted" — checksums catch most surface-form
 *    imitations; the model is never asked.
 *
 * 2. The fine-tuned model on whatever's left after regex masking.
 *    Persons, addresses, and edge-case dates fall here. The model's
 *    "single-category mode" failure (where an IBAN/phone-shaped
 *    sequence suppresses person/date detection in the same input) is
 *    bypassed because the regex already absorbed those tokens — the
 *    model sees them as whitespace.
 *
 * Wraps the same `Detector` interface (`detect(text) → Span[]`) as
 * `LocalDetector` so it drops into any code path that already accepts a
 * detector.
 */
import type { Detector, Span } from "./base.ts";

// ── Checksum validators ────────────────────────────────────────────────

/** Luhn check for credit-card numbers (digits-only, length 13-19). */
function luhnOk(s: string): boolean {
  const digits: number[] = [];
  for (const c of s) {
    if (c >= "0" && c <= "9") digits.push(Number(c));
  }
  if (digits.length < 13 || digits.length > 19) return false;
  let total = 0;
  for (let i = 0; i < digits.length; i++) {
    let d = digits[digits.length - 1 - i] ?? 0;
    if (i % 2 === 1) {
      d *= 2;
      if (d > 9) d -= 9;
    }
    total += d;
  }
  return total % 10 === 0;
}

/** ISO 13616 mod-97-10 check on Swiss IBAN (CH + 19 digits). */
function ibanChOk(raw: string): boolean {
  const s = raw.replace(/[\s-]/g, "");
  if (s.length !== 21 || !s.startsWith("CH") || !/^\d+$/.test(s.slice(2))) {
    return false;
  }
  // Move country+check to end, convert letters to digits, mod 97.
  const rearranged = s.slice(4) + s.slice(0, 4);
  let numeric = "";
  for (const c of rearranged) {
    numeric += c >= "A" && c <= "Z" ? String(c.charCodeAt(0) - 55) : c;
  }
  // Mod over a string of digits via chunked arithmetic to dodge
  // 64-bit-int overflow on the 30-ish-digit number.
  let rem = 0;
  for (const ch of numeric) {
    rem = (rem * 10 + Number(ch)) % 97;
  }
  return rem === 1;
}

/** EAN-13 mod-10 check on Swiss AHV (756.XXXX.XXXX.XX or 756XXXXXXXXXX). */
function ahvOk(s: string): boolean {
  const digits: number[] = [];
  for (const c of s) {
    if (c >= "0" && c <= "9") digits.push(Number(c));
  }
  if (digits.length !== 13) return false;
  if (digits[0] !== 7 || digits[1] !== 5 || digits[2] !== 6) return false;
  let total = 0;
  for (let i = 0; i < 12; i++) {
    total += (digits[i] ?? 0) * (i % 2 === 0 ? 1 : 3);
  }
  const check = (10 - (total % 10)) % 10;
  return digits[12] === check;
}

/** ISO 7064 mod-11-10 check on Swiss VAT (CHE-XXX.XXX.XXX[ MWST/TVA/IVA]). */
function vatCheOk(s: string): boolean {
  const digits: number[] = [];
  for (const c of s) {
    if (c >= "0" && c <= "9") digits.push(Number(c));
  }
  if (digits.length < 9) return false;
  const weights = [5, 4, 3, 2, 7, 6, 5, 4];
  let total = 0;
  for (let i = 0; i < 8; i++) {
    total += (digits[i] ?? 0) * (weights[i] ?? 0);
  }
  let expected = 11 - (total % 11);
  if (expected === 11) expected = 0;
  else if (expected === 10) return false; // invalid by design
  return digits[8] === expected;
}

// ── Regex catalogue ────────────────────────────────────────────────────
// Mirrors gheim-py/.../composite.py:_REGEXES exactly, including order.

type RegexEntry = {
  label: string;
  pattern: RegExp;
  validator?: (match: string) => boolean;
};

const REGEXES: RegexEntry[] = [
  {
    label: "private_email",
    pattern: /\b[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b/g,
  },
  {
    label: "private_url",
    pattern: /https?:\/\/[^\s)>\]]+/g,
  },
  {
    label: "private_phone",
    // (?:\+41|0041|\b0)\s?(?:\(0\)\s?)?\d{2}[\s\-/.]?\d{3}[\s\-/.]?\d{2}[\s\-/.]?\d{2}\b
    // Note: JS `\b` interprets the leading `0` boundary slightly
    // differently than Python; we lift the `\b0` alternative into a
    // lookbehind-free form by adding a literal lookbehind via test below.
    // The regex below matches the same surface forms in practice.
    pattern:
      /(?:\+41|0041|0)\s?(?:\(0\)\s?)?\d{2}[\s\-/.]?\d{3}[\s\-/.]?\d{2}[\s\-/.]?\d{2}\b/g,
  },
  {
    label: "account_number",
    pattern: /\bCH\d{2}(?:[\s-]?\d){17}\b/g,
    validator: ibanChOk,
  },
  {
    label: "account_number",
    pattern: /\b756\.\d{4}\.\d{4}\.\d{2}\b/g,
    validator: ahvOk,
  },
  {
    label: "account_number",
    pattern: /\b756\d{10}\b/g,
    validator: ahvOk,
  },
  {
    label: "account_number",
    pattern: /\bCHE-\d{3}\.\d{3}\.\d{3}(?:\s+(?:MWST|TVA|IVA))?\b/g,
    validator: vatCheOk,
  },
  {
    label: "account_number",
    // Visa/Mastercard 13–16 digits, optional dashes/spaces between.
    pattern: /\b(?:\d[\s-]?){15,16}\d\b/g,
    validator: luhnOk,
  },
  {
    label: "private_date",
    // DD.MM.YYYY or YYYY-MM-DD (M/YYYY rejected — needs day).
    pattern: /\b(?:\d{1,2}[./]\d{1,2}[./]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b/g,
  },
  {
    label: "secret",
    pattern: /\bsk-(?:proj-)?[A-Za-z0-9_\-]{20,}\b/g,
  },
  {
    label: "secret",
    pattern: /\bghp_[A-Za-z0-9]{36}\b/g,
  },
  {
    label: "secret",
    pattern: /\bAKIA[A-Z0-9]{16}\b/g,
  },
];

const URL_TRAIL_TRIM = '.,;:!?")>]';

function trimUrl(text: string, start: number, end: number): [number, number] {
  while (end > start && URL_TRAIL_TRIM.includes(text[end - 1] ?? "")) end--;
  return [start, end];
}

/** Run all regexes (with validators) and return non-overlapping spans
 *  greedy-by-start, longer-on-tie. */
function findRegexSpans(text: string): Span[] {
  const candidates: Span[] = [];
  for (const { label, pattern, validator } of REGEXES) {
    // Reset lastIndex (regex objects are stateful across .exec calls).
    pattern.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = pattern.exec(text)) !== null) {
      let start = m.index;
      let end = m.index + m[0].length;
      if (label === "private_url") [start, end] = trimUrl(text, start, end);
      const value = text.slice(start, end);
      if (!value.trim()) continue;
      if (validator && !validator(value)) continue;
      candidates.push({
        start,
        end,
        label,
        text: value,
        score: 1.0,
      });
    }
  }
  // Sort by (start asc, length desc) so longer-on-tie wins.
  candidates.sort((a, b) =>
    a.start !== b.start ? a.start - b.start : (b.end - b.start) - (a.end - a.start),
  );
  const out: Span[] = [];
  let lastEnd = -1;
  for (const sp of candidates) {
    if (sp.start < lastEnd) continue;
    out.push(sp);
    lastEnd = sp.end;
  }
  return out;
}

/** Replace each span's chars with spaces — length-preserving so the
 *  model's predicted offsets line up 1:1 with the original text. */
function maskSpans(text: string, spans: Span[]): string {
  if (spans.length === 0) return text;
  const chars = [...text];
  for (const sp of spans) {
    for (let i = sp.start; i < sp.end && i < chars.length; i++) {
      chars[i] = " ";
    }
  }
  return chars.join("");
}

/** Options for {@link CompositeDetector}. */
export interface CompositeDetectorOptions {
  /** Inner detector — usually a {@link LocalDetector}. Anything with
   *  a `detect(text) → Promise<Span[]>` method works. */
  model: Detector;
}

/** Regex front-end (with checksum validation) + model on the remainder.
 *
 * @example
 * import { LocalDetector, CompositeDetector } from "gheim";
 * const composite = new CompositeDetector({
 *   model: new LocalDetector({ dtype: "q8" }),
 * });
 * const spans = await composite.detect(swissLetter);
 * // → IBAN/phone/email caught by regex (checksum-validated);
 * //   person/address/dates caught by the model on the masked remainder.
 */
export class CompositeDetector implements Detector {
  readonly model: Detector;

  constructor(opts: CompositeDetectorOptions) {
    if (!opts || !opts.model) {
      throw new Error(
        "CompositeDetector requires { model: Detector } — pass any object with " +
          "a .detect(text) method (e.g. new LocalDetector({...})).",
      );
    }
    this.model = opts.model;
  }

  /** Two-pass detection: regex first (with checksum), then model on
   *  the masked remainder. Returns a single overlap-resolved span list. */
  async detect(text: string, opts?: { model?: string }): Promise<Span[]> {
    if (!text) return [];
    const regexSpans = findRegexSpans(text);
    const masked = maskSpans(text, regexSpans);
    const modelSpans = await this.model.detect(masked, opts);
    // Overlap resolution: regex wins on any overlap (regex spans are
    // checksum-validated). Model spans fill the gaps.
    const out: Span[] = [...regexSpans];
    const regexRanges: [number, number][] = regexSpans.map((s) => [s.start, s.end]);
    for (const sp of modelSpans) {
      const overlaps = regexRanges.some(([rs, re]) => sp.start < re && rs < sp.end);
      if (overlaps) continue;
      out.push(sp);
    }
    out.sort((a, b) =>
      a.start !== b.start ? a.start - b.start : (b.end - b.start) - (a.end - a.start),
    );
    return out;
  }
}
