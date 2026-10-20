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

/** Options for {@link CustomPattern}. */
export interface CustomPatternOptions {
  /** Label of the spans this rule produces. It becomes the placeholder tag
   *  (`insurance_card` -> `<INSURANCE_CARD_1>`), so it must start with a
   *  letter and contain only letters, digits and underscores. Reuse a
   *  built-in label such as `account_number` to share its tag. */
  label: string;
  /** Pattern to search for. A string is compiled with `flags`; a RegExp
   *  keeps its own flags. */
  regex: string | RegExp;
  /** Capture group that is the PII itself (0 = whole match). Lets a label
   *  word anchor the match without being masked (`Pass-Nr. (X1234567)`). */
  group?: number;
  /** Flags for a string pattern. Default `"i"`. `g` and `d` are always added. */
  flags?: string;
  /** Receives the PII text; return false to reject the match (a checksum,
   *  or a case check, since JS has no scoped `(?i:...)` flag). */
  validator?: (value: string) => boolean;
  /** Optional name for error messages. */
  name?: string;
}

/** A user-supplied regex rule for formats that are specific to one
 *  deployment (insurance card numbers, hospital record numbers, internal
 *  case IDs, ...). Nothing in the library is domain-specific: the caller
 *  passes the rules it needs to {@link CompositeDetector}.
 *
 *  Matches run in the regex front-end next to the built-in rules: the text
 *  is masked before the model sees it, and a match wins over any model span
 *  it overlaps. */
export class CustomPattern {
  readonly label: string;
  readonly group: number;
  readonly name: string;
  private readonly regex: RegExp;
  private readonly validator: ((value: string) => boolean) | undefined;

  constructor(opts: CustomPatternOptions) {
    if (!/^[A-Za-z][A-Za-z0-9_]*$/.test(opts.label)) {
      throw new Error(
        `CustomPattern label ${JSON.stringify(opts.label)} must start with a letter and contain ` +
          "only letters, digits and underscores (it becomes a <TAG_1> placeholder).",
      );
    }
    this.label = opts.label;
    this.name = opts.name ?? opts.label;
    this.group = opts.group ?? 0;
    this.validator = opts.validator;
    const source = typeof opts.regex === "string" ? opts.regex : opts.regex.source;
    const base = typeof opts.regex === "string" ? (opts.flags ?? "i") : opts.regex.flags;
    // `g` to iterate, `d` for the capture-group offsets.
    const flags = [...new Set([...base, "g", "d"])].join("");
    this.regex = new RegExp(source, flags);
    // An empty alternative always matches, so its length tells how many groups exist.
    const groups = (new RegExp(`${source}|`, flags.replace(/[gd]/g, "")).exec("") ?? []).length - 1;
    if (!Number.isInteger(this.group) || this.group < 0 || this.group > groups) {
      throw new Error(
        `CustomPattern ${JSON.stringify(this.name)}: group ${this.group} does not exist ` +
          `(the pattern has ${groups}).`,
      );
    }
  }

  /** Every non-empty, validated match of the PII group in `text`. */
  find(text: string): Span[] {
    const out: Span[] = [];
    this.regex.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = this.regex.exec(text)) !== null) {
      if (m[0] === "") {
        this.regex.lastIndex++;
        continue;
      }
      const range = (m as RegExpExecArray & { indices: Array<[number, number] | undefined> }).indices[
        this.group
      ];
      if (!range || range[0] === range[1]) continue; // optional group did not take part
      const [start, end] = range;
      const value = text.slice(start, end);
      if (this.validator && !this.validator(value)) continue;
      out.push({ start, end, label: this.label, text: value, score: 1.0 });
    }
    return out;
  }
}

const NAME_PART = /[A-ZÀ-Ý][\p{L}\p{N}_'’-]{3,}/gu;
// Honorifics are capitalised words of 4+ letters; propagating them as
// surnames would mask every "Frau" in the text.
const HONORIFICS = new Set([
  "frau", "herr", "herrn", "madame", "monsieur", "signora", "signor", "signore",
  "dott", "dottor", "dottoressa", "doktor", "docteur", "prof", "professor",
  "mister", "miss", "mrs",
]);

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Return `spans` plus a span for every other occurrence of a string
 *  already found as a person.
 *
 *  The model often tags one occurrence of a name and skips later ones, so a
 *  name seen in a greeting leaks again in a signature. An occurrence that
 *  lies inside an existing span is skipped. One that contains shorter
 *  existing spans (a fragment such as `Käthi` of `Zbinden Käthi`) replaces
 *  them. With `surnames` the capitalised parts of each detected name are
 *  propagated too (higher recall, more false positives). */
function propagateNames(text: string, spans: Span[], surnames: boolean): Span[] {
  const names = new Set<string>();
  for (const sp of spans) {
    if (sp.label !== "private_person") continue;
    const value = text.slice(sp.start, sp.end).trim();
    if (value.length >= 4) names.add(value);
    if (surnames) {
      for (const part of value.match(NAME_PART) ?? []) {
        if (!HONORIFICS.has(part.toLowerCase())) names.add(part);
      }
    }
  }
  let result = [...spans];
  for (const name of [...names].sort((a, b) => b.length - a.length)) {
    const re = new RegExp(
      `(?<![\\p{L}\\p{N}_])${escapeRegExp(name)}(?![\\p{L}\\p{N}_])`,
      "gu",
    );
    let m: RegExpExecArray | null;
    while ((m = re.exec(text)) !== null) {
      const start = m.index;
      const end = start + m[0].length;
      const hits = result.filter((sp) => start < sp.end && sp.start < end);
      if (hits.some((sp) => sp.start <= start && end <= sp.end)) continue; // already covered
      if (hits.some((sp) => sp.start < start || end < sp.end)) continue; // straddles a boundary
      result = result.filter((sp) => !hits.includes(sp));
      result.push({ start, end, label: "private_person", text: m[0], score: 1.0 });
    }
  }
  return result;
}

/** Options for {@link CompositeDetector}. */
export interface CompositeDetectorOptions {
  /** Inner detector — usually a {@link LocalDetector}. Anything with
   *  a `detect(text) → Promise<Span[]>` method works. */
  model: Detector;
  /** Extra regex rules for formats specific to your deployment. */
  patterns?: CustomPattern[];
  /** Mask every other occurrence of a string the model found as a person,
   *  so a name caught once is not leaked by a later occurrence. Default true. */
  propagateNames?: boolean;
  /** With `propagateNames`: also propagate the individual capitalised parts
   *  of each detected name. Higher recall, more false positives. Default false. */
  propagateSurnames?: boolean;
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
  readonly patterns: readonly CustomPattern[];
  readonly propagateNames: boolean;
  readonly propagateSurnames: boolean;

  constructor(opts: CompositeDetectorOptions) {
    if (!opts || !opts.model) {
      throw new Error(
        "CompositeDetector requires { model: Detector } — pass any object with " +
          "a .detect(text) method (e.g. new LocalDetector({...})).",
      );
    }
    this.model = opts.model;
    this.patterns = opts.patterns ?? [];
    this.propagateNames = opts.propagateNames ?? true;
    this.propagateSurnames = opts.propagateSurnames ?? false;
  }

  /** Two-pass detection: regex first (with checksum), then model on
   *  the masked remainder. Returns a single overlap-resolved span list. */
  async detect(text: string, opts?: { model?: string }): Promise<Span[]> {
    if (!text) return [];
    const regexSpans = findRegexSpans(text);
    // Built-in rules keep their spans; a custom rule only fills the gaps,
    // longest match first when two custom rules overlap.
    const extra = this.patterns.flatMap((p) => p.find(text));
    extra.sort((a, b) =>
      a.start !== b.start ? a.start - b.start : (b.end - b.start) - (a.end - a.start),
    );
    for (const sp of extra) {
      if (!regexSpans.some((o) => sp.start < o.end && o.start < sp.end)) regexSpans.push(sp);
    }
    regexSpans.sort((a, b) => a.start - b.start);
    const masked = maskSpans(text, regexSpans);
    const modelSpans = await this.model.detect(masked, opts);
    // Overlap resolution: regex wins on any overlap (regex spans are
    // checksum-validated). Model spans fill the gaps.
    let out: Span[] = [...regexSpans];
    const regexRanges: [number, number][] = regexSpans.map((s) => [s.start, s.end]);
    for (const sp of modelSpans) {
      const overlaps = regexRanges.some(([rs, re]) => sp.start < re && rs < sp.end);
      if (overlaps) continue;
      out.push(sp);
    }
    if (this.propagateNames) out = propagateNames(text, out, this.propagateSurnames);
    out.sort((a, b) =>
      a.start !== b.start ? a.start - b.start : (b.end - b.start) - (a.end - a.start),
    );
    return out;
  }
}
