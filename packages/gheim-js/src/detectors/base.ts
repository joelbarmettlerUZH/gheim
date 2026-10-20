/** Detector protocol — anything that maps text to PII spans. */

/**
 * Strip leading/trailing whitespace from a span range. The upstream
 * privacy-filter model tends to include leading whitespace in spans as a BPE
 * tokenization artifact; trimming fixes cosmetically-broken restored text.
 * Returns null if the trimmed range is empty.
 */
export function trimSpan(
  text: string,
  start: number,
  end: number,
): { start: number; end: number } | null {
  while (start < end && /\s/.test(text[start] ?? "")) start += 1;
  while (end > start && /\s/.test(text[end - 1] ?? "")) end -= 1;
  if (start >= end) return null;
  return { start, end };
}

export interface Span {
  /** e.g. "private_person", "private_email" */
  label: string;
  /** Inclusive character offset in the original string. */
  start: number;
  /** Exclusive. */
  end: number;
  /** Original surface form (text.slice(start, end)). */
  text: string;
  score?: number;
}

export interface Detector {
  detect(text: string, opts?: { model?: string }): Promise<Span[]>;
}

const WORD_CHAR = /[\p{L}\p{N}_]/u;
const isWordChar = (ch: string | undefined): boolean => ch !== undefined && WORD_CHAR.test(ch);

/**
 * Widen spans that start or end inside a word to the whole word, then merge
 * same-label spans that overlap or touch.
 *
 * A token classifier sometimes tags only part of a word (`gano` of `Lugano`,
 * `Zbind` + `en` of `Zbinden`), and a window boundary can cut a word in half.
 * Masking a fragment leaves the rest of the word readable and splits one entity
 * into two placeholders. A span is widened only when the characters on both
 * sides of its edge are word characters, so edges at spaces or punctuation
 * (`Gay` in `Gay-Crosier`) are left alone. The merged span's score is the
 * length-weighted mean of its parts.
 */
export function snapToWords(text: string, spans: Span[]): Span[] {
  const snapped: Span[] = spans.map((sp) => {
    let { start, end } = sp;
    while (start > 0 && isWordChar(text[start - 1]) && isWordChar(text[start])) start -= 1;
    while (end < text.length && isWordChar(text[end - 1]) && isWordChar(text[end])) end += 1;
    return start === sp.start && end === sp.end ? sp : { ...sp, start, end, text: text.slice(start, end) };
  });

  snapped.sort((a, b) => a.start - b.start || a.end - b.end);
  const merged: Span[] = [];
  const lastByLabel = new Map<string, number>();
  for (const sp of snapped) {
    const i = lastByLabel.get(sp.label);
    const prev = i === undefined ? undefined : merged[i];
    if (i !== undefined && prev !== undefined && sp.start <= prev.end) {
      const end = Math.max(prev.end, sp.end);
      const wPrev = prev.end - prev.start;
      const wSp = sp.end - sp.start;
      const score = ((prev.score ?? 1) * wPrev + (sp.score ?? 1) * wSp) / (wPrev + wSp);
      merged[i] = { ...prev, end, text: text.slice(prev.start, end), score };
    } else {
      lastByLabel.set(sp.label, merged.length);
      merged.push(sp);
    }
  }
  return merged.sort((a, b) => a.start - b.start || a.end - b.end);
}
