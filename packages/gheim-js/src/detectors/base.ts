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
