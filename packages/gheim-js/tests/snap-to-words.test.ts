import { describe, expect, test } from "bun:test";
import { snapToWords } from "../src/detectors/base.ts";
import type { Span } from "../src/detectors/base.ts";

function span(text: string, part: string, label = "private_address", score = 1.0): Span {
  const start = text.indexOf(part);
  return { label, start, end: start + part.length, text: part, score };
}
const texts = (spans: Span[]) => spans.map((s) => s.text);

describe("snapToWords", () => {
  test("a fragment at the end of a word is widened", () => {
    const t = "Treuhand, Lugano.";
    expect(texts(snapToWords(t, [span(t, "gano")]))).toEqual(["Lugano"]);
  });

  test("a fragment at the start of a word is widened", () => {
    const t = "Herr Tschudi kam.";
    expect(texts(snapToWords(t, [span(t, "Tsch", "private_person")]))).toEqual(["Tschudi"]);
  });

  test("both edges inside words are widened", () => {
    const t = "Herr Zimmermann kam.";
    expect(texts(snapToWords(t, [span(t, "mmerm", "private_person")]))).toEqual(["Zimmermann"]);
  });

  test("edges at spaces and punctuation are left alone", () => {
    const t = "Dr. Gay-Crosier, Bern.";
    expect(texts(snapToWords(t, [span(t, "Gay", "private_person")]))).toEqual(["Gay"]);
    expect(texts(snapToWords(t, [span(t, "Dr. Gay-Crosier", "private_person")]))).toEqual(["Dr. Gay-Crosier"]);
  });

  test("touching fragments of one word become one span with a weighted score", () => {
    const t = "Frau Käthi Zbinden kam.";
    const out = snapToWords(t, [
      span(t, "Frau Käthi Zbind", "private_person", 0.9),
      span(t, "en", "private_person", 0.5),
    ]);
    expect(texts(out)).toEqual(["Frau Käthi Zbinden"]);
    expect(out[0]!.score!).toBeGreaterThan(0.5);
    expect(out[0]!.score!).toBeLessThan(0.9);
  });

  test("spans separated by a space are not merged", () => {
    const t = "Anna Keller";
    const out = snapToWords(t, [span(t, "Anna", "private_person"), span(t, "Keller", "private_person")]);
    expect(texts(out)).toEqual(["Anna", "Keller"]);
  });

  test("different labels are never merged", () => {
    const t = "Meier-Bern";
    const out = snapToWords(t, [span(t, "Meier", "private_person"), span(t, "Bern", "private_address")]);
    expect(texts(out)).toEqual(["Meier", "Bern"]);
  });

  test("overlapping same-label spans merge", () => {
    const t = "Frau Käthi Zbinden";
    const out = snapToWords(t, [span(t, "Frau Käthi", "private_person"), span(t, "Käthi Zbinden", "private_person")]);
    expect(texts(out)).toEqual(["Frau Käthi Zbinden"]);
  });

  test("digits count as word characters", () => {
    const t = "Tel 0311234567 bitte";
    expect(texts(snapToWords(t, [span(t, "1234567", "private_phone")]))).toEqual(["0311234567"]);
  });

  test("empty input and untouched spans", () => {
    expect(snapToWords("", [])).toEqual([]);
    const t = "Anna kam.";
    const sp = span(t, "Anna", "private_person");
    expect(snapToWords(t, [sp])).toEqual([sp]);
  });
});
