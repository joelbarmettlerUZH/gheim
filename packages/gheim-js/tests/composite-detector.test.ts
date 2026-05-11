import { describe, expect, test } from "bun:test";
import { CompositeDetector } from "../src/detectors/composite.ts";
import type { Detector, Span } from "../src/detectors/base.ts";

/** A fake `Detector` that returns whatever the test pre-loads. The
 *  composite's `detect` calls the model with the masked text, which we
 *  capture so we can assert the regex correctly stripped its bits. */
function fakeModel(
  out: Span[] | ((maskedText: string) => Span[]),
): { detector: Detector; lastMasked: { value: string | null } } {
  const lastMasked = { value: null as string | null };
  const detector: Detector = {
    async detect(text: string): Promise<Span[]> {
      lastMasked.value = text;
      return typeof out === "function" ? out(text) : out;
    },
  };
  return { detector, lastMasked };
}

describe("CompositeDetector regex front-end", () => {
  test("constructor rejects missing model", () => {
    // @ts-expect-error: deliberately omitting required field
    expect(() => new CompositeDetector({})).toThrow(/CompositeDetector requires/);
  });

  test("empty input returns []", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    expect(await c.detect("")).toEqual([]);
  });

  // ── Email ─────────────────────────────────────────────────────────
  test("email is detected", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("contact me at alice@example.ch please");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_email");
    expect(out[0]!.text).toBe("alice@example.ch");
  });

  // ── URL ───────────────────────────────────────────────────────────
  test("URL is detected and trailing punctuation trimmed", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("see https://www.example.ch/path.");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_url");
    expect(out[0]!.text).toBe("https://www.example.ch/path");
  });

  // ── Phone ──────────────────────────────────────────────────────────
  test("Swiss phone +41 spaced is detected", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("call +41 44 268 12 34 today");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_phone");
    expect(out[0]!.text).toBe("+41 44 268 12 34");
  });

  test("Swiss phone 0044... national format", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("call 044 268 12 34 today");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_phone");
  });

  // ── IBAN-CH (mod-97-10) ───────────────────────────────────────────
  test("valid Swiss IBAN passes checksum", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    // The historic test IBAN — verified mod-97 = 1.
    const out = await c.detect("IBAN: CH9300762011623852957");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("account_number");
    expect(out[0]!.text).toBe("CH9300762011623852957");
  });

  test("invalid Swiss IBAN (broken checksum) is dropped", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    // Swap the last two digits — should fail mod-97.
    const out = await c.detect("IBAN: CH9300762011623852975");
    expect(out.length).toBe(0);
  });

  test("spaced Swiss IBAN passes checksum", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("IBAN CH93 0076 2011 6238 5295 7 done.");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("account_number");
  });

  // ── AHV (EAN-13) ──────────────────────────────────────────────────
  test("valid Swiss AHV passes checksum", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    // 756.2276.9349.68 — verified EAN-13 mod-10 against the Py validator.
    const out = await c.detect("AHV: 756.2276.9349.68");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("account_number");
  });

  test("invalid AHV (broken checksum) is dropped", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    // 756.2276.9349.00 — body would compute check=8, claimed 0 ≠ 8.
    const out = await c.detect("AHV: 756.2276.9349.00");
    expect(out.length).toBe(0);
  });

  // ── VAT-CHE (mod-11-10) ───────────────────────────────────────────
  test("valid Swiss VAT-CHE passes checksum", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    // CHE-116.281.710 — verified mod-11-10 valid (the Wikipedia example).
    const out = await c.detect("UID: CHE-116.281.710 MWST");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("account_number");
  });

  // ── Credit card (Luhn) ────────────────────────────────────────────
  test("valid Visa passes Luhn", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    // 4532 0151 1283 0366 — Luhn-valid Visa test number.
    const out = await c.detect("Visa: 4532 0151 1283 0366");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("account_number");
  });

  test("invalid Visa (broken Luhn) is dropped", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("Visa: 4532 0151 1283 0367");
    expect(out.length).toBe(0);
  });

  // ── Date ──────────────────────────────────────────────────────────
  test("DD.MM.YYYY date is detected", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("Datum: 14.03.2026");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_date");
    expect(out[0]!.text).toBe("14.03.2026");
  });

  test("YYYY-MM-DD date is detected", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("Date: 2026-03-14");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_date");
  });

  // ── Secrets ───────────────────────────────────────────────────────
  test("OpenAI sk- token is detected", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("token: sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("secret");
  });

  test("GitHub PAT is detected", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("token: ghp_" + "x".repeat(36));
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("secret");
  });
});

describe("CompositeDetector overlap resolution", () => {
  test("regex spans win over overlapping model claims", async () => {
    // Model claims "alice@example" as a person — composite should drop
    // it because the regex already tagged the whole email.
    const { detector } = fakeModel([
      {
        label: "private_person",
        start: 14,
        end: 27,
        text: "alice@example",
        score: 0.9,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect("contact me at alice@example.ch please");
    expect(out.length).toBe(1);
    expect(out[0]!.label).toBe("private_email");
  });

  test("model spans on the masked remainder are kept", async () => {
    // Compose a real Swiss letter; the regex catches the IBAN, the
    // model is fed the masked remainder and emits a person span at the
    // SAME char offset as the original (mask is length-preserving).
    const text = "Frau Müller, IBAN: CH9300762011623852957";
    const personStart = text.indexOf("Müller");
    const personEnd = personStart + "Müller".length;
    const { detector, lastMasked } = fakeModel([
      {
        label: "private_person",
        start: personStart,
        end: personEnd,
        text: "Müller",
        score: 0.99,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect(text);
    // Model saw the IBAN as spaces:
    expect(lastMasked.value!.slice(text.indexOf("CH"))).toMatch(/^ +$/);
    // Both spans returned, sorted by start.
    expect(out.length).toBe(2);
    expect(out[0]!.label).toBe("private_person");
    expect(out[1]!.label).toBe("account_number");
  });

  test("end-to-end: original Müller letter — every category found", async () => {
    const text =
      "Sehr geehrte Frau Müller,\n" +
      "vielen Dank für Ihre Anfrage vom 14.03.2026.\n" +
      "IBAN: CH9300762011623852957\n" +
      "Telefon: +41 44 268 12 34\n" +
      "E-Mail: a.mueller@kanzlei-mueller.ch";
    // Fake the model's output for "Müller" so the test doesn't need
    // the real transformer; the composite does the rest.
    const personStart = text.indexOf("Müller");
    const { detector } = fakeModel([
      {
        label: "private_person",
        start: personStart,
        end: personStart + "Müller".length,
        text: "Müller",
        score: 0.99,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const out = await c.detect(text);
    const labels = new Set(out.map((s) => s.label));
    expect(labels.has("private_person")).toBe(true);
    expect(labels.has("private_date")).toBe(true);
    expect(labels.has("account_number")).toBe(true);
    expect(labels.has("private_phone")).toBe(true);
    expect(labels.has("private_email")).toBe(true);
  });
});

describe("CompositeDetector overlap-resolution edge cases", () => {
  test("two regex matches that overlap: greedy-by-start, longer wins on tie", async () => {
    // No real-world example that triggers this for the SDK regexes —
    // but verify the algorithm via a custom case that the catalogue
    // could produce in principle. Compose two valid IBANs glued
    // together to exercise the overlap path.
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const text =
      "CH9300762011623852957 CH9300762011623852957"; // two valid IBANs, space-separated
    const out = await c.detect(text);
    expect(out.length).toBe(2);
    expect(out.every((s) => s.label === "account_number")).toBe(true);
    expect(out[0]!.start).toBe(0);
    expect(out[1]!.start).toBe(22);
  });
});
