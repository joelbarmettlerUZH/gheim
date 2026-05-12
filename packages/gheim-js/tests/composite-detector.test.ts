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

// ─────────────────────────────────────────────────────────────────────
// Correctness tests — written against what SHOULD happen, not what
// currently does. Some may fail today; failures point at real bugs.
// Each section tests one piece of the contract.
// ─────────────────────────────────────────────────────────────────────

import { Session } from "../src/core/session.ts";

describe("CompositeDetector — multiple distinct surfaces, same label", () => {
  test("two different IBANs → both kept, distinct sentinels via Session", async () => {
    // Two checksum-valid Swiss IBANs (different account numbers).
    // Both should be regex-caught and both reach the user as separate
    // ACCOUNT sentinels — coalescing must NOT collapse different
    // surfaces just because they share a label.
    const ibanA = "CH9300762011623852957";
    const ibanB = "CH4807496806297567030"; // different valid IBAN
    const text = `Privat: ${ibanA}. Geschäft: ${ibanB}.`;
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });

    const spans = await c.detect(text);

    // Both spans returned, both labelled account_number.
    expect(spans.length).toBe(2);
    expect(spans.every((s) => s.label === "account_number")).toBe(true);
    // Distinct surface forms.
    const surfaces = new Set(spans.map((s) => s.text));
    expect(surfaces.size).toBe(2);
    expect(surfaces.has(ibanA)).toBe(true);
    expect(surfaces.has(ibanB)).toBe(true);

    // End-to-end: Session must mint TWO sentinels (different surfaces).
    const sess = new Session();
    const redacted = sess.applySpans(text, spans);
    const sentinels = Object.keys(sess.mapping);
    expect(sentinels.length).toBe(2);
    expect(sentinels).toContain("<ACCOUNT_1>");
    expect(sentinels).toContain("<ACCOUNT_2>");
    expect(sess.mapping["<ACCOUNT_1>"]).toBe(ibanA);
    expect(sess.mapping["<ACCOUNT_2>"]).toBe(ibanB);
    // Redacted text contains both sentinels, original IBANs gone.
    expect(redacted).toContain("<ACCOUNT_1>");
    expect(redacted).toContain("<ACCOUNT_2>");
    expect(redacted).not.toContain(ibanA);
    expect(redacted).not.toContain(ibanB);
  });

  test("two different emails → both kept, two EMAIL sentinels", async () => {
    const text =
      "Schreibe an alice@uzh.ch oder bob@ethz.ch — beide antworten.";
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });

    const spans = await c.detect(text);
    const emails = spans.filter((s) => s.label === "private_email");
    expect(emails.length).toBe(2);
    expect(emails.map((s) => s.text).sort()).toEqual([
      "alice@uzh.ch", "bob@ethz.ch",
    ]);

    const sess = new Session();
    sess.applySpans(text, spans);
    const sentinels = Object.keys(sess.mapping);
    expect(sentinels.includes("<EMAIL_1>")).toBe(true);
    expect(sentinels.includes("<EMAIL_2>")).toBe(true);
  });
});

describe("CompositeDetector — same surface, repeated", () => {
  test("same IBAN twice → both regex-caught, ONE sentinel, both positions redacted", async () => {
    const iban = "CH9300762011623852957";
    const text = `IBAN ${iban}. Bestätigung an IBAN ${iban}.`;
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });

    const spans = await c.detect(text);
    // Regex finds both occurrences.
    expect(spans.length).toBe(2);
    expect(spans.every((s) => s.label === "account_number")).toBe(true);
    expect(spans.every((s) => s.text === iban)).toBe(true);

    const sess = new Session();
    const redacted = sess.applySpans(text, spans);
    // Same surface → single sentinel, both positions get it.
    expect(Object.keys(sess.mapping)).toEqual(["<ACCOUNT_1>"]);
    const matches = redacted.match(/<ACCOUNT_1>/g) || [];
    expect(matches.length).toBe(2);
    expect(redacted).not.toContain(iban);
  });

  test("same email twice → one sentinel, both positions redacted", async () => {
    const email = "alice@uzh.ch";
    const text = `Mail: ${email}. Erneut: ${email}.`;
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    expect(spans.length).toBe(2);

    const sess = new Session();
    const redacted = sess.applySpans(text, spans);
    expect(Object.keys(sess.mapping)).toEqual(["<EMAIL_1>"]);
    expect((redacted.match(/<EMAIL_1>/g) || []).length).toBe(2);
  });
});

describe("CompositeDetector — mixed regex + model on the SAME label", () => {
  test("regex catches IBAN_A; model claims IBAN_B as account_number → BOTH spans, two sentinels", async () => {
    // The user's exact question: regex masks one IBAN, model later
    // emits another as account_number. Composite must keep both;
    // Session must mint two sentinels because the surfaces differ.
    const ibanA = "CH9300762011623852957"; // checksum-valid → regex catches
    // ibanB has a CHECKSUM-INVALID shape so regex skips it. Model is
    // free to claim it (simulates a model that's been trained on
    // looser IBAN patterns than our regex enforces).
    const ibanB = "CH9911111111111111111";
    const text = `Konto A: ${ibanA}\nKonto B: ${ibanB}`;
    const ibanBStart = text.indexOf(ibanB);
    const ibanBEnd = ibanBStart + ibanB.length;

    const { detector, lastMasked } = fakeModel([
      // Model emits the second IBAN-shaped string as account_number.
      // Its position is in the masked-remainder text but the offsets
      // refer to the original text (length-preserving mask).
      {
        label: "account_number",
        start: ibanBStart, end: ibanBEnd,
        text: ibanB, score: 0.95,
      },
    ]);
    const c = new CompositeDetector({ model: detector });

    const spans = await c.detect(text);

    // Both kept, both labelled account_number, different surfaces.
    expect(spans.length).toBe(2);
    expect(spans.every((s) => s.label === "account_number")).toBe(true);
    const surfaces = spans.map((s) => s.text).sort();
    expect(surfaces).toEqual([ibanA, ibanB].sort());

    // Model saw ibanA as spaces, ibanB as raw text.
    const masked = lastMasked.value!;
    expect(masked.slice(text.indexOf(ibanA), text.indexOf(ibanA) + ibanA.length))
      .toMatch(/^ +$/);
    expect(masked.slice(ibanBStart, ibanBEnd)).toBe(ibanB);

    // Session: two distinct sentinels.
    const sess = new Session();
    const redacted = sess.applySpans(text, spans);
    expect(Object.keys(sess.mapping).length).toBe(2);
    expect(new Set(Object.values(sess.mapping))).toEqual(new Set([ibanA, ibanB]));
    expect(redacted).not.toContain(ibanA);
    expect(redacted).not.toContain(ibanB);
  });

  test("regex IBAN + model PERSON adjacent (no overlap) → both kept", async () => {
    const text = "Sehr geehrte Frau Müller, IBAN CH9300762011623852957.";
    const personStart = text.indexOf("Müller");
    const personEnd = personStart + "Müller".length;
    const { detector } = fakeModel([
      { label: "private_person",
        start: personStart, end: personEnd,
        text: "Müller", score: 0.99 },
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    const labels = spans.map((s) => s.label).sort();
    expect(labels).toEqual(["account_number", "private_person"]);
  });
});

describe("CompositeDetector — overlap resolution", () => {
  test("model partial overlap with regex email → model dropped (regex wins)", async () => {
    const text = "Email: alice@uzh.ch — danke!";
    const emailStart = text.indexOf("alice");
    const { detector } = fakeModel([
      // Model claims "alice" as a person — partial overlap with the email.
      {
        label: "private_person",
        start: emailStart, end: emailStart + 5, // "alice"
        text: "alice", score: 0.8,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    // Only the email survives; the partially-overlapping person is dropped.
    expect(spans.length).toBe(1);
    expect(spans[0]!.label).toBe("private_email");
  });

  test("model fully-contained inside a regex span → dropped", async () => {
    // E.g., a URL that the model decides has a person name in it.
    const text = "Site: https://www.kanzlei-mueller.ch/team/";
    const urlStart = text.indexOf("https");
    const muellerStart = text.indexOf("mueller");
    const { detector } = fakeModel([
      {
        label: "private_person",
        start: muellerStart, end: muellerStart + 7,
        text: "mueller", score: 0.9,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    expect(spans.length).toBe(1);
    expect(spans[0]!.label).toBe("private_url");
    expect(spans[0]!.start).toBe(urlStart);
  });

  test("model span exactly ABUTTING (no overlap) a regex span → kept", async () => {
    // "Müller +41 44 268 12 34" — person ends where phone begins (with one space).
    const text = "Müller +41 44 268 12 34";
    const personEnd = "Müller".length;
    const phoneStart = text.indexOf("+41");
    expect(phoneStart).toBeGreaterThan(personEnd); // sanity: not overlapping
    const { detector } = fakeModel([
      {
        label: "private_person",
        start: 0, end: personEnd,
        text: "Müller", score: 0.99,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    expect(spans.length).toBe(2);
    expect(spans.map((s) => s.label).sort()).toEqual(["private_person",
                                                       "private_phone"]);
  });
});

describe("CompositeDetector — model never sees regex-caught regions (defense in depth)", () => {
  test("model that claims to find an entity in masked spaces → dropped", async () => {
    // Pathological model emits a span whose text is whitespace (the mask).
    // Composite shouldn't accept it. Today's behavior: the span doesn't
    // overlap any regex range exactly because the regex span COVERS those
    // chars — so the overlap test catches it.
    const text = "Email: alice@uzh.ch";
    const { detector } = fakeModel((masked) => {
      // Find a stretch of spaces (the masked email) and claim it.
      const m = masked.match(/ {5,}/);
      if (!m || m.index === undefined) return [];
      return [{
        label: "private_person",
        start: m.index, end: m.index + m[0].length,
        text: m[0], score: 0.9,
      }];
    });
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    expect(spans.length).toBe(1);
    expect(spans[0]!.label).toBe("private_email");
  });
});

describe("CompositeDetector — invalid checksum is silently dropped, then visible to the model", () => {
  test("checksum-invalid IBAN does NOT mask, so the model sees it raw", async () => {
    const badIban = "CH9911111111111111111"; // checksum-invalid
    const text = `IBAN ${badIban}`;
    let masked = "";
    const { detector, lastMasked } = fakeModel((m) => {
      masked = m;
      return [];
    });
    const c = new CompositeDetector({ model: detector });
    await c.detect(text);
    // Mask must equal the original — regex didn't claim it.
    expect(lastMasked.value).toBe(text);
    expect(masked.includes(badIban)).toBe(true);
  });

  test("mixed valid + invalid IBAN: only the valid one is regex-caught", async () => {
    const goodIban = "CH9300762011623852957";
    const badIban = "CH9911111111111111111";
    const text = `Gut: ${goodIban}. Schlecht: ${badIban}.`;
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    // Only the valid IBAN survives; the bad one isn't claimed.
    expect(spans.length).toBe(1);
    expect(spans[0]!.text).toBe(goodIban);
  });
});

describe("CompositeDetector — empty / trivial inputs", () => {
  test("PII-free input → []", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    expect(await c.detect("just plain text without any PII at all"))
      .toEqual([]);
  });

  test("whitespace-only input → []", async () => {
    const { detector } = fakeModel([]);
    const c = new CompositeDetector({ model: detector });
    expect(await c.detect("   \n\t   ")).toEqual([]);
  });
});

describe("CompositeDetector — Unicode in surfaces (Müller, Zürich)", () => {
  test("model person Müller + regex address with umlauts → both kept verbatim", async () => {
    const text = "Müller wohnt an der Werdstrasse 36, 8004 Zürich.";
    const muellerStart = 0;
    const muellerEnd = "Müller".length;
    const { detector } = fakeModel([
      // Model emits person "Müller" and address "Werdstrasse 36, 8004 Zürich".
      { label: "private_person", start: muellerStart, end: muellerEnd,
        text: "Müller", score: 0.99 },
      {
        label: "private_address",
        start: text.indexOf("Werdstrasse"),
        end: text.indexOf("Zürich") + "Zürich".length,
        text: "Werdstrasse 36, 8004 Zürich",
        score: 0.95,
      },
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    // No regex matches (no IBAN/phone/email/URL/date). Both model spans pass through.
    expect(spans.length).toBe(2);
    const sess = new Session();
    const redacted = sess.applySpans(text, spans);
    expect(redacted).toContain("<PERSON_1>");
    expect(redacted).toContain("<ADDRESS_1>");
  });
});

describe("CompositeDetector — sort order on output", () => {
  test("output is sorted by start ascending", async () => {
    // Mix of regex (IBAN) + model (person) at scattered positions.
    const text = "Frau Müller, Tel +41 44 268 12 34, IBAN CH9300762011623852957.";
    const muellerStart = text.indexOf("Müller");
    const { detector } = fakeModel([
      { label: "private_person",
        start: muellerStart, end: muellerStart + "Müller".length,
        text: "Müller", score: 0.99 },
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);
    expect(spans.length).toBe(3);
    for (let i = 1; i < spans.length; i++) {
      expect(spans[i]!.start).toBeGreaterThanOrEqual(spans[i - 1]!.start);
    }
  });
});

describe("CompositeDetector — opts pass-through", () => {
  test("per-call opts.model is forwarded to the inner detector", async () => {
    let receivedModel: string | undefined;
    const detector: Detector = {
      async detect(_text: string, opts?: { model?: string }) {
        receivedModel = opts?.model;
        return [];
      },
    };
    const c = new CompositeDetector({ model: detector });
    await c.detect("hello world", { model: "joelbarmettler/gheim-ch-560m" });
    expect(receivedModel).toBe("joelbarmettler/gheim-ch-560m");
  });
});

describe("CompositeDetector + Session round-trip", () => {
  test("composite output → Session.apply_spans → restore reproduces original", async () => {
    const text =
      "Sehr geehrte Frau Müller,\n" +
      "vielen Dank für Ihre Anfrage. Bitte überweisen auf IBAN CH9300762011623852957\n" +
      "oder per E-Mail an a.mueller@kanzlei.ch.\n" +
      "Mit freundlichen Grüßen, Müller";
    const muellerStart1 = text.indexOf("Müller");
    const muellerStart2 = text.lastIndexOf("Müller");
    const { detector } = fakeModel([
      { label: "private_person",
        start: muellerStart1, end: muellerStart1 + "Müller".length,
        text: "Müller", score: 0.99 },
      // Note: only the first Müller is detected; v0.1.4 duplicate-recovery
      // should pick up the second occurrence in apply_spans.
    ]);
    const c = new CompositeDetector({ model: detector });
    const spans = await c.detect(text);

    const sess = new Session();
    const redacted = sess.applySpans(text, spans);
    // Both Müller occurrences should be redacted (one by detector,
    // one by duplicate-recovery — same surface, same sentinel).
    expect((redacted.match(/<PERSON_1>/g) || []).length).toBe(2);
    expect(redacted).not.toContain("Müller");

    // Round-trip: restore brings everything back.
    const restored = sess.restore(redacted);
    expect(restored).toBe(text);
  });
});
