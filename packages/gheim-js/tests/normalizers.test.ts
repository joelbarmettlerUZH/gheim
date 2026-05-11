/**
 * Tests for opt-in surface-form normalizers. Mirrors gheim-py's
 * test_normalizers.py.
 *
 * Covers the cases that NFKC+lowercase deliberately doesn't catch
 * (format-level variants of phone numbers and dates) and verifies the
 * fallback behaviour when a normalizer can't parse its input.
 */
import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";
import { e164, germanTransliteration, isoDate, resolveNormalizer } from "../src/normalizers.ts";

// ---------------------------------------------------------------------------
// e164: phone-number canonicalization
// ---------------------------------------------------------------------------

describe("e164 normalizer", () => {
  test("collapses Swiss phone format variants", () => {
    const s = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    const a = s.allocate("private_phone", "+41 44 268 12 34");
    const b = s.allocate("private_phone", "0041 44 268 12 34");
    const c = s.allocate("private_phone", "044 268 12 34");
    const d = s.allocate("private_phone", "(044) 268-12-34");
    expect(a).toBe("<PHONE_1>");
    expect(b).toBe("<PHONE_1>");
    expect(c).toBe("<PHONE_1>");
    expect(d).toBe("<PHONE_1>");
  });

  test("does not collapse truly different numbers", () => {
    const s = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    const a = s.allocate("private_phone", "+41 44 268 12 34");
    const b = s.allocate("private_phone", "+41 44 268 12 35");
    expect(a).not.toBe(b);
  });

  test("falls back to default normalizer on garbage input", () => {
    const s = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    const a = s.allocate("private_phone", "call me later");
    const b = s.allocate("private_phone", "Call Me Later");
    expect(a).toBe("<PHONE_1>");
    expect(b).toBe("<PHONE_1>");  // collapses via NFKC+lowercase fallback
    const c = s.allocate("private_phone", "different garbage");
    expect(c).toBe("<PHONE_2>");
  });

  test("local-form unparseable without region", () => {
    const sNoRegion = new Session({ normalizers: { private_phone: e164() } });
    const a = sNoRegion.allocate("private_phone", "044 268 12 34");
    const b = sNoRegion.allocate("private_phone", "+41 44 268 12 34");
    expect(a).not.toBe(b);
  });
});

// ---------------------------------------------------------------------------
// isoDate: date canonicalization
// ---------------------------------------------------------------------------

describe("isoDate normalizer", () => {
  test("collapses format variants", () => {
    const s = new Session({ normalizers: { private_date: isoDate() } });
    const a = s.allocate("private_date", "1990-01-02");
    const b = s.allocate("private_date", "January 2 1990");
    const c = s.allocate("private_date", "Jan 2, 1990");
    expect(a).toBe("<DATE_1>");
    expect(b).toBe("<DATE_1>");
    expect(c).toBe("<DATE_1>");
  });

  test("distinguishes different days", () => {
    const s = new Session({ normalizers: { private_date: isoDate() } });
    const a = s.allocate("private_date", "2024-01-02");
    const b = s.allocate("private_date", "2024-01-03");
    expect(a).not.toBe(b);
  });

  test("falls back on unparseable input", () => {
    const s = new Session({ normalizers: { private_date: isoDate() } });
    const a = s.allocate("private_date", "not a date at all");
    const b = s.allocate("private_date", "NOT A DATE AT ALL");
    expect(a).toBe("<DATE_1>");
    expect(b).toBe("<DATE_1>");
  });
});

// ---------------------------------------------------------------------------
// germanTransliteration: ü/ö/ä/ß → ue/oe/ae/ss
// ---------------------------------------------------------------------------

describe("germanTransliteration normalizer", () => {
  test("collapses umlaut variants of names", () => {
    const s = new Session({ normalizers: { private_person: germanTransliteration() } });
    expect(s.allocate("private_person", "Müller")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "Mueller")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "MÜLLER")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "MUELLER")).toBe("<PERSON_1>");
  });

  test("collapses eszett and umlauts in addresses", () => {
    const s = new Session({ normalizers: { private_address: germanTransliteration() } });
    const a = s.allocate("private_address", "Bahnhofstraße 1, 8001 Zürich");
    const b = s.allocate("private_address", "Bahnhofstrasse 1, 8001 Zuerich");
    const c = s.allocate("private_address", "BAHNHOFSTRASSE 1, 8001 ZÜRICH");
    expect(a).toBe("<ADDRESS_1>");
    expect(b).toBe("<ADDRESS_1>");
    expect(c).toBe("<ADDRESS_1>");
  });

  test("does not merge different names", () => {
    const s = new Session({ normalizers: { private_person: germanTransliteration() } });
    expect(s.allocate("private_person", "Müller")).not.toBe(s.allocate("private_person", "Schmidt"));
  });

  test("via resolveNormalizer string shortcut", () => {
    const fn = resolveNormalizer("german");
    expect(fn("Müller")).toBe("mueller");
    expect(fn("Müller")).toBe(fn("Mueller"));
  });
});

// ---------------------------------------------------------------------------
// Composition
// ---------------------------------------------------------------------------

describe("normalizers + Session", () => {
  test("only apply to their configured label", () => {
    const s = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    s.allocate("private_phone", "+41 44 268 12 34");
    s.allocate("private_phone", "0041 44 268 12 34");
    expect(Object.keys(s.mapping).filter((k) => k.startsWith("<PHONE_"))).toHaveLength(1);
    s.allocate("private_person", "Joel");
    s.allocate("private_person", "JOEL");
    expect(Object.keys(s.mapping).filter((k) => k.startsWith("<PERSON_"))).toHaveLength(1);
  });

  test("user-supplied callable normalizer", () => {
    const upcase = (s: string): string => s.toUpperCase();
    const s = new Session({ normalizers: { private_person: upcase } });
    s.allocate("private_person", "Joel");
    s.allocate("private_person", "joel");
    s.allocate("private_person", "JOEL");
    expect(Object.keys(s.mapping).filter((k) => k.startsWith("<PERSON_"))).toHaveLength(1);
    expect(s.mapping["<PERSON_1>"]).toBe("Joel");
  });

  test("resolveNormalizer string shortcut", () => {
    const fn = resolveNormalizer("e164");
    expect(typeof fn).toBe("function");
    expect(fn("+41 44 268 12 34")).toBe("+41442681234");
    expect(fn("044 268 12 34")).toBeNull();  // no region default
  });

  test("resolveNormalizer unknown name throws", () => {
    expect(() => resolveNormalizer("does_not_exist")).toThrow(/unknown built-in normalizer/);
  });
});

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

describe("Session.fromJSON with normalizers", () => {
  test("continues to dedup phone format variants on reload", () => {
    const s1 = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    s1.allocate("private_phone", "+41 44 268 12 34");
    const raw = s1.toJSON();
    const s2 = Session.fromJSON(raw, { normalizers: { private_phone: e164({ region: "CH" }) } });
    expect(s2.allocate("private_phone", "044 268 12 34")).toBe("<PHONE_1>");
    expect(s2.allocate("private_phone", "0041 44 268 12 34")).toBe("<PHONE_1>");
  });

  test("without normalizers, falls back to default", () => {
    const s1 = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    s1.allocate("private_phone", "+41 44 268 12 34");
    const s2 = Session.fromJSON(s1.toJSON());
    expect(s2.mapping).toEqual({ "<PHONE_1>": "+41 44 268 12 34" });
    expect(s2.allocate("private_phone", "044 268 12 34")).toBe("<PHONE_2>");
  });
});

describe("Restoration", () => {
  test("returns first-seen phone surface", () => {
    const s = new Session({ normalizers: { private_phone: e164({ region: "CH" }) } });
    s.allocate("private_phone", "+41 44 268 12 34");
    s.allocate("private_phone", "044 268 12 34");
    expect(s.restore("Call <PHONE_1> please.")).toBe("Call +41 44 268 12 34 please.");
  });
});
