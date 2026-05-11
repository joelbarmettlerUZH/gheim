/**
 * Variant-name dedup tests for Session.allocate. Mirrors gheim-py's
 * test_session_normalize.py.
 *
 * These would have caught the dedup gap that motivated the
 * NFKC + lowercase + whitespace-collapse normalization: "Joel" /
 * "JOEL" / "joel" previously got three different sentinels in the
 * same session.
 *
 * Note: JS's `toLowerCase()` is not full Unicode casefold — German
 * "ß" stays as-is (Python's casefold() expands to "ss"), so the
 * eszett-collapse case is intentionally absent here.
 */
import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";

describe("Session normalization", () => {
  test("casing variants collapse to one sentinel", () => {
    const s = new Session();
    expect(s.allocate("private_person", "Joel")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "JOEL")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "joel")).toBe("<PERSON_1>");
    expect(s.mapping).toEqual({ "<PERSON_1>": "Joel" });
  });

  test("whitespace variants collapse", () => {
    const s = new Session();
    const a = s.allocate("private_person", "Joel  Barmettler");
    const b = s.allocate("private_person", "Joel\tBarmettler");
    const c = s.allocate("private_person", "  Joel Barmettler  ");
    const d = s.allocate("private_person", "Joel\nBarmettler");
    expect(a).toBe("<PERSON_1>");
    expect(b).toBe("<PERSON_1>");
    expect(c).toBe("<PERSON_1>");
    expect(d).toBe("<PERSON_1>");
  });

  test("NFKC collapses decomposed unicode", () => {
    const s = new Session();
    const composed = "Müller";          // "Müller" with precomposed U+00FC
    const decomposed = "Müller";       // "Müller" as M + u + combining diaeresis
    expect(composed).not.toBe(decomposed);   // sanity: they are different bytes
    expect(s.allocate("private_person", composed)).toBe("<PERSON_1>");
    expect(s.allocate("private_person", decomposed)).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "MÜLLER")).toBe("<PERSON_1>");  // MÜLLER
  });

  test("NFKC collapses compatibility ligatures", () => {
    const s = new Session();
    expect(s.allocate("private_person", "Soﬁa")).toBe("<PERSON_1>");  // Soﬁa (FI ligature)
    expect(s.allocate("private_person", "Sofia")).toBe("<PERSON_1>");
  });

  test("email is case-insensitive", () => {
    const s = new Session();
    expect(s.allocate("private_email", "Alice@Example.com")).toBe("<EMAIL_1>");
    expect(s.allocate("private_email", "ALICE@EXAMPLE.COM")).toBe("<EMAIL_1>");
    expect(s.allocate("private_email", "alice@example.com")).toBe("<EMAIL_1>");
  });

  test("truly distinct surfaces get distinct sentinels", () => {
    const s = new Session();
    expect(s.allocate("private_person", "Joel")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "Alice")).toBe("<PERSON_2>");
    expect(s.allocate("private_person", "Bob")).toBe("<PERSON_3>");
  });

  test("normalization does not cross label boundaries", () => {
    const s = new Session();
    expect(s.allocate("private_person", "joel")).toBe("<PERSON_1>");
    expect(s.allocate("private_email", "joel")).toBe("<EMAIL_1>");
    expect(s.allocate("private_person", "JOEL")).toBe("<PERSON_1>");
    expect(s.allocate("private_email", "JOEL")).toBe("<EMAIL_1>");
  });

  test("round-trip preserves first-seen casing", () => {
    const s = new Session();
    s.allocate("private_person", "JOEL");
    s.allocate("private_person", "Joel");
    expect(s.restore("Hello <PERSON_1>!")).toBe("Hello JOEL!");
  });

  test("fromJSON preserves normalized dedup", () => {
    const s1 = new Session();
    s1.allocate("private_person", "Joel");
    const s2 = Session.fromJSON(s1.toJSON());
    expect(s2.allocate("private_person", "JOEL")).toBe("<PERSON_1>");
    expect(s2.allocate("private_person", "joel")).toBe("<PERSON_1>");
    expect(s2.mapping).toEqual({ "<PERSON_1>": "Joel" });
  });
});
