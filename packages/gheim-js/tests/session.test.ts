import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";
import type { Span } from "../src/detectors/base.ts";

const span = (label: string, start: number, end: number, text: string): Span => ({
  label,
  start,
  end,
  text,
});

describe("Session.allocate", () => {
  test("returns stable sentinel for same surface", () => {
    const s = new Session();
    expect(s.allocate("private_person", "Joel")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "Joel")).toBe("<PERSON_1>");
  });
  test("increments per-tag", () => {
    const s = new Session();
    expect(s.allocate("private_person", "Alice")).toBe("<PERSON_1>");
    expect(s.allocate("private_person", "Bob")).toBe("<PERSON_2>");
    expect(s.allocate("private_email", "a@b.com")).toBe("<EMAIL_1>");
    expect(s.allocate("private_person", "Carol")).toBe("<PERSON_3>");
  });
});

describe("Session.applySpans", () => {
  test("basic", () => {
    const s = new Session();
    const text = "Hi, my name is Joel and my email is joel@example.com.";
    const out = s.applySpans(text, [
      span("private_person", 15, 19, "Joel"),
      span("private_email", 36, 52, "joel@example.com"),
    ]);
    expect(out).toBe("Hi, my name is <PERSON_1> and my email is <EMAIL_1>.");
  });
  test("unsorted input", () => {
    const s = new Session();
    const out = s.applySpans("Joel and Alice", [
      span("private_person", 9, 14, "Alice"),
      span("private_person", 0, 4, "Joel"),
    ]);
    expect(out).toBe("<PERSON_1> and <PERSON_2>");
  });
  test("merges adjacent same-label spans", () => {
    const s = new Session();
    const out = s.applySpans("Email me at alice@example.com please.", [
      span("private_email", 12, 25, "alice@example"),
      span("private_email", 25, 29, ".com"),
    ]);
    expect(out).toBe("Email me at <EMAIL_1> please.");
    expect(s.mapping).toEqual({ "<EMAIL_1>": "alice@example.com" });
  });

  test("does not merge different labels", () => {
    const s = new Session();
    const out = s.applySpans("Alicejoel@example.com", [
      span("private_person", 0, 5, "Alice"),
      span("private_email", 5, 21, "joel@example.com"),
    ]);
    expect(out).toBe("<PERSON_1><EMAIL_1>");
  });

  test("skips overlap", () => {
    const s = new Session();
    const out = s.applySpans("abcdef", [
      span("private_person", 0, 4, "abcd"),
      span("private_person", 2, 6, "cdef"),
    ]);
    expect(out).toBe("<PERSON_1>ef");
  });
  test("coalesces repeated surface", () => {
    const s = new Session();
    const out = s.applySpans("Joel told Joel", [
      span("private_person", 0, 4, "Joel"),
      span("private_person", 10, 14, "Joel"),
    ]);
    expect(out).toBe("<PERSON_1> told <PERSON_1>");
  });
});

describe("Session.restore", () => {
  test("replaces known sentinels and passes through unknown", () => {
    const s = new Session();
    s.allocate("private_person", "Joel");
    expect(s.restore("Hello <PERSON_1>, code <SECRET_99>.")).toBe("Hello Joel, code <SECRET_99>.");
  });
});

describe("Session JSON round-trip", () => {
  test("preserves mapping and counters", () => {
    const s = new Session();
    s.allocate("private_person", "Joel");
    s.allocate("private_email", "a@b.com");
    s.allocate("private_person", "Alice");
    const r = Session.fromJSON(s.toJSON());
    expect(r.mapping).toEqual(s.mapping);
    expect(r.allocate("private_person", "Bob")).toBe("<PERSON_3>");
  });
});
