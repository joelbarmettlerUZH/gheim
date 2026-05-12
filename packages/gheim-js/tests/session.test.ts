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

describe("Session duplicate-surface recovery", () => {
  test("redacts a name that the detector only flagged once", () => {
    // Simulates the demo-letter case: detector found "Lukas Brunner"
    // in "mein Name ist..." but missed the signature occurrence.
    const text =
      "Hi Team,\nmein Name ist Lukas Brunner.\n" +
      "Freundliche Grüsse,\nLukas Brunner";
    const sess = new Session();
    const detected: Span[] = [
      { start: 23, end: 36, label: "private_person",
        text: "Lukas Brunner", score: 0.99 },
    ];
    const out = sess.applySpans(text, detected);
    // Both occurrences should be replaced with the SAME sentinel.
    expect(out).toBe("Hi Team,\nmein Name ist <PERSON_1>.\n" +
                     "Freundliche Grüsse,\n<PERSON_1>");
  });

  test("does not over-match inside a longer word (Müller / Müllergasse)", () => {
    const text = "Hallo Müller, ich wohne in der Müllergasse.";
    const sess = new Session();
    const detected: Span[] = [
      { start: 6, end: 12, label: "private_person",
        text: "Müller", score: 0.99 },
    ];
    const out = sess.applySpans(text, detected);
    // Should redact "Müller" but NOT the "Müller" inside "Müllergasse".
    expect(out).toBe("Hallo <PERSON_1>, ich wohne in der Müllergasse.");
  });

  test("longest-first prevents short-name shadowing the longer one", () => {
    // "Joel" alone and "Joel Barmettler" both sentinels — the long one
    // must be replaced first so its substring "Joel" isn't pre-empted.
    const text =
      "Joel Barmettler ist hier, und Joel auch. Joel Barmettler signed.";
    const sess = new Session();
    const detected: Span[] = [
      // Detector caught both surfaces in one go.
      { start: 0, end: 15, label: "private_person",
        text: "Joel Barmettler", score: 0.99 },
      { start: 30, end: 34, label: "private_person",
        text: "Joel", score: 0.95 },
    ];
    const out = sess.applySpans(text, detected);
    // Expectation: PERSON_1 = "Joel Barmettler", PERSON_2 = "Joel".
    // "Joel Barmettler" later in the text should still go to PERSON_1
    // (not PERSON_2's prefix), and the standalone "Joel" stays PERSON_2.
    expect(out).toContain("<PERSON_1> ist hier");
    expect(out).toContain("und <PERSON_2> auch");
    expect(out).toContain("<PERSON_1> signed");
  });

  test("recovers email/phone duplicates too", () => {
    const text =
      "Reach me at lukas.brunner@example.ch (also lukas.brunner@example.ch later).";
    const sess = new Session();
    const detected: Span[] = [
      { start: 12, end: 36, label: "private_email",
        text: "lukas.brunner@example.ch", score: 0.99 },
    ];
    const out = sess.applySpans(text, detected);
    // Both occurrences should be redacted to <EMAIL_1>.
    const matches = out.match(/<EMAIL_1>/g) || [];
    expect(matches.length).toBe(2);
  });

  test("no-op when there are no duplicates", () => {
    const text = "Hi Joel.";
    const sess = new Session();
    const detected: Span[] = [
      { start: 3, end: 7, label: "private_person", text: "Joel", score: 0.99 },
    ];
    const out = sess.applySpans(text, detected);
    expect(out).toBe("Hi <PERSON_1>.");
  });

  test("merged span list contains the recovered duplicate occurrences", () => {
    // Regression test: prior to v0.1.6, recovery only patched the
    // redacted string. UI consumers (bars view) read the merged span
    // list and only saw the ONE detected span, so the second
    // occurrence rendered without a redaction overlay.
    const text =
      "Hi Team,\nmein Name ist Lukas Brunner.\n" +
      "Freundliche Grüsse,\nLukas Brunner";
    const sess = new Session();
    const detected: Span[] = [
      { start: 23, end: 36, label: "private_person",
        text: "Lukas Brunner", score: 0.99 },
    ];
    const { redacted, merged } = sess.applySpansDetailed(text, detected);
    expect(redacted).toContain("<PERSON_1>");
    expect(merged.length).toBe(2);
    expect(text.slice(merged[0]!.start, merged[0]!.end)).toBe("Lukas Brunner");
    expect(text.slice(merged[1]!.start, merged[1]!.end)).toBe("Lukas Brunner");
    expect(merged[0]!.label).toBe("private_person");
    expect(merged[1]!.label).toBe("private_person");
    // Sorted by start.
    expect(merged[0]!.start).toBeLessThan(merged[1]!.start);
  });

  test("recovered merged spans don't overlap detector spans", () => {
    // If the detector already found both occurrences, recovery should
    // be a no-op on the merged list (no double-counting).
    const text = "Hi Lukas Brunner. Best, Lukas Brunner.";
    const sess = new Session();
    const detected: Span[] = [
      { start: 3, end: 16, label: "private_person",
        text: "Lukas Brunner", score: 0.99 },
      { start: 24, end: 37, label: "private_person",
        text: "Lukas Brunner", score: 0.99 },
    ];
    const { merged } = sess.applySpansDetailed(text, detected);
    expect(merged.length).toBe(2);
  });
});
