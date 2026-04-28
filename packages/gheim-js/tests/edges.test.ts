/**
 * Correctness edge cases (Batch B) — unicode, overflow, empty, repeated surface,
 * cross-language Session JSON round-trip.
 *
 * Mirrors test_edges.py.
 */
import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";
import { StreamDeanonymizer } from "../src/core/stream.ts";
import { anonymizeText, deanonymizeText } from "../src/plain.ts";
import type { Detector, Span } from "../src/detectors/base.ts";

const MAX_SENTINEL_LEN = 64; // mirrors core/stream.ts

class NoopDetector implements Detector {
  async detect(): Promise<Span[]> {
    return [];
  }
}

class SurfaceDetector implements Detector {
  constructor(private surfaces: Record<string, string>) {}
  async detect(text: string): Promise<Span[]> {
    const out: Span[] = [];
    for (const [surface, label] of Object.entries(this.surfaces)) {
      let i = 0;
      while (true) {
        const idx = text.indexOf(surface, i);
        if (idx === -1) break;
        out.push({ label, start: idx, end: idx + surface.length, text: surface });
        i = idx + surface.length;
      }
    }
    return out;
  }
}

function withDetector(d: Detector): Session {
  const s = new Session();
  (s as unknown as { detector: Detector }).detector = d;
  return s;
}

// ---------- B.1 multibyte unicode in stream ----------

describe("Batch B: stream + unicode", () => {
  test("CJK around sentinel reassembles correctly", () => {
    const buf = new StreamDeanonymizer({ "<PERSON_1>": "Joel" });
    let out = "";
    for (const c of ["你好 ", "<PER", "SON_1>", " 안녕"]) out += buf.feed(c);
    out += buf.flush();
    expect(out).toBe("你好 Joel 안녕");
  });

  test("emoji around sentinel reassembles correctly", () => {
    const buf = new StreamDeanonymizer({ "<PERSON_1>": "Joel" });
    let out = "";
    for (const c of ["🚀 hello ", "<PERSON_1>", " 🎉"]) out += buf.feed(c);
    out += buf.flush();
    expect(out).toBe("🚀 hello Joel 🎉");
  });

  test("unicode in restored surface", () => {
    const buf = new StreamDeanonymizer({ "<PERSON_1>": "André" });
    let out = "";
    for (const c of ["Bonjour ", "<PER", "SON_1>", " — ravi"]) out += buf.feed(c);
    out += buf.flush();
    expect(out).toBe("Bonjour André — ravi");
  });

  test("emoji that uses surrogate pair adjacent to sentinel", () => {
    // 👋 (U+1F44B) is a surrogate pair in UTF-16. JS string indexing uses code units.
    const buf = new StreamDeanonymizer({ "<PERSON_1>": "Joel" });
    let out = "";
    for (const c of ["👋 ", "<", "PERSON_1>", "!"]) out += buf.feed(c);
    out += buf.flush();
    expect(out).toBe("👋 Joel!");
  });

  test("unicode round-trip via Session.applySpans", () => {
    const text = "Hello 你好 André at andré@exämple.com.";
    const personStart = text.indexOf("André");
    const emailStart = text.indexOf("andré@exämple.com");
    const sess = new Session();
    const redacted = sess.applySpans(text, [
      { label: "private_person", start: personStart, end: personStart + "André".length, text: "André" },
      { label: "private_email", start: emailStart, end: emailStart + "andré@exämple.com".length, text: "andré@exämple.com" },
    ]);
    expect(redacted).toBe("Hello 你好 <PERSON_1> at <EMAIL_1>.");
    expect(sess.restore(redacted)).toBe(text);
  });
});

// ---------- B.2 overflow ----------

describe("Batch B: overflow / pathological '<' patterns", () => {
  test("'<' followed by long block of A's emits verbatim", () => {
    const junk = "<" + "A".repeat(MAX_SENTINEL_LEN + 16);
    const buf = new StreamDeanonymizer({ "<PERSON_1>": "Joel" });
    let out = buf.feed(junk);
    out += buf.flush();
    expect(out).toBe(junk);
  });

  test("'<' + huge block + '>' + real sentinel after", () => {
    const pre = "<" + "A".repeat(MAX_SENTINEL_LEN + 4) + ">";
    const text = pre + " then <PERSON_1>";
    const buf = new StreamDeanonymizer({ "<PERSON_1>": "Joel" });
    let out = "";
    for (const c of [...text]) out += buf.feed(c);
    out += buf.flush();
    expect(out).toBe(pre + " then Joel");
  });

  test("many consecutive '<' do not stall", () => {
    const buf = new StreamDeanonymizer({});
    let out = buf.feed("<<<<<<<<");
    out += buf.flush();
    expect(out).toBe("<<<<<<<<");
  });
});

// ---------- B.3 empty / whitespace / lone '<' ----------

describe("Batch B: empty + whitespace inputs", () => {
  test("anonymize empty text", async () => {
    const sess = withDetector(new NoopDetector());
    expect(await anonymizeText("", sess)).toBe("");
    expect(sess.mapping).toEqual({});
  });

  test("anonymize whitespace-only text", async () => {
    const sess = withDetector(new NoopDetector());
    expect(await anonymizeText("   \n\t  ", sess)).toBe("   \n\t  ");
  });

  test("deanonymize empty text", () => {
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    expect(deanonymizeText("", sess)).toBe("");
  });

  test("lone '<' across chunks", () => {
    const buf = new StreamDeanonymizer({});
    let out = "";
    for (const c of ["<", " ", "x"]) out += buf.feed(c);
    out += buf.flush();
    expect(out).toBe("< x");
  });

  test("lone '<' at end of stream is flushed verbatim", () => {
    const buf = new StreamDeanonymizer({});
    let out = buf.feed("text <");
    out += buf.flush();
    expect(out).toBe("text <");
  });
});

// ---------- B.4 repeated surface across calls ----------

describe("Batch B: repeated-surface coalescing across calls", () => {
  test("repeated surface across anonymize calls keeps same sentinel", async () => {
    const det = new SurfaceDetector({ Joel: "private_person" });
    const sess = withDetector(det);
    const a = await anonymizeText("Joel was here.", sess);
    const b = await anonymizeText("Then Joel left.", sess);
    const c = await anonymizeText("Joel is back.", sess);
    expect(a).toBe("<PERSON_1> was here.");
    expect(b).toBe("Then <PERSON_1> left.");
    expect(c).toBe("<PERSON_1> is back.");
    expect(sess.mapping).toEqual({ "<PERSON_1>": "Joel" });
  });

  test("distinct labels for the same-looking but different surfaces get distinct sentinels", async () => {
    const det = new SurfaceDetector({ Joel: "private_person", "joel@x.ch": "private_email" });
    const sess = withDetector(det);
    const out = await anonymizeText("Joel at joel@x.ch wrote", sess);
    expect(out).toBe("<PERSON_1> at <EMAIL_1> wrote");
  });
});

// ---------- B.5 cross-language Session JSON round-trip ----------

describe("Batch B: cross-language Session JSON", () => {
  test("toJSON shape is just { mapping } — same as Python", () => {
    const sess = new Session();
    sess.allocate("private_person", "Joel");
    sess.allocate("private_email", "joel@example.com");
    sess.allocate("private_person", "Alice");
    const raw = sess.toJSON();
    expect(Object.keys(raw)).toEqual(["mapping"]);
    expect(raw.mapping).toEqual({
      "<PERSON_1>": "Joel",
      "<EMAIL_1>": "joel@example.com",
      "<PERSON_2>": "Alice",
    });
  });

  test("fromJSON accepts a payload Python would produce", () => {
    const foreign = {
      mapping: {
        "<PERSON_1>": "Joel",
        "<EMAIL_1>": "joel@example.com",
      },
    };
    const sess = Session.fromJSON(foreign);
    expect(sess.mapping).toEqual(foreign.mapping);
    // Counters survive — next Alice gets <PERSON_2>, repeated Joel coalesces.
    expect(sess.allocate("private_person", "Alice")).toBe("<PERSON_2>");
    expect(sess.allocate("private_person", "Joel")).toBe("<PERSON_1>");
  });

  test("JSON round-trip with unicode surfaces", () => {
    const sess = new Session();
    sess.allocate("private_person", "André");
    sess.allocate("private_email", "anna@bürgerportal.ch");
    const rebuilt = Session.fromJSON(sess.toJSON());
    expect(rebuilt.mapping).toEqual(sess.mapping);
  });
});
