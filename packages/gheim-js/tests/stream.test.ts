import { describe, expect, test } from "bun:test";
import { StreamDeanonymizer } from "../src/core/stream.ts";

const MAPPING: Record<string, string> = {
  "<PERSON_1>": "Joel",
  "<EMAIL_1>": "joel@example.com",
  "<PERSON_2>": "Alice",
};

function drive(chunks: string[]): string {
  const d = new StreamDeanonymizer(MAPPING);
  let out = "";
  for (const c of chunks) out += d.feed(c);
  out += d.flush();
  return out;
}

describe("StreamDeanonymizer", () => {
  test("passthrough with no sentinels", () => {
    expect(drive(["Hello, world!"])).toBe("Hello, world!");
  });
  test("single sentinel in one chunk", () => {
    expect(drive(["Hi <PERSON_1>!"])).toBe("Hi Joel!");
  });
  test("split across two chunks", () => {
    expect(drive(["Hi <PER", "SON_1>!"])).toBe("Hi Joel!");
  });
  test("character-by-character split", () => {
    const text = "Hello <PERSON_1>, your email is <EMAIL_1>.";
    const expected = "Hello Joel, your email is joel@example.com.";
    expect(drive([...text])).toBe(expected);
  });
  test("incidental angle brackets", () => {
    expect(drive(["a < b and c < d"])).toBe("a < b and c < d");
    expect(drive(["<html><body>hi</body></html>"])).toBe("<html><body>hi</body></html>");
  });
  test("unknown sentinel passes through", () => {
    expect(drive(["Code <SECRET_99> here"])).toBe("Code <SECRET_99> here");
  });
  test("two sentinels back to back", () => {
    expect(drive(["<PERSON_1><PERSON_2>"])).toBe("JoelAlice");
  });
  test("aggressive single-split coverage", () => {
    const text = "<PERSON_1> wrote to <PERSON_2> at <EMAIL_1>.";
    const expected = "Joel wrote to Alice at joel@example.com.";
    for (let i = 0; i <= text.length; i++) {
      expect(drive([text.slice(0, i), text.slice(i)])).toBe(expected);
    }
  });
  test("three-way split coverage", () => {
    const text = "Hi <PERSON_1>!";
    const expected = "Hi Joel!";
    for (let i = 0; i <= text.length; i++) {
      for (let j = i; j <= text.length; j++) {
        expect(drive([text.slice(0, i), text.slice(i, j), text.slice(j)])).toBe(expected);
      }
    }
  });
  test("flush emits truncated prefix verbatim", () => {
    expect(drive(["truncated <PERSON_"])).toBe("truncated <PERSON_");
  });
  test("lowercase sentinel is not substituted", () => {
    expect(drive(["<person_1>"])).toBe("<person_1>");
  });
  test("markdown wrapping", () => {
    expect(drive(["**<PERSON_1>**"])).toBe("**Joel**");
  });
  test("max-length overflow emits verbatim", () => {
    const long = "<" + "A".repeat(80);
    expect(drive([long])).toBe(long);
  });
  test("repeated known sentinel", () => {
    expect(drive(["<PERSON_1> and <PERSON_1>"])).toBe("Joel and Joel");
  });
  test("empty chunks ignored", () => {
    expect(drive(["", "<PERSON_1>", "", "!"])).toBe("Joel!");
  });
});
