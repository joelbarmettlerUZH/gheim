import { describe, expect, test } from "bun:test";
import {
  isPossibleSentinelPrefix,
  labelTag,
  parseSentinel,
  renderSentinel,
} from "../src/core/sentinels.ts";

describe("labelTag", () => {
  test("known labels", () => {
    expect(labelTag("private_person")).toBe("PERSON");
    expect(labelTag("private_email")).toBe("EMAIL");
    expect(labelTag("account_number")).toBe("ACCOUNT");
  });
  test("unknown falls back to upper-cased label", () => {
    expect(labelTag("custom_thing")).toBe("CUSTOM_THING");
  });
});

describe("renderSentinel / parseSentinel", () => {
  test("round-trip", () => {
    const s = { tag: "PERSON", index: 7 };
    expect(renderSentinel(s)).toBe("<PERSON_7>");
    expect(parseSentinel("<PERSON_7>")).toEqual(s);
  });
  test("rejects non-sentinels", () => {
    expect(parseSentinel("PERSON_7")).toBeNull();
    expect(parseSentinel("<PERSON>")).toBeNull();
    expect(parseSentinel("<person_7>")).toBeNull();
    expect(parseSentinel("<PERSON_>")).toBeNull();
    expect(parseSentinel("<PERSON_7> extra")).toBeNull();
  });
});

describe("isPossibleSentinelPrefix", () => {
  test("positive cases", () => {
    for (const s of ["<", "<P", "<PE", "<PERSON", "<PERSON_", "<PERSON_1", "<PERSON_12"]) {
      expect(isPossibleSentinelPrefix(s)).toBe(true);
    }
  });
  test("negative cases", () => {
    expect(isPossibleSentinelPrefix("<PERSON_1>")).toBe(false); // already complete
    expect(isPossibleSentinelPrefix("<person")).toBe(false);
    expect(isPossibleSentinelPrefix("<PERSON ")).toBe(false);
    expect(isPossibleSentinelPrefix("PERSON_1>")).toBe(false);
    expect(isPossibleSentinelPrefix("")).toBe(false);
    expect(isPossibleSentinelPrefix("<PERSON_1>x")).toBe(false);
  });
});
