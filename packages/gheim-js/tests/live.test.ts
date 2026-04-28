/**
 * Real-model end-to-end tests for LocalDetector + plain surface.
 *
 * These download `openai/privacy-filter` from HuggingFace on first run and run
 * inference via @huggingface/transformers. Skipped unless GHEIM_RUN_LIVE=1.
 */
import { describe, expect, test } from "bun:test";
import { LocalDetector } from "../src/detectors/local.ts";
import { Session } from "../src/core/session.ts";
import {
  anonymizeText,
  deanonymizeStream,
  deanonymizeText,
} from "../src/plain.ts";

const LIVE = ["1", "true", "yes"].includes(
  (process.env.GHEIM_RUN_LIVE ?? "").toLowerCase(),
);
const t = LIVE ? test : test.skip;

describe("live: LocalDetector against the real model", () => {
  t("detects a person name", async () => {
    const det = new LocalDetector({ device: "cpu", dtype: "fp32" });
    const spans = await det.detect("Hi, my name is Alice Smith.");
    const persons = spans.filter((s) => s.label === "private_person");
    expect(persons.length).toBeGreaterThan(0);
  }, 600_000);

  t("detects an email", async () => {
    const det = new LocalDetector({ device: "cpu", dtype: "fp32" });
    const spans = await det.detect("Email me at alice@example.com please.");
    const emails = spans.filter((s) => s.label === "private_email");
    expect(emails.length).toBeGreaterThan(0);
  }, 600_000);

  t("anonymize → deanonymize round trip", async () => {
    const det = new LocalDetector({ device: "cpu", dtype: "fp32" });
    const session = new Session();
    (session as unknown as { detector: LocalDetector }).detector = det;
    const text = "Hi, my name is Alice and my email is alice@example.com.";
    const redacted = await anonymizeText(text, session);
    expect(redacted.includes("<PERSON_") || redacted.includes("<EMAIL_")).toBe(true);
    const restored = deanonymizeText(redacted, session);
    expect(restored).toBe(text);
  }, 600_000);

  t("streaming round trip with split sentinel", async () => {
    const det = new LocalDetector({ device: "cpu", dtype: "fp32" });
    const session = new Session();
    (session as unknown as { detector: LocalDetector }).detector = det;
    await anonymizeText("Hi, my name is Alice.", session);
    const personSentinel = Object.keys(session.mapping).find((k) =>
      k.startsWith("<PERSON_"),
    )!;
    const chunks = [
      "Nice to meet you, ",
      personSentinel.slice(0, 4),
      personSentinel.slice(4),
      "!",
    ];
    let out = "";
    for await (const c of deanonymizeStream(chunks, session)) out += c;
    expect(out).toContain("Alice");
    expect(out.includes(personSentinel)).toBe(false);
  }, 600_000);
});
