/**
 * Real-model integration test for gheim-ch-XXXm (Swiss-tuned) via transformers.js.
 *
 * Verifies the fine-tuned model end-to-end:
 *   - Loads via @huggingface/transformers (CPU/WASM, the default Node + Bun path)
 *   - Detects Swiss-shape PII (CH-IBAN, AHV, +41 phone, Swiss-format address)
 *   - Detects English-shape secrets (API keys) — covers the synthetic gap-fill cells
 *   - Round-trips cleanly through anonymize → deanonymize
 *
 * Gated by GHEIM_RUN_LIVE=1; pick the model via GHEIM_LIVE_MODEL.
 *
 * Run:
 *   GHEIM_RUN_LIVE=1 \
 *   GHEIM_LIVE_MODEL=joelbarmettler/gheim-ch-XXXm \
 *   bun test packages/gheim-js/tests/live-swiss.test.ts
 *
 * If GHEIM_LIVE_MODEL is unset the test is skipped (so CI without the model
 * available doesn't fail). When the bake-off picks a leader, set the env var
 * to its hub id (e.g. joelbarmettler/gheim-ch-550m).
 */
import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";
import { LocalDetector } from "../src/detectors/local.ts";
import { anonymizeText, deanonymizeText } from "../src/plain.ts";

const LIVE = ["1", "true", "yes"].includes(
  (process.env.GHEIM_RUN_LIVE ?? "").toLowerCase(),
);
const MODEL = process.env.GHEIM_LIVE_MODEL ?? "";
const enabled = LIVE && MODEL.length > 0;
const t = enabled ? test : test.skip;

const TIMEOUT = 600_000; // model load on first call dominates wall time

describe("live-swiss: gheim-ch-XXXm CPU/WASM via transformers.js", () => {
  t("loads the model on Node/Bun CPU without throwing", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "fp32" });
    const spans = await det.detect("Hello.");
    // Loading must not throw; spans for a PII-free chunk may be 0 or empty.
    expect(Array.isArray(spans)).toBe(true);
  }, TIMEOUT);

  t("detects a Swiss IBAN with checksum", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "fp32" });
    // Real-format CH IBAN, mod-97-10 valid.
    const text = "Bitte überweisen an IBAN CH9300762011623852957 bis Ende Monat.";
    const spans = await det.detect(text);
    const account = spans.filter((s) => s.label === "account_number");
    expect(account.length).toBeGreaterThan(0);
    expect(account[0]!.text).toContain("CH9300762011623852957");
  }, TIMEOUT);

  t("detects a Swiss-format address (Strasse + PLZ + Ort)", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "fp32" });
    const text = "Die Praxis befindet sich an der Werdstrasse 36, 8004 Zürich.";
    const spans = await det.detect(text);
    const addrs = spans.filter((s) => s.label === "private_address");
    expect(addrs.length).toBeGreaterThan(0);
  }, TIMEOUT);

  t("detects a +41 phone number", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "fp32" });
    const text = "Erreichbar unter +41 44 268 12 34 von Montag bis Freitag.";
    const spans = await det.detect(text);
    const phones = spans.filter((s) => s.label === "private_phone");
    expect(phones.length).toBeGreaterThan(0);
  }, TIMEOUT);

  t("detects an API key as secret", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "fp32" });
    const text =
      "Hey, please rotate the staging key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl asap.";
    const spans = await det.detect(text);
    const secrets = spans.filter((s) => s.label === "secret");
    expect(secrets.length).toBeGreaterThan(0);
  }, TIMEOUT);

  t("anonymize → deanonymize round trip on a multi-PII Swiss chunk", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "fp32" });
    const session = new Session();
    (session as unknown as { detector: LocalDetector }).detector = det;

    const text =
      "Bonjour, je suis Marc Dubois (marc.dubois@vd.ch). " +
      "Mon adresse: Avenue de la Gare 12, 1003 Lausanne. " +
      "Téléphone: +41 21 555 12 34.";

    const redacted = await anonymizeText(text, session);
    // At least person, email, and address should be redacted to sentinels.
    const tagsSeen = new Set(
      Object.keys(session.mapping).map(
        (k) => k.match(/^<([A-Z]+)_/)?.[1] ?? "",
      ),
    );
    expect(tagsSeen.has("PERSON")).toBe(true);
    expect(tagsSeen.has("EMAIL")).toBe(true);
    // Address coverage may vary; we don't fail if the model misses it here.

    // Round trip must restore the original text exactly.
    const restored = deanonymizeText(redacted, session);
    expect(restored).toBe(text);
    // No sentinel residue.
    expect(restored.includes("<PERSON_")).toBe(false);
    expect(restored.includes("<EMAIL_")).toBe(false);
  }, TIMEOUT);
});
