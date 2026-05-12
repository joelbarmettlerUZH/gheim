/**
 * Real-model integration test for the published gheim model via transformers.js.
 *
 * Verifies the fine-tuned model end-to-end:
 *   - Loads via @huggingface/transformers (CPU/WASM, the default Node + Bun path)
 *   - Detects Swiss-shape PII (CH-IBAN, AHV, +41 phone, Swiss-format address)
 *   - Detects English-shape secrets (API keys) — covers the synthetic gap-fill cells
 *   - Round-trips cleanly through anonymize → deanonymize
 *
 * Always runs (no env gate). On first run this downloads ~1 GB of ONNX
 * weights from the Hub; the workflow caches `~/.cache/huggingface/hub`
 * across runs.
 */
import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";
import { LocalDetector } from "../src/detectors/local.ts";
import { anonymizeText, deanonymizeText } from "../src/plain.ts";

const MODEL = process.env.GHEIM_LIVE_MODEL ?? "joelbarmettler/gheim-ch-560m";
const TIMEOUT = 600_000; // model load on first call dominates wall time

describe("live-swiss: published gheim model on CPU/WASM via transformers.js", () => {
  test("loads the model on Node/Bun CPU without throwing", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "q8" });
    const spans = await det.detect("Hello.");
    // Loading must not throw; spans for a PII-free chunk may be 0 or empty.
    expect(Array.isArray(spans)).toBe(true);
  }, TIMEOUT);

  test("detects a Swiss IBAN with checksum", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "q8" });
    // Real-format CH IBAN, mod-97-10 valid.
    const text = "Bitte überweisen an IBAN CH9300762011623852957 bis Ende Monat.";
    const spans = await det.detect(text);
    const account = spans.filter((s) => s.label === "account_number");
    expect(account.length).toBeGreaterThan(0);
    expect(account[0]!.text).toContain("CH9300762011623852957");
  }, TIMEOUT);

  test("detects a Swiss-format address (Strasse + PLZ + Ort)", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "q8" });
    const text = "Die Praxis befindet sich an der Werdstrasse 36, 8004 Zürich.";
    const spans = await det.detect(text);
    const addrs = spans.filter((s) => s.label === "private_address");
    expect(addrs.length).toBeGreaterThan(0);
  }, TIMEOUT);

  test("detects a +41 phone number", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "q8" });
    const text = "Erreichbar unter +41 44 268 12 34 von Montag bis Freitag.";
    const spans = await det.detect(text);
    const phones = spans.filter((s) => s.label === "private_phone");
    expect(phones.length).toBeGreaterThan(0);
  }, TIMEOUT);

  test("detects an API key as secret", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "q8" });
    const text =
      "Hey, please rotate the staging key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl asap.";
    const spans = await det.detect(text);
    const secrets = spans.filter((s) => s.label === "secret");
    expect(secrets.length).toBeGreaterThan(0);
  }, TIMEOUT);

  test("anonymize → deanonymize round trip on a multi-PII Swiss chunk", async () => {
    const det = new LocalDetector({ model: MODEL, device: "cpu", dtype: "q8" });
    const session = new Session();
    (session as unknown as { detector: LocalDetector }).detector = det;

    const text =
      "Bonjour, je suis Marc Dubois (marc.dubois@vd.ch). " +
      "Mon adresse: Avenue de la Gare 12, 1003 Lausanne. " +
      "Téléphone: +41 21 555 12 34.";

    const redacted = await anonymizeText(text, session);
    // At least person and email should be redacted to sentinels.
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

import { CalibratedDetector } from "../src/detectors/calibrated.ts";

describe("live-swiss: CalibratedDetector recovers the documented suppression", () => {
  // The exact failing input from the probe rounds: v1 + raw model emits
  // ZERO non-O tokens for "Müller" when an IBAN follows. With the
  // calibration default (oBias=0.5) the model recovers all three
  // categories (person, date, account_number).
  const FAILING = (
    "Sehr geehrte Frau Müller,\n" +
    "vielen Dank für Ihre Anfrage vom 14. März 2026.\n" +
    "IBAN: CH9300762011623852957"
  );

  test("default oBias=0.5 recovers Müller + date when IBAN follows", async () => {
    const det = new CalibratedDetector({
      model: MODEL, device: "cpu", dtype: "q8",
    });
    const spans = await det.detect(FAILING);
    const cats = new Set(spans.map((s) => s.label));
    expect(cats.has("private_person")).toBe(true);
    expect(cats.has("private_date")).toBe(true);
    expect(cats.has("account_number")).toBe(true);
  }, TIMEOUT);

  test("calibration is monotone: oBias>=0 never removes spans the model would find", async () => {
    // Contract: subtracting from the O logit before argmax can only
    // make non-O classes win MORE often, never less. So the set of
    // detected categories at oBias=0.5 must be a superset of (or equal
    // to) the set at oBias=0.0. Verifies our O-bias path is implemented
    // correctly (subtracts from O, not from a wrong class).
    //
    // Note: the fp32 suppression bug doesn't always reproduce on the q8
    // ONNX (quantization noise nudges argmax differently), so we don't
    // assert specific categories at oBias=0 — only the monotonicity.
    const detLow = new CalibratedDetector({
      model: MODEL, device: "cpu", dtype: "q8", oBias: 0,
    });
    const detHigh = new CalibratedDetector({
      model: MODEL, device: "cpu", dtype: "q8", oBias: 0.5,
    });
    const lowCats = new Set((await detLow.detect(FAILING)).map((s) => s.label));
    const highCats = new Set((await detHigh.detect(FAILING)).map((s) => s.label));
    for (const cat of lowCats) {
      expect(highCats.has(cat)).toBe(true);
    }
  }, TIMEOUT);

  test("no false positives on PII-free German legal text", async () => {
    const det = new CalibratedDetector({
      model: MODEL, device: "cpu", dtype: "q8",
    });
    const spans = await det.detect(
      "Das Bundesgericht hat in seinem Entscheid die Argumentation der Vorinstanz bestätigt.",
    );
    expect(spans.length).toBe(0);
  }, TIMEOUT);
});
