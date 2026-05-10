/**
 * Local-checkpoint variant of live-swiss.test.ts.
 *
 * Loads the gheim-ch-560m ONNX export from a local directory (no HF Hub
 * needed). Used pre-publication to verify the JS detector works end-to-end
 * against our actual fine-tuned model.
 *
 * Skipped unless both env vars are set:
 *   GHEIM_RUN_LIVE=1
 *   GHEIM_LOCAL_MODEL_DIR=/tmp/gheim-local-models   # parent dir
 *
 * The model directory structure transformers.js expects:
 *   $GHEIM_LOCAL_MODEL_DIR/
 *     gheim-ch-560m/
 *       config.json
 *       tokenizer.json
 *       tokenizer_config.json
 *       special_tokens_map.json
 *       onnx/
 *         model.onnx
 *         model.onnx_data
 *
 * Run:
 *   GHEIM_RUN_LIVE=1 \
 *   GHEIM_LOCAL_MODEL_DIR=/tmp/gheim-local-models \
 *   bun test packages/gheim-js/tests/live-swiss-local.test.ts
 */
import { describe, expect, test } from "bun:test";
import { LocalDetector } from "../src/detectors/local.ts";

const LIVE = ["1", "true", "yes"].includes(
  (process.env.GHEIM_RUN_LIVE ?? "").toLowerCase(),
);
const LOCAL_DIR = process.env.GHEIM_LOCAL_MODEL_DIR ?? "";
const enabled = LIVE && LOCAL_DIR.length > 0;
const t = enabled ? test : test.skip;

const TIMEOUT = 600_000;
const MODEL_NAME = "gheim-ch-560m";

if (enabled) {
  // Configure transformers.js to load from local disk only.
  // Must happen before any import that would touch the env singleton.
  const { env } = await import("@huggingface/transformers");
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = LOCAL_DIR.endsWith("/") ? LOCAL_DIR : `${LOCAL_DIR}/`;
  console.log(`[live-swiss-local] loading from ${env.localModelPath}${MODEL_NAME}`);
}

describe("live-swiss-local: gheim-ch-560m ONNX from local disk", () => {
  t("loads the model from the local ONNX export", async () => {
    const det = new LocalDetector({ model: MODEL_NAME, device: "cpu", dtype: "fp32" });
    const spans = await det.detect("Hello.");
    expect(Array.isArray(spans)).toBe(true);
  }, TIMEOUT);

  t("detects a Swiss IBAN", async () => {
    const det = new LocalDetector({ model: MODEL_NAME, device: "cpu", dtype: "fp32" });
    const spans = await det.detect(
      "Bitte überweisen an IBAN CH9300762011623852957.",
    );
    const account = spans.filter((s) => s.label === "account_number");
    expect(account.length).toBeGreaterThan(0);
    expect(account[0]!.text).toContain("CH9300762011623852957");
  }, TIMEOUT);

  t("detects a Swiss-format address", async () => {
    const det = new LocalDetector({ model: MODEL_NAME, device: "cpu", dtype: "fp32" });
    const spans = await det.detect(
      "Die Praxis befindet sich an der Werdstrasse 36, 8004 Zürich.",
    );
    const addrs = spans.filter((s) => s.label === "private_address");
    expect(addrs.length).toBeGreaterThan(0);
  }, TIMEOUT);

  t("detects a +41 phone", async () => {
    const det = new LocalDetector({ model: MODEL_NAME, device: "cpu", dtype: "fp32" });
    const spans = await det.detect(
      "Erreichbar unter +41 44 268 12 34 von Montag bis Freitag.",
    );
    const phones = spans.filter((s) => s.label === "private_phone");
    expect(phones.length).toBeGreaterThan(0);
  }, TIMEOUT);

  t("detects an API key as secret", async () => {
    const det = new LocalDetector({ model: MODEL_NAME, device: "cpu", dtype: "fp32" });
    const spans = await det.detect(
      "Hey, please rotate the staging key sk-proj-AbCdEfGhIjKlMnOpQrStUvWxYz0123456789AbCdEfGhIjKl asap.",
    );
    const secrets = spans.filter((s) => s.label === "secret");
    expect(secrets.length).toBeGreaterThan(0);
  }, TIMEOUT);
});
