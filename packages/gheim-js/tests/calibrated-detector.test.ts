import { describe, expect, test } from "bun:test";
import {
  CalibratedDetector,
  defaultDetector,
  RemoteDetector,
} from "../src/index.ts";

describe("CalibratedDetector — construction", () => {
  test("default oBias is 0.5", () => {
    const det = new CalibratedDetector();
    expect(det.oBias).toBe(0.5);
  });

  test("default model is gheim-ch-560m", () => {
    const det = new CalibratedDetector();
    expect(det.model).toBe("joelbarmettler/gheim-ch-560m");
  });

  test("oBias can be overridden", () => {
    expect(new CalibratedDetector({ oBias: 0 }).oBias).toBe(0);
    expect(new CalibratedDetector({ oBias: 1.5 }).oBias).toBe(1.5);
  });

  test("rejects chunk_overlap >= chunk_tokens", () => {
    expect(
      () => new CalibratedDetector({ chunkTokens: 10, chunkOverlapTokens: 10 }),
    ).toThrow(/must be smaller than/);
    expect(
      () => new CalibratedDetector({ chunkTokens: 10, chunkOverlapTokens: 11 }),
    ).toThrow(/must be smaller than/);
  });

  test("trimWhitespace defaults to true", () => {
    expect(new CalibratedDetector().trimWhitespace).toBe(true);
    expect(new CalibratedDetector({ trimWhitespace: false }).trimWhitespace).toBe(false);
  });

  test("empty input short-circuits without loading the model", async () => {
    const det = new CalibratedDetector();
    const out = await det.detect("");
    expect(out).toEqual([]);
  });

  test("per-call model override that doesn't match constructor → throws", async () => {
    const det = new CalibratedDetector({ model: "some/model" });
    await expect(det.detect("anything", { model: "other/model" })).rejects.toThrow(
      /per-call model override/,
    );
  });
});

describe("defaultDetector wiring", () => {
  test("returns CalibratedDetector when no GHEIM_API_KEY in env", () => {
    const prev = process.env.GHEIM_API_KEY;
    delete process.env.GHEIM_API_KEY;
    try {
      const det = defaultDetector();
      expect(det).toBeInstanceOf(CalibratedDetector);
      // And the calibration knob is set to the production default.
      expect((det as CalibratedDetector).oBias).toBe(0.5);
    } finally {
      if (prev !== undefined) process.env.GHEIM_API_KEY = prev;
    }
  });

  test("returns RemoteDetector when GHEIM_API_KEY is set", () => {
    const prev = process.env.GHEIM_API_KEY;
    process.env.GHEIM_API_KEY = "test-key";
    try {
      const det = defaultDetector();
      expect(det).toBeInstanceOf(RemoteDetector);
    } finally {
      if (prev === undefined) delete process.env.GHEIM_API_KEY;
      else process.env.GHEIM_API_KEY = prev;
    }
  });
});
