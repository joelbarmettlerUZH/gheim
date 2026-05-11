import { describe, expect, test } from "bun:test";
import { LocalDetector } from "../src/detectors/local.ts";

/** Wrap a single-input pipeline fake so it satisfies the batched contract
 *  LocalDetector now uses internally (string | string[]) → result | result[]). */
function batched(
  scan: (text: string) => unknown[] | Promise<unknown[]>,
): (texts: string | string[]) => Promise<unknown[] | unknown[][]> {
  return async (texts) => {
    if (typeof texts === "string") return await scan(texts);
    return Promise.all(texts.map((t) => Promise.resolve(scan(t))));
  };
}

/** Fake transformers.js pipeline — returns hits for "Joel" only. */
function makeFakePipeline() {
  return batched((text: string) => {
    const out: unknown[] = [];
    let i = 0;
    while (true) {
      const idx = text.indexOf("Joel", i);
      if (idx === -1) break;
      out.push({
        entity_group: "private_person",
        score: 0.95,
        start: idx,
        end: idx + 4,
        word: " Joel",
      });
      i = idx + 4;
    }
    return out;
  });
}

describe("LocalDetector", () => {
  test("uses injected pipelineInstance and returns Span[]", async () => {
    const det = new LocalDetector({ pipelineInstance: makeFakePipeline() });
    const spans = await det.detect("Hi, my name is Joel and Joel says hi.");
    expect(spans.length).toBe(2);
    expect(spans[0]).toEqual({
      label: "private_person",
      start: 15,
      end: 19,
      text: "Joel",
      score: 0.95,
    });
    expect(spans[1]!.start).toBe(24);
  });

  test("returns [] for empty input without loading the pipeline", async () => {
    const det = new LocalDetector(); // no pipelineInstance — would normally try to import
    const out = await det.detect("");
    expect(out).toEqual([]);
  });

  test("rejects per-call model override that doesn't match constructor model", async () => {
    const det = new LocalDetector({ pipelineInstance: makeFakePipeline() });
    await expect(det.detect("anything", { model: "other/model" })).rejects.toThrow(
      /model override/,
    );
  });

  test("accepts per-call model override that matches constructor model", async () => {
    const det = new LocalDetector({ pipelineInstance: makeFakePipeline() });
    // Pass the default model id so the matching-model branch in detect() runs.
    const spans = await det.detect("Joel", { model: "joelbarmettler/gheim-ch-560m" });
    expect(spans.length).toBe(1);
  });

  test("recovers char offsets when the pipeline doesn't return start/end", async () => {
    // Mirrors transformers.js v4 output: entity_group + word, no offsets.
    const noOffsetPipeline = batched(() => [
      { entity_group: "PER", word: "Alice", score: 0.95 },
      { entity_group: "ORG", word: "Microsoft", score: 0.99 },
    ]);
    const det = new LocalDetector({ pipelineInstance: noOffsetPipeline });
    const spans = await det.detect("My name is Alice and I work at Microsoft.");
    expect(spans.length).toBe(2);
    expect(spans[0]).toMatchObject({ label: "PER", start: 11, end: 16, text: "Alice" });
    expect(spans[1]).toMatchObject({ label: "ORG", start: 31, end: 40, text: "Microsoft" });
  });

  test("skips entities whose surface can't be located in the text", async () => {
    const phantomPipeline = batched(() => [
      { entity_group: "PER", word: "Alice", score: 0.95 },
      { entity_group: "PER", word: "Bob", score: 0.90 }, // not in text
    ]);
    const det = new LocalDetector({ pipelineInstance: phantomPipeline });
    const spans = await det.detect("Alice was here.");
    expect(spans.length).toBe(1);
    expect(spans[0]!.text).toBe("Alice");
  });

  test("preserves left-to-right ordering for repeated surfaces", async () => {
    const repeatedPipeline = batched(() => [
      { entity_group: "PER", word: "Joel", score: 0.95 },
      { entity_group: "PER", word: "Joel", score: 0.95 },
    ]);
    const det = new LocalDetector({ pipelineInstance: repeatedPipeline });
    const spans = await det.detect("Joel wrote to Joel.");
    expect(spans.length).toBe(2);
    expect(spans[0]!.start).toBe(0);
    expect(spans[1]!.start).toBe(14);
  });

  test("concurrent detect calls share the same lazy-loaded pipeline", async () => {
    let loadCount = 0;
    let firstResolve: (() => void) | null = null;
    const firstReady = new Promise<void>((res) => {
      firstResolve = res;
    });

    // Pipeline that records how many "loads" happen — really how many times the
    // LocalDetector tries to (re-)await the load promise. We can't directly
    // count loads of @huggingface/transformers without injecting; instead, we
    // verify behavior by injecting a pre-built pipeline (which bypasses load)
    // and asserting all parallel calls return.
    const slowPipeline = batched(async () => {
      loadCount += 1;
      // Suspend the very first call until all callers are queued.
      if (loadCount === 1) {
        await firstReady;
      }
      return [{ entity_group: "PER", word: "Joel", score: 0.9, start: 0, end: 4 }];
    });

    const det = new LocalDetector({ pipelineInstance: slowPipeline });
    // Fire 10 concurrent detect calls.
    const promises = Array.from({ length: 10 }, () => det.detect("Joel here"));
    // Let the event loop schedule them all.
    await new Promise((r) => setTimeout(r, 10));
    firstResolve!();
    const all = await Promise.all(promises);
    expect(all.length).toBe(10);
    expect(all.every((s) => s.length === 1 && s[0]!.text === "Joel")).toBe(true);
  });

  test("strips ## wordpiece markers from the surface before searching", async () => {
    const wordpiecePipeline = batched(() => [
      { entity_group: "ORG", word: "Micro##soft", score: 0.99 },
    ]);
    const det = new LocalDetector({ pipelineInstance: wordpiecePipeline });
    const spans = await det.detect("Works at Microsoft today.");
    expect(spans.length).toBe(1);
    expect(spans[0]!.text).toBe("Microsoft");
  });
});
