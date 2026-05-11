/**
 * Long-input chunking + batching tests for LocalDetector. Mirrors
 * gheim-py's test_chunking.py.
 *
 * Would have caught the silent-truncation bug where the underlying
 * @huggingface/transformers pipeline truncates anything past ~512
 * tokens without warning, so PII near the end of a long document
 * went un-redacted.
 *
 * Uses a fake pipeline + fake tokenizer where 1 token = 1 character,
 * making the chunking math exactly predictable.
 */
import { describe, expect, test } from "bun:test";
import { LocalDetector } from "../src/detectors/local.ts";

interface DetectionItem {
  start?: number;
  end?: number;
  word?: string;
  entity?: string;
  entity_group?: string;
  score?: number;
}

interface PipelineCall {
  n: number;
  batch_size?: number;
}

/** 1-token-per-character tokenizer; token offsets == char offsets. */
function makeFakeTokenizer() {
  return async (text: string, _opts: unknown) => ({
    offset_mapping: {
      tolist: () => text.split("").map((_, i) => [i, i + 1]),
    },
  });
}

/** Returns spans for every occurrence of `needle` in the input text. */
function makeFakePipeline(needle = "Joel", label = "private_person") {
  const calls: PipelineCall[] = [];

  function scan(text: string): DetectionItem[] {
    const out: DetectionItem[] = [];
    let i = 0;
    while (true) {
      const idx = text.indexOf(needle, i);
      if (idx === -1) break;
      out.push({
        start: idx,
        end: idx + needle.length,
        entity_group: label,
        score: 0.99,
      });
      i = idx + needle.length;
    }
    return out;
  }

  const pipe = async (
    texts: string | string[],
    opts?: { batch_size?: number },
  ): Promise<DetectionItem[] | DetectionItem[][]> => {
    if (typeof texts === "string") {
      calls.push({ n: 1, batch_size: opts?.batch_size });
      return scan(texts);
    }
    calls.push({ n: texts.length, batch_size: opts?.batch_size });
    return texts.map(scan);
  };
  // Attach the tokenizer the way @huggingface/transformers exposes it on
  // a real pipeline so LocalDetector can introspect for token-aware chunking.
  (pipe as unknown as { tokenizer: ReturnType<typeof makeFakeTokenizer> }).tokenizer = makeFakeTokenizer();
  return { pipe, calls };
}

function makeDetector(
  pipe: unknown,
  opts: { chunkTokens?: number; chunkOverlapTokens?: number; batchSize?: number } = {},
) {
  return new LocalDetector({
    pipelineInstance: pipe,
    chunkTokens: opts.chunkTokens ?? 100,
    chunkOverlapTokens: opts.chunkOverlapTokens ?? 20,
    batchSize: opts.batchSize ?? 4,
  });
}

// ---------------------------------------------------------------------------
// Single-input chunking
// ---------------------------------------------------------------------------

describe("LocalDetector chunking", () => {
  test("short input uses fast path", async () => {
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe, { chunkTokens: 100 });
    const spans = await det.detect("Hello Joel.");
    expect(spans).toHaveLength(1);
    expect(spans[0].start).toBe(6);
    expect(spans[0].end).toBe(10);
    expect(spans[0].text).toBe("Joel");
    expect(calls).toHaveLength(1);
    expect(calls[0].n).toBe(1);
  });

  test("long input finds PII at the end (the truncation regression)", async () => {
    const text = "x".repeat(950) + "Joel was here." + "y".repeat(50);
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe, { chunkTokens: 200, chunkOverlapTokens: 20 });
    const spans = await det.detect(text);
    expect(spans).toHaveLength(1);
    expect(text.slice(spans[0].start, spans[0].end)).toBe("Joel");
    expect(spans[0].start).toBe(950);
    expect(calls[0].n).toBeGreaterThanOrEqual(5);
  });

  test("finds PII at start, middle, and end of long doc", async () => {
    const blocks = [
      "a".repeat(50) + "Joel" + "b".repeat(50),    // 104 chars, Joel at 50
      "c".repeat(400),
      "d".repeat(50) + "Joel" + "e".repeat(50),    // Joel at 554
      "f".repeat(400),
      "g".repeat(50) + "Joel" + "h".repeat(50),    // Joel at 1058
    ];
    const text = blocks.join("");
    const { pipe } = makeFakePipeline();
    const det = makeDetector(pipe, { chunkTokens: 200, chunkOverlapTokens: 32 });
    const spans = await det.detect(text);
    const starts = spans.map((s) => s.start).sort((a, b) => a - b);
    expect(starts).toEqual([50, 554, 1058]);
    for (const s of spans) expect(text.slice(s.start, s.end)).toBe("Joel");
  });

  test("PII in overlap region is deduplicated", async () => {
    // chunk_tokens=100, overlap=20, stride=80.
    // Chunk 1: [0, 100). Chunk 2: [80, 180). Joel at [85, 89) is in both.
    const text = "x".repeat(85) + "Joel" + "y".repeat(100);
    const { pipe } = makeFakePipeline();
    const det = makeDetector(pipe, { chunkTokens: 100, chunkOverlapTokens: 20 });
    const spans = await det.detect(text);
    expect(spans).toHaveLength(1);
    expect(spans[0].start).toBe(85);
    expect(spans[0].end).toBe(89);
  });

  test("rejects overlap >= chunk_tokens at construction", () => {
    expect(
      () => new LocalDetector({ chunkTokens: 10, chunkOverlapTokens: 10 }),
    ).toThrow();
  });

  test("empty input returns empty without a pipeline call", async () => {
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe);
    expect(await det.detect("")).toEqual([]);
    expect(calls).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Cross-input batching (detectBatch)
// ---------------------------------------------------------------------------

describe("LocalDetector detectBatch", () => {
  test("one batched pipeline call for many short inputs", async () => {
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe, { chunkTokens: 100, batchSize: 4 });
    const results = await det.detectBatch(["I am Joel.", "Joel and Joel.", "Nobody."]);
    expect(results).toHaveLength(3);
    expect(results[0]).toHaveLength(1);
    expect(results[1]).toHaveLength(2);
    expect(results[2]).toHaveLength(0);
    expect(calls).toHaveLength(1);
    expect(calls[0].n).toBe(3);
    expect(calls[0].batch_size).toBe(4);
  });

  test("long inputs split into chunks share one batched call", async () => {
    const short = "Joel here.";
    const long_ = "x".repeat(200) + "Joel" + "y".repeat(200);
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe, {
      chunkTokens: 180,
      chunkOverlapTokens: 20,
      batchSize: 8,
    });
    const results = await det.detectBatch([short, long_]);
    expect(results[0]).toHaveLength(1);
    expect(results[1]).toHaveLength(1);
    expect(results[1][0].start).toBe(200);
    expect(calls).toHaveLength(1);
    expect(calls[0].n).toBeGreaterThanOrEqual(3);
  });

  test("ordering preserved with empty inputs interleaved", async () => {
    const { pipe } = makeFakePipeline();
    const det = makeDetector(pipe);
    const results = await det.detectBatch(["I am Joel.", "", "Hi Joel!", ""]);
    expect(results).toHaveLength(4);
    expect(results[0]).toHaveLength(1);
    expect(results[1]).toEqual([]);
    expect(results[2]).toHaveLength(1);
    expect(results[3]).toEqual([]);
  });

  test("empty list returns empty", async () => {
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe);
    expect(await det.detectBatch([])).toEqual([]);
    expect(calls).toHaveLength(0);
  });

  test("all-empty inputs skip the pipeline entirely", async () => {
    const { pipe, calls } = makeFakePipeline();
    const det = makeDetector(pipe);
    expect(await det.detectBatch(["", "", ""])).toEqual([[], [], []]);
    expect(calls).toHaveLength(0);
  });
});
