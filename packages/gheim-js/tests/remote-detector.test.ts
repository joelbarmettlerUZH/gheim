/**
 * RemoteDetector tests using a fetch mock — no live server.
 * Mirrors test_remote_detector.py.
 */
import { describe, expect, test } from "bun:test";
import { RemoteDetector } from "../src/detectors/remote.ts";

interface CapturedRequest {
  url: string;
  method: string;
  headers: Record<string, string>;
  body: unknown;
}

function makeFetch(
  reply: { status?: number; body?: unknown },
  captured: CapturedRequest[],
): typeof fetch {
  return async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = typeof input === "string" ? input : input.toString();
    const headers: Record<string, string> = {};
    if (init?.headers) {
      const h = new Headers(init.headers);
      h.forEach((v, k) => {
        headers[k.toLowerCase()] = v;
      });
    }
    captured.push({
      url,
      method: init?.method ?? "GET",
      headers,
      body: typeof init?.body === "string" ? JSON.parse(init.body) : init?.body,
    });
    const status = reply.status ?? 200;
    return new Response(JSON.stringify(reply.body ?? {}), {
      status,
      headers: { "Content-Type": "application/json" },
    });
  };
}

describe("RemoteDetector (mocked fetch)", () => {
  test("posts the right body and parses spans", async () => {
    const captured: CapturedRequest[] = [];
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      apiKey: "secret-abc",
      fetch: makeFetch(
        {
          body: {
            model: "openai/privacy-filter",
            spans: [{ label: "private_person", start: 11, end: 16, score: 0.99 }],
          },
        },
        captured,
      ),
    });
    const spans = await det.detect("My name is Alice.");
    expect(captured.length).toBe(1);
    expect(captured[0]!.method).toBe("POST");
    expect(captured[0]!.url).toBe("http://mock/v1/detect");
    expect(captured[0]!.body).toEqual({
      model: "openai/privacy-filter",
      text: "My name is Alice.",
    });
    expect(captured[0]!.headers.authorization).toBe("Bearer secret-abc");
    expect(spans.length).toBe(1);
    expect(spans[0]!.label).toBe("private_person");
    expect(spans[0]!.text).toBe("Alice");
    expect(spans[0]!.score).toBeCloseTo(0.99);
  });

  test("omits Authorization when apiKey is not set", async () => {
    const captured: CapturedRequest[] = [];
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch({ body: { spans: [] } }, captured),
    });
    await det.detect("hi");
    expect(captured[0]!.headers.authorization).toBeUndefined();
  });

  test("per-call model override goes into body", async () => {
    const captured: CapturedRequest[] = [];
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch({ body: { spans: [] } }, captured),
    });
    await det.detect("hi", { model: "custom/model" });
    expect((captured[0]!.body as { model: string }).model).toBe("custom/model");
  });

  test("empty text short-circuits without fetching", async () => {
    const captured: CapturedRequest[] = [];
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch({ body: { spans: [] } }, captured),
    });
    const out = await det.detect("");
    expect(out).toEqual([]);
    expect(captured.length).toBe(0);
  });

  test("500 response throws", async () => {
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch({ status: 500, body: { detail: "boom" } }, []),
    });
    await expect(det.detect("anything")).rejects.toThrow(/500/);
  });

  test("401 response throws", async () => {
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch({ status: 401, body: { detail: "missing bearer token" } }, []),
    });
    await expect(det.detect("anything")).rejects.toThrow(/401/);
  });

  test("trim_whitespace strips BPE leading-space artifact by default", async () => {
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch(
        {
          body: {
            spans: [{ label: "private_person", start: 14, end: 20, score: 0.99 }],
          },
        },
        [],
      ),
    });
    const spans = await det.detect("Hi, my name is Alice.");
    expect(spans[0]!.text).toBe("Alice");
    expect(spans[0]!.start).toBe(15);
  });

  test("trim_whitespace can be disabled", async () => {
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      trimWhitespace: false,
      fetch: makeFetch(
        {
          body: {
            spans: [{ label: "private_person", start: 14, end: 20, score: 0.99 }],
          },
        },
        [],
      ),
    });
    const spans = await det.detect("Hi, my name is Alice.");
    expect(spans[0]!.text).toBe(" Alice");
  });

  test("whitespace-only span dropped when trimming", async () => {
    const det = new RemoteDetector({
      baseUrl: "http://mock",
      fetch: makeFetch(
        { body: { spans: [{ label: "x", start: 0, end: 0, score: 0.5 }] } },
        [],
      ),
    });
    expect(await det.detect("abcdef")).toEqual([]);
  });
});
