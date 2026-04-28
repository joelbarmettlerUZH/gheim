import { describe, expect, test } from "bun:test";
import { Session } from "../src/core/session.ts";
import {
  anonymizeMessages,
  anonymizeText,
  deanonymizeStream,
  deanonymizeText,
} from "../src/plain.ts";
import type { Detector, Span } from "../src/detectors/base.ts";

class FakeDetector implements Detector {
  constructor(private surfaces: Record<string, string>) {}
  async detect(text: string): Promise<Span[]> {
    const out: Span[] = [];
    for (const [surface, label] of Object.entries(this.surfaces)) {
      let start = 0;
      while (true) {
        const idx = text.indexOf(surface, start);
        if (idx === -1) break;
        out.push({ label, start: idx, end: idx + surface.length, text: surface });
        start = idx + surface.length;
      }
    }
    return out;
  }
}

function sessionWith(detector: Detector): Session {
  const s = new Session();
  (s as unknown as { detector: Detector }).detector = detector;
  return s;
}

describe("plain API", () => {
  test("anonymize/deanonymize round-trip", async () => {
    const s = sessionWith(new FakeDetector({ Joel: "private_person", "joel@x.ch": "private_email" }));
    const clean = await anonymizeText("Joel at joel@x.ch", s);
    expect(clean).toBe("<PERSON_1> at <EMAIL_1>");
    const restored = deanonymizeText("Hi <PERSON_1>, your email <EMAIL_1>.", s);
    expect(restored).toBe("Hi Joel, your email joel@x.ch.");
  });

  test("anonymizeMessages handles string and multimodal content", async () => {
    const s = sessionWith(new FakeDetector({ Joel: "private_person" }));
    const out = await anonymizeMessages(
      [
        { role: "system", content: "You speak to Joel." },
        {
          role: "user",
          content: [
            { type: "text", text: "Joel here" },
            { type: "image_url", image_url: { url: "https://x/y.png" } },
          ],
        },
      ],
      s,
    );
    expect(out[0]!.content).toBe("You speak to <PERSON_1>.");
    const parts = out[1]!.content as Array<{ type: string; text?: string }>;
    expect(parts[0]).toEqual({ type: "text", text: "<PERSON_1> here" });
    expect(parts[1]!.type).toBe("image_url");
  });

  test("deanonymizeStream restores split sentinels", async () => {
    const s = sessionWith(new FakeDetector({ Joel: "private_person" }));
    await anonymizeText("Hi Joel", s);
    const chunks = ["Hi ", "<PER", "SON_1>", "!"];
    let out = "";
    for await (const c of deanonymizeStream(chunks, s)) out += c;
    expect(out).toBe("Hi Joel!");
  });
});
