/**
 * End-to-end JS smoke against a running gheim-server.
 * Uses RemoteDetector — no local model load.
 *
 * Run: GHEIM_BASE_URL=http://127.0.0.1:8765 bun experiments/e2e_js.ts
 */
import {
  RemoteDetector,
  Session,
  anonymizeText,
  deanonymizeStream,
  deanonymizeText,
} from "../packages/gheim-js/src/index.ts";

const baseUrl = process.env.GHEIM_BASE_URL ?? "http://127.0.0.1:8765";
console.log(`Server: ${baseUrl}`);

const det = new RemoteDetector({ baseUrl });
const session = new Session();
(session as unknown as { detector: typeof det }).detector = det;

// 1. anonymize
const text = "Hi, my name is Alice and my email is alice@example.com.";
const redacted = await anonymizeText(text, session);
console.log("\n--- anonymize ---");
console.log("original:", text);
console.log("redacted:", redacted);
console.log("mapping :", JSON.stringify(session.mapping, null, 2));

// 2. non-streaming restore using sentinels in a simulated LLM response
const personSentinel = Object.keys(session.mapping).find((k) => k.startsWith("<PERSON_"));
const emailSentinel = Object.keys(session.mapping).find((k) => k.startsWith("<EMAIL_"));
if (!personSentinel) {
  console.error("ERROR: no <PERSON_*> sentinel allocated");
  process.exit(2);
}
const simulated = `Of course, ${personSentinel}, I'd be happy to help. Your email ${emailSentinel ?? "<EMAIL_X>"} is on file.`;
const restored = deanonymizeText(simulated, session);
console.log("\n--- non-streaming restore ---");
console.log("from LLM :", simulated);
console.log("to user  :", restored);
if (restored.includes("<PERSON_") || restored.includes("<EMAIL_")) {
  console.error("ERROR: sentinel leaked through restore");
  process.exit(3);
}

// 3. streaming restore — sentinel split mid-token across chunks
const chunks = ["Of course, ", personSentinel.slice(0, 4), personSentinel.slice(4), "! Happy to help."];
let collected = "";
for await (const c of deanonymizeStream(chunks, session)) collected += c;
console.log("\n--- streaming restore (split sentinel) ---");
console.log("chunks  :", chunks);
console.log("combined:", collected);
if (collected.includes(personSentinel)) {
  console.error("ERROR: split sentinel leaked in streaming path");
  process.exit(4);
}

console.log("\nOK");
