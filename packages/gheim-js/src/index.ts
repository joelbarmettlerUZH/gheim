/**
 * gheim — PII round-trip for LLM APIs.
 *
 * Public API:
 *   - Session, Span, Detector
 *   - LocalDetector (planned v0.2), RemoteDetector
 *   - anonymizeText / deanonymizeText / anonymizeMessages / deanonymizeStream
 *   - StreamDeanonymizer (low-level)
 */
export { Session } from "./core/session.ts";
export type { Normalizer, SessionOptions } from "./core/session.ts";
export {
  LABEL_TO_TAG,
  SENTINEL_RE,
  isPossibleSentinelPrefix,
  labelTag,
  parseSentinel,
  renderSentinel,
} from "./core/sentinels.ts";
export type { Sentinel } from "./core/sentinels.ts";
export { StreamDeanonymizer } from "./core/stream.ts";
export type { Detector, Span } from "./detectors/base.ts";
export {
  DEFAULT_BASE_URL,
  REMOTE_DEFAULT_MODEL,
  RemoteDetector,
  LocalDetector,
  CompositeDetector,
  defaultDetector,
} from "./detectors/index.ts";
export type {
  RemoteDetectorOptions,
  LocalDetectorOptions,
  LocalDetectorLoadEvent,
  CompositeDetectorOptions,
} from "./detectors/index.ts";
export {
  anonymizeMessages,
  anonymizeText,
  deanonymizeStream,
  deanonymizeText,
} from "./plain.ts";
export type { ChatMessage } from "./plain.ts";

// Note: gheim/openai is exposed as a separate subpath export.
// Import from "gheim/openai" to use OpenAI-typed helpers.

export const VERSION = "0.1.3";

export { mergeAdjacent } from "./core/session.ts";
export type { MergedSpan } from "./core/session.ts";
