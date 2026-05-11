export type { Detector, Span } from "./base.ts";
export {
  DEFAULT_BASE_URL,
  DEFAULT_MODEL as REMOTE_DEFAULT_MODEL,
  RemoteDetector,
} from "./remote.ts";
export type { RemoteDetectorOptions } from "./remote.ts";
export { LocalDetector } from "./local.ts";
export type { LocalDetectorOptions, LocalDetectorLoadEvent } from "./local.ts";
export { CompositeDetector } from "./composite.ts";
export type { CompositeDetectorOptions } from "./composite.ts";

import type { Detector } from "./base.ts";
import { LocalDetector } from "./local.ts";
import { RemoteDetector } from "./remote.ts";

/** Pick a reasonable default: remote if `GHEIM_API_KEY` is set in env, else local. */
export function defaultDetector(): Detector {
  const hasKey =
    typeof process !== "undefined" && Boolean(process.env?.GHEIM_API_KEY);
  return hasKey ? new RemoteDetector() : new LocalDetector();
}
