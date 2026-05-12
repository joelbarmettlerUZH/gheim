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
export { CalibratedDetector } from "./calibrated.ts";
export type { CalibratedDetectorOptions } from "./calibrated.ts";

import type { Detector } from "./base.ts";
import { CalibratedDetector } from "./calibrated.ts";
import { RemoteDetector } from "./remote.ts";

/** Pick a reasonable default: remote when `GHEIM_API_KEY` is set in env,
 *  else a calibrated local model.
 *
 *  The calibrated default (`oBias=0.5`) recovers ~95% of the
 *  multi-entity-letter cases that the raw v1 model fails on, with
 *  ~0.4pp test_v1 strict-F1 cost. See eval/calibration_sweep.json. */
export function defaultDetector(): Detector {
  const hasKey =
    typeof process !== "undefined" && Boolean(process.env?.GHEIM_API_KEY);
  return hasKey ? new RemoteDetector() : new CalibratedDetector();
}
