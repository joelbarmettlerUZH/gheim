import { ref, shallowRef } from "vue";
import { LocalDetector, type LocalDetectorLoadEvent, type Span } from "gheim";

/* In-browser PII detector — thin reactive wrapper around the gheim SDK's
 * LocalDetector. The SDK does the WebGPU adapter probe + WASM fallback
 * and surfaces a normalized progress event; this composable just maps
 * that into Vue refs and turns the phase strings into UI copy.            */

export type DetectorState =
  | "idle"
  | "loading"
  | "ready"
  | "running"
  | "error"
  | "unsupported";

export type DetectedSpan = Span;

const MODEL_ID = "joelbarmettler/gheim-ch-560m";

function progressLabel(e: LocalDetectorLoadEvent): string {
  switch (e.phase) {
    case "progress": {
      const mb = e.loadedBytes != null ? (e.loadedBytes / 1024 / 1024).toFixed(0) : "—";
      const total = e.totalBytes != null ? (e.totalBytes / 1024 / 1024).toFixed(0) : "—";
      return `Downloading model · ${mb} / ${total} MB`;
    }
    case "ready":
      return "Warming up…";
    case "download":
      return `Fetching ${e.file ?? "weights"}…`;
    case "initiate":
      return `Preparing ${e.file ?? "files"}…`;
    case "done":
      return "Ready.";
    default:
      // Future-proofing: a new phase the SDK adds shouldn't crash the demo.
      return "Loading…";
  }
}

export function useDetector() {
  const state = ref<DetectorState>("idle");
  const progress = ref(0);
  const progressLabelRef = ref<string>("");
  const error = ref<string | null>(null);
  const device = ref<string>("");
  const sdkDetector = shallowRef<LocalDetector | null>(null);

  let loadPromise: Promise<LocalDetector> | null = null;

  async function load(): Promise<LocalDetector> {
    if (sdkDetector.value) return sdkDetector.value;
    if (loadPromise) return loadPromise;
    state.value = "loading";
    progress.value = 0;
    progressLabelRef.value = "Connecting to model…";

    loadPromise = (async () => {
      const det = new LocalDetector({
        model: MODEL_ID,
        // SDK probes WebGPU + falls back to WASM; "auto" dtype picks
        // fp16 on WebGPU (byte-equivalent to fp32) and q8 elsewhere.
        device: "auto",
        dtype: "auto",
        onProgress: (e: LocalDetectorLoadEvent) => {
          if (e.fraction != null) progress.value = e.fraction;
          progressLabelRef.value = progressLabel(e);
          if (e.device) device.value = e.device;
        },
      });
      try {
        await det.load();
        sdkDetector.value = det;
        if (det.actualDevice) device.value = det.actualDevice;
        state.value = "ready";
        progress.value = 1;
        return det;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        error.value = msg;
        state.value = "error";
        throw e;
      }
    })();

    return loadPromise;
  }

  async function detect(text: string): Promise<DetectedSpan[]> {
    if (!text.trim()) return [];
    const det = await load();
    state.value = "running";
    try {
      return await det.detect(text);
    } finally {
      state.value = "ready";
    }
  }

  return {
    state,
    progress,
    progressLabel: progressLabelRef,
    error,
    device,
    load,
    detect,
  };
}
