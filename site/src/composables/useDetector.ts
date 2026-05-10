import { ref, shallowRef } from "vue";

/* -----------------------------------------------------------
 * In-browser PII detector via @huggingface/transformers.
 *
 * The 560M xlm-roberta model is exported as int8 (~280–560 MB).
 * transformers.js fetches the ONNX from HF Hub on first load and
 * caches it in IndexedDB; subsequent visits load instantly.
 *
 * This composable returns reactive state plus a `detect(text)` fn.
 * ----------------------------------------------------------- */

export type DetectorState =
  | "idle"
  | "loading"
  | "ready"
  | "running"
  | "error"
  | "unsupported";

export interface DetectedSpan {
  start: number;
  end: number;
  label: string;
  text: string;
  score: number;
}

interface ProgressInfo {
  status: string;
  file?: string;
  progress?: number;
  loaded?: number;
  total?: number;
}

const MODEL_ID = "joelbarmettler/gheim-ch-560m";

export function useDetector() {
  const state = ref<DetectorState>("idle");
  const progress = ref(0);
  const progressLabel = ref<string>("");
  const error = ref<string | null>(null);
  const device = ref<string>("");
  const pipe = shallowRef<unknown>(null);

  let loadPromise: Promise<unknown> | null = null;

  async function load() {
    if (pipe.value) return pipe.value;
    if (loadPromise) return loadPromise;
    state.value = "loading";
    progress.value = 0;
    progressLabel.value = "Connecting to model…";

    loadPromise = (async () => {
      // Dynamic import so transformers.js is split out of the main chunk
      const tx = await import("@huggingface/transformers");

      // Try webgpu, fall back to wasm
      let preferredDevice = "wasm";
      if (
        typeof navigator !== "undefined" &&
        (navigator as { gpu?: unknown }).gpu
      ) {
        preferredDevice = "webgpu";
      }

      try {
        const built = await tx.pipeline("token-classification", MODEL_ID, {
          device: preferredDevice as "webgpu" | "wasm",
          dtype: "q8",
          progress_callback: (p: ProgressInfo) => {
            if (p.status === "progress" && p.progress != null) {
              progress.value = p.progress;
              const mb =
                p.loaded != null ? (p.loaded / 1024 / 1024).toFixed(0) : "—";
              const total =
                p.total != null ? (p.total / 1024 / 1024).toFixed(0) : "—";
              progressLabel.value = `Downloading model · ${mb} / ${total} MB`;
            } else if (p.status === "ready") {
              progressLabel.value = "Warming up…";
            } else if (p.status === "download") {
              progressLabel.value = `Fetching ${p.file ?? "weights"}…`;
            } else if (p.status === "initiate") {
              progressLabel.value = `Preparing ${p.file ?? "files"}…`;
            } else if (p.status === "done") {
              progressLabel.value = "Ready.";
            }
          },
        });
        pipe.value = built;
        device.value = preferredDevice;
        state.value = "ready";
        progress.value = 1;
        return built;
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        error.value = msg;
        if (/webgpu/i.test(msg) && preferredDevice === "webgpu") {
          // Retry on WASM
          progressLabel.value = "WebGPU unavailable, retrying with WASM…";
          try {
            const built = await tx.pipeline("token-classification", MODEL_ID, {
              device: "wasm" as const,
              dtype: "q8",
            });
            pipe.value = built;
            device.value = "wasm";
            state.value = "ready";
            error.value = null;
            return built;
          } catch (e2) {
            const m2 = e2 instanceof Error ? e2.message : String(e2);
            error.value = m2;
            state.value = "error";
            throw e2;
          }
        }
        state.value = "error";
        throw e;
      }
    })();

    return loadPromise;
  }

  async function detect(text: string): Promise<DetectedSpan[]> {
    if (!text.trim()) return [];
    const p = await load();
    state.value = "running";
    type RawItem = {
      start?: number;
      end?: number;
      word?: string;
      entity_group?: string;
      entity?: string;
      score?: number;
    };
    type PipeFn = (
      input: string,
      opts?: { aggregation_strategy?: string },
    ) => Promise<RawItem[]>;
    const items = await (p as PipeFn)(text, { aggregation_strategy: "simple" });

    const spans: DetectedSpan[] = [];
    for (const it of items) {
      const start = it.start ?? -1;
      const end = it.end ?? -1;
      if (start < 0 || end <= start) continue;
      const label = (it.entity_group ?? it.entity ?? "").replace(/^[BIES]-/, "");
      if (!label || label === "O") continue;
      spans.push({
        start,
        end,
        label,
        text: text.slice(start, end),
        score: it.score ?? 0,
      });
    }
    spans.sort((a, b) => a.start - b.start);
    state.value = "ready";
    return spans;
  }

  return {
    state,
    progress,
    progressLabel,
    error,
    device,
    load,
    detect,
  };
}
