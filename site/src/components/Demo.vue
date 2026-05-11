<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue";
import { useDetector, type DetectedSpan } from "../composables/useDetector";
import { substitute } from "../lib/sentinels";

const EXAMPLE = `Sehr geehrte Frau Müller,

vielen Dank für Ihre Anfrage vom 14. März 2026. Bitte überweisen Sie den
Restbetrag von CHF 1'480.00 bis spätestens 28. März 2026 auf unser Konto
bei der Zürcher Kantonalbank: IBAN CH93 0076 2011 6238 5295 7.

Für Rückfragen erreichen Sie mich werktags zwischen 9 und 17 Uhr unter
+41 44 268 12 34 oder per E-Mail an a.mueller@kanzlei-mueller.ch.

Unsere Praxis befindet sich an der Werdstrasse 36, 8004 Zürich.

Freundliche Grüsse,
Dr. Andrea Müller`;

const detector = useDetector();
const input = ref(EXAMPLE);
const spans = ref<DetectedSpan[]>([]);
const view = ref<"redacted" | "sentinel">("redacted");
const requestId = ref(0);
const lastRunMs = ref<number | null>(null);

let debounceHandle: number | undefined;
const dirty = ref(false);

async function run() {
  const id = ++requestId.value;
  const t0 = performance.now();
  try {
    const out = await detector.detect(input.value);
    if (id !== requestId.value) return; // stale
    spans.value = out;
    lastRunMs.value = Math.round(performance.now() - t0);
    dirty.value = false;
  } catch {
    /* error already in detector.error */
  }
}

// Per-keystroke detection saturates the WASM thread on long inputs
// (detection is 0.5–2 s for a typical paragraph), making typing visibly
// laggy. So: mark the input dirty immediately, but only auto-run after
// 1.2 s of typing inactivity. The user can also click "Redact now" to
// run immediately.
watch(input, () => {
  dirty.value = true;
  window.clearTimeout(debounceHandle);
  debounceHandle = window.setTimeout(() => {
    if (detector.state.value === "ready" || detector.state.value === "running") {
      void run();
    }
  }, 1200);
});

async function start() {
  await detector.load();
  await run();
}

function reset() {
  input.value = EXAMPLE;
  void run();
}

onMounted(() => {
  // do not auto-load: the user opts in by clicking "Load model"
});

const sentinelOutput = computed(() => substitute(input.value, spans.value));

interface Segment {
  text: string;
  span?: DetectedSpan;
}

const segments = computed<Segment[]>(() => {
  const text = input.value;
  const sorted = [...spans.value].sort((a, b) => a.start - b.start);
  const out: Segment[] = [];
  let cursor = 0;
  for (const s of sorted) {
    if (s.start < cursor) continue;
    if (s.start > cursor) out.push({ text: text.slice(cursor, s.start) });
    out.push({ text: text.slice(s.start, s.end), span: s });
    cursor = s.end;
  }
  if (cursor < text.length) out.push({ text: text.slice(cursor) });
  return out;
});

const counts = computed(() => {
  const c: Record<string, number> = {};
  for (const s of spans.value) c[s.label] = (c[s.label] ?? 0) + 1;
  return c;
});

const labelDisplay: Record<string, string> = {
  private_person: "Person",
  private_email: "Email",
  private_phone: "Phone",
  private_address: "Address",
  private_date: "Date",
  private_url: "URL",
  account_number: "Account",
  secret: "Secret",
};
</script>

<template>
  <section id="demo" class="section-rule pt-14 sm:pt-20 pb-16 sm:pb-24">
    <div class="grid sm:grid-cols-12 gap-6 sm:gap-10 mb-10">
      <div class="sm:col-span-7">
        <p class="kicker mb-3">§ 01 — Live demo</p>
        <h2 class="font-serif text-3xl sm:text-4xl leading-tight tracking-tight">
          Type something. Watch the model<br />
          <span class="italic text-accent">redact it in real time.</span>
        </h2>
      </div>
      <div class="sm:col-span-5 self-end">
        <p class="text-ink-soft text-sm leading-relaxed">
          Inference runs entirely in your browser via
          <code class="font-mono text-[0.85em]">@huggingface/transformers</code>
          on WebGPU (or WASM). No text leaves this page. The model is fetched
          once from Hugging Face and cached.
        </p>
      </div>
    </div>

    <!-- Loader gate -->
    <div
      v-if="detector.state.value === 'idle'"
      class="border border-rule rounded-md p-8 sm:p-12 bg-paper-dim text-center"
    >
      <p class="kicker mb-3">Model not loaded</p>
      <p class="font-serif text-xl mb-2">
        First run downloads ~280 MB from Hugging Face
      </p>
      <p class="text-ink-soft text-sm mb-6">
        Cached in IndexedDB after that. Roughly 30–60 s on broadband.
      </p>
      <button class="btn btn-primary" @click="start">Load model and run</button>
    </div>

    <div
      v-else-if="detector.state.value === 'loading'"
      class="border border-rule rounded-md p-8 sm:p-12 bg-paper-dim"
    >
      <p class="kicker mb-3">Loading model</p>
      <p class="font-serif text-xl mb-4">{{ detector.progressLabel.value }}</p>
      <div class="w-full h-2 bg-rule rounded-sm overflow-hidden">
        <div
          class="h-full bg-ink transition-[width] duration-300"
          :style="{
            width: `${Math.min(100, Math.max(0, Math.round(detector.progress.value * 100)))}%`,
          }"
        ></div>
      </div>
      <p class="mt-3 text-xs text-ink-soft font-mono">
        {{ Math.min(100, Math.max(0, Math.round(detector.progress.value * 100))) }}%
      </p>
    </div>

    <div
      v-else-if="detector.state.value === 'error'"
      class="border border-accent rounded-md p-8 bg-paper-dim"
    >
      <p class="kicker mb-2 text-accent">Loading failed</p>
      <p class="font-mono text-xs text-ink-soft mb-4 break-all">
        {{ detector.error.value }}
      </p>
      <button class="btn btn-ghost" @click="start">Retry</button>
    </div>

    <!-- Working pane -->
    <div v-else class="grid lg:grid-cols-2 gap-5 items-stretch">
      <!-- INPUT -->
      <div class="border border-rule rounded-md bg-paper flex flex-col">
        <div
          class="flex items-center justify-between px-4 border-b border-rule gap-3 h-12 shrink-0"
        >
          <span class="kicker">Input · plaintext</span>
          <div class="flex items-center gap-3">
            <button
              v-if="dirty"
              class="text-xs font-mono px-2 py-1 rounded-sm bg-ink text-paper hover:opacity-90 cursor-pointer"
              @click="run"
            >
              ↻ Redact now
            </button>
            <button
              class="text-xs font-mono text-ink-soft hover:text-accent cursor-pointer"
              @click="reset"
            >
              ↺ reset example
            </button>
          </div>
        </div>
        <textarea
          v-model="input"
          spellcheck="false"
          class="demo-text demo-text-input w-full px-4 py-3 bg-transparent resize-none outline-none flex-1"
        />
      </div>

      <!-- OUTPUT -->
      <div class="border border-rule rounded-md bg-paper flex flex-col">
        <div
          class="flex items-center justify-between px-4 border-b border-rule gap-3 h-12 shrink-0"
        >
          <span class="kicker">
            <span v-if="view === 'redacted'">Output · redacted view</span>
            <span v-else>Output · what the LLM sees</span>
          </span>
          <div class="flex items-center gap-2">
            <button
              class="text-xs font-mono px-2 py-1 rounded-sm transition-colors cursor-pointer"
              :class="
                view === 'redacted'
                  ? 'bg-ink text-paper'
                  : 'text-ink-soft hover:text-accent'
              "
              @click="view = 'redacted'"
            >
              Bars
            </button>
            <button
              class="text-xs font-mono px-2 py-1 rounded-sm transition-colors cursor-pointer"
              :class="
                view === 'sentinel'
                  ? 'bg-ink text-paper'
                  : 'text-ink-soft hover:text-accent'
              "
              @click="view = 'sentinel'"
            >
              Sentinels
            </button>
          </div>
        </div>
        <div class="px-4 py-3 flex-1 overflow-auto">
          <div
            v-if="view === 'redacted'"
            class="demo-text whitespace-pre-wrap break-words"
          >
            <template v-for="(seg, i) in segments" :key="`${requestId}-${i}`">
              <span
                v-if="seg.span"
                class="redact"
                :data-cat="seg.span.label"
                :title="`${labelDisplay[seg.span.label] ?? seg.span.label} · ${(seg.span.score * 100).toFixed(0)}%`"
                >{{ seg.text }}</span
              ><template v-else>{{ seg.text }}</template>
            </template>
            <span
              v-if="detector.state.value === 'running' && spans.length === 0"
              class="text-ink-soft italic"
              >detecting…</span
            >
          </div>
          <pre
            v-else
            class="demo-text whitespace-pre-wrap break-words m-0"
            >{{ sentinelOutput.redacted }}</pre>
        </div>
      </div>

      <!-- META -->
      <div class="lg:col-span-2 grid sm:grid-cols-3 gap-3 text-xs">
        <div class="border border-rule rounded-sm p-3">
          <div class="kicker mb-1">Detections</div>
          <div v-if="spans.length === 0" class="text-ink-soft font-mono">
            (none)
          </div>
          <div v-else class="flex flex-wrap gap-2 font-mono">
            <span
              v-for="(n, label) in counts"
              :key="label"
              class="inline-flex items-center gap-1.5 px-2 py-0.5 border border-rule rounded-sm"
            >
              <span
                class="inline-block w-2 h-2 rounded-full"
                :style="{
                  background: `var(--color-cat-${(label as string).replace('private_', '').replace('account_number', 'account')})`,
                }"
              ></span>
              {{ labelDisplay[label as string] ?? label }} · {{ n }}
            </span>
          </div>
        </div>
        <div class="border border-rule rounded-sm p-3">
          <div class="kicker mb-1">Backend</div>
          <div class="font-mono">
            {{ detector.device.value || "—" }}
            <span v-if="lastRunMs" class="text-ink-soft">
              · {{ lastRunMs }} ms</span
            >
          </div>
        </div>
        <div class="border border-rule rounded-sm p-3">
          <div class="kicker mb-1">Model</div>
          <a
            href="https://huggingface.co/joelbarmettler/gheim-ch-560m"
            target="_blank"
            rel="noopener"
            class="font-mono text-ink hover:text-accent break-all"
            >joelbarmettler/gheim-ch-560m · q8</a
          >
        </div>
      </div>
    </div>
  </section>
</template>
