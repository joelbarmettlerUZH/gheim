<script setup lang="ts">
const steps = [
  {
    n: "01",
    title: "Detect",
    body: "A token-classification model scans the prompt for names, addresses, phones, IBANs, dates, URLs, emails, and secrets across five languages.",
  },
  {
    n: "02",
    title: "Substitute",
    body: "Each span is replaced with a stable sentinel — <PERSON_1>, <EMAIL_1>, <ACCOUNT_1> — that the LLM treats as opaque tokens.",
  },
  {
    n: "03",
    title: "Send",
    body: "The redacted prompt goes to whichever LLM you use. Streaming, function calls, multi-turn chat — all transparent.",
  },
  {
    n: "04",
    title: "Restore",
    body: "On the way back, every sentinel in the response is rewritten to the original value. Works mid-stream so the user sees Joel, never <PERSON_1>.",
  },
];
</script>

<template>
  <section id="how" class="section-rule pt-14 sm:pt-20 pb-16 sm:pb-24">
    <div class="grid sm:grid-cols-12 gap-6 sm:gap-10 mb-10">
      <div class="sm:col-span-5">
        <p class="kicker mb-3">§ 02 — How it works</p>
        <h2 class="font-serif text-3xl sm:text-4xl leading-tight tracking-tight">
          Four moves between you<br />
          and any <span class="italic text-accent">remote model.</span>
        </h2>
      </div>
      <div class="sm:col-span-7 self-end">
        <p class="text-ink-soft text-base leading-relaxed">
          gheim is framework-agnostic. It plugs into the OpenAI Python and Node
          SDKs as a drop-in client, and it works against any HTTP LLM you call
          yourself. The detection model is yours to swap — the Swiss-tuned
          <code class="font-mono text-[0.85em]">gheim-ch-560m</code> is the
          default, and any <code class="font-mono text-[0.85em]">openai/privacy-filter</code>
          -compatible token-classification model drops in.
        </p>
      </div>
    </div>

    <ol class="grid sm:grid-cols-2 lg:grid-cols-4 gap-px bg-rule rounded-sm overflow-hidden border border-rule">
      <li v-for="step in steps" :key="step.n" class="bg-paper p-6 flex flex-col gap-3">
        <span class="kicker text-accent">{{ step.n }}</span>
        <h3 class="font-serif text-xl">{{ step.title }}</h3>
        <p class="text-sm text-ink-soft leading-relaxed">{{ step.body }}</p>
      </li>
    </ol>

    <!-- Visual flow strip -->
    <div class="mt-10 border border-rule rounded-md bg-paper-dim p-6 sm:p-8">
      <p class="kicker mb-4">Round-trip in one paragraph</p>
      <div class="font-mono text-sm leading-loose break-words">
        <span>Hi, my name is </span
        ><span class="redact" data-static="true" data-cat="private_person"
          >Joel</span
        ><span> and my email is </span
        ><span class="redact" data-static="true" data-cat="private_email"
          >alice@example.ch</span
        ><span>.</span>
        <span class="text-ink-soft mx-2">→</span>
        <span class="sentinel">&lt;PERSON_1&gt;</span>
        <span class="text-ink-soft">·</span>
        <span class="sentinel">&lt;EMAIL_1&gt;</span>
        <span class="text-ink-soft mx-2">→ LLM →</span>
        <span class="sentinel">&lt;PERSON_1&gt;</span>,
        <span class="sentinel">&lt;EMAIL_1&gt;</span>
        <span class="text-ink-soft mx-2">→</span>
        <span>Joel, alice@example.ch</span>
      </div>
    </div>
  </section>
</template>
