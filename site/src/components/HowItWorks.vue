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
    <div class="mt-10 border border-rule rounded-md bg-paper-dim">
      <div class="px-6 sm:px-8 pt-6 sm:pt-8 pb-3">
        <p class="kicker">Round-trip in one paragraph</p>
      </div>
      <div class="grid sm:grid-cols-4 divide-y sm:divide-y-0 sm:divide-x divide-rule">
        <!-- 01 — what the user types -->
        <div class="p-6 sm:p-7 flex flex-col gap-3">
          <span class="kicker text-ink-soft">01 — User input</span>
          <p class="font-mono text-[0.78rem] leading-relaxed">
            Please draft a polite reminder to <strong class="font-semibold">Joel Barmettler</strong>
            (<strong class="font-semibold">jbarmettler@proton.me</strong>) that
            invoice <strong class="font-semibold">CH9300762011623852957</strong>
            is overdue since <strong class="font-semibold">15 May 2026</strong>.
          </p>
        </div>
        <!-- 02 — what crosses the wire -->
        <div class="p-6 sm:p-7 flex flex-col gap-3">
          <span class="kicker text-ink-soft">02 — Sent to OpenAI</span>
          <p class="font-mono text-[0.78rem] leading-relaxed">
            Please draft a polite reminder to
            <span class="sentinel">&lt;PERSON_1&gt;</span>
            (<span class="sentinel">&lt;EMAIL_1&gt;</span>) that invoice
            <span class="sentinel">&lt;ACCOUNT_1&gt;</span>
            is overdue since
            <span class="sentinel">&lt;DATE_1&gt;</span>.
          </p>
        </div>
        <!-- 03 — LLM response (with sentinels) -->
        <div class="p-6 sm:p-7 flex flex-col gap-3">
          <span class="kicker text-ink-soft">03 — LLM response</span>
          <p class="font-mono text-[0.78rem] leading-relaxed">
            Subject: Friendly reminder. Dear
            <span class="sentinel">&lt;PERSON_1&gt;</span>, this is a friendly
            reminder that invoice
            <span class="sentinel">&lt;ACCOUNT_1&gt;</span>
            has been outstanding since
            <span class="sentinel">&lt;DATE_1&gt;</span>. We will follow up by
            email at <span class="sentinel">&lt;EMAIL_1&gt;</span>.
          </p>
        </div>
        <!-- 04 — restored to the user -->
        <div class="p-6 sm:p-7 flex flex-col gap-3">
          <span class="kicker text-ink-soft">04 — Restored for user</span>
          <p class="font-mono text-[0.78rem] leading-relaxed">
            Subject: Friendly reminder. Dear
            <strong class="font-semibold">Joel Barmettler</strong>, this is a
            friendly reminder that invoice
            <strong class="font-semibold">CH9300762011623852957</strong>
            has been outstanding since
            <strong class="font-semibold">15 May 2026</strong>. We will follow
            up by email at
            <strong class="font-semibold">jbarmettler@proton.me</strong>.
          </p>
        </div>
      </div>
    </div>
  </section>
</template>
