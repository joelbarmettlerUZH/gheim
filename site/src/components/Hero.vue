<script setup lang="ts">
function scrollTo(id: string) {
  document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
}
</script>

<template>
  <section class="pt-4 pb-16 sm:pb-24">
    <!-- Top bar -->
    <header class="flex items-center justify-between mb-16 sm:mb-24">
      <a href="/" class="flex items-center gap-3">
        <img src="/logo.png" alt="gheim" class="h-7 w-auto" />
      </a>
      <nav class="hidden sm:flex items-center gap-8 kicker">
        <a href="#demo" class="hover:text-accent transition-colors">Demo</a>
        <a href="#how" class="hover:text-accent transition-colors">How</a>
        <a href="#install" class="hover:text-accent transition-colors">Install</a>
        <a
          href="https://github.com/joelbarmettlerUZH/gheim"
          target="_blank"
          rel="noopener"
          class="hover:text-accent transition-colors"
          >GitHub</a
        >
      </nav>
    </header>

    <!-- Headline -->
    <div class="grid sm:grid-cols-12 gap-6 sm:gap-10 items-end">
      <div class="sm:col-span-8">
        <p class="kicker mb-5">PII round-trip · MIT-licensed · runs in your browser</p>
        <h1
          class="font-serif font-medium leading-[1.05] text-[2.6rem] sm:text-[4.2rem] tracking-[-0.015em]"
        >
          Anonymise<br />
          <span class="redact" data-static="true" aria-label="redacted">PII</span>
          before it leaves<br />
          your <span class="italic text-accent">browser.</span>
        </h1>
        <p
          class="mt-7 max-w-2xl font-serif text-lg sm:text-xl text-ink-soft leading-relaxed"
        >
          gheim detects names, addresses, phones, IBANs and more across the four
          official Swiss languages plus English, swaps them for stable sentinels
          before you call any LLM, and restores the originals when the response
          comes back — even mid-stream.
        </p>
      </div>
      <div class="sm:col-span-4 flex sm:justify-end">
        <div class="flex flex-col gap-3 sm:items-end w-full sm:w-auto">
          <button
            class="btn btn-primary w-full sm:w-auto justify-center"
            @click="scrollTo('demo')"
          >
            Try the demo →
          </button>
          <a
            href="https://github.com/joelbarmettlerUZH/gheim"
            target="_blank"
            rel="noopener"
            class="btn btn-ghost w-full sm:w-auto justify-center"
            >Read on GitHub</a
          >
        </div>
      </div>
    </div>

    <!-- Stat strip -->
    <div
      class="mt-20 sm:mt-28 grid grid-cols-2 sm:grid-cols-4 gap-px bg-rule rounded-sm overflow-hidden border border-rule"
    >
      <div class="bg-paper p-5">
        <div class="kicker mb-1.5">Test F1</div>
        <div class="font-serif text-3xl">0.916</div>
      </div>
      <div class="bg-paper p-5">
        <div class="kicker mb-1.5">Languages</div>
        <div class="font-serif text-3xl">5</div>
        <div class="text-xs text-ink-soft mt-1">de_CH · fr_CH · it_CH · rm · en</div>
      </div>
      <div class="bg-paper p-5">
        <div class="kicker mb-1.5">Categories</div>
        <div class="font-serif text-3xl">8</div>
        <div class="text-xs text-ink-soft mt-1">person · address · email · …</div>
      </div>
      <div class="bg-paper p-5">
        <div class="kicker mb-1.5">Parameters</div>
        <div class="font-serif text-3xl">560M</div>
        <div class="text-xs text-ink-soft mt-1">xlm-roberta-large</div>
      </div>
    </div>
  </section>
</template>
