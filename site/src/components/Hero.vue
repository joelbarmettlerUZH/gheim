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
          href="/paper.pdf"
          target="_blank"
          rel="noopener"
          class="hover:text-accent transition-colors"
        >Paper</a>
        <a
          href="https://github.com/joelbarmettlerUZH/gheim"
          target="_blank"
          rel="noopener"
          class="inline-flex items-center gap-1.5 hover:text-accent transition-colors"
        >
          <!-- GitHub mark -->
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 16 16"
            class="w-4 h-4 fill-current"
            aria-hidden="true"
          >
            <path
              d="M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82a7.43 7.43 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
            />
          </svg>
          GitHub
        </a>
      </nav>
    </header>

    <!-- Headline -->
    <div class="grid sm:grid-cols-12 gap-6 sm:gap-10 items-end">
      <div class="sm:col-span-8">
        <p class="kicker mb-5">PII round-trip · Apache 2.0 · runs on your device</p>
        <h1
          class="font-serif font-medium leading-[1.05] text-[2.6rem] sm:text-[4.2rem] tracking-[-0.015em]"
        >
          <span class="redact" data-static="true" aria-label="redacted">Anonymise</span><br />
          PII before it leaves<br />
          your <span class="italic text-accent">device.</span>
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
        <div class="font-serif text-3xl">0.910</div>
        <div class="text-xs text-ink-soft mt-1">P 0.890 · R 0.932 · char 0.946</div>
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
