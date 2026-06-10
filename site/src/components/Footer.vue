<script setup lang="ts">
const links = {
  Library: [
    { label: "GitHub", href: "https://github.com/joelbarmettlerUZH/gheim" },
    { label: "PyPI · gheim", href: "https://pypi.org/project/gheim/" },
    { label: "npm · gheim", href: "https://www.npmjs.com/package/gheim" },
  ],
  "Hugging Face": [
    {
      label: "Model · gheim-ch-560m",
      href: "https://huggingface.co/joelbarmettler/gheim-ch-560m",
    },
    {
      label: "Model · gheim-ch-560m-research",
      href: "https://huggingface.co/joelbarmettler/gheim-ch-560m-research",
    },
    {
      label: "Dataset · gheim-ch-pii-212k",
      href: "https://huggingface.co/datasets/joelbarmettler/gheim-ch-pii-212k",
    },
  ],
  About: [
    {
      label: "Technical report (PDF)",
      href: "/paper.pdf",
    },
    {
      label: "Author",
      href: "https://joelbarmettler.xyz",
    },
    {
      label: "Issues",
      href: "https://github.com/joelbarmettlerUZH/gheim/issues",
    },
    {
      label: "Apache 2.0",
      href: "https://www.apache.org/licenses/LICENSE-2.0",
    },
    {
      label: "Impressum",
      href: "/impressum",
    },
  ],
};
</script>

<template>
  <footer class="section-rule pt-14 pb-10">
    <div class="grid sm:grid-cols-12 gap-6 sm:gap-10">
      <div class="sm:col-span-5">
        <img src="/logo.png" alt="gheim" class="h-7 w-auto mb-5" />
        <p class="text-sm text-ink-soft leading-relaxed max-w-md">
          gheim is a small open-source toolkit for redacting PII before it
          reaches an LLM and restoring it on the way back. Built in Switzerland.
          Apache 2.0 licensed.
        </p>
      </div>

      <div
        v-for="(group, name) in links"
        :key="name"
        class="sm:col-span-2 flex flex-col gap-2"
      >
        <p class="kicker mb-1">{{ name }}</p>
        <a
          v-for="l in group"
          :key="l.href"
          :href="l.href"
          :target="l.href.startsWith('http') ? '_blank' : undefined"
          :rel="l.href.startsWith('http') ? 'noopener' : undefined"
          class="text-sm hover:text-accent transition-colors"
          >{{ l.label }}</a
        >
      </div>
    </div>

    <div
      class="mt-12 pt-5 border-t border-rule flex flex-wrap items-center justify-between gap-4 text-xs text-ink-soft font-mono"
    >
      <span>© {{ new Date().getFullYear() }} LatentSpace Labs GmbH · gheim.ch</span>
      <span>Inference runs on your device. No telemetry.</span>
    </div>
  </footer>
</template>
