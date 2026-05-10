<script setup lang="ts">
import { ref } from "vue";

type Tab = "python" | "node" | "browser";
const tab = ref<Tab>("python");

const tabs: { id: Tab; label: string; sub: string }[] = [
  { id: "python", label: "Python", sub: "openai SDK · drop-in" },
  { id: "node", label: "Node / Bun", sub: "openai SDK · drop-in" },
  { id: "browser", label: "Browser", sub: "transformers.js · in-process" },
];

function copy(id: string) {
  const el = document.getElementById(id);
  if (!el) return;
  navigator.clipboard.writeText(el.innerText).catch(() => {});
}
</script>

<template>
  <section id="install" class="section-rule pt-14 sm:pt-20 pb-16 sm:pb-24">
    <div class="grid sm:grid-cols-12 gap-6 sm:gap-10 mb-10">
      <div class="sm:col-span-7">
        <p class="kicker mb-3">§ 03 — Install</p>
        <h2 class="font-serif text-3xl sm:text-4xl leading-tight tracking-tight">
          Three minutes from<br />
          <span class="italic text-accent">pip install</span> to redacted prompts.
        </h2>
      </div>
      <div class="sm:col-span-5 self-end">
        <p class="text-ink-soft text-sm leading-relaxed">
          The Python and Node packages ship a drop-in replacement for the
          official OpenAI client. Anonymisation and round-trip restoration
          happen automatically — including in streamed responses.
        </p>
      </div>
    </div>

    <!-- Tab strip -->
    <div class="flex gap-px bg-rule rounded-sm overflow-hidden border border-rule mb-3">
      <button
        v-for="t in tabs"
        :key="t.id"
        class="flex-1 px-4 py-3 text-left transition-colors"
        :class="
          tab === t.id
            ? 'bg-ink text-paper'
            : 'bg-paper text-ink hover:bg-paper-dim'
        "
        @click="tab = t.id"
      >
        <div class="kicker mb-0.5" :class="tab === t.id ? 'text-paper' : ''">
          {{ t.label }}
        </div>
        <div
          class="text-xs font-mono"
          :class="tab === t.id ? 'text-paper-dim' : 'text-ink-soft'"
        >
          {{ t.sub }}
        </div>
      </button>
    </div>

    <!-- Code panes -->
    <div v-show="tab === 'python'">
      <div class="grid lg:grid-cols-2 gap-3">
        <div class="border border-rule rounded-md bg-paper">
          <div class="flex items-center justify-between px-4 py-2 border-b border-rule">
            <span class="kicker">Install</span>
            <button class="text-xs font-mono text-ink-soft hover:text-accent" @click="copy('py-install')">copy</button>
          </div>
          <pre id="py-install" class="codeblock m-0 rounded-none rounded-b-md"><span class="tok-comment"># Core + drop-in OpenAI wrapper</span>
uv add <span class="tok-string">"gheim[openai]"</span>

<span class="tok-comment"># Add on-device detection (torch + transformers)</span>
uv add <span class="tok-string">"gheim[local,openai]"</span></pre>
        </div>

        <div class="border border-rule rounded-md bg-paper">
          <div class="flex items-center justify-between px-4 py-2 border-b border-rule">
            <span class="kicker">Drop-in OpenAI client</span>
            <button class="text-xs font-mono text-ink-soft hover:text-accent" @click="copy('py-code')">copy</button>
          </div>
          <pre id="py-code" class="codeblock m-0 rounded-none rounded-b-md"><span class="tok-keyword">from</span> gheim.openai <span class="tok-keyword">import</span> OpenAI

client = <span class="tok-fn">OpenAI</span>()  <span class="tok-comment"># same args as openai.OpenAI</span>

r = client.chat.completions.<span class="tok-fn">create</span>(
    model=<span class="tok-string">"gpt-4o"</span>,
    messages=[{<span class="tok-string">"role"</span>: <span class="tok-string">"user"</span>,
                <span class="tok-string">"content"</span>: <span class="tok-string">"Hi, my name is Joel"</span>}],
)

<span class="tok-comment"># r.choices[0].message.content contains "Joel".</span>
<span class="tok-comment"># OpenAI only ever saw &lt;PERSON_1&gt;.</span></pre>
        </div>
      </div>
    </div>

    <div v-show="tab === 'node'">
      <div class="grid lg:grid-cols-2 gap-3">
        <div class="border border-rule rounded-md bg-paper">
          <div class="flex items-center justify-between px-4 py-2 border-b border-rule">
            <span class="kicker">Install</span>
            <button class="text-xs font-mono text-ink-soft hover:text-accent" @click="copy('node-install')">copy</button>
          </div>
          <pre id="node-install" class="codeblock m-0 rounded-none rounded-b-md"><span class="tok-comment"># Server-side / Bun / Node 18+</span>
npm install gheim openai

<span class="tok-comment"># Optional: on-device detection in Node</span>
npm install <span class="tok-string">@huggingface/transformers</span></pre>
        </div>

        <div class="border border-rule rounded-md bg-paper">
          <div class="flex items-center justify-between px-4 py-2 border-b border-rule">
            <span class="kicker">Drop-in OpenAI client</span>
            <button class="text-xs font-mono text-ink-soft hover:text-accent" @click="copy('node-code')">copy</button>
          </div>
          <pre id="node-code" class="codeblock m-0 rounded-none rounded-b-md"><span class="tok-keyword">import</span> { OpenAI } <span class="tok-keyword">from</span> <span class="tok-string">"gheim/openai"</span>;

<span class="tok-keyword">const</span> client = <span class="tok-keyword">new</span> <span class="tok-fn">OpenAI</span>();

<span class="tok-keyword">const</span> r = <span class="tok-keyword">await</span> client.chat.completions.<span class="tok-fn">create</span>({
  model: <span class="tok-string">"gpt-4o"</span>,
  messages: [{ role: <span class="tok-string">"user"</span>, content: <span class="tok-string">"Hi, my name is Joel"</span> }],
});

<span class="tok-comment">// r.choices[0].message.content contains "Joel".</span>
<span class="tok-comment">// OpenAI only ever saw &lt;PERSON_1&gt;.</span></pre>
        </div>
      </div>
    </div>

    <div v-show="tab === 'browser'">
      <div class="grid lg:grid-cols-2 gap-3">
        <div class="border border-rule rounded-md bg-paper">
          <div class="flex items-center justify-between px-4 py-2 border-b border-rule">
            <span class="kicker">Install</span>
            <button class="text-xs font-mono text-ink-soft hover:text-accent" @click="copy('br-install')">copy</button>
          </div>
          <pre id="br-install" class="codeblock m-0 rounded-none rounded-b-md"><span class="tok-comment"># Bundle for the browser</span>
npm install gheim <span class="tok-string">@huggingface/transformers</span></pre>
        </div>

        <div class="border border-rule rounded-md bg-paper">
          <div class="flex items-center justify-between px-4 py-2 border-b border-rule">
            <span class="kicker">In-browser detection · WebGPU</span>
            <button class="text-xs font-mono text-ink-soft hover:text-accent" @click="copy('br-code')">copy</button>
          </div>
          <pre id="br-code" class="codeblock m-0 rounded-none rounded-b-md"><span class="tok-keyword">import</span> { Session, LocalDetector,
         anonymizeText, deanonymizeText } <span class="tok-keyword">from</span> <span class="tok-string">"gheim"</span>;

<span class="tok-keyword">const</span> session = <span class="tok-keyword">new</span> <span class="tok-fn">Session</span>();
session.detector = <span class="tok-keyword">new</span> <span class="tok-fn">LocalDetector</span>({
  model: <span class="tok-string">"joelbarmettler/gheim-ch-560m"</span>,
  device: <span class="tok-string">"webgpu"</span>,
  dtype:  <span class="tok-string">"q8"</span>,
});

<span class="tok-keyword">const</span> clean = <span class="tok-keyword">await</span> <span class="tok-fn">anonymizeText</span>(<span class="tok-string">"Hi, my name is Joel"</span>, session);
<span class="tok-comment">// → "Hi, my name is &lt;PERSON_1&gt;"</span>
<span class="tok-comment">// ...send `clean` to any LLM endpoint, then:</span>
<span class="tok-keyword">const</span> final = <span class="tok-fn">deanonymizeText</span>(responseText, session);</pre>
        </div>
      </div>
    </div>

    <p class="mt-6 text-xs text-ink-soft font-mono">
      Full reference:
      <a
        href="https://github.com/joelbarmettlerUZH/gheim#readme"
        target="_blank"
        rel="noopener"
        class="underline hover:text-accent"
        >README on GitHub</a
      >.
    </p>
  </section>
</template>
