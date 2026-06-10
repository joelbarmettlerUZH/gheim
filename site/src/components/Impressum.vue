<script setup lang="ts">
import { ref, onMounted } from "vue";

// --- Email obfuscation ------------------------------------------------------
// The contact address is legally required (Swiss UWG Art. 3 para. 1 lit. s,
// EU Directive 2000/31/EC Art. 5), but we don't want naive harvesters to scrape
// it. We never emit the literal `user@domain` string into the served HTML or as
// a plain literal in the JS bundle: it is stored base64-encoded AND reversed,
// then decoded at runtime to build both the visible text and the mailto href.
// (A contact form is the only fully scraper-proof option, but needs a backend.)
const EMAIL_TOKEN = "=UWbu0GcAV2YhB3c05WZ0FGb";

const email = ref("");
const mailto = ref("");

onMounted(() => {
  try {
    const decoded = atob(EMAIL_TOKEN.split("").reverse().join(""));
    email.value = decoded;
    mailto.value = "mailto:" + decoded;
  } catch {
    /* leave empty — the static fallback note stays visible */
  }
});

const updated = "10 June 2026";
</script>

<template>
  <main class="mx-auto max-w-3xl px-6 sm:px-10 py-10 sm:py-14">
    <!-- Top bar -->
    <header class="flex items-center justify-between mb-14 sm:mb-20">
      <a href="/" class="flex items-center gap-3">
        <img src="/logo.png" alt="gheim" class="h-7 w-auto" />
      </a>
      <nav class="flex items-center gap-8 kicker">
        <a href="/" class="hover:text-accent transition-colors">← Back to gheim.ch</a>
      </nav>
    </header>

    <p class="kicker mb-4">Legal notice · Impressum</p>
    <h1
      class="font-serif font-medium leading-[1.08] text-[2.2rem] sm:text-[3rem] tracking-[-0.015em] mb-6"
    >
      Impressum &amp;<br />Data protection
    </h1>
    <p class="font-serif text-lg text-ink-soft leading-relaxed max-w-2xl">
      Information provided pursuant to Art. 3 para. 1 lit. s of the Swiss Federal
      Act against Unfair Competition (UWG) and Art. 5 of EU Directive
      2000/31/EC on electronic commerce.
    </p>

    <!-- =========================================================== -->
    <section class="section-rule mt-12 pt-8">
      <p class="kicker mb-4">Site operator · Diensteanbieter</p>
      <div class="font-mono text-sm leading-relaxed text-ink">
        <p class="font-semibold">LatentSpace Labs GmbH</p>
        <p>c/o Joel Barmettler</p>
        <p>Hirschgartnerweg 18</p>
        <p>8057 Zürich</p>
        <p>Switzerland</p>
      </div>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Legal form &amp; commercial register</p>
      <dl class="grid grid-cols-1 sm:grid-cols-[12rem_1fr] gap-x-6 gap-y-2 font-mono text-sm">
        <dt class="text-ink-soft">Legal form</dt>
        <dd>Gesellschaft mit beschränkter Haftung (GmbH) — Swiss limited liability company</dd>

        <dt class="text-ink-soft">Registered office</dt>
        <dd>Zürich, Switzerland</dd>

        <dt class="text-ink-soft">Commercial register</dt>
        <dd>Handelsregister des Kantons Zürich</dd>

        <dt class="text-ink-soft">Enterprise ID (UID)</dt>
        <dd>
          <a
            href="https://www.uid.admin.ch/Detail.aspx?uid_id=CHE170986632"
            target="_blank"
            rel="noopener"
            class="hover:text-accent transition-colors"
            >CHE-170.986.632</a
          >
        </dd>

        <dt class="text-ink-soft">Register ID (CH-ID)</dt>
        <dd>CH-020-4092420-2</dd>

        <dt class="text-ink-soft">EHRA-ID</dt>
        <dd>1753104</dd>

        <dt class="text-ink-soft">VAT (MWST)</dt>
        <dd>Not registered for Swiss VAT. The UID becomes the VAT number once the company is liable for VAT.</dd>
      </dl>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Authorised representative</p>
      <p class="font-mono text-sm leading-relaxed">
        Joel Barmettler, Managing Director (Geschäftsführer), with sole
        signatory authority (Einzelunterschrift).
      </p>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Contact</p>
      <p class="font-mono text-sm leading-relaxed">
        Email:
        <template v-if="email">
          <a :href="mailto" class="hover:text-accent transition-colors">{{ email }}</a>
        </template>
        <template v-else>
          <span class="text-ink-soft">[ enable JavaScript to reveal the address ]</span>
        </template>
      </p>
      <p class="text-xs text-ink-soft mt-2">
        The address is assembled in your browser to deter automated harvesting.
      </p>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Disclaimer · Haftungsausschluss</p>
      <div class="font-serif text-base leading-relaxed text-ink-soft space-y-4 max-w-2xl">
        <p>
          <span class="font-semibold text-ink">Content.</span> The contents of
          this website were created with the greatest possible care. We accept
          no liability, however, for the accuracy, completeness or timeliness of
          the content. The use of the website and its content is at the user's
          own risk.
        </p>
        <p>
          <span class="font-semibold text-ink">External links.</span> This
          website may contain links to third-party websites. We have no
          influence over their content and accept no responsibility for it.
          Responsibility for linked sites lies solely with their respective
          operators.
        </p>
        <p>
          <span class="font-semibold text-ink">Software.</span> The gheim
          software is published under the Apache License 2.0 and is provided
          “as is”, without warranty of any kind. The applicable terms are set
          out in the
          <a
            href="https://www.apache.org/licenses/LICENSE-2.0"
            target="_blank"
            rel="noopener"
            class="text-accent hover:underline"
            >license</a
          >.
        </p>
      </div>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Copyright · Urheberrecht</p>
      <p class="font-serif text-base leading-relaxed text-ink-soft max-w-2xl">
        © {{ new Date().getFullYear() }} LatentSpace Labs GmbH. The texts,
        images and graphics on this website are protected by copyright. The
        gheim source code, models and datasets are licensed separately under the
        terms stated in their respective repositories.
      </p>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Data protection · Datenschutzerklärung</p>
      <div class="font-serif text-base leading-relaxed text-ink-soft space-y-4 max-w-2xl">
        <p>
          This notice explains how personal data is handled on gheim.ch in
          accordance with the Swiss Federal Act on Data Protection (FADP/revDSG)
          and, where applicable, the EU General Data Protection Regulation
          (GDPR, Regulation (EU) 2016/679).
        </p>
        <p>
          <span class="font-semibold text-ink">Controller.</span> The
          controller responsible for data processing is the site operator named
          above. For EU/EEA matters, the controller can be reached at the
          contact address above; no separate EU representative under Art. 27
          GDPR has been appointed, as processing is occasional and low-risk.
        </p>
        <p>
          <span class="font-semibold text-ink">On-device processing.</span> The
          PII-detection demo runs entirely in your browser. The model is
          downloaded to your device and the text you enter is processed locally.
          It is <span class="font-semibold text-ink">never transmitted to us</span>
          or to any server. We operate no analytics, tracking pixels, advertising
          or telemetry, and we set no tracking cookies.
        </p>
        <p>
          <span class="font-semibold text-ink">Hosting &amp; server logs.</span>
          The website is hosted on GitHub Pages (GitHub, Inc., USA). When you
          visit, GitHub automatically processes connection data such as your IP
          address, the requested resource and the time of access in order to
          deliver the site and ensure its security (legal basis: legitimate
          interest, Art. 6(1)(f) GDPR). See GitHub's privacy statement for
          details.
        </p>
        <p>
          <span class="font-semibold text-ink">Web fonts &amp; model files.</span>
          Fonts are loaded from Google Fonts (Google Ireland/LLC) and model
          weights are downloaded from the Hugging Face CDN. Calling these
          services transmits your IP address to the respective provider, which
          may be located outside Switzerland and the EU/EEA. We use them solely
          to present the site and run the on-device demo.
        </p>
        <p>
          <span class="font-semibold text-ink">Your rights.</span> Subject to
          the applicable law, you have the right to access, rectify, delete and
          restrict the processing of your personal data, the right to data
          portability and the right to object. You may also lodge a complaint
          with a supervisory authority — in Switzerland the Federal Data
          Protection and Information Commissioner (FDPIC), or your local data
          protection authority in the EU/EEA. To exercise your rights, contact us
          at the address above.
        </p>
      </div>
    </section>

    <!-- =========================================================== -->
    <section class="section-rule mt-10 pt-8">
      <p class="kicker mb-4">Governing law &amp; dispute resolution</p>
      <div class="font-serif text-base leading-relaxed text-ink-soft space-y-4 max-w-2xl">
        <p>
          These terms and any use of this website are governed by Swiss law, to
          the extent permitted by mandatory consumer-protection provisions of the
          user's country of residence. The exclusive place of jurisdiction is
          Zürich, Switzerland, where legally permissible.
        </p>
        <p>
          We are neither obliged nor willing to participate in dispute
          resolution proceedings before a consumer arbitration board. (Note: the
          EU Online Dispute Resolution platform was discontinued on 20 July 2025
          and is therefore no longer referenced here.)
        </p>
      </div>
    </section>

    <!-- =========================================================== -->
    <footer
      class="section-rule mt-14 pt-6 flex flex-wrap items-center justify-between gap-4 text-xs text-ink-soft font-mono"
    >
      <span>© {{ new Date().getFullYear() }} LatentSpace Labs GmbH · gheim.ch</span>
      <span>Last updated: {{ updated }}</span>
    </footer>
  </main>
</template>
