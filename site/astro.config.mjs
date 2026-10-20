import { defineConfig } from "astro/config";
import vue from "@astrojs/vue";
import sitemap from "@astrojs/sitemap";
import tailwind from "@tailwindcss/vite";

// gheim.ch — custom domain → base "/". Fully static output: every page is
// pre-rendered to HTML at build time so crawlers see real content. The only
// client-side JS is the Vue island for the in-browser demo.
export default defineConfig({
  site: "https://gheim.ch",
  output: "static",
  integrations: [vue(), sitemap()],
  vite: {
    plugins: [tailwind()],
    build: {
      target: "es2022",
    },
    optimizeDeps: {
      exclude: ["@huggingface/transformers"],
    },
    worker: {
      format: "es",
    },
  },
});
