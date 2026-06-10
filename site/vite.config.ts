import { defineConfig } from "vite";
import { resolve } from "node:path";
import vue from "@vitejs/plugin-vue";
import tailwind from "@tailwindcss/vite";

// gheim.ch — custom domain → base "/"
export default defineConfig({
  plugins: [vue(), tailwind()],
  base: "/",
  build: {
    target: "es2022",
    sourcemap: false,
    rollupOptions: {
      // Multi-page build: the landing page plus the standalone /impressum/
      // legal notice. Each HTML entry is emitted as a real, deep-linkable
      // file (impressum/index.html → served at /impressum), so the legal
      // page survives a hard refresh without any SPA fallback routing.
      input: {
        main: resolve(__dirname, "index.html"),
        impressum: resolve(__dirname, "impressum/index.html"),
      },
      output: {
        manualChunks: {
          transformers: ["@huggingface/transformers"],
        },
      },
    },
  },
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },
  worker: {
    format: "es",
  },
});
