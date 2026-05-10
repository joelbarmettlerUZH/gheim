import { defineConfig } from "vite";
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
