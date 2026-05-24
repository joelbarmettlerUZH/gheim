// Stages paper/paper.pdf into site/public/ so Vite serves it at /paper.pdf.
// Runs from pre{dev,build,preview} hooks; warns instead of failing when the
// PDF is missing (e.g. a fresh checkout that hasn't built the LaTeX yet) so
// the site can still serve everything else.
import { copyFile, mkdir, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../../paper/paper.pdf");
const dst = resolve(here, "../public/paper.pdf");

try {
  const s = await stat(src);
  await mkdir(dirname(dst), { recursive: true });
  await copyFile(src, dst);
  const kb = (s.size / 1024).toFixed(0);
  console.log(`[copy-paper] ${src} → public/paper.pdf (${kb} KB)`);
} catch (err) {
  if (err.code === "ENOENT") {
    console.warn(`[copy-paper] WARN: ${src} not found — /paper.pdf will 404. ` +
      "Build the LaTeX (tectonic paper/paper.tex) and re-run.");
  } else {
    throw err;
  }
}
