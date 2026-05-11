/**
 * Optional surface-form normalizers for Session.
 *
 * A normalizer canonicalises one specific category of PII so that
 * trivially-different formats of the same identity collapse to a single
 * sentinel — beyond what NFKC + lowercase + whitespace-collapse alone
 * catches.
 *
 * Examples that the default key does NOT collapse and that an opt-in
 * normalizer here does:
 *   - Phone formats: "+41 44 268 12 34" / "0041 44 268 1234" / "044 268 12 34"
 *     (with region "CH") → all "+41442681234" under {@link e164}.
 *   - Date formats: "1990-01-02" / "Jan 2 1990" → both "1990-01-02"
 *     under {@link isoDate}.
 *
 * Wire it up by passing a `{label: normalizer}` map to Session:
 *
 * ```ts
 * import { Session } from "gheim";
 * import { e164, isoDate } from "gheim/normalizers";
 *
 * const session = new Session({
 *   normalizers: {
 *     private_phone: e164({ region: "CH" }),
 *     private_date:  isoDate(),
 *   },
 * });
 * ```
 *
 * A normalizer returns `null` for un-parseable input; when that happens,
 * the session falls back to the default NFKC+lowercase key so the
 * surface still gets a sentinel — it just won't collapse with format
 * variants.
 *
 * Both `libphonenumber-js` and `chrono-node` are optional peer
 * dependencies. They are loaded synchronously on first use and fail
 * with an actionable error if missing:
 *
 * ```bash
 * bun add libphonenumber-js chrono-node
 * ```
 */

/**
 * A normalizer takes a raw surface and returns a canonical form, or
 * `null` if the surface is unparseable (caller falls back to the
 * default NFKC+lowercase key). Re-exported from the core for ergonomics
 * — the canonical declaration lives in `core/session.ts` so that the
 * main `gheim` entry doesn't pull in `node:module`.
 */
export type { Normalizer } from "./core/session.ts";
type Normalizer = (s: string) => string | null;

/** Options for {@link e164}. */
export interface E164Options {
  /**
   * Default region (ISO 3166-1 alpha-2, e.g. "CH") for phone numbers
   * without an explicit country code. Undefined disables parsing of
   * non-fully-qualified numbers.
   */
  region?: string;
}

interface PhoneLib {
  parsePhoneNumberFromString(
    s: string,
    region?: string,
  ): { isValid: () => boolean; format: (fmt: "E.164") => string } | undefined;
}

// Synchronous loader for optional peer deps via Node's createRequire.
// Lazy-initialised so the module bundles cleanly to CJS (no top-level
// await) and so browser builds that never call a normalizer never
// touch node:module.
import { createRequire } from "node:module";

type RequireFn = (mod: string) => unknown;
let _require: RequireFn | null = null;
let _requireResolved = false;

function _getRequire(): RequireFn | null {
  if (_requireResolved) return _require;
  _requireResolved = true;
  try {
    _require = createRequire(import.meta.url);
  } catch {
    _require = null;
  }
  return _require;
}

function loadSync<T>(name: string, errMsg: string): T {
  const req = _getRequire();
  if (!req) throw new Error(errMsg);
  try {
    return req(name) as T;
  } catch {
    throw new Error(errMsg);
  }
}

function loadPhoneLib(): PhoneLib {
  return loadSync<PhoneLib>(
    "libphonenumber-js",
    "the e164 normalizer requires libphonenumber-js. " +
      "Install with: bun add libphonenumber-js",
  );
}

/**
 * Phone-number canonicaliser via Google's libphonenumber-js port.
 * Returns the E.164 form (+<country><subscriber>) of valid phone
 * numbers; returns `null` for invalid or unparseable input.
 */
export function e164(opts: E164Options = {}): Normalizer {
  // Eagerly load on construction so missing-dep errors surface at
  // factory-call time rather than buried inside Session.allocate.
  const lib = loadPhoneLib();
  return (s: string): string | null => {
    try {
      const parsed = lib.parsePhoneNumberFromString(s, opts.region);
      if (!parsed || !parsed.isValid()) return null;
      return parsed.format("E.164");
    } catch {
      return null;
    }
  };
}

/** Options for {@link isoDate}. */
export interface IsoDateOptions {
  /**
   * Locale name to look up on chrono-node's exported parser objects
   * (e.g. "en", "de", "fr"). Defaults to chrono's universal parser.
   */
  locale?: "en" | "de" | "fr" | "ja" | "nl" | "pt" | "ru" | "uk" | "zh";
}

interface ChronoLike {
  parseDate(s: string): Date | null;
}

function loadChrono(opts: IsoDateOptions): ChronoLike {
  const mod = loadSync<Record<string, ChronoLike> & ChronoLike>(
    "chrono-node",
    "the isoDate normalizer requires chrono-node. " +
      "Install with: bun add chrono-node",
  );
  if (opts.locale) {
    const localeParser = mod[opts.locale];
    if (localeParser) return localeParser;
  }
  return mod;
}

/**
 * Date canonicaliser to ISO-8601 (YYYY-MM-DD) via chrono-node. Returns
 * `null` for input that doesn't parse as a date.
 */
export function isoDate(opts: IsoDateOptions = {}): Normalizer {
  const parser = loadChrono(opts);
  return (s: string): string | null => {
    const d = parser.parseDate(s);
    if (!d) return null;
    // ISO-8601 date portion only (YYYY-MM-DD), stripping time/zone noise.
    return d.toISOString().slice(0, 10);
  };
}

/**
 * German DIN-5007-2 transliteration on top of NFKC + lowercase. Collapses
 * `ä`→`ae`, `ö`→`oe`, `ü`→`ue`, `ß`→`ss` so that German bureaucracy
 * surface variants — `Müller` / `Mueller` / `MÜLLER` / `MUELLER` — collapse
 * to one sentinel. The codepoint `ü` doesn't decompose to `ue` under NFKC,
 * which is why the default key alone misses this case.
 *
 * Pure-JS — no extra dependencies.
 */
export function germanTransliteration(): Normalizer {
  const table: Record<string, string> = {
    ä: "ae", Ä: "ae",
    ö: "oe", Ö: "oe",
    ü: "ue", Ü: "ue",
    ß: "ss",
  };
  return (s: string): string | null => {
    let out = "";
    for (const ch of s.normalize("NFKC")) {
      out += table[ch] ?? ch;
    }
    return out.toLowerCase().split(/\s+/).filter(Boolean).join(" ");
  };
}

const BUILTINS: Record<string, () => Normalizer> = {
  e164: () => e164(),
  iso_date: () => isoDate(),
  german: () => germanTransliteration(),
};

/** Resolve a built-in name or pass through a custom callable. */
export function resolveNormalizer(spec: string | Normalizer): Normalizer {
  if (typeof spec === "function") return spec;
  const factory = BUILTINS[spec];
  if (!factory) {
    throw new Error(
      `unknown built-in normalizer: ${JSON.stringify(spec)}. ` +
        `Available: ${Object.keys(BUILTINS).join(", ")}, or pass a callable.`,
    );
  }
  return factory();
}
