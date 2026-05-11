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
 * default NFKC+lowercase key).
 */
export type Normalizer = (s: string) => string | null;

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

// ESM-safe synchronous loader. createRequire resolves CJS/ESM packages
// from the running module via Node's module resolver. Browsers don't have
// node:module — there the user must pre-bundle the optional dep with their
// own toolchain (esbuild / vite / webpack will resolve the import) or pass
// a custom callable normalizer instead.
let _createRequire: ((url: string) => (mod: string) => unknown) | null = null;
async function _getCreateRequire(): Promise<typeof _createRequire> {
  if (_createRequire) return _createRequire;
  try {
    const m = await import("node:module");
    _createRequire = m.createRequire as never;
  } catch {
    _createRequire = null;
  }
  return _createRequire;
}
// Trigger the load eagerly (top-level await is supported in ESM).
await _getCreateRequire();

function loadSync<T>(name: string, errMsg: string): T {
  if (!_createRequire) throw new Error(errMsg);
  try {
    const req = _createRequire(import.meta.url);
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

const BUILTINS: Record<string, () => Normalizer> = {
  e164: () => e164(),
  iso_date: () => isoDate(),
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
