/** Strict-mode proxy: raises on access to unwrapped endpoints. */

export const STRICT_REASONS: Record<string, string> = {
  beta: (
    "the Assistants API (beta.assistants.*, beta.threads.*) is stateful and " +
    "stores instructions + messages on OpenAI's servers across calls — " +
    "wrapping it correctly requires cross-call session persistence which is " +
    "not yet implemented. OpenAI is steering everyone to the new Responses " +
    "API; consider client.responses.create(...) instead."
  ),
  batches: (
    "the Batch API takes a JSONL file containing nested chat/completions/embeddings " +
    "request bodies; wrapping requires JSONL rewrite + per-line restoration which " +
    "is not yet implemented. Anonymize each line yourself with anonymizeText " +
    "before uploading the batch input file."
  ),
  files: (
    "files.create uploads binary blobs (PDFs, CSVs, audio); document-level " +
    "redaction is outside gheim's scope. Use a dedicated PDF/document redactor."
  ),
  uploads: "uploads.* is the multi-step file upload protocol; same reason as files.*.",
  fineTuning: (
    "fine-tuning sends a training data file by ID — gheim doesn't redact files. " +
    "Pre-anonymize your training data file before uploading."
  ),
  vectorStores: "vector_stores ingest pre-uploaded files; redact those files first.",
  containers: "containers manage server-side execution environments; not in scope.",
};

export class GheimStrictError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GheimStrictError";
  }
}

function strictMessage(attr: string, sub: string, reason: string): string {
  return (
    `gheim/openai blocks client.${attr}.${sub} in strict mode: ${reason}\n` +
    `To call it anyway with raw PII flowing to OpenAI, use ` +
    `client.raw.${attr}.${sub}(...). Or construct the client with ` +
    `gheimStrict: false to silence this check (one-time warning).`
  );
}

/**
 * Build a Proxy that throws on any property access or call. Each level of
 * sub-property nesting (client.beta.assistants.create) gets its own raise so
 * the error message names the leaf operation.
 */
export function makeStrictProxy(attrName: string, reason: string, path: string[] = []): unknown {
  return new Proxy(function () {}, {
    get(_target, prop) {
      if (typeof prop === "symbol") return undefined;
      // Allow Promise-detection probes through; never block these.
      if (prop === "then" || prop === "catch") return undefined;
      const newPath = [...path, prop];
      return makeStrictProxy(attrName, reason, newPath);
    },
    apply() {
      const sub = path.join(".");
      throw new GheimStrictError(strictMessage(attrName, sub || "*", reason));
    },
  });
}

const warnedAttrs = new Set<string>();

/** Build a forwarder that warns once per attribute then forwards to the inner object. */
export function makeWarningProxy<T extends object>(
  attrName: string,
  inner: T,
  reason: string,
): T {
  return new Proxy(inner, {
    get(target, prop, receiver) {
      if (!warnedAttrs.has(attrName)) {
        warnedAttrs.add(attrName);
        // eslint-disable-next-line no-console
        console.warn(
          `gheim/openai: client.${attrName}.* sends data to OpenAI WITHOUT ` +
            `gheim PII redaction. ${reason}`,
        );
      }
      return Reflect.get(target, prop, receiver);
    },
  });
}

export function _resetWarnings(): void {
  warnedAttrs.clear();
}
