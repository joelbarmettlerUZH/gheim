"""End-to-end Python smoke test against a running gheim-server.

Uses RemoteDetector (no local model load), demonstrates full round-trip.
"""
from __future__ import annotations

import json
import os
import sys

from gheim import RemoteDetector, Session, anonymize_text, deanonymize_stream, deanonymize_text


def main() -> int:
    base_url = os.environ.get("GHEIM_BASE_URL", "http://127.0.0.1:8765")
    det = RemoteDetector(base_url=base_url, api_key=None)

    print(f"Server: {base_url}")

    # 1. anonymize
    session = Session(detector=det)
    text = "Hi, my name is Alice and my email is alice@example.com."
    redacted = anonymize_text(text, session)
    print("\n--- anonymize ---")
    print("original:", text)
    print("redacted:", redacted)
    print("mapping :", json.dumps(session.mapping, indent=2))

    # 2. simulate an LLM response that references the sentinel
    person_sentinel = next(iter(k for k in session.mapping if k.startswith("<PERSON_")), None)
    if not person_sentinel:
        print("ERROR: no <PERSON_*> sentinel allocated — detection broken?", file=sys.stderr)
        return 2
    simulated_response = f"Of course, {person_sentinel}, I'd be happy to help. Your email {list(session.mapping)[-1]} is on file."

    # 3. one-shot deanonymize
    restored = deanonymize_text(simulated_response, session)
    print("\n--- non-streaming restore ---")
    print("from LLM :", simulated_response)
    print("to user  :", restored)

    if person_sentinel in restored or "<EMAIL_" in restored:
        print("ERROR: sentinels leaked through restore", file=sys.stderr)
        return 3

    # 4. streaming deanonymize — sentinel split across chunks
    chunks = [
        "Of course, ",
        person_sentinel[:4],
        person_sentinel[4:],
        "! Happy to help.",
    ]
    streamed = "".join(deanonymize_stream(chunks, session))
    print("\n--- streaming restore (split sentinel) ---")
    print("chunks  :", chunks)
    print("combined:", streamed)

    if person_sentinel in streamed:
        print("ERROR: split sentinel leaked in streaming path", file=sys.stderr)
        return 4

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
