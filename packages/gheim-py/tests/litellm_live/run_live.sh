#!/usr/bin/env bash
# Live test: fake upstream (host) + official LiteLLM docker image with the gheim
# guardrail mounted from source.
#
#   run_live.sh [image] [python-with-litellm+openai+pytest]
#
# Env: CONFIG (default config.yaml), TESTFILE (default test_live_proxy.py),
#      USE_INSTALLED (skip mounting source), HF_HOME_HOST (mounted as the container's HF cache; for the real model).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$(cd "$HERE/../.." && pwd)"
IMAGE="${1:-ghcr.io/berriai/litellm:v1.83.10-stable}"
PY="${2:-python}"
CONFIG="${CONFIG:-config.yaml}"
TESTFILE="${TESTFILE:-test_live_proxy.py}"
PORT=4100

EXTRA=()
# USE_INSTALLED=1: test the gheim already installed in the image (e.g. from
# PyPI) instead of mounting the source tree.
SRC=(-v "$PKG/src/gheim:/app/gheim:ro")
PYPATH=/app:/cfg
if [ -n "${USE_INSTALLED:-}" ]; then SRC=(); PYPATH=/cfg; fi
if [ -n "${HF_HOME_HOST:-}" ]; then
  EXTRA+=(-v "$HF_HOME_HOST:/hf" -e HF_HOME=/hf)
fi

$PY "$HERE/fake_upstream.py" 9100 & UP=$!
docker rm -f gheim-litellm-live >/dev/null 2>&1 || true
trap 'kill $UP 2>/dev/null; docker logs gheim-litellm-live >"$HERE/proxy.log" 2>&1; docker rm -f gheim-litellm-live >/dev/null 2>&1' EXIT

docker run -d --name gheim-litellm-live --network host \
  "${SRC[@]}" -v "$HERE:/cfg:ro" "${EXTRA[@]}" \
  -e PYTHONPATH=$PYPATH -e DISABLE_SCHEMA_UPDATE=true \
  "$IMAGE" --config "/cfg/$CONFIG" --port $PORT >/dev/null

for _ in $(seq 300); do
  curl -sf "http://127.0.0.1:$PORT/health/liveliness" >/dev/null && break; sleep 1
done
GHEIM_LITELLM_URL="http://127.0.0.1:$PORT" $PY -m pytest "$HERE/$TESTFILE" -v -p no:cacheprovider
