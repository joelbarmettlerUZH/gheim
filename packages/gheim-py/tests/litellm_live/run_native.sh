#!/usr/bin/env bash
# Same suite against a pip-installed LiteLLM (no docker), direct import path.
#   run_native.sh <venv-bin-dir>   (venv needs litellm[proxy], openai, pytest)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V="${1:?venv bin dir}"
export PYTHONPATH="$HERE/../../src:$HERE" DISABLE_SCHEMA_UPDATE=true
"$V/python" "$HERE/fake_upstream.py" 9100 & UP=$!
"$V/litellm" --config "$HERE/config_direct.yaml" --port 4101 >"$HERE/native.log" 2>&1 & PX=$!
trap 'kill $UP $PX 2>/dev/null' EXIT
for _ in $(seq 90); do curl -sf http://127.0.0.1:4101/health/liveliness >/dev/null && break; sleep 1; done
"$V/python" -c "import importlib.metadata as m; print('litellm', m.version('litellm'))"
GHEIM_LITELLM_URL=http://127.0.0.1:4101 "$V/python" -m pytest "$HERE/test_live_proxy.py" -p no:cacheprovider
