# Multi-stage build: bakes the gheim-ch-560m weights into the image so the
# container works in air-gapped environments with no runtime HuggingFace downloads.
# To serve another model, rebuild with --build-arg GHEIM_DEFAULT_MODEL=<hf-id>; the same
# value becomes the runtime default, so the baked weights and the served model always match.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --upgrade pip && pip install --prefix=/install .

# Pre-download the model so the runtime image ships with weights baked in.
ARG GHEIM_DEFAULT_MODEL=joelbarmettler/gheim-ch-560m
ENV PYTHONPATH=/install/lib/python3.12/site-packages
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${GHEIM_DEFAULT_MODEL}')"

# ---- runtime ----
FROM python:3.12-slim

# Re-declared: an ARG does not cross a stage boundary. Without this the baked model and the
# default the server serves (registry.py) could differ.
ARG GHEIM_DEFAULT_MODEL=joelbarmettler/gheim-ch-560m
ENV GHEIM_DEFAULT_MODEL=${GHEIM_DEFAULT_MODEL}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8080

WORKDIR /app

COPY --from=builder /install /usr/local
COPY --from=builder /opt/hf /opt/hf

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health').status==200 else 1)" || exit 1

ENTRYPOINT ["gheim-server"]
