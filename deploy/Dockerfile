# TianXiMem application image.
#
# Build from the repository root:
#   docker build -f deploy/Dockerfile -t tianximem:local .
#
# Every external download is a build argument so a restricted or mirrored
# network can be served without editing this file:
#   docker build -f deploy/Dockerfile \
#     --build-arg PYTHON_IMAGE=<mirror>/library/python:3.12-slim-bookworm \
#     --build-arg PYPI_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
#     -t tianximem:local .
#
# The image contains no secrets and no runtime state. SQLite, the embedding
# cache and the metric snapshot all live under /data and must be a volume;
# without a volume every container restart loses the source of truth.

# python:3.12-slim-bookworm matches pyproject's ">=3.11,<3.13" and the
# .python-version pin that CI and `uv` resolve against.
ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# ── builder ────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS builder

# `uv` is installed from the Python package index instead of the ghcr.io image
# so the build works on networks where only PyPI is reachable.
ARG UV_VERSION=0.12.15
ARG PYPI_INDEX=https://pypi.org/simple

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_NO_CACHE=1

RUN pip install --index-url "${PYPI_INDEX}" "uv==${UV_VERSION}"

WORKDIR /app

# Dependency layer first: it is cached until pyproject.toml or uv.lock change.
# uv.lock pins https://pypi.org/simple, so the default index must stay the one
# recorded there; pass PYPI_INDEX only when that mirror proxies PyPI verbatim.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

# Warm the tokenizer cache at build time. `tiktoken` downloads the BPE file on
# first use; doing it here keeps the container usable on an offline host and
# turns a network problem into a build failure instead of a first-request 500.
# The check that the cache directory is non-empty is what makes the guarantee
# real: `get_encoding` succeeds only after the file has been written.
ARG TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
RUN mkdir -p "${TIKTOKEN_CACHE_DIR}" \
 && TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR}" \
    .venv/bin/python -c "import tiktoken; assert tiktoken.get_encoding('o200k_base').encode('ok')" \
 && test -n "$(ls -A "${TIKTOKEN_CACHE_DIR}")"

# ── runtime ────────────────────────────────────────────────────────────
FROM ${PYTHON_IMAGE} AS runtime

ARG APP_UID=10001
ARG APP_GID=10001

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    TIANXIMEM_PROFILE=default \
    TIANXIMEM_SQLITE_PATH=/data/tianxi.db \
    TIANXIMEM_EMBED_CACHE_DIR=/data/embed_cache \
    TIANXIMEM_METRICS_PATH=/data/metrics.json \
    TIANXIMEM_WORKERS=1

# A fixed, unprivileged identity: the /data volume must stay writable by the
# same UID across rebuilds, otherwise a fresh image cannot open the database.
RUN groupadd --gid "${APP_GID}" app \
 && useradd --uid "${APP_UID}" --gid "${APP_GID}" --no-create-home \
        --shell /usr/sbin/nologin app \
 && mkdir -p /data/embed_cache \
 && chown -R app:app /data

WORKDIR /app

COPY --from=builder --chown=root:root /app/.venv /app/.venv
COPY --from=builder --chown=root:root /opt/tiktoken-cache /opt/tiktoken-cache
COPY --chown=root:root configs/ /app/configs/

USER app
EXPOSE 8000

# Liveness only: the endpoint deliberately does not probe Qdrant or the model
# gateways, so a dependency blip does not mark a healthy process unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys,httpx; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code == 200 else 1)"]

# `--workers 1` is a correctness constraint, not a tuning knob: the SQLite
# writer and the in-process Qdrant client assume a single process.
ENTRYPOINT ["uvicorn", "tianximem.service.app:create_app_from_env", \
            "--factory", "--workers", "1", \
            "--host", "0.0.0.0", "--port", "8000"]
