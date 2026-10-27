# TianXiMem application image.
# Build from the repository root:
#   docker build -f deploy/Dockerfile -t tianximem:local .

FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

ARG UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_DEFAULT_INDEX=${UV_DEFAULT_INDEX}

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
RUN uv sync --frozen --no-dev --no-editable

ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache
RUN .venv/bin/python -c "import tiktoken; assert tiktoken.get_encoding('o200k_base').encode('ok')"

FROM python:3.12-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/app/.venv/bin:$PATH"

RUN groupadd --gid 10001 app \
 && useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin app \
 && mkdir -p /data/embed_cache /data/capture \
 && chown -R app:app /data

WORKDIR /app

COPY --from=builder --chown=root:root /app/.venv /app/.venv
COPY --from=builder --chown=root:root /opt/tiktoken-cache /opt/tiktoken-cache
COPY --chown=root:root configs/ /app/configs/

ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken-cache \
    TIANXIMEM_SQLITE_PATH=/data/tianxi.db \
    TIANXIMEM_EMBED_CACHE_DIR=/data/embed_cache \
    TIANXIMEM_CAPTURE_PATH=/data/capture/requests.jsonl \
    TIANXIMEM_WORKERS=1

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys,httpx; sys.exit(0 if httpx.get('http://127.0.0.1:8000/health', timeout=4).status_code == 200 else 1)"]

ENTRYPOINT ["uvicorn", "tianximem.service.app:create_app_from_env", \
            "--factory", "--workers", "1", \
            "--host", "0.0.0.0", "--port", "8000"]
