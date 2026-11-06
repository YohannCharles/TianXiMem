from __future__ import annotations

from typing import Final

ENV_PROFILE: Final[str] = "TIANXIMEM_PROFILE"
ENV_CONFIG_DIR: Final[str] = "TIANXIMEM_CONFIG_DIR"
ENV_SQLITE_PATH: Final[str] = "TIANXIMEM_SQLITE_PATH"
ENV_QDRANT_URL: Final[str] = "TIANXIMEM_QDRANT_URL"
ENV_EMBED_BASE_URL: Final[str] = "AML_EMB_BASE_URL"
ENV_EMBED_API_KEY: Final[str] = "AML_EMB_API_KEY"
ENV_EMBED_CACHE_DIR: Final[str] = "TIANXIMEM_EMBED_CACHE_DIR"
ENV_WORKERS: Final[str] = "TIANXIMEM_WORKERS"

ENV_RERANKER_BASE_URL: Final[str] = "TIANXIMEM_RERANKER_BASE_URL"
ENV_RERANKER_API_KEY: Final[str] = "TIANXIMEM_RERANKER_API_KEY"
ENV_RERANKER_MODEL: Final[str] = "TIANXIMEM_RERANKER_MODEL"

ENV_LLM_BASE_URL: Final[str] = "AML_BASE_URL"
ENV_LLM_API_KEY: Final[str] = "AML_API_KEY"
ENV_LLM_MODEL: Final[str] = "AML_MODEL"
ENV_FILE: Final[str] = "TIANXIMEM_ENV_FILE"

ENV_QDRANT_IMAGE: Final[str] = "TIANXIMEM_QDRANT_IMAGE"

ENV_METRICS_PATH: Final[str] = "TIANXIMEM_METRICS_PATH"

DEFAULT_PROFILE: Final[str] = "default"
DEFAULT_CONFIG_DIR: Final[str] = "configs"
DEFAULT_ENV_FILE: Final[str] = ".env"

RRF_K: Final[int] = 61

DEFAULT_EXPANSION_SEED_LIMIT: Final[int] = 1000
DEFAULT_RADIUS: Final[int] = 2

SEED_PLACEMENTS: Final[tuple[str, ...]] = ("keep", "front", "echo")


DEFAULT_PREFETCH_LIMIT: Final[int] = 200
DEFAULT_WEIGHTS: Final[tuple[float, float]] = (0.5, 0.5)

DEFAULT_FACT_TIMEOUT_S: Final[float] = 10.0
DEFAULT_FACT_MAX_OUTPUT_TOKENS: Final[int] = 256

DEFAULT_RERANK_TIMEOUT_S: Final[float] = 30.0

DEFAULT_RERANK_ENVELOPE: Final[str] = "query"

RERANK_ENVELOPES: Final[frozenset[str]] = frozenset({"queries", "query", "dashscope"})

DEFAULT_ROUTER_TIMEOUT_S: Final[float] = 10.0

DEFAULT_ROUTER_MAX_OUTPUT_TOKENS: Final[int] = 8

DEFAULT_ROUTER_PROMPT_VERSION: Final[str] = "v5"

DEFAULT_COQ_MAX_ROUNDS: Final[int] = 3

DEFAULT_COQ_PER_ROUND_TOP_K: Final[int] = 20

DEFAULT_COQ_SUFFICIENCY_THRESHOLD: Final[float] = 0.8

DEFAULT_COQ_TIMEOUT_S: Final[float] = 30.0

DEFAULT_COQ_MAX_OUTPUT_TOKENS: Final[int] = 512

DEFAULT_COQ_PROMPT_VERSION: Final[str] = "coq-original"

DEFAULT_AGENTIC_MODE: Final[str] = "react"

AGENTIC_MODES: Final[tuple[str, ...]] = ("react", "plan")

DEFAULT_PLAN_MAX_OUTPUT_TOKENS: Final[int] = 1536

DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION: Final[str] = "plan-decompose-v2"

DEFAULT_PLAN_STEP_PROMPT_VERSION: Final[str] = "plan-step-v2"

FINAL_RERANK_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")

DEFAULT_EXTRACTION_MIN_SCORE: Final[int] = 3

DEFAULT_EXTRACTION_TIMEOUT_S: Final[float] = 30.0

DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS: Final[int] = 2048

DEFAULT_EXTRACTION_PROMPT_VERSION: Final[str] = "v3"

DEFAULT_QUERY_ENTITIES_PROMPT_VERSION: Final[str] = "v1"
