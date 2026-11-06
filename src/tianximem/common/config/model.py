from __future__ import annotations

from dataclasses import dataclass, field

from tianximem.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

from .constants import (
    DEFAULT_AGENTIC_MODE,
    DEFAULT_COQ_MAX_OUTPUT_TOKENS,
    DEFAULT_COQ_MAX_ROUNDS,
    DEFAULT_COQ_PER_ROUND_TOP_K,
    DEFAULT_COQ_PROMPT_VERSION,
    DEFAULT_COQ_SUFFICIENCY_THRESHOLD,
    DEFAULT_COQ_TIMEOUT_S,
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS,
    DEFAULT_EXTRACTION_MIN_SCORE,
    DEFAULT_EXTRACTION_PROMPT_VERSION,
    DEFAULT_EXTRACTION_TIMEOUT_S,
    DEFAULT_FACT_MAX_OUTPUT_TOKENS,
    DEFAULT_FACT_TIMEOUT_S,
    DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION,
    DEFAULT_PLAN_MAX_OUTPUT_TOKENS,
    DEFAULT_PLAN_STEP_PROMPT_VERSION,
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_PROFILE,
    DEFAULT_RADIUS,
    DEFAULT_RERANK_ENVELOPE,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_ROUTER_MAX_OUTPUT_TOKENS,
    DEFAULT_ROUTER_PROMPT_VERSION,
    DEFAULT_ROUTER_TIMEOUT_S,
    DEFAULT_WEIGHTS,
    RRF_K,
)


class ConfigError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SqliteConfig:
    path: str = "var/tianxi.db"
    busy_timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection: str = "memories"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    sqlite: SqliteConfig = field(default_factory=SqliteConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


@dataclass(frozen=True, slots=True)
class EmbedCacheConfig:
    dir: str = "var/embed_cache"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    embed: EmbedCacheConfig = field(default_factory=EmbedCacheConfig)


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    embedder: str = "Qwen/Qwen3-Embedding-4B"

    embed_dim: int = 1024


@dataclass(frozen=True, slots=True)
class RrfConfig:
    k: int = RRF_K
    weights: tuple[float, float] = DEFAULT_WEIGHTS


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    query_instruction: str = ""
    fact_augmentation: bool = False
    fact_timeout_seconds: float = DEFAULT_FACT_TIMEOUT_S
    fact_max_output_tokens: int = DEFAULT_FACT_MAX_OUTPUT_TOKENS
    fact_temperature: float = 0.0
    rrf: RrfConfig = field(default_factory=RrfConfig)


@dataclass(frozen=True, slots=True)
class NeighborConfig:
    expansion_seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT
    radius: int = DEFAULT_RADIUS
    seed_placement: str = "keep"


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    max_tokens: int = MAX_INPUT_TOKENS
    tokenizer: str = DEFAULT_TOKENIZER


@dataclass(frozen=True, slots=True)
class PackagingConfig:
    inject_abs_time: bool = True

    annotate_relatives: bool = True
    annotate_unit_totals: bool = False


@dataclass(frozen=True, slots=True)
class RerankConfig:
    enabled: bool = True
    timeout_seconds: float = DEFAULT_RERANK_TIMEOUT_S
    envelope: str = DEFAULT_RERANK_ENVELOPE


@dataclass(frozen=True, slots=True)
class RouterConfig:
    enabled: bool = False
    timeout_seconds: float = DEFAULT_ROUTER_TIMEOUT_S
    max_output_tokens: int = DEFAULT_ROUTER_MAX_OUTPUT_TOKENS
    temperature: float = 0.0
    prompt_version: str = DEFAULT_ROUTER_PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class AgenticConfig:
    enabled: bool = False

    mode: str = DEFAULT_AGENTIC_MODE

    force: bool = False

    max_rounds: int = DEFAULT_COQ_MAX_ROUNDS

    per_round_top_k: int = DEFAULT_COQ_PER_ROUND_TOP_K

    sufficiency_threshold: float = DEFAULT_COQ_SUFFICIENCY_THRESHOLD

    max_output_tokens: int = DEFAULT_COQ_MAX_OUTPUT_TOKENS

    plan_max_output_tokens: int = DEFAULT_PLAN_MAX_OUTPUT_TOKENS

    timeout_seconds: float = DEFAULT_COQ_TIMEOUT_S

    temperature: float = 0.0

    round_rerank_enabled: bool = True

    final_rerank_query_mode: str = "combined"

    prompt_version: str = DEFAULT_COQ_PROMPT_VERSION

    plan_decompose_prompt_version: str = DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION

    plan_step_prompt_version: str = DEFAULT_PLAN_STEP_PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class ServerConfig:
    workers: int = 1


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    enabled: bool = False

    min_score: int = DEFAULT_EXTRACTION_MIN_SCORE

    timeout_seconds: float = DEFAULT_EXTRACTION_TIMEOUT_S

    max_output_tokens: int = DEFAULT_EXTRACTION_MAX_OUTPUT_TOKENS

    temperature: float = 0.0

    prompt_version: str = DEFAULT_EXTRACTION_PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class AppConfig:
    profile: str = DEFAULT_PROFILE
    storage: StorageConfig = field(default_factory=StorageConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    neighbor: NeighborConfig = field(default_factory=NeighborConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    embed_base_url: str = ""
    embed_api_key: str = ""

    reranker_base_url: str = ""
    reranker_api_key: str = ""
    reranker_model: str = ""

    metrics_path: str = ""

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
