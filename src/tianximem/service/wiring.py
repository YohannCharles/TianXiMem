from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from tianximem.agent import CoqAgent, PlanAgent
from tianximem.common.config import (
    ENV_RERANKER_API_KEY,
    ENV_RERANKER_BASE_URL,
    AppConfig,
    ConfigError,
)
from tianximem.common.render import template_version
from tianximem.common.tokens import load_counter
from tianximem.embed.base import (
    CachingEmbedder,
    DiskVectorCache,
    EmbeddingCoordinate,
    OpenAICompatEmbedder,
)
from tianximem.embed.query_instruction import QueryInstructionEmbedder
from tianximem.embed.qwen3_embedding import Qwen3EmbeddingEmbedder
from tianximem.embed.text_embedding_v4 import DEFAULT_MODEL as TEXT_EMBEDDING_V4_MODEL
from tianximem.embed.text_embedding_v4 import TextEmbeddingV4Embedder
from tianximem.facts.llm_extract import ModelExtractor
from tianximem.llm import LlmClient, OpenAiCompatChat
from tianximem.observability import MetricsSink, NullMetricsSink, SnapshotMetricsSink
from tianximem.rank import RemoteReranker, Reranker
from tianximem.retrieve import DenseArm, EvidenceChecker, HybridRetriever, make_hybrid_params
from tianximem.retrieve.router import QueryRouter
from tianximem.service.pipeline import AddPipeline, AgenticSearch, SearchPipeline
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    store: SqliteStore
    qdrant: QdrantStore
    embedder: QueryInstructionEmbedder
    add: AddPipeline
    search: SearchPipeline

    def close(self) -> None:
        self.qdrant.close()
        wrapped = self.embedder
        inner = wrapped.inner
        if isinstance(inner, CachingEmbedder):
            inner = inner.inner
        if isinstance(inner, OpenAICompatEmbedder):
            inner.close()
        reranker = self.search.reranker
        if isinstance(reranker, RemoteReranker):
            reranker.close()
        router = self.search.router
        if router is not None and isinstance(router.client, OpenAiCompatChat):
            router.client.close()
        agentic = self.search.agentic
        if isinstance(agentic, (CoqAgent, PlanAgent)) and isinstance(agentic.llm, OpenAiCompatChat):
            agentic.llm.close()
        extractor = self.add.extractor
        if extractor is not None and isinstance(extractor.llm, OpenAiCompatChat):
            extractor.llm.close()


def build_services(config: AppConfig) -> Services:

    store = SqliteStore.open(
        config.storage.sqlite.path,
        busy_timeout_ms=config.storage.sqlite.busy_timeout_ms,
    )
    qdrant = QdrantStore(
        url=config.storage.qdrant.url,
        collection=config.storage.qdrant.collection,
        hybrid=make_hybrid_params(
            prefetch_limit=config.retrieval.prefetch_limit,
            weights=config.retrieval.rrf.weights,
            rrf_k=config.retrieval.rrf.k,
        ),
    )
    inner = build_embedder(config)
    cache = DiskVectorCache(
        config.cache.embed.dir,
        EmbeddingCoordinate(
            model=inner.model,
            render_template=template_version(inject_abs_time=config.packaging.inject_abs_time),
            dim=config.models.embed_dim,
        ),
    )
    embedder = QueryInstructionEmbedder(
        CachingEmbedder(inner, cache),
        instruction=config.retrieval.query_instruction,
    )
    dense = DenseArm(embedder)
    retriever = HybridRetriever(store=qdrant, dense=dense)
    counter = load_counter(config.budget.tokenizer)
    reranker = build_reranker(config)
    extractor = build_extractor(config)
    fact_llm = build_fact_llm(config)
    return Services(
        store=store,
        qdrant=qdrant,
        embedder=embedder,
        add=AddPipeline(
            store=store,
            qdrant=qdrant,
            embedder=embedder,
            inject_abs_time=config.packaging.inject_abs_time,
            extractor=extractor,
        ),
        search=SearchPipeline(
            store=store,
            qdrant=qdrant,
            retriever=retriever,
            checker=EvidenceChecker(),
            counter=counter,
            budget_tokens=config.budget.max_tokens,
            seed_limit=config.neighbor.expansion_seed_limit,
            radius=config.neighbor.radius,
            reranker=reranker,
            inject_abs_time=config.packaging.inject_abs_time,
            seed_placement=config.neighbor.seed_placement,
            annotate_relatives=config.packaging.annotate_relatives,
            annotate_unit_totals=config.packaging.annotate_unit_totals,
            router=build_query_router(config),
            agentic=build_agentic(
                config,
                store=store,
                retriever=retriever,
                reranker=reranker,
            ),
            agentic_force=config.agentic.force,
            fact_llm=fact_llm,
            fact_max_output_tokens=config.retrieval.fact_max_output_tokens,
            fact_temperature=config.retrieval.fact_temperature,
            metrics=build_metrics_sink(config),
        ),
    )


def build_llm(config: AppConfig, *, timeout: float | None = None) -> LlmClient | None:
    if not (
        config.router.enabled
        or config.agentic.enabled
        or config.extraction.enabled
        or config.retrieval.fact_augmentation
    ):
        return None
    return OpenAiCompatChat(
        base_url=config.llm_base_url,
        api_key=config.llm_api_key,
        model=config.llm_model,
        timeout=config.router.timeout_seconds if timeout is None else timeout,
        counter=load_counter(config.budget.tokenizer),
    )


def build_query_router(config: AppConfig) -> QueryRouter | None:
    if not config.router.enabled:
        return None
    client = build_llm(config, timeout=config.router.timeout_seconds)
    if client is None:
        raise ConfigError("router.enabled = true 但 LLM 客户端没构造出来（端点缺失？）")
    return QueryRouter(
        client,
        max_output_tokens=config.router.max_output_tokens,
        temperature=config.router.temperature,
        prompt_version=config.router.prompt_version,
    )


def build_extractor(config: AppConfig) -> ModelExtractor | None:
    if not config.extraction.enabled:
        return None
    client = build_llm(config, timeout=config.extraction.timeout_seconds)
    if client is None:
        return None
    return ModelExtractor(
        client,
        max_output_tokens=config.extraction.max_output_tokens,
        temperature=config.extraction.temperature,
        min_score=config.extraction.min_score,
    )


def build_fact_llm(config: AppConfig) -> LlmClient | None:

    if not config.retrieval.fact_augmentation:
        return None
    try:
        return build_llm(config, timeout=config.retrieval.fact_timeout_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "事实补充的模型客户端构造失败，已关闭该功能：%s: %s", type(exc).__name__, exc
        )
        return None


def build_agentic(
    config: AppConfig,
    *,
    store: SqliteStore,
    retriever: HybridRetriever,
    reranker: Reranker | None,
) -> AgenticSearch | None:
    if not config.agentic.enabled:
        return None
    client = build_llm(config, timeout=config.agentic.timeout_seconds)
    if client is None:
        raise ConfigError("agentic.enabled = true 但 LLM 客户端没构造出来（端点缺失？）")
    if config.agentic.force:
        logger.warning("`agentic.force = true`：绕过路由器，所有查询按 MULTI_HOP 处理")
    if config.agentic.mode == "plan":
        return PlanAgent(
            store=store,
            retriever=retriever,
            llm=client,
            reranker=reranker,
            max_rounds=config.agentic.max_rounds,
            per_round_top_k=config.agentic.per_round_top_k,
            max_output_tokens=config.agentic.plan_max_output_tokens,
            temperature=config.agentic.temperature,
            round_rerank_enabled=config.agentic.round_rerank_enabled,
            final_rerank_query_mode=config.agentic.final_rerank_query_mode,
            inject_abs_time=config.packaging.inject_abs_time,
            decompose_prompt_version=config.agentic.plan_decompose_prompt_version,
            step_prompt_version=config.agentic.plan_step_prompt_version,
        )
    return CoqAgent(
        store=store,
        retriever=retriever,
        llm=client,
        reranker=reranker,
        max_rounds=config.agentic.max_rounds,
        per_round_top_k=config.agentic.per_round_top_k,
        sufficiency_threshold=config.agentic.sufficiency_threshold,
        max_output_tokens=config.agentic.max_output_tokens,
        temperature=config.agentic.temperature,
        round_rerank_enabled=config.agentic.round_rerank_enabled,
        final_rerank_query_mode=config.agentic.final_rerank_query_mode,
        inject_abs_time=config.packaging.inject_abs_time,
        prompt_version=config.agentic.prompt_version,
    )


def build_metrics_sink(config: AppConfig) -> MetricsSink:
    if not config.metrics_path:
        return NullMetricsSink()
    return SnapshotMetricsSink(path=Path(config.metrics_path))


def build_reranker(config: AppConfig) -> RemoteReranker | None:
    if not config.rerank.enabled:
        return None
    missing = [
        name
        for name, value in (
            (ENV_RERANKER_BASE_URL, config.reranker_base_url),
            (ENV_RERANKER_API_KEY, config.reranker_api_key),
        )
        if not value
    ]
    if missing:
        logger.warning("rerank.enabled = true，但 %s 为空 ⇒ 不构造 reranker", " 与 ".join(missing))
        return None
    return RemoteReranker(
        base_url=config.reranker_base_url,
        api_key=config.reranker_api_key,
        model=config.reranker_model,
        timeout=config.rerank.timeout_seconds,
        envelope=config.rerank.envelope,
    )


def build_embedder(config: AppConfig) -> OpenAICompatEmbedder:
    if config.models.embedder == TEXT_EMBEDDING_V4_MODEL:
        return TextEmbeddingV4Embedder(
            base_url=config.embed_base_url,
            api_key=config.embed_api_key,
        )
    return Qwen3EmbeddingEmbedder(
        base_url=config.embed_base_url,
        api_key=config.embed_api_key,
        model=config.models.embedder,
    )
