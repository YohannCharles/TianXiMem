"""Dependency wiring for the HTTP service.

This module owns construction of the service object graph.  HTTP lifecycle and
route registration remain in :mod:`tianximem.service.app`; keeping those
concerns separate makes the graph testable without importing FastAPI.
"""

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
from tianximem.llm import LlmClient, OpenAiCompatChat
from tianximem.observability import MetricsSink, NullMetricsSink, SnapshotMetricsSink
from tianximem.rank import RemoteReranker, Reranker
from tianximem.retrieve import DenseArm, EvidenceChecker, HybridRetriever, make_hybrid_params
from tianximem.retrieve.router import QueryRouter
from tianximem.service.capture import RequestCapture
from tianximem.service.grounded_planner import GroundedPlanner
from tianximem.service.pipeline import AddPipeline, AgenticSearch, SearchPipeline
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import SqliteStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Services:
    """装配好的对象图。测试可以直接构造它，绕开 HTTP 与网络。"""

    store: SqliteStore
    qdrant: QdrantStore
    embedder: QueryInstructionEmbedder
    add: AddPipeline
    search: SearchPipeline
    capture: RequestCapture | None = None

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
        planner = self.search.grounded_planner
        if planner is not None and isinstance(planner.llm, OpenAiCompatChat):
            planner.llm.close()
        if self.capture is not None:
            self.capture.close()


def build_services(config: AppConfig) -> Services:
    """从配置装配全部依赖。"""

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
    grounded_planner = build_grounded_planner(config)
    return Services(
        store=store,
        qdrant=qdrant,
        embedder=embedder,
        add=AddPipeline(
            store=store,
            qdrant=qdrant,
            embedder=embedder,
            inject_abs_time=config.packaging.inject_abs_time,
            grounded_evidence=config.retrieval.grounded_evidence,
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
            grounded_evidence=config.retrieval.grounded_evidence,
            evidence_limit=config.retrieval.evidence_limit,
            evidence_hop_limit=config.retrieval.evidence_hop_limit,
            fact_backfill_limit=config.retrieval.fact_backfill_limit,
            grounded_planner=grounded_planner,
            router=build_query_router(config),
            agentic=build_agentic(
                config,
                store=store,
                retriever=retriever,
                reranker=reranker,
            ),
            agentic_force=config.agentic.force,
            metrics=build_metrics_sink(config),
        ),
        capture=build_capture(config),
    )


def build_llm(config: AppConfig, *, timeout: float | None = None) -> LlmClient | None:
    if not (
        config.router.enabled
        or config.agentic.enabled
        or (config.retrieval.grounded_evidence and config.retrieval.grounded_planner.enabled)
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


def build_grounded_planner(config: AppConfig) -> GroundedPlanner | None:
    """Build the optional LLM interpreter for deterministic evidence plans.

    It shares the configured Qwen/OpenAI-compatible endpoint with Search's
    other LLM consumers.  The planner is useful only when grounded evidence
    is enabled; otherwise no extra client or request is created.
    """
    planner_config = config.retrieval.grounded_planner
    if not (config.retrieval.grounded_evidence and planner_config.enabled):
        return None
    client = build_llm(config, timeout=planner_config.timeout_seconds)
    if client is None:
        return None
    return GroundedPlanner(
        client,
        max_output_tokens=planner_config.max_output_tokens,
        temperature=planner_config.temperature,
        prompt_version=planner_config.prompt_version,
    )


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


def build_capture(config: AppConfig) -> RequestCapture | None:
    if not config.capture.enabled:
        return None
    capture = RequestCapture(path=config.capture.path, max_bytes=config.capture.max_bytes)
    if not capture.open():
        return None
    return capture


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
