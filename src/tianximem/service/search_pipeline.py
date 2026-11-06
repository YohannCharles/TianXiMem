from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from time import monotonic
from typing import Protocol, runtime_checkable

from tianximem.agent import AgenticResult, PlanResult
from tianximem.common.config import (
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_RADIUS,
)
from tianximem.common.render import render_pair
from tianximem.common.tokens import TokenCounter
from tianximem.observability import MetricsSink, NullMetricsSink, SearchObservation
from tianximem.rank import (
    PackagedResponse,
    Reranker,
    RerankUnavailable,
    expand_neighbors,
    merge_segments,
    package,
)
from tianximem.rank.packaging import ResponseItem
from tianximem.retrieve import EvidenceChecker, HybridRetriever, dedup_candidates
from tianximem.retrieve.fusion import Candidate
from tianximem.retrieve.router import QueryRouter, Route
from tianximem.service.fact_snippets import FactSuffix, collect_fact_snippets
from tianximem.store.qdrant_store import QdrantStore
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = ["AgenticSearch", "SearchPipeline"]


@runtime_checkable
class AgenticSearch(Protocol):
    def search(self, *, user_id: str, query: str, top_k: int) -> AgenticResult | PlanResult: ...


@dataclass(slots=True)
class _SearchTrace:
    fact_status: str = "disabled"
    fact_snippets: int = 0
    rerank_calls: int = 0
    rerank_disabled: int = 0
    rerank_degraded: int = 0
    route: str = "disabled"
    route_status: str = "disabled"
    route_reason: str | None = None
    route_fallback: str | None = None
    route_latency_ms: float | None = None
    agentic_status: str = "disabled"
    agentic_rounds: int = 0
    agentic_stop_reason: str | None = None
    agentic_llm_calls: int = 0
    agentic_rerank_calls: int = 0
    agentic_selected_evidence: int = 0
    agentic_candidates: int = 0
    agentic_joint_queries: int = 0
    agentic_latency_ms: float | None = None


class SearchPipeline:
    def __init__(
        self,
        *,
        store: SqliteStore,
        qdrant: QdrantStore,
        retriever: HybridRetriever,
        checker: EvidenceChecker,
        counter: TokenCounter,
        budget_tokens: int,
        seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT,
        radius: int = DEFAULT_RADIUS,
        reranker: Reranker | None = None,
        inject_abs_time: bool = False,
        seed_placement: str = "keep",
        annotate_relatives: bool = False,
        annotate_unit_totals: bool = False,
        router: QueryRouter | None = None,
        agentic: AgenticSearch | None = None,
        agentic_force: bool = False,
        fact_llm: object | None = None,
        fact_max_output_tokens: int = 256,
        fact_temperature: float = 0.0,
        metrics: MetricsSink | None = None,
    ) -> None:
        self._store = store
        self._qdrant = qdrant
        self._retriever = retriever
        self._checker = checker
        self._counter = counter
        self._budget_tokens = budget_tokens
        self._seed_limit = seed_limit
        self._radius = radius
        self._reranker = reranker
        self._inject_abs_time = inject_abs_time
        self._seed_placement = seed_placement
        self._annotate_relatives = annotate_relatives
        self._annotate_unit_totals = annotate_unit_totals
        self._router = router
        self._agentic = agentic
        self._agentic_force = agentic_force
        self._fact_llm = fact_llm
        self._fact_max_output_tokens = fact_max_output_tokens
        self._fact_temperature = fact_temperature
        self._metrics = metrics if metrics is not None else NullMetricsSink()
        self.rerank_calls = 0
        self.rerank_degraded = 0
        self.rerank_disabled = 0

    @property
    def reranker(self) -> Reranker | None:
        return self._reranker

    @property
    def metrics(self) -> MetricsSink:
        return self._metrics

    @property
    def router(self) -> QueryRouter | None:
        return self._router

    @property
    def agentic(self) -> AgenticSearch | None:
        return self._agentic

    def run(self, *, user_id: str, query: str, top_k: int) -> PackagedResponse:
        started = monotonic()
        trace = _SearchTrace()
        response = self._search(user_id=user_id, query=query, top_k=top_k, trace=trace)
        self._metrics.record(
            SearchObservation(
                latency_ms=round((monotonic() - started) * 1000.0, 3),
                rerank_calls=trace.rerank_calls,
                rerank_disabled=trace.rerank_disabled,
                rerank_degraded=trace.rerank_degraded,
                rerank_name=self._reranker.name if self._reranker is not None else None,
                fact_status=trace.fact_status,
                fact_snippets=trace.fact_snippets,
                route=trace.route,
                route_status=trace.route_status,
                route_reason=trace.route_reason,
                route_fallback=trace.route_fallback,
                route_latency_ms=trace.route_latency_ms,
                agentic_status=trace.agentic_status,
                agentic_rounds=trace.agentic_rounds,
                agentic_stop_reason=trace.agentic_stop_reason,
                agentic_llm_calls=trace.agentic_llm_calls,
                agentic_rerank_calls=trace.agentic_rerank_calls,
                agentic_selected_evidence=trace.agentic_selected_evidence,
                agentic_candidates=trace.agentic_candidates,
                agentic_joint_queries=trace.agentic_joint_queries,
                agentic_latency_ms=trace.agentic_latency_ms,
                returned_segments=len(response.items),
                truncated_by_budget=response.truncated_by_budget,
                truncated_by_top_k=response.truncated_by_top_k,
                dropped_missing=response.dropped_missing,
            )
        )
        return response

    def _search(
        self, *, user_id: str, query: str, top_k: int, trace: _SearchTrace | None = None
    ) -> PackagedResponse:
        if trace is None:
            trace = _SearchTrace()
        if top_k <= 0:
            return PackagedResponse(items=())
        if not query.strip():
            return PackagedResponse(items=())

        route = self._route(query, trace)
        if not self._qdrant.exists():
            return PackagedResponse(items=())

        rerank_query: str | None = None
        if route is Route.MULTI_HOP and self._agentic is not None:
            outcome = self._agentic.search(user_id=user_id, query=query, top_k=top_k)
            self._record_agentic(trace, outcome)
            if outcome.fallback_reason is not None:
                trace.route_fallback = outcome.fallback_reason
                candidates: Sequence[Candidate] = self._retriever.search(
                    user_id=user_id, query=query, top_k=top_k
                )
            else:
                candidates = outcome.candidates
                rerank_query = outcome.rerank_query
        else:
            candidates = self._retriever.search(user_id=user_id, query=query, top_k=top_k)

        packaged = self._rank_and_package(
            candidates, query=query, top_k=top_k, trace=trace, rerank_query=rerank_query
        )
        return self._append_facts(
            user_id=user_id, query=query, top_k=top_k, packaged=packaged, trace=trace
        )

    def _append_facts(
        self,
        *,
        user_id: str,
        query: str,
        top_k: int,
        packaged: PackagedResponse,
        trace: _SearchTrace,
    ) -> PackagedResponse:

        room = top_k - len(packaged.items)
        if room <= 0:
            trace.fact_status = "no_room"
            return packaged

        facts: FactSuffix = collect_fact_snippets(
            store=self._store,
            llm=self._fact_llm,  # type: ignore[arg-type]
            user_id=user_id,
            query=query,
            max_output_tokens=self._fact_max_output_tokens,
            temperature=self._fact_temperature,
            limit=room,
        )
        trace.fact_status = facts.status
        if not facts:
            return packaged

        existing = {item.content for item in packaged.items}
        extras: list[ResponseItem] = []
        for snippet in facts.snippets:
            if len(extras) >= room:
                break
            if snippet in existing:
                continue
            existing.add(snippet)
            extras.append(
                ResponseItem(
                    id=f"fact:{len(packaged.items) + len(extras)}",
                    content=snippet,
                    created_at="",
                    score=0.0,
                )
            )
        if not extras:
            return packaged
        trace.fact_snippets = len(extras)
        return replace(
            packaged,
            items=(*packaged.items, *extras),
            truncated_by_top_k=packaged.truncated_by_top_k,
        )

    def _record_agentic(self, trace: _SearchTrace, outcome: AgenticResult | PlanResult) -> None:
        stats = outcome.stats
        trace.agentic_status = "success" if outcome.fallback_reason is None else "degraded"
        trace.agentic_rounds = len(stats.rounds)
        trace.agentic_stop_reason = stats.stop_reason
        trace.agentic_llm_calls = stats.llm_calls
        trace.agentic_rerank_calls = stats.rerank_calls
        trace.agentic_selected_evidence = stats.selected_evidence_count
        trace.agentic_candidates = stats.unique_candidate_count
        trace.agentic_joint_queries = stats.joint_query_count
        trace.agentic_latency_ms = stats.latency_ms

    def _route(self, query: str, trace: _SearchTrace) -> Route | None:
        if self._agentic_force and self._agentic is not None:
            trace.route = Route.MULTI_HOP.value
            trace.route_status = "forced"
            return Route.MULTI_HOP
        if self._router is None:
            return None
        decision = self._router.route(query)
        trace.route = decision.route.value
        trace.route_status = decision.status
        trace.route_reason = decision.reason
        trace.route_latency_ms = decision.latency_ms
        if decision.route is Route.MULTI_HOP and self._agentic is None:
            trace.route_fallback = "agentic_unavailable"
        return decision.route

    def _rank_and_package(
        self,
        candidates: Sequence[Candidate],
        *,
        query: str,
        top_k: int,
        trace: _SearchTrace,
        rerank_query: str | None = None,
    ) -> PackagedResponse:
        ranked = dedup_candidates(candidates)

        self._checker.decide(query=query)

        ranked = self._maybe_rerank(
            query=query if rerank_query is None else rerank_query, ranked=ranked, trace=trace
        )

        expansion = expand_neighbors(
            ranked, store=self._store, seed_limit=self._seed_limit, radius=self._radius
        )
        segments = merge_segments(
            expansion.selected,
            counter=self._counter,
            inject_abs_time=self._inject_abs_time,
            seed_placement=self._seed_placement,
            annotate_relatives=self._annotate_relatives,
        )

        return package(
            segments,
            top_k=top_k,
            counter=self._counter,
            max_tokens=self._budget_tokens,
            dropped_missing=expansion.missing_rows,
        )

    def _record_rerank(self, trace: _SearchTrace | None, field: str) -> None:
        setattr(self, field, getattr(self, field) + 1)
        if trace is not None:
            setattr(trace, field, getattr(trace, field) + 1)

    def _maybe_rerank(
        self, *, query: str, ranked: list[Candidate], trace: _SearchTrace | None = None
    ) -> list[Candidate]:
        if self._reranker is None or not ranked:
            self._record_rerank(trace, "rerank_disabled")
            return ranked

        pairs = self._fetch_for_rerank([c.memory_id for c in ranked])
        documents = [
            render_pair(p, inject_abs_time=self._inject_abs_time) for p in pairs if p is not None
        ]
        if len(documents) != len(ranked):
            self._record_rerank(trace, "rerank_degraded")
            return ranked

        self._record_rerank(trace, "rerank_calls")
        try:
            scores = self._reranker.score(query=query, documents=documents)
        except RerankUnavailable:
            self._record_rerank(trace, "rerank_degraded")
            return ranked

        if len(scores) != len(ranked):
            self._record_rerank(trace, "rerank_degraded")
            return ranked

        order = sorted(range(len(ranked)), key=lambda i: (-float(scores[i]), i))
        return [
            Candidate(memory_id=ranked[i].memory_id, rank=new_rank)
            for new_rank, i in enumerate(order)
        ]

    def _fetch_for_rerank(self, ids: list[str]) -> list[QaPair | None]:
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, ids)
        by_id = {p.id: p for p in pairs}
        return [by_id.get(memory_id) for memory_id in ids]
