from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Final

from tianximem.agent.coq_prompt import COQ_PROMPT, COQ_PROMPT_SLOTS, COQ_PROMPT_VERSION
from tianximem.agent.parse import CoqDecision, InvalidCoqOutput, parse_coq_decision
from tianximem.common.render import day_granularity, render_pair
from tianximem.llm import LlmClient, LlmUnavailable
from tianximem.rank import Reranker, RerankUnavailable
from tianximem.retrieve.fusion import Candidate, HybridRetriever
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "FINAL_QUERY_MODES",
    "STOP_REASONS",
    "AgenticResult",
    "AgenticStats",
    "CoqAgent",
    "RoundRecord",
    "normalize_query",
    "render_coq_prompt",
]

STOP_REASONS: Final[tuple[str, ...]] = (
    "sufficient",
    "low_confidence",
    "max_rounds",
    "empty_query",
    "duplicate_query",
    "invalid_output",
    "llm_unavailable",
    "llm_error",
    "retrieval_error",
)

FINAL_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")

_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(slot) for slot in COQ_PROMPT_SLOTS)
)


def normalize_query(query: str) -> str:

    return " ".join(query.casefold().split())


def render_coq_prompt(
    *, original_query: str, used_queries: Sequence[str], documents: Sequence[str]
) -> str:

    values = {
        "{original_query}": original_query,
        "{used_query}": "\n".join(used_queries),
        "{retrieved_episodes}": "\n\n".join(documents),
    }
    missing = [slot for slot in COQ_PROMPT_SLOTS if slot not in COQ_PROMPT]
    if missing:
        raise ValueError(f"提示词缺少占位符：{missing}")
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(0)], COQ_PROMPT)


@dataclass(frozen=True, slots=True)
class RoundRecord:
    index: int
    query: str
    recalled_candidates: int
    document_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    selected_ids: tuple[str, ...]
    dropped_documents: int
    rerank_status: str
    is_sufficient: bool | None = None
    confidence_score: float | None = None
    new_query: str | None = None
    extraction: str | None = None
    stop_reason: str | None = None
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AgenticStats:
    rounds: tuple[RoundRecord, ...]
    stop_reason: str
    fallback_reason: str | None
    candidate_count: int
    unique_candidate_count: int
    selected_evidence_count: int
    joint_query_count: int
    llm_calls: int
    rerank_calls: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class AgenticResult:
    candidates: tuple[Candidate, ...]
    rerank_query: str | None
    stats: AgenticStats
    fallback_reason: str | None = None


@dataclass(slots=True)
class _State:
    executed: list[str] = field(default_factory=list)
    normalized: set[str] = field(default_factory=set)
    ordered_ids: dict[str, None] = field(default_factory=dict)
    selected: list[str] = field(default_factory=list)
    bodies: dict[str, str] = field(default_factory=dict)
    days: dict[str, str] = field(default_factory=dict)


class CoqAgent:
    def __init__(
        self,
        *,
        store: SqliteStore,
        retriever: HybridRetriever,
        llm: LlmClient,
        reranker: Reranker | None = None,
        max_rounds: int = 3,
        per_round_top_k: int = 20,
        sufficiency_threshold: float = 0.8,
        max_output_tokens: int = 512,
        temperature: float = 0.0,
        round_rerank_enabled: bool = True,
        final_rerank_query_mode: str = "combined",
        inject_abs_time: bool = False,
        prompt_version: str = COQ_PROMPT_VERSION,
    ) -> None:
        if prompt_version != COQ_PROMPT_VERSION:
            raise ValueError(
                f"agentic.prompt_version = {prompt_version!r}，"
                f"但代码里只有一版提示词 {COQ_PROMPT_VERSION!r}"
            )
        if max_rounds < 1:
            raise ValueError("agentic.max_rounds 必须 ≥ 1")
        if per_round_top_k < 1:
            raise ValueError("agentic.per_round_top_k 必须 ≥ 1")
        if not 0.0 <= sufficiency_threshold <= 1.0:
            raise ValueError("agentic.sufficiency_threshold 必须在 0..1 之间")
        if max_output_tokens <= 0:
            raise ValueError("agentic.max_output_tokens 必须为正")
        if temperature < 0:
            raise ValueError("agentic.temperature 不得为负")
        if final_rerank_query_mode not in FINAL_QUERY_MODES:
            raise ValueError(
                f"agentic.final_rerank_query_mode 必须是 {FINAL_QUERY_MODES} 之一，"
                f"收到 {final_rerank_query_mode!r}"
            )

        self._store = store
        self._retriever = retriever
        self._llm = llm
        self._reranker = reranker
        self._max_rounds = max_rounds
        self._per_round_top_k = per_round_top_k
        self._sufficiency_threshold = sufficiency_threshold
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._round_rerank = round_rerank_enabled
        self._final_query_mode = final_rerank_query_mode
        self._inject_abs_time = inject_abs_time

    @property
    def llm(self) -> LlmClient:

        return self._llm

    def search(self, *, user_id: str, query: str, top_k: int) -> AgenticResult:
        started = monotonic()
        state = _State()
        records: list[RoundRecord] = []
        llm_calls = 0
        rerank_calls = 0
        stop_reason: str | None = None
        fallback_reason: str | None = None
        current = query

        for index in range(1, self._max_rounds + 1):
            round_query = current
            round_started = monotonic()

            try:
                recalled = self._retriever.search(user_id=user_id, query=round_query, top_k=top_k)
            except Exception:  # noqa: BLE001
                stop_reason = "retrieval_error"
                if not state.ordered_ids:
                    fallback_reason = "retrieval_error"
                records.append(
                    RoundRecord(
                        index=index,
                        query=round_query,
                        recalled_candidates=0,
                        document_ids=(),
                        candidate_ids=(),
                        selected_ids=(),
                        dropped_documents=0,
                        rerank_status="disabled",
                        stop_reason=stop_reason,
                        retrieval_latency_ms=_ms(round_started),
                    )
                )
                break
            retrieval_ms = _ms(round_started)

            rerank_started = monotonic()
            ranked, rerank_status = self._rerank_round(query=round_query, candidates=recalled)
            if rerank_status == "called":
                rerank_calls += 1
            rerank_ms = _ms(rerank_started)

            top = ranked[: self._per_round_top_k]
            top_ids = tuple(candidate.memory_id for candidate in top)
            self._cache_documents(state, top_ids)
            for memory_id in top_ids:
                state.ordered_ids.setdefault(memory_id, None)

            state.executed.append(round_query)
            state.normalized.add(normalize_query(round_query))

            documents, document_ids, dropped = self._build_documents(state, top_ids)

            llm_started = monotonic()
            llm_calls += 1
            decision, failure = self._decide(
                original=query, used=state.executed, documents=documents
            )
            llm_ms = _ms(llm_started)
            input_tokens, output_tokens = _usage(self._llm)

            picked: tuple[str, ...] = ()
            if decision is None:
                stop_reason = failure or "llm_error"
                input_tokens = output_tokens = None
            else:
                picked = tuple(document_ids[i] for i in decision.evidence_indices)
                for memory_id in picked:
                    if memory_id not in state.selected:
                        state.selected.append(memory_id)

                sufficient = decision.is_sufficient
                confident = decision.confidence_score >= self._sufficiency_threshold
                if sufficient and confident:
                    stop_reason = "sufficient"
                else:
                    next_query = decision.new_query.strip()
                    if not next_query or normalize_query(next_query) in state.normalized:
                        if sufficient:
                            stop_reason = "low_confidence"
                        elif not next_query:
                            stop_reason = "empty_query"
                        else:
                            stop_reason = "duplicate_query"
                    elif index >= self._max_rounds:
                        stop_reason = "max_rounds"
                    else:
                        current = next_query

            records.append(
                RoundRecord(
                    index=index,
                    query=round_query,
                    recalled_candidates=len(recalled),
                    document_ids=document_ids,
                    candidate_ids=top_ids,
                    selected_ids=picked,
                    dropped_documents=dropped,
                    rerank_status=rerank_status,
                    is_sufficient=None if decision is None else decision.is_sufficient,
                    confidence_score=None if decision is None else decision.confidence_score,
                    new_query=None if decision is None else decision.new_query,
                    extraction=None if decision is None else decision.extraction,
                    stop_reason=stop_reason,
                    retrieval_latency_ms=retrieval_ms,
                    rerank_latency_ms=rerank_ms,
                    llm_latency_ms=llm_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            if stop_reason is not None:
                break

        return self._finish(
            state=state,
            records=records,
            stop_reason=stop_reason or "max_rounds",
            fallback_reason=fallback_reason,
            llm_calls=llm_calls,
            rerank_calls=rerank_calls,
            started=started,
        )

    def _rerank_round(
        self, *, query: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str]:

        if not self._round_rerank or self._reranker is None or not candidates:
            return candidates, "disabled"
        pairs = self._fetch([candidate.memory_id for candidate in candidates])
        documents = [
            render_pair(pair, inject_abs_time=self._inject_abs_time)
            for pair in pairs
            if pair is not None
        ]
        if len(documents) != len(candidates):
            return candidates, "degraded"
        try:
            scores = self._reranker.score(query=query, documents=documents)
        except RerankUnavailable:
            return candidates, "degraded"
        if len(scores) != len(candidates):
            return candidates, "degraded"
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return (
            [Candidate(memory_id=candidates[i].memory_id, rank=new) for new, i in enumerate(order)],
            "called",
        )

    def _build_documents(
        self, state: _State, top_ids: Sequence[str]
    ) -> tuple[list[str], tuple[str, ...], int]:

        history_ids = [memory_id for memory_id in state.selected if memory_id in state.bodies]
        history = [(memory_id, self._document_body(state, memory_id)) for memory_id in history_ids]
        selected_set = set(history_ids)
        fresh: list[tuple[str, str]] = []
        dropped = 0
        for memory_id in top_ids:
            if memory_id in selected_set:
                continue
            if memory_id not in state.bodies:
                dropped += 1
                continue
            fresh.append((memory_id, self._document_body(state, memory_id)))

        ordered = fresh + history
        documents = [
            _render_document(index, body, state.days.get(memory_id, ""))
            for index, (memory_id, body) in enumerate(ordered)
        ]
        return documents, tuple(memory_id for memory_id, _ in ordered), dropped

    def _decide(
        self, *, original: str, used: Sequence[str], documents: Sequence[str]
    ) -> tuple[CoqDecision | None, str | None]:

        prompt = render_coq_prompt(original_query=original, used_queries=used, documents=documents)
        try:
            raw = self._llm.complete(
                prompt, max_tokens=self._max_output_tokens, temperature=self._temperature
            )
        except LlmUnavailable:
            return None, "llm_unavailable"
        except Exception:  # noqa: BLE001
            return None, "llm_error"
        try:
            return parse_coq_decision(raw, document_count=len(documents)), None
        except InvalidCoqOutput:
            return None, "invalid_output"

    def _finish(
        self,
        *,
        state: _State,
        records: list[RoundRecord],
        stop_reason: str,
        fallback_reason: str | None,
        llm_calls: int,
        rerank_calls: int,
        started: float,
    ) -> AgenticResult:
        candidates = tuple(
            Candidate(memory_id=memory_id, rank=rank)
            for rank, memory_id in enumerate(state.ordered_ids)
        )
        executed = list(state.executed)
        rerank_query = (
            "\n".join(executed) if self._final_query_mode == "combined" and executed else None
        )
        return AgenticResult(
            candidates=candidates,
            rerank_query=rerank_query,
            stats=AgenticStats(
                rounds=tuple(records),
                stop_reason=stop_reason,
                fallback_reason=fallback_reason,
                candidate_count=sum(record.recalled_candidates for record in records),
                unique_candidate_count=len(candidates),
                selected_evidence_count=len(state.selected),
                joint_query_count=len(executed),
                llm_calls=llm_calls,
                rerank_calls=rerank_calls,
                latency_ms=_ms(started),
            ),
            fallback_reason=fallback_reason,
        )

    def _cache_documents(self, state: _State, memory_ids: Sequence[str]) -> None:

        missing = [memory_id for memory_id in memory_ids if memory_id not in state.bodies]
        if not missing:
            return
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, missing)
        by_id = {pair.id: pair for pair in pairs}
        for memory_id in missing:
            pair = by_id.get(memory_id)
            if pair is None:
                continue
            state.bodies[memory_id] = render_pair(pair, inject_abs_time=self._inject_abs_time)
            state.days[memory_id] = day_granularity(getattr(pair, "event_time", None))

    def _document_body(self, state: _State, memory_id: str) -> str:

        return state.bodies[memory_id]

    def _fetch(self, memory_ids: Sequence[str]) -> list[QaPair | None]:
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, list(memory_ids))
        by_id = {pair.id: pair for pair in pairs}
        return [by_id.get(memory_id) for memory_id in memory_ids]


def _render_document(index: int, body: str, day: str = "") -> str:

    head = f"[{index}] Time: {day}" if day else f"[{index}]"
    return f"{head}\n{body}"


def _ms(started: float) -> float:
    return round((monotonic() - started) * 1000.0, 3)


def _usage(client: LlmClient) -> tuple[int | None, int | None]:

    usage = getattr(client, "last_usage", None)
    if not isinstance(usage, dict):
        return None, None
    return _as_tokens(usage.get("prompt_tokens")), _as_tokens(usage.get("completion_tokens"))


def _as_tokens(value: object) -> int | None:

    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
