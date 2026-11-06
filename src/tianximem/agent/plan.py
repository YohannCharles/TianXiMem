from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from time import monotonic
from typing import Final

from tianximem.agent.coq import _ms, _render_document, _usage
from tianximem.agent.plan_parse import (
    InvalidPlanOutput,
    Plan,
    PlanStep,
    StepAnswer,
    bind_placeholders,
    parse_plan,
    parse_step_answer,
    render_plan,
)
from tianximem.agent.plan_prompts import (
    PLAN_DECOMPOSE_PROMPT,
    PLAN_DECOMPOSE_PROMPT_VERSION,
    PLAN_DECOMPOSE_SLOTS,
    PLAN_STEP_PROMPT,
    PLAN_STEP_PROMPT_VERSION,
    PLAN_STEP_SLOTS,
)
from tianximem.common.render import day_granularity, render_pair
from tianximem.llm import LlmClient, LlmUnavailable
from tianximem.rank import Reranker, RerankUnavailable
from tianximem.retrieve.fusion import Candidate, HybridRetriever
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "FINAL_QUERY_MODES",
    "PLAN_MAX_STEPS",
    "PLAN_RECOVERY_STATUSES",
    "PLAN_RECOVERY_TRIGGERS",
    "PLAN_STOP_REASONS",
    "PlanAgent",
    "PlanRecovery",
    "PlanResult",
    "PlanStats",
    "PlanStepRecord",
    "render_completed_history",
    "render_plan_decompose_prompt",
    "render_plan_step_prompt",
    "step_line",
]

PLAN_STOP_REASONS: Final[tuple[str, ...]] = (
    "plan_complete",
    "dependency_unresolved",
    "binding_conflict",
    "unbound_placeholder",
    "invalid_output",
    "plan_invalid",
    "max_rounds",
    "llm_unavailable",
    "llm_error",
    "retrieval_error",
)

PLAN_MAX_STEPS: Final[int] = 4


PLAN_RECOVERY_STATUSES: Final[tuple[str, ...]] = (
    "not_triggered",
    "success",
    "empty",
    "error",
    "skipped_pool_full",
)

PLAN_RECOVERY_TRIGGERS: Final[frozenset[str]] = frozenset(
    {
        "dependency_unresolved",
        "binding_conflict",
        "invalid_output",
        "llm_unavailable",
        "plan_invalid",
    }
)

FINAL_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")

_DECOMPOSE_FILLER: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(slot) for slot in PLAN_DECOMPOSE_SLOTS)
)
_STEP_FILLER: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(slot) for slot in PLAN_STEP_SLOTS)
)


@dataclass(frozen=True, slots=True)
class PlanStepRecord:
    index: int
    step_id: str
    template: str
    executed_query: str
    recalled_candidates: int
    document_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    evidence_quotes: tuple[str, ...]
    quotes_unverified: int
    status: str | None
    answer: str | None
    canonical_next_query: str | None
    history_text: str
    dropped_documents: int
    rerank_status: str
    extraction: str | None
    stop_reason: str | None
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class PlanRecovery:
    entered: bool = False
    status: str = "not_triggered"
    executed_rounds: int = 0
    target_top_k: int = 0
    pool_before: int = 0
    recalled: int = 0
    reranked: str = "disabled"
    added_unique: int = 0
    latency_ms: float = 0.0
    rerank_query_mode: str = "joint"


@dataclass(frozen=True, slots=True)
class PlanStats:
    rounds: tuple[PlanStepRecord, ...]
    stop_reason: str
    fallback_reason: str | None
    candidate_count: int
    unique_candidate_count: int
    selected_evidence_count: int
    joint_query_count: int
    llm_calls: int
    rerank_calls: int
    latency_ms: float
    plan: tuple[PlanStep, ...] = ()
    decompose_llm_calls: int = 0
    step_llm_calls: int = 0
    plan_steps: int = 0
    steps_executed: int = 0
    dependencies_repaired: int = 0


@dataclass(frozen=True, slots=True)
class PlanResult:
    candidates: tuple[Candidate, ...]
    rerank_query: str | None
    stats: PlanStats
    fallback_reason: str | None = None
    plan_history: tuple[PlanStepRecord, ...] = ()
    plan: tuple[PlanStep, ...] = ()
    recovery: PlanRecovery = PlanRecovery()


@dataclass(slots=True)
class _State:
    executed: list[str] = field(default_factory=list)
    executed_for: dict[str, str] = field(default_factory=dict)
    ordered_ids: dict[str, None] = field(default_factory=dict)
    evidence: list[str] = field(default_factory=list)
    bodies: dict[str, str] = field(default_factory=dict)
    days: dict[str, str] = field(default_factory=dict)
    answers: dict[str, str] = field(default_factory=dict)
    sources: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _fill(
    pattern: re.Pattern[str], template: str, slots: Sequence[str], values: dict[str, str]
) -> str:
    missing = [slot for slot in slots if slot not in template]
    if missing:
        raise ValueError(f"提示词缺少占位符：{missing}")
    return pattern.sub(lambda match: values[match.group(0)], template)


def render_plan_decompose_prompt(*, original_question: str) -> str:
    return _fill(
        _DECOMPOSE_FILLER,
        PLAN_DECOMPOSE_PROMPT,
        PLAN_DECOMPOSE_SLOTS,
        {"{original_question}": original_question},
    )


def step_line(step: PlanStep) -> str:
    return f"{step.id}: {step.query}"


def render_plan_step_prompt(
    *,
    original_question: str,
    retrieval_plan: str,
    completed_history: str,
    current_step: str,
    executed_query: str,
    current_documents: Sequence[str],
) -> str:
    return _fill(
        _STEP_FILLER,
        PLAN_STEP_PROMPT,
        PLAN_STEP_SLOTS,
        {
            "{original_question}": original_question,
            "{retrieval_plan}": retrieval_plan,
            "{completed_history}": completed_history,
            "{current_step}": current_step,
            "{executed_query}": executed_query,
            "{current_documents}": "\n\n".join(current_documents),
        },
    )


def render_completed_history(plan: Plan, state: _State) -> str:
    blocks: list[str] = []
    for step in plan.steps:
        answer = state.answers.get(step.id)
        if answer is None:
            continue
        sources = ", ".join(state.sources.get(step.id, ())) or "(none)"
        blocks.append(
            f"{step.id} question: {state.executed_for.get(step.id, step.query)}\n"
            f"{step.id} answer: {answer}\n"
            f"{step.id} sources: {sources}"
        )
    return "\n\n".join(blocks) if blocks else "(none yet)"


class PlanAgent:
    def __init__(
        self,
        *,
        store: SqliteStore,
        retriever: HybridRetriever,
        llm: LlmClient,
        reranker: Reranker | None = None,
        max_rounds: int = 4,
        per_round_top_k: int = 20,
        max_output_tokens: int = 1536,
        temperature: float = 0.0,
        round_rerank_enabled: bool = True,
        final_rerank_query_mode: str = "combined",
        inject_abs_time: bool = False,
        max_plan_steps: int = PLAN_MAX_STEPS,
        decompose_prompt_version: str = PLAN_DECOMPOSE_PROMPT_VERSION,
        step_prompt_version: str = PLAN_STEP_PROMPT_VERSION,
    ) -> None:
        if decompose_prompt_version != PLAN_DECOMPOSE_PROMPT_VERSION:
            raise ValueError(
                f"agentic.plan_decompose_prompt_version = {decompose_prompt_version!r}，"
                f"但代码里只有一版提示词 {PLAN_DECOMPOSE_PROMPT_VERSION!r}"
            )
        if step_prompt_version != PLAN_STEP_PROMPT_VERSION:
            raise ValueError(
                f"agentic.plan_step_prompt_version = {step_prompt_version!r}，"
                f"但代码里只有一版提示词 {PLAN_STEP_PROMPT_VERSION!r}"
            )
        if max_rounds < 1:
            raise ValueError("agentic.max_rounds 必须 ≥ 1")
        if per_round_top_k < 1:
            raise ValueError("agentic.per_round_top_k 必须 ≥ 1")
        if max_output_tokens <= 0:
            raise ValueError("agentic.plan_max_output_tokens 必须为正")
        if temperature < 0:
            raise ValueError("agentic.temperature 不得为负")
        if not 1 <= max_plan_steps <= PLAN_MAX_STEPS:
            raise ValueError(f"计划步数上限必须在 1..{PLAN_MAX_STEPS} 之间")
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
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._round_rerank = round_rerank_enabled
        self._final_query_mode = final_rerank_query_mode
        self._inject_abs_time = inject_abs_time
        self._max_plan_steps = max_plan_steps

    @property
    def llm(self) -> LlmClient:
        return self._llm

    def search(self, *, user_id: str, query: str, top_k: int) -> PlanResult:
        started = monotonic()
        state = _State()
        records: list[PlanStepRecord] = []
        rerank_calls = 0
        stop_reason: str | None = None
        fallback_reason: str | None = None

        plan, failure = self._decompose(query)
        if plan is None:
            reason = failure or "plan_invalid"
            recovery = self._recover(
                state=state,
                user_id=user_id,
                original=query,
                stop_reason=reason,
                top_k=top_k,
            )
            rerank_calls = 1 if recovery.reranked == "called" else 0
            return self._finish(
                state=state,
                records=records,
                stop_reason=reason,
                fallback_reason=None if recovery.entered and state.ordered_ids else reason,
                rerank_calls=rerank_calls,
                decompose_llm_calls=1,
                step_llm_calls=0,
                plan=None,
                original=query,
                recovery=recovery,
                started=started,
            )

        steps = plan.steps[: self._max_rounds]
        truncated = len(plan.steps) > len(steps)

        for index, step in enumerate(steps, start=1):
            executed_query, unbound = bind_placeholders(step.query, state.answers)
            if unbound:
                stop_reason = "unbound_placeholder"
                records.append(self._bare_record(index, step, executed_query, stop_reason))
                break

            round_started = monotonic()
            try:
                recalled = self._retriever.search(
                    user_id=user_id, query=executed_query, top_k=top_k
                )
            except Exception:  # noqa: BLE001
                stop_reason = "retrieval_error"
                if not state.ordered_ids:
                    fallback_reason = "retrieval_error"
                records.append(
                    self._bare_record(
                        index,
                        step,
                        executed_query,
                        stop_reason,
                        retrieval_latency_ms=_ms(round_started),
                    )
                )
                break
            retrieval_ms = _ms(round_started)

            rerank_started = monotonic()
            ranked, rerank_status = self._rerank_round(query=executed_query, candidates=recalled)
            if rerank_status == "called":
                rerank_calls += 1
            rerank_ms = _ms(rerank_started)

            top = ranked[: self._per_round_top_k]
            top_ids = tuple(candidate.memory_id for candidate in top)
            self._cache_documents(state, top_ids)
            for memory_id in top_ids:
                state.ordered_ids.setdefault(memory_id, None)
            state.executed.append(executed_query)
            state.executed_for[step.id] = executed_query

            next_step = steps[index] if index < len(steps) else None
            history_text = render_completed_history(plan, state)
            plan_text = render_plan(plan)
            documents, document_ids, dropped = self._build_documents(state=state, top_ids=top_ids)

            llm_started = monotonic()
            decision, failure = self._answer_step(
                original=query,
                plan_text=plan_text,
                history=history_text,
                step=step,
                executed_query=executed_query,
                documents=documents,
            )
            llm_ms = _ms(llm_started)
            input_tokens, output_tokens = _usage(self._llm)
            if decision is None:
                input_tokens = output_tokens = None
                stop_reason = failure or "llm_error"

            evidence_ids: tuple[str, ...] = ()
            quotes: tuple[str, ...] = ()
            unverified = 0
            canonical: str | None = None
            if decision is not None:
                evidence_ids = tuple(document_ids[ref.document_index] for ref in decision.evidence)
                quotes = tuple(ref.quote for ref in decision.evidence)
                unverified = sum(1 for ref in decision.evidence if not ref.verified)
                for memory_id in evidence_ids:
                    if memory_id not in state.evidence:
                        state.evidence.append(memory_id)

                if decision.status != "answered":
                    stop_reason = (
                        "dependency_unresolved"
                        if decision.status == "unresolved"
                        else "binding_conflict"
                    )
                else:
                    state.answers[step.id] = decision.answer or ""
                    state.sources[step.id] = evidence_ids
                    if next_step is None:
                        stop_reason = "max_rounds" if truncated else "plan_complete"
                    else:
                        canonical, still_unbound = bind_placeholders(next_step.query, state.answers)
                        if still_unbound:
                            stop_reason = "unbound_placeholder"

            records.append(
                PlanStepRecord(
                    index=index,
                    step_id=step.id,
                    template=step.query,
                    executed_query=executed_query,
                    recalled_candidates=len(recalled),
                    document_ids=document_ids,
                    candidate_ids=top_ids,
                    evidence_ids=evidence_ids,
                    evidence_quotes=quotes,
                    quotes_unverified=unverified,
                    status=None if decision is None else decision.status,
                    answer=None if decision is None else decision.answer,
                    canonical_next_query=canonical,
                    history_text=history_text,
                    dropped_documents=dropped,
                    rerank_status=rerank_status,
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

        recovery = self._recover(
            state=state,
            user_id=user_id,
            original=query,
            stop_reason=stop_reason or "plan_complete",
            top_k=top_k,
        )
        if recovery.reranked == "called":
            rerank_calls += 1

        return self._finish(
            state=state,
            records=records,
            stop_reason=stop_reason or "plan_complete",
            fallback_reason=fallback_reason,
            rerank_calls=rerank_calls,
            decompose_llm_calls=1,
            step_llm_calls=len(records),
            plan=plan,
            original=query,
            recovery=recovery,
            started=started,
        )

    def _recover(
        self, *, state: _State, user_id: str, original: str, stop_reason: str, top_k: int
    ) -> PlanRecovery:
        if stop_reason not in PLAN_RECOVERY_TRIGGERS:
            return PlanRecovery()
        executed_rounds = len(state.executed)
        target = top_k
        pool_before = len(state.ordered_ids)
        if pool_before >= target:
            return PlanRecovery(
                entered=True,
                status="skipped_pool_full",
                executed_rounds=executed_rounds,
                target_top_k=target,
                pool_before=pool_before,
            )
        started = monotonic()
        try:
            extra = self._retriever.search(user_id=user_id, query=original, top_k=target)
        except Exception:  # noqa: BLE001
            return PlanRecovery(
                entered=True,
                status="error",
                executed_rounds=executed_rounds,
                target_top_k=target,
                pool_before=pool_before,
                latency_ms=_ms(started),
            )
        ranked, rerank_status = self._rerank_fallback(query=original, candidates=list(extra))
        added = 0
        for candidate in ranked:
            if len(state.ordered_ids) >= target:
                break
            if candidate.memory_id not in state.ordered_ids:
                added += 1
            state.ordered_ids.setdefault(candidate.memory_id, None)
        return PlanRecovery(
            entered=True,
            status="success" if extra else "empty",
            executed_rounds=executed_rounds,
            target_top_k=target,
            pool_before=pool_before,
            recalled=len(extra),
            reranked=rerank_status,
            added_unique=added,
            latency_ms=_ms(started),
        )

    def _decompose(self, original: str) -> tuple[Plan | None, str | None]:
        prompt = render_plan_decompose_prompt(original_question=original)
        try:
            raw = self._llm.complete(
                prompt, max_tokens=self._max_output_tokens, temperature=self._temperature
            )
        except LlmUnavailable:
            return None, "llm_unavailable"
        except Exception:  # noqa: BLE001
            return None, "llm_error"
        try:
            return parse_plan(raw, max_steps=self._max_plan_steps), None
        except InvalidPlanOutput:
            return None, "plan_invalid"

    def _answer_step(
        self,
        *,
        original: str,
        plan_text: str,
        history: str,
        step: PlanStep,
        executed_query: str,
        documents: Sequence[str],
    ) -> tuple[StepAnswer | None, str | None]:
        prompt = render_plan_step_prompt(
            original_question=original,
            retrieval_plan=plan_text,
            completed_history=history,
            current_step=step_line(step),
            executed_query=executed_query,
            current_documents=documents,
        )
        try:
            raw = self._llm.complete(
                prompt, max_tokens=self._max_output_tokens, temperature=self._temperature
            )
        except LlmUnavailable:
            return None, "llm_unavailable"
        except Exception:  # noqa: BLE001
            return None, "llm_error"
        try:
            return parse_step_answer(raw, step_id=step.id, documents=documents), None
        except InvalidPlanOutput:
            return None, "invalid_output"

    def _rerank_round(
        self, *, query: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str]:
        return self._rerank_candidates(
            query=query, candidates=candidates, respect_round_switch=True
        )

    def _rerank_fallback(
        self, *, query: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str]:
        return self._rerank_candidates(
            query=query, candidates=candidates, respect_round_switch=False
        )

    def _rerank_candidates(
        self, *, query: str, candidates: list[Candidate], respect_round_switch: bool
    ) -> tuple[list[Candidate], str]:
        if (
            (respect_round_switch and not self._round_rerank)
            or self._reranker is None
            or not candidates
        ):
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
        self,
        *,
        state: _State,
        top_ids: Sequence[str],
    ) -> tuple[list[str], tuple[str, ...], int]:
        documents: list[str] = []
        document_ids: list[str] = []
        dropped = 0
        for memory_id in top_ids:
            body = state.bodies.get(memory_id)
            if body is None:
                dropped += 1
                continue
            documents.append(
                _render_document(len(document_ids), body, state.days.get(memory_id, ""))
            )
            document_ids.append(memory_id)
        return documents, tuple(document_ids), dropped

    def _finish(
        self,
        *,
        state: _State,
        records: list[PlanStepRecord],
        stop_reason: str,
        fallback_reason: str | None,
        rerank_calls: int,
        decompose_llm_calls: int,
        step_llm_calls: int,
        plan: Plan | None,
        original: str,
        recovery: PlanRecovery,
        started: float,
    ) -> PlanResult:
        candidates = tuple(
            Candidate(memory_id=memory_id, rank=rank)
            for rank, memory_id in enumerate(state.ordered_ids)
        )
        executed = list(state.executed)
        recovery = replace(recovery, rerank_query_mode="original" if recovery.entered else "joint")
        rerank_query = _final_rerank_query(
            mode=self._final_query_mode,
            executed=executed,
            recovery=recovery,
            original=original,
        )
        steps = () if plan is None else plan.steps
        return PlanResult(
            candidates=candidates,
            rerank_query=rerank_query,
            stats=PlanStats(
                rounds=tuple(records),
                stop_reason=stop_reason,
                fallback_reason=fallback_reason,
                candidate_count=sum(record.recalled_candidates for record in records),
                unique_candidate_count=len(candidates),
                selected_evidence_count=len(state.evidence),
                joint_query_count=len(executed),
                llm_calls=decompose_llm_calls + step_llm_calls,
                rerank_calls=rerank_calls,
                latency_ms=_ms(started),
                plan=steps,
                decompose_llm_calls=decompose_llm_calls,
                step_llm_calls=step_llm_calls,
                plan_steps=len(steps),
                steps_executed=len(executed),
                dependencies_repaired=sum(1 for step in steps if step.dependencies_repaired),
            ),
            fallback_reason=fallback_reason,
            plan_history=tuple(records),
            plan=steps,
            recovery=recovery,
        )

    @staticmethod
    def _bare_record(
        index: int,
        step: PlanStep,
        executed_query: str,
        stop_reason: str,
        *,
        recalled: int = 0,
        candidate_ids: tuple[str, ...] = (),
        dropped: int = 0,
        rerank_status: str = "disabled",
        history_text: str = "",
        retrieval_latency_ms: float = 0.0,
        rerank_latency_ms: float = 0.0,
    ) -> PlanStepRecord:
        return PlanStepRecord(
            index=index,
            step_id=step.id,
            template=step.query,
            executed_query=executed_query,
            recalled_candidates=recalled,
            document_ids=(),
            candidate_ids=candidate_ids,
            evidence_ids=(),
            evidence_quotes=(),
            quotes_unverified=0,
            status=None,
            answer=None,
            canonical_next_query=None,
            history_text=history_text,
            dropped_documents=dropped,
            rerank_status=rerank_status,
            extraction=None,
            stop_reason=stop_reason,
            retrieval_latency_ms=retrieval_latency_ms,
            rerank_latency_ms=rerank_latency_ms,
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

    def _fetch(self, memory_ids: Sequence[str]) -> list[QaPair | None]:
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, list(memory_ids))
        by_id = {pair.id: pair for pair in pairs}
        return [by_id.get(memory_id) for memory_id in memory_ids]


def _final_rerank_query(
    *, mode: str, executed: Sequence[str], recovery: PlanRecovery, original: str
) -> str | None:
    if recovery.entered:
        return original if original.strip() else None
    if mode == "combined" and executed:
        return chr(10).join(executed)
    return None
