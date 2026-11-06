# ruff: noqa: E501


from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tianximem.llm.base import DEFAULT_MAX_OUTPUT_TOKENS, LlmClient, LlmUnavailable

__all__ = [
    "NONE_OUTPUT",
    "ROUTER_PROMPT",
    "ROUTER_PROMPT_VERSION",
    "ROUTE_STATUSES",
    "Route",
    "RouteDecision",
    "QueryRouter",
    "parse_route",
]


class Route(StrEnum):
    MULTI_HOP = "MULTI_HOP"
    DEFAULT = "DEFAULT"


ROUTER_PROMPT_VERSION: Final[str] = "v5"

ROUTER_PROMPT: Final[
    str
] = """You are a  router. Your task is to select exactly ONE tool name from the provided list that best fits the user query. Do not call any tools. Use only the text in {query}; do not assume external context or missing metadata. You may interpret dependency only from explicit linguistic structure (e.g., "use X to find Y", possessive/relationship chains).

GOAL
- Choose exactly one of: MULTI_HOP, DEFAULT.

MECHANISM (what to do, then how to do it)

1) Classify the query type from {query} using ONLY the operational criteria below (pick exactly one):

A) MULTI-HOP (dependency chain; later step depends on earlier result)
Choose MULTI-HOP if the query requires two or more dependent steps where you must first determine X, then use X to determine Y (or more).
- Explicit dependency signals: "then", "after", "using that", "based on that result", "from there", "which of those", "once you find", "given the answer to", "trace/derive".
- Relationship-chain dependency: queries that require resolving an intermediate entity/attribute to proceed (e.g., "X's spouse's company", "author of the book that inspired the film").
- Comparison/timeline becomes MULTI-HOP ONLY if it requires derived attributes that must be found first (e.g., "Which is older, the CEO of A or the founder of B?" requires finding CEO/founder first).
- Tie-breaker: If any explicit dependency chain exists, classify as MULTI-HOP even if multiple entities/keywords appear.

B) DEFAULT (direct or independent retrieval)

Choose DEFAULT when the query is a straightforward lookup or contains multiple independent lookups that do not require dependent resolution.

- Multiple known entities, subjects, or questions without dependency belong here.
- Direct comparisons of independently identifiable facts belong here.
- Use DEFAULT when A does not apply.

2) Deterministic routing (hard mapping)

- If MULTI-HOP dependency exists -> MULTI_HOP.
- Else -> DEFAULT.

AVAILABLE ROUTES

MULTI_HOP: Iterative retrieval that searches step by step, uses retrieved evidence to identify missing relationships, and generates subsequent queries.

DEFAULT: Standard hybrid memory retrieval without iterative query rewriting.

INPUT
Query:
{query}

OUTPUT FORMAT (strict)
Output a single line containing only:
MULTI_HOP
  DEFAULT

Do not output explanations, confidence scores, JSON, or any additional text.

CLASSIFICATION EXAMPLES (for calibration; do not output these)

1) "Who is the author of Dune?" -> DEFAULT
2) "Give the capitals of Spain and Portugal." -> DEFAULT
3) "Find the spouse of Marie Curie, then name his primary field." -> MULTI_HOP
4) "Compare GDP per capita of Japan and South Korea." -> DEFAULT
5) "Which is older: the founder of Company A or the CEO of Company B?" -> MULTI_HOP
6) "What type of car did Evan get after his old Prius broke down?" -> MULTI_HOP
7) "How many conferences did the person who hired Melanie attend?" -> MULTI_HOP"""


NONE_OUTPUT: Final[str] = "NONE"


def is_none_output(text: str) -> bool:
    return text.strip().upper() == NONE_OUTPUT


def parse_route(text: str) -> Route | None:
    parsed = text.strip().upper()
    for route in Route:
        if parsed == route.value:
            return route
    return None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    route: Route
    status: str
    reason: str | None
    raw: str
    latency_ms: float


ROUTE_STATUSES: Final[tuple[str, ...]] = ("success", "degraded")


class QueryRouter:
    def __init__(
        self,
        client: LlmClient,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        prompt_version: str = ROUTER_PROMPT_VERSION,
    ) -> None:
        if prompt_version != ROUTER_PROMPT_VERSION:
            raise ValueError(
                f"`router.prompt_version` 必须是 {ROUTER_PROMPT_VERSION!r}，"
                f"收到 {prompt_version!r}。\n"
                "  它是一条【断言】不是说明：本实现带的是哪一版提示词，配置就得声称哪一版。\n"
                "  ⚠ 换了提示词，就必须同时改 `ROUTER_PROMPT_VERSION` 与 `configs/*.yaml`——"
                "否则两轮实验的 run record 会声称同一份提示词，而那种漂移查不出来。"
            )
        if max_output_tokens <= 0:
            raise ValueError(f"`router.max_output_tokens` 必须为正：{max_output_tokens}")
        if temperature < 0:
            raise ValueError(f"`router.temperature` 不得为负：{temperature}")
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._prompt_version = prompt_version
        self._prompt = ROUTER_PROMPT

    @property
    def client(self) -> LlmClient:
        return self._client

    @property
    def model(self) -> str:
        return self._client.name

    @property
    def prompt_version(self) -> str:
        return self._prompt_version

    def route(self, query: str) -> RouteDecision:
        started = time.perf_counter()
        prompt = self._prompt.replace("{query}", query)
        try:
            raw = self._client.complete(
                prompt,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )
        except LlmUnavailable as exc:
            return self._degrade("llm_unavailable", str(exc), started)
        except Exception as exc:  # noqa: BLE001
            return self._degrade("llm_error", f"{type(exc).__name__}: {exc}", started)

        if is_none_output(raw):
            return self._degrade("model_none", None, started, raw=raw)
        route = parse_route(raw)
        if route is None:
            return self._degrade("invalid_output", None, started, raw=raw)
        return RouteDecision(
            route=route,
            status="success",
            reason=None,
            raw=raw,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    def _degrade(
        self, reason: str, detail: str | None, started: float, *, raw: str = ""
    ) -> RouteDecision:
        return RouteDecision(
            route=Route.DEFAULT,
            status="degraded",
            reason=f"{reason}: {detail}" if detail else reason,
            raw=raw,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
