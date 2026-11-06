from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tianximem.common.render import SEGMENT_SEP, day_granularity
from tianximem.common.tokens import TokenCounter
from tianximem.rank.neighbor import ContextSegment

__all__ = [
    "PackagedResponse",
    "ResponseItem",
    "day_granularity",
    "package",
    "placeholder_score",
]

UTC_ONLY: Final[str] = "UTC"


def placeholder_score(rank: int) -> float:
    if rank < 0:
        raise ValueError(f"rank 不得为负：{rank}")
    return 1.0 / (rank + 1.0)


@dataclass(frozen=True, slots=True)
class ResponseItem:
    id: str
    content: str
    created_at: str
    score: float


@dataclass(frozen=True, slots=True)
class PackagedResponse:
    items: tuple[ResponseItem, ...]
    dropped_missing: int = 0
    considered_segments: int = 0
    truncated_by_top_k: bool = False
    truncated_by_budget: bool = False
    output_tokens: int = 0

    @property
    def count(self) -> int:
        return len(self.items)


def _fit_by_budget(
    segments: Sequence[ContextSegment],
    *,
    counter: TokenCounter,
    max_tokens: int,
) -> tuple[list[ContextSegment], bool]:
    if max_tokens <= 0:
        return [], bool(segments)

    separator_tokens = counter.count(SEGMENT_SEP)
    chosen: list[ContextSegment] = []
    used = 0
    for segment in segments:
        extra = segment.token_count + (separator_tokens if chosen else 0)
        if used + extra > max_tokens:
            return chosen, True
        used += extra
        chosen.append(segment)
    return chosen, False


def package(
    segments: Sequence[ContextSegment],
    *,
    top_k: int,
    counter: TokenCounter,
    max_tokens: int,
    dropped_missing: int = 0,
) -> PackagedResponse:
    total_segments = len(segments)
    if top_k <= 0 or total_segments == 0:
        return PackagedResponse(items=(), dropped_missing=dropped_missing)

    head = list(segments)[:top_k]
    truncated_by_top_k = total_segments > len(head)

    chosen, truncated_by_budget = _fit_by_budget(head, counter=counter, max_tokens=max_tokens)

    items = tuple(
        ResponseItem(
            id=segment.anchor_memory_id,
            content=segment.content,
            created_at=day_granularity(segment.anchor_event_time),
            score=placeholder_score(output_rank),
        )
        for output_rank, segment in enumerate(chosen)
    )

    output_text = SEGMENT_SEP.join(segment.content for segment in chosen)
    return PackagedResponse(
        items=items,
        dropped_missing=dropped_missing,
        considered_segments=total_segments,
        truncated_by_top_k=truncated_by_top_k,
        truncated_by_budget=truncated_by_budget,
        output_tokens=counter.count(output_text),
    )
