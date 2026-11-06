from __future__ import annotations

from typing import Final

from tianximem.common.tokens import (
    EMBED_MAX_TOKENS,
    O200kCounter,
    TokenCounterError,
    load_counter,
)

__all__ = ["TRUNCATION_MARKER", "truncate_for_embedding"]

TRUNCATION_MARKER: Final[str] = "\n...\n"

_BUDGET_RATIO: Final[float] = 0.56

_DECODE_SLACK: Final[int] = 16

_DEFAULT_LIMIT: Final[int] = int(EMBED_MAX_TOKENS * _BUDGET_RATIO)


def _o200k() -> O200kCounter:
    counter = load_counter()
    if not isinstance(counter, O200kCounter):
        raise TokenCounterError(
            "truncate_for_embedding 需要能切分 token id 的 O200kCounter，"
            f"当前拿到的是 {type(counter).__name__}"
        )
    return counter


def truncate_for_embedding(text: str, *, max_tokens: int | None = None) -> str:
    limit = _DEFAULT_LIMIT if max_tokens is None else max_tokens
    if limit <= 0:
        raise ValueError(f"max_tokens 必须为正，收到 {limit}")
    counter = _o200k()
    ids = counter.encode_ids(text)
    if len(ids) <= limit:
        return text

    marker_ids = counter.encode_ids(TRUNCATION_MARKER)
    budget = limit - len(marker_ids) - _DECODE_SLACK
    if budget < 2:
        return counter.decode_ids(ids[:limit])

    head_n = budget // 2
    tail_n = budget - head_n
    head = counter.decode_ids(ids[:head_n])
    tail = counter.decode_ids(ids[len(ids) - tail_n :])
    return f"{head}{TRUNCATION_MARKER}{tail}"
