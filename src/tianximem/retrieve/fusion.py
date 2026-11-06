from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tianximem.common.config import DEFAULT_PREFETCH_LIMIT, DEFAULT_WEIGHTS
from tianximem.retrieve.bm25 import lexical_query
from tianximem.retrieve.dense import DenseArm
from tianximem.store.qdrant_store import HybridParams, QdrantStore

__all__ = [
    "DEFAULT_PREFETCH_LIMIT",
    "DEFAULT_WEIGHTS",
    "RRF_K",
    "Candidate",
    "HybridRetriever",
    "dedup_candidates",
    "make_hybrid_params",
]


RRF_K: Final[int] = 61


class RetrievalParamError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Candidate:
    memory_id: str
    rank: int


def make_hybrid_params(
    *,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    rrf_k: int = RRF_K,
) -> HybridParams:
    if prefetch_limit <= 0:
        raise RetrievalParamError(f"prefetch_limit 必须为正：{prefetch_limit}")
    if len(weights) != 2:
        raise RetrievalParamError(
            f"weights 必须是两个（先后对应 bm25、dense 的 prefetch 顺序）：{weights!r}"
        )
    w_bm25, w_dense = float(weights[0]), float(weights[1])
    if w_bm25 < 0 or w_dense < 0:
        raise RetrievalParamError(f"权重不得为负：{(w_bm25, w_dense)!r}")
    if w_bm25 == 0 and w_dense == 0:
        raise RetrievalParamError("两个权重不能同时为 0——那等于不检索")
    if rrf_k != RRF_K:
        raise RetrievalParamError(
            f"rrf_k 必须是 {RRF_K}，收到 {rrf_k}。\n"
            "  它是【正确性常量】不是可调项：Qdrant 的 k 默认是 2、RRF 文献是 60，"
            f"Qdrant 的秩从 0 开始，只有 {RRF_K} 才等价于文献公式中的 60。\n"
            "  填错【不会报错】——两种写法的名次都看起来正常，只有分数差一个数量级。"
        )
    return HybridParams(
        prefetch_limit=prefetch_limit,
        weights=(w_bm25, w_dense),
        rrf_k=rrf_k,
    )


def dedup_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen: dict[str, int] = {}
    out: list[Candidate] = []
    for candidate in candidates:
        if candidate.memory_id in seen:
            continue
        seen[candidate.memory_id] = len(out)
        out.append(Candidate(memory_id=candidate.memory_id, rank=len(out)))
    return out


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    store: QdrantStore
    dense: DenseArm

    def search(self, *, user_id: str, query: str, top_k: int) -> list[Candidate]:
        if top_k <= 0:
            return []
        dense_vector = self.dense.encode_query(query)
        hits = self.store.hybrid_search(
            user_id=user_id,
            query_text=lexical_query(query),
            dense_vector=dense_vector,
        )
        ordered = sorted(hits, key=lambda h: (-h.score, h.memory_id))[:top_k]
        return [Candidate(memory_id=h.memory_id, rank=rank) for rank, h in enumerate(ordered)]
