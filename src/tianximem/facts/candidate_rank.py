"""Offline scoring for entity/relation candidates.

This is an experiment sidecar.  It does not persist facts or alter the Add /
Search path.  A cheap lexical score is always available; an injected embedder
adds semantic similarity and an injected reranker can make the final ordering
model-first without changing the candidate set.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from tianximem.facts.entities import ExtractionCandidates, RelationCandidate

try:  # Optional imports keep this sidecar usable without the model extras.
    from tianximem.embed import Embedder
except ImportError:  # pragma: no cover - only relevant to minimal installs
    Embedder = Any  # type: ignore[misc,assignment]

try:
    from tianximem.rank import Reranker, RerankUnavailable
except ImportError:  # pragma: no cover - only relevant to minimal installs
    RerankUnavailable = RuntimeError  # type: ignore[misc,assignment]
    Reranker = Any  # type: ignore[misc,assignment]

__all__ = ["RankedRelation", "RelationCandidateRanker"]


_TOKEN = re.compile(r"\w+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class RankedRelation:
    candidate: RelationCandidate
    score: float
    lexical_score: float
    embedding_score: float | None
    rerank_score: float | None


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN.findall(value)}


def _lexical_score(query: str, candidate: RelationCandidate) -> float:
    """Score exact endpoint/relation overlap without inventing semantics."""
    query_tokens = _tokens(query)
    candidate_tokens = _tokens(candidate.subject.text)
    candidate_tokens.update(_tokens(candidate.relation))
    if candidate.object is not None:
        candidate_tokens.update(_tokens(candidate.object.text))
    if not query_tokens or not candidate_tokens:
        return 0.0
    overlap = len(query_tokens & candidate_tokens) / len(query_tokens)
    endpoint_hits = sum(
        bool(_tokens(endpoint.text) <= query_tokens)
        for endpoint in (candidate.subject, candidate.object)
        if endpoint is not None
    )
    return min(1.0, overlap + 0.25 * endpoint_hits)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left or not right:
        return 0.0
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def _document(candidate: RelationCandidate) -> str:
    object_text = candidate.object.text if candidate.object is not None else ""
    return f"{candidate.subject.text} {candidate.relation} {object_text}".strip()


class RelationCandidateRanker:
    """Rank candidates while preserving the complete candidate set."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker

    @property
    def mode(self) -> str:
        if self._reranker is not None and self._embedder is not None:
            return "lexical+embedding+rerank"
        if self._reranker is not None:
            return "lexical+rerank"
        if self._embedder is not None:
            return "lexical+embedding"
        return "lexical"

    def rank(
        self,
        query: str,
        extraction: ExtractionCandidates,
        *,
        top_k: int | None = None,
    ) -> tuple[RankedRelation, ...]:
        candidates = extraction.relations
        if not candidates:
            return ()
        documents = [_document(candidate) for candidate in candidates]
        lexical = [_lexical_score(query, candidate) for candidate in candidates]
        embeddings: list[float | None] = [None] * len(candidates)
        if self._embedder is not None:
            vectors = self._embedder.encode([query, *documents])
            if len(vectors) != len(documents) + 1:
                raise ValueError("embedder 返回的向量数量与候选数量不一致")
            embeddings = [
                _cosine(vectors[0], vectors[index + 1]) for index in range(len(documents))
            ]
        rerank_scores: list[float | None] = [None] * len(candidates)
        if self._reranker is not None:
            try:
                scores = list(self._reranker.score(query=query, documents=documents))
            except RerankUnavailable:
                scores = []
            if scores and len(scores) != len(candidates):
                raise ValueError("reranker 返回的分数数量与候选数量不一致")
            if scores:
                rerank_scores = [float(score) for score in scores]

        ranked = []
        for index, candidate in enumerate(candidates):
            rerank_score = rerank_scores[index]
            embedding_score = embeddings[index]
            if rerank_score is not None:
                final_score = rerank_score
            elif embedding_score is not None:
                final_score = 0.65 * embedding_score + 0.35 * lexical[index]
            else:
                final_score = lexical[index]
            ranked.append(
                RankedRelation(
                    candidate=candidate,
                    score=float(final_score),
                    lexical_score=lexical[index],
                    embedding_score=embeddings[index],
                    rerank_score=rerank_scores[index],
                )
            )
        ranked.sort(key=lambda item: item.score, reverse=True)
        if top_k is not None:
            if top_k < 0:
                raise ValueError("top_k 不能为负数")
            ranked = ranked[:top_k]
        return tuple(ranked)
