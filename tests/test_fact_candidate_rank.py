from __future__ import annotations

import pytest

from tianximem.facts.candidate_rank import RelationCandidateRanker
from tianximem.facts.entities import EntityCandidate, ExtractionCandidates, RelationCandidate


def _extraction() -> ExtractionCandidates:
    alice = EntityCandidate("Alice", "PERSON", 0, 5, "alice", "test")
    acme = EntityCandidate("Acme Labs", "ORG", 15, 25, "acme labs", "test")
    oxford = EntityCandidate("Oxford", "ORG", 15, 21, "oxford", "test")
    return ExtractionCandidates(
        entities=(alice, acme, oxford),
        relations=(
            RelationCandidate(alice, "employee", acme, "Alice works for Acme Labs.", "test"),
            RelationCandidate(alice, "student", oxford, "Alice studied at Oxford.", "test"),
        ),
        backend="test",
        model_name="test",
    )


def test_lexical_ranker_is_available_without_model() -> None:
    result = RelationCandidateRanker().rank("Who works for Acme Labs?", _extraction())

    assert result[0].candidate.relation == "employee"
    assert result[0].embedding_score is None
    assert result[0].rerank_score is None


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]][: len(texts)]


class FakeReranker:
    def score(self, *, query: str, documents: list[str]) -> list[float]:
        return [0.2, 0.8]


def test_reranker_controls_order_and_embedding_is_recorded() -> None:
    result = RelationCandidateRanker(
        embedder=FakeEmbedder(), reranker=FakeReranker()
    ).rank("which school", _extraction())

    assert result[0].candidate.relation == "student"
    assert result[0].rerank_score == 0.8
    assert result[0].embedding_score is not None
    assert result[0].score == 0.8


def test_rerank_unavailable_falls_back_to_embedding() -> None:
    class Unavailable:
        def score(self, *, query: str, documents: list[str]) -> list[float]:
            from tianximem.rank import RerankUnavailable

            raise RerankUnavailable("offline")

    result = RelationCandidateRanker(
        embedder=FakeEmbedder(), reranker=Unavailable()
    ).rank("which school", _extraction(), top_k=1)

    assert len(result) == 1
    assert all(item.rerank_score is None for item in result)


def test_invalid_top_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="top_k"):
        RelationCandidateRanker().rank("q", _extraction(), top_k=-1)
