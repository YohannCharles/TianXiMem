from __future__ import annotations

from tianximem.store.qdrant_store import BM25_MODEL, SPARSE_VECTOR

__all__ = [
    "BM25_MODEL",
    "SPARSE_VECTOR",
    "lexical_query",
]


def lexical_query(query: str) -> str:
    if not query.strip():
        raise ValueError("查询不得为空（query 是必填字段）")
    return query
