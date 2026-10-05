"""检索与判据 —— rerank 之前的那一半（§7.1–§7.3、§8）。

**本目录只负责"拿到候选 + 判断够不够"。** rerank 之后的链路
（**Neighbor Expansion、Context Packaging**）在 [`../rank/`](../rank/)。

## 与 `store/` 的分界

**参数归本目录，执行归 `store/qdrant_store.py`**：`prefetch_limit` / `weights` / `k` /
`top_k` 的取值、**校验**、标定在这里，翻成 `prefetch` + `rrf` 的 Qdrant 调用在那里。
⇒ 一句话：**本目录说"用什么参数"，`store/` 说"怎么发给 Qdrant"**（[`CLAUDE.md`](./CLAUDE.md)）。
"""

from tianximem.retrieve.bm25 import BM25_MODEL, SPARSE_VECTOR, lexical_query
from tianximem.retrieve.checker import (
    CheckerDecision,
    CheckerInstrument,
    CheckerThresholds,
    EvidenceChecker,
    InMemoryCheckerInstrument,
    NullCheckerInstrument,
    rank_consistency,
)
from tianximem.retrieve.dense import DenseArm, DenseArmError
from tianximem.retrieve.fusion import (
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_WEIGHTS,
    RRF_K,
    Candidate,
    HybridRetriever,
    RetrievalParamError,
    dedup_candidates,
    make_hybrid_params,
)

__all__ = [
    "BM25_MODEL",
    "DEFAULT_PREFETCH_LIMIT",
    "DEFAULT_WEIGHTS",
    "RRF_K",
    "SPARSE_VECTOR",
    "Candidate",
    "dedup_candidates",
    "CheckerDecision",
    "CheckerInstrument",
    "CheckerThresholds",
    "DenseArm",
    "DenseArmError",
    "EvidenceChecker",
    "HybridRetriever",
    "InMemoryCheckerInstrument",
    "NullCheckerInstrument",
    "RetrievalParamError",
    "lexical_query",
    "make_hybrid_params",
    "rank_consistency",
]
