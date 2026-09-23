"""Embedding 后端（§7.4 / §7.2 / §2.3）。

**架构必须 embedder-agnostic**：`text-embedding-v4` 不提供 sparse 或 ColBERT 输出，
因此任何依赖 BGE-M3 多向量能力的代码在提交时都是**死重**（D2）。

⚠ 本目录**不自己拼字符串**——索引侧的输入是 [`../common/render.py`](../common/render.py)
的那一份渲染（§7.2 要求 embedding 的输入与返回的 `content` 是同一份）。
"""

from tianxi_am.embed.base import (
    CachingEmbedder,
    DimensionMismatchError,
    DimNotKnownError,
    DiskVectorCache,
    Embedder,
    EmbeddingCoordinate,
    EmbeddingError,
    OpenAICompatEmbedder,
    to_float32,
)
from tianxi_am.embed.bge_m3 import BGEM3Embedder

__all__ = [
    "BGEM3Embedder",
    "CachingEmbedder",
    "DimNotKnownError",
    "DimensionMismatchError",
    "DiskVectorCache",
    "Embedder",
    "EmbeddingCoordinate",
    "EmbeddingError",
    "OpenAICompatEmbedder",
    "to_float32",
]
