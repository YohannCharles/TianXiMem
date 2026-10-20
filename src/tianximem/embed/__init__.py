"""Embedding 后端。

开发期的 Qwen3-Embedding-4B 与提交期的 `text-embedding-v4` 都只提供 dense 输出，
因此实现不依赖 sparse、ColBERT 或其他多向量能力。

本包不自行拼接文档文本；索引输入来自 [`../common/render.py`](../common/render.py)，
并与返回 `content` 共享同一份基础渲染。
"""

from tianximem.embed.base import (
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
from tianximem.embed.query_instruction import (
    DEFAULT_TASK,
    DEFAULT_TEMPLATE,
    QueryInstructionEmbedder,
    apply_query_instruction,
)
from tianximem.embed.qwen3_embedding import Qwen3EmbeddingEmbedder
from tianximem.embed.text_embedding_v4 import TextEmbeddingV4Embedder
from tianximem.embed.truncate import TRUNCATION_MARKER, truncate_for_embedding

__all__ = [
    "DEFAULT_TASK",
    "DEFAULT_TEMPLATE",
    "Qwen3EmbeddingEmbedder",
    "QueryInstructionEmbedder",
    "TRUNCATION_MARKER",
    "TextEmbeddingV4Embedder",
    "apply_query_instruction",
    "truncate_for_embedding",
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
