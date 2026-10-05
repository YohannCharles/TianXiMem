"""Embedding 后端（§7.4 / §7.2 / §2.3）。

**架构必须 embedder-agnostic**：开发期的 Qwen3-Embedding-8B 与提交期的 `text-embedding-v4`
**都不提供** sparse 或 ColBERT 输出，因此任何依赖多向量能力的代码都是**死重**（D2）。

⚠ 本目录**不自己拼字符串**——索引侧的输入是 [`../common/render.py`](../common/render.py)
的那一份渲染（§7.2 要求 embedding 的输入与返回的 `content` 是同一份）。
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

__all__ = [
    "DEFAULT_TASK",
    "DEFAULT_TEMPLATE",
    "Qwen3EmbeddingEmbedder",
    "QueryInstructionEmbedder",
    "TextEmbeddingV4Embedder",
    "apply_query_instruction",
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
