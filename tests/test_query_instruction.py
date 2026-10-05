"""查询侧 instruction 兼容层（§7.2 / `src/tianximem/embed/query_instruction.py`）。

**这组测试守的是一条静默失败的边界**：前缀加错位置时结果依然"看起来正常"，
所以必须由测试来钉——而不是靠读代码。
"""

from __future__ import annotations

from collections.abc import Sequence

from tianximem.embed.base import CachingEmbedder, DiskVectorCache, EmbeddingCoordinate
from tianximem.embed.query_instruction import (
    DEFAULT_TASK,
    DEFAULT_TEMPLATE,
    QueryInstructionEmbedder,
    apply_query_instruction,
)


class RecordingEmbedder:
    """把收到的文本逐字记下来，并返回一个**随文本长度变化**的向量。

    向量随文本变，是为了让"前缀有没有真的送进去"变成可断言的事实——
    固定向量会让加没加前缀看起来一样。
    """

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.seen: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        self.seen.extend(texts)
        return [[float(len(text))] * self._dim for text in texts]


# ── 模板本身 ───────────────────────────────────────────────────────────


def test_template_shape_has_no_space_after_query_colon() -> None:
    """`Query:` 之后**没有空格**——这是模型卡的逐字形状，不要"顺手"加上。"""
    assert DEFAULT_TEMPLATE == "Instruct: {task}\nQuery:{query}"
    assert apply_query_instruction("英雄联盟", "T") == "Instruct: T\nQuery:英雄联盟"


def test_empty_instruction_is_bytewise_identity() -> None:
    """空 instruction ⇒ **逐字节原样**。v1 的规格（query 不改写）由此保持不变。"""
    assert apply_query_instruction("q", "") == "q"
    inner = RecordingEmbedder()
    QueryInstructionEmbedder(inner).encode_query(["q"])
    assert inner.seen == ["q"]


# ── 两侧必须不同 ───────────────────────────────────────────────────────


def test_documents_are_never_prefixed_and_queries_are() -> None:
    inner = RecordingEmbedder()
    emb = QueryInstructionEmbedder(inner, instruction="T")
    emb.encode(["doc"])
    emb.encode_query(["q"])
    assert inner.seen == ["doc", "Instruct: T\nQuery:q"]


def test_dim_is_forwarded() -> None:
    inner = RecordingEmbedder(dim=7)
    assert QueryInstructionEmbedder(inner, instruction="T").dim == 7


# ── 那条不可违反的顺序：前缀在缓存**之上** ─────────────────────────────


def test_prefix_lands_above_the_cache(tmp_path) -> None:
    """命中缓存与未命中必须给出**同一个**向量，且缓存键是**带前缀**的那份文本。

    若把前缀放到缓存之下，这里会看到"第二次的向量与第一次不同"——而生产里
    它只会表现为"检索结果随缓存状态波动"（不报错）。
    """
    inner = RecordingEmbedder()
    cache = DiskVectorCache(tmp_path, EmbeddingCoordinate(model="qwen3-embedding-8b"))
    emb = QueryInstructionEmbedder(CachingEmbedder(inner, cache), instruction=DEFAULT_TASK)

    first = emb.encode_query(["q"])
    calls_after_miss = len(inner.seen)
    second = emb.encode_query(["q"])  # 应命中缓存

    assert second == first
    assert len(inner.seen) == calls_after_miss, "缓存命中时不该再调用内层"
    assert cache.get(apply_query_instruction("q", DEFAULT_TASK)) is not None
    assert cache.get("q") is None, "裸查询不该进缓存——否则前缀等于没加"


def test_changing_the_instruction_changes_the_cache_entry(tmp_path) -> None:
    """换 instruction ⇒ 换缓存键 ⇒ 内层被重新调用（不会静默复用另一套前缀的向量）。"""
    inner = RecordingEmbedder()
    cache = DiskVectorCache(tmp_path, EmbeddingCoordinate(model="qwen3-embedding-8b"))
    caching = CachingEmbedder(inner, cache)

    QueryInstructionEmbedder(caching, instruction="A").encode_query(["q"])
    calls = len(inner.seen)
    QueryInstructionEmbedder(caching, instruction="B").encode_query(["q"])

    assert len(inner.seen) > calls, "换了 instruction 却命中了旧缓存——这是静默取到错向量"
