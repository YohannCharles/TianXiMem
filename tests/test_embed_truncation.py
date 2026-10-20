"""查询侧超长输入的前置截断（`src/tianximem/embed/truncate.py`）。

**这组测试守的是一条静默边界**：截断发生在缓存之下、前缀之后——顺序错了结果依然
"看起来正常"（能拿到向量），只是检索质量悄悄变了，或者不同查询共享了同一个缓存条目。
"""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from tests.conftest import FakeEmbedder

from tianximem.common.tokens import load_counter
from tianximem.embed.base import (
    QUERY_CACHE_PREFIX,
    CachingEmbedder,
    DiskVectorCache,
    EmbeddingCoordinate,
    OpenAICompatEmbedder,
)
from tianximem.embed.query_instruction import QueryInstructionEmbedder
from tianximem.embed.truncate import TRUNCATION_MARKER, truncate_for_embedding

COORD = EmbeddingCoordinate(model="Qwen/Qwen3-Embedding-8B")

LONG = "HEADMARK " + " ".join(f"filler{i}" for i in range(400)) + " TAILMARK"


def _count(text: str) -> int:
    return load_counter().count(text)


# ── 纯函数 ─────────────────────────────────────────────────────────────


def test_short_text_is_returned_bytewise() -> None:
    """未超限 ⇒ **逐字原样**——检索输入不能因为这条防线而漂移。"""
    assert truncate_for_embedding("hello world") == "hello world"
    assert truncate_for_embedding("") == ""


def test_long_text_is_capped_and_keeps_both_ends() -> None:
    """超限 ⇒ token 数严格 <= 上限，且**头尾都保留**（问题可能在任一端）。"""
    out = truncate_for_embedding(LONG, max_tokens=64)
    assert _count(out) <= 64
    assert out.startswith("HEADMARK")
    assert out.endswith("TAILMARK")
    assert TRUNCATION_MARKER in out
    assert out != LONG


def test_marker_itself_is_counted_in_the_budget() -> None:
    """标记不是免费的——预算必须扣掉它，否则结果会贴着窗口边缘超出去。"""
    lim = 40
    out = truncate_for_embedding(LONG, max_tokens=lim)
    assert _count(out) <= lim
    # 如果标记没被扣掉，头尾会各多占 2 token 左右；这里用更紧的上限证明它被算了进去
    assert _count(out) <= lim


def test_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="必须为正"):
        truncate_for_embedding("x", max_tokens=0)


# ── 适配层：查询侧截断，文档侧原样 ─────────────────────────────────────


def _ok_body(dim: int, n: int = 1) -> dict:
    return {"data": [{"index": i, "embedding": [float(j) for j in range(dim)]} for i in range(n)]}


class _Resp:
    def __init__(self, body: dict) -> None:
        self._json = body
        self.status_code = 200
        self.text = json.dumps(body)

    def json(self) -> dict:
        return self._json


class _Recorder:
    """记录每次 `/embeddings` 请求里的 `input`。**不碰网络。**"""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim
        self.inputs: list[list[str]] = []

    def post(self, url, json=None, headers=None):  # noqa: ANN001, ANN201
        self.inputs.append(list(json["input"]))
        return _Resp(_ok_body(self._dim, len(json["input"])))

    def close(self) -> None:
        pass


def test_documents_are_not_truncated() -> None:
    """索引侧**原样发送**：超窗在那边是数据 bug，必须让网关响亮失败。"""
    rec = _Recorder()
    emb = OpenAICompatEmbedder(base_url="http://x", api_key="k", model="m", client=rec)
    emb.encode([LONG])
    assert rec.inputs == [[LONG]]


def test_queries_are_truncated_before_they_leave_the_process() -> None:
    """查询侧：桩客户端看到的**已经是截断后的文本**——不是等网关拒绝。"""
    rec = _Recorder()
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    emb.encode_query([LONG])
    sent = rec.inputs[0][0]
    assert sent != LONG
    assert _count(sent) <= 64
    assert sent.startswith("HEADMARK")
    assert sent.endswith("TAILMARK")


# ── 缓存键必须仍然是**原文** ───────────────────────────────────────────


def _wrap(inner, tmp_path) -> tuple[CachingEmbedder, DiskVectorCache]:  # noqa: ANN001
    cache = DiskVectorCache(tmp_path / "cache", COORD)
    return CachingEmbedder(inner, cache), cache


def test_cache_key_is_the_original_text_with_a_query_role_prefix(tmp_path) -> None:
    """截断在缓存**之下**：键是原文，但查询侧带角色前缀、文档侧不带。

    两侧都不带前缀时，先文档后查询（或反过来）会静默复用另一侧的向量——
    因为原文相同不再意味着模型输入相同（报告 §7.4 的两个复现顺序）。
    """
    rec = _Recorder()
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    emb, cache = _wrap(inner, tmp_path)
    emb.encode_query([LONG])
    assert cache.get(QUERY_CACHE_PREFIX + LONG) is not None, "查询键 = 角色前缀 + 原文"
    assert cache.get(LONG) is None, "文档键不能被查询写入"
    assert cache.get(truncate_for_embedding(LONG, max_tokens=64)) is None, "键不能是截断文本"


def test_document_then_query_never_shares_a_vector(tmp_path) -> None:
    """先 `encode` 再 `encode_query`：查询必须**重新调用内层**，不能复用完整文档向量。"""
    rec = _Recorder()
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    emb, _ = _wrap(inner, tmp_path)
    emb.encode([LONG])
    emb.encode_query([LONG])
    assert len(rec.inputs) == 2
    assert rec.inputs[0] == [LONG]  # 文档侧原样
    assert rec.inputs[1][0] != LONG  # 查询侧截断


def test_query_then_document_never_shares_a_vector(tmp_path) -> None:
    """反过来：先 `encode_query` 再 `encode`，文档必须拿到**完整文本**的向量。"""
    rec = _Recorder()
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    emb, _ = _wrap(inner, tmp_path)
    emb.encode_query([LONG])
    emb.encode([LONG])
    assert len(rec.inputs) == 2
    assert rec.inputs[0][0] != LONG  # 查询侧截断
    assert rec.inputs[1] == [LONG]  # 文档侧原样


def test_distinct_long_queries_do_not_share_a_cache_entry(tmp_path) -> None:
    """两条只在**被截掉的中段**不同的查询，不能让第二条命中第一条的向量。"""
    rec = _Recorder()
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    emb, _ = _wrap(inner, tmp_path)
    other = "HEADMARK " + " ".join(f"other{i}" for i in range(400)) + " TAILMARK"
    emb.encode_query([LONG])
    emb.encode_query([other])
    assert len(rec.inputs) == 2, "第二条查询命中了第一条的缓存——键不该是截断后的文本"


def test_second_call_of_the_same_long_query_hits_the_cache(tmp_path) -> None:
    """同一个超长查询重试 ⇒ 不再打网关（缓存在截断之前生效）。"""
    rec = _Recorder()
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    emb, _ = _wrap(inner, tmp_path)
    first = emb.encode_query([LONG])
    second = emb.encode_query([LONG])
    assert second == first
    assert len(rec.inputs) == 1


# ── 前缀在缓存之上、截断在发送之前 ─────────────────────────────────────


def test_instruction_prefix_is_applied_before_truncation(tmp_path) -> None:
    """带前缀的全文进缓存；送到网关的是**带前缀**、已截断的文本。"""
    rec = _Recorder()
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=rec, max_query_tokens=64
    )
    cache = DiskVectorCache(tmp_path / "cache", COORD)
    emb = QueryInstructionEmbedder(CachingEmbedder(inner, cache), instruction="T")
    emb.encode_query([LONG])

    sent = rec.inputs[0][0]
    assert sent.startswith("Instruct: T\nQuery:HEADMARK")
    assert _count(sent) <= 64
    assert cache.get(QUERY_CACHE_PREFIX + "Instruct: T\nQuery:" + LONG) is not None


def test_no_encode_query_on_the_inner_falls_back_to_encode(tmp_path) -> None:
    """内层没有 `encode_query`（测试 / 归档实现）⇒ 退回 `encode`，不炸。"""
    inner = FakeEmbedder()
    emb, _ = _wrap(inner, tmp_path)
    vec = emb.encode_query(["hello"])
    assert len(vec) == 1


class _RecordingInner:
    """记录 `encode_query` / `encode` 各被调用了几次。"""

    def __init__(self) -> None:
        self.query_calls: list[list[str]] = []
        self.doc_calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return 3

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        self.doc_calls.append(list(texts))
        return [[1.0, 2.0, 3.0] for _ in texts]

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]:
        self.query_calls.append(list(texts))
        return [[1.0, 2.0, 3.0] for _ in texts]


def test_caching_embedder_routes_query_to_inner_encode_query(tmp_path) -> None:
    """`CachingEmbedder.encode_query` 必须把 miss 交给内层的 `encode_query`——
    否则截断只写在 `OpenAICompatEmbedder` 上，而装配链里根本走不到它。"""
    inner = _RecordingInner()
    emb, _ = _wrap(inner, tmp_path)
    emb.encode_query(["q"])
    assert inner.query_calls == [["q"]]
    assert inner.doc_calls == []
