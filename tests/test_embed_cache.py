"""Embedding 后端：维度来源、落盘缓存、坐标系失效、HTTP 响应解析。

对应 [`../tests/README.md`](../tests/README.md) 与 embed/README.md §缓存。
⚠ **全部用例都不碰网络**——缓存与维度用 `FakeEmbedder`，HTTP 用桩客户端。
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest
from tests.conftest import FakeEmbedder

from tianxi_am.common.render import TEMPLATE_VERSION
from tianxi_am.embed.base import (
    CachingEmbedder,
    DimensionMismatchError,
    DimNotKnownError,
    DiskVectorCache,
    EmbeddingCoordinate,
    EmbeddingError,
    OpenAICompatEmbedder,
)
from tianxi_am.embed.bge_m3 import DEFAULT_MODEL, BGEM3Embedder

COORD = EmbeddingCoordinate(model="BAAI/bge-m3")


def _wrap(inner: FakeEmbedder, tmp_path, coord: EmbeddingCoordinate = COORD):
    cache = DiskVectorCache(tmp_path / "cache", coord)
    return CachingEmbedder(inner, cache), cache


# ── 维度必须来自真实返回 ────────────────────────────────────────────────


def test_dim_is_unknown_before_first_call() -> None:
    """**维度只从真实返回里取**——第一次调用之前访问 `dim` 必须响亮地失败。

    这是"不许写死常量"的落点：集合的向量维度必须来自接口的真实输出。
    """
    subject = OpenAICompatEmbedder(base_url="http://x", api_key="k", model="m")
    with pytest.raises(DimNotKnownError, match="尚未确定"):
        _ = subject.dim


def test_dim_comes_from_actual_response() -> None:
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient(_ok_body(5))
    )
    assert emb.encode(["a"]) == [[float(i) for i in range(5)]]
    assert emb.dim == 5  # ← 来自响应，不是声明


def test_dim_change_across_calls_fails_loudly() -> None:
    """同一坐标系下维度变了 ⇒ 响亮失败（否则只会表现为"检索结果很差"）。"""
    client = _StubClient(_ok_body(4))
    emb = OpenAICompatEmbedder(base_url="http://x", api_key="k", model="m", client=client)
    emb.encode(["a"])
    client.body = _ok_body(7)
    with pytest.raises(DimensionMismatchError, match="先前返回 4 维"):
        emb.encode(["b"])


def test_cached_embedder_falls_back_to_cache_dim(tmp_path) -> None:
    """内层还没调过时，若缓存里只有一个维度，可以用它——但它来自**缓存**，不是常量。

    ⚠ 这里必须用**真正的懒维度实现**（`OpenAICompatEmbedder`），不能用 `FakeEmbedder`
    ——后者总是立刻知道自己的维度，会掩盖这条回退路径。
    """
    inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient(_ok_body(6))
    )
    wrapped, _ = _wrap(inner, tmp_path)
    with pytest.raises(DimNotKnownError):
        _ = wrapped.dim  # 内层没调过、缓存也空 ⇒ 维度未知

    wrapped.encode(["hello world"])

    fresh_inner = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient(_ok_body(6))
    )
    fresh, _ = _wrap(fresh_inner, tmp_path)  # 同一个缓存目录
    assert fresh.dim == 6  # 来自缓存里已出现的维度
    fresh.encode(["hello world"])
    assert fresh_inner._client.calls == []  # 而且真的命中了缓存


# ── 缓存：键 = 渲染文本的内容哈希 ───────────────────────────────────────


def test_first_call_hits_inner_second_call_does_not(tmp_path) -> None:
    """**缓存命中时不重复调用远程模型**——这是 §7.2 那条"一次性成本"的落点。"""
    inner = FakeEmbedder()
    wrapped, _ = _wrap(inner, tmp_path)

    first = wrapped.encode(["Q: a\nA: b"])
    assert inner.call_count == 1
    second = wrapped.encode(["Q: a\nA: b"])

    assert inner.call_count == 1  # ← 没有第二次调用
    assert first == second  # 逐位相同


def test_same_text_across_instances_reuses_cache(tmp_path) -> None:
    """**相同渲染文本、不同 memory_id 必须能复用。**

    缓存键里根本没有 `id`——所以"不同 memory_id、相同文本"复用是**结构性**的，
    不依赖任何调用方记得传对 id。
    """
    inner_a = FakeEmbedder()
    wrapped_a, _ = _wrap(inner_a, tmp_path)
    vec = wrapped_a.encode(["Q: same\nA: same"])[0]

    inner_b = FakeEmbedder()  # 全新内层（模拟另一个 memory_id / 另一个进程）
    wrapped_b, _ = _wrap(inner_b, tmp_path)
    assert wrapped_b.encode(["Q: same\nA: same"])[0] == vec
    assert inner_b.call_count == 0  # 直接命中


def test_different_text_does_not_reuse_cache(tmp_path) -> None:
    """**不同渲染文本不能错误复用。**"""
    inner = FakeEmbedder()
    wrapped, _ = _wrap(inner, tmp_path)
    a = wrapped.encode(["Q: a"])[0]
    b = wrapped.encode(["Q: b"])[0]
    assert a != b
    assert inner.call_count == 2


def test_cache_is_persisted_on_disk(tmp_path) -> None:
    """必须落盘——否则每次重启重付一遍整个数据集。"""
    inner = FakeEmbedder()
    cache = DiskVectorCache(tmp_path / "cache", COORD)
    wrapped = CachingEmbedder(inner, cache)
    wrapped.encode(["Q: persisted"])
    cache.close()

    assert cache.path.exists()

    inner2 = FakeEmbedder()
    cache2 = DiskVectorCache(tmp_path / "cache", COORD)  # 同一个目录，重新打开
    wrapped2 = CachingEmbedder(inner2, cache2)
    wrapped2.encode(["Q: persisted"])
    assert inner2.call_count == 0  # 跨"进程"命中


def test_batch_deduplicates_within_one_call(tmp_path) -> None:
    inner = FakeEmbedder()
    wrapped, _ = _wrap(inner, tmp_path)

    out = wrapped.encode(["same", "same", "other"])

    assert len(out) == 3
    assert out[0] == out[1]
    assert inner.encoded_texts().count("same") == 1  # 批内去重


def test_empty_input_calls_nothing(tmp_path) -> None:
    inner = FakeEmbedder()
    wrapped, _ = _wrap(inner, tmp_path)
    assert wrapped.encode([]) == []
    assert inner.call_count == 0


def test_hit_and_miss_return_identical_precision(tmp_path) -> None:
    """**缓存命中与未命中必须返回逐位相同的向量。**

    否则"检索结果依赖缓存状态" ⇒ 不可复现，而这是本项目最不能接受的失败类型。
    """
    inner = FakeEmbedder()
    wrapped, _ = _wrap(inner, tmp_path)
    fresh = wrapped.encode(["Q: precision"])[0]
    cached = wrapped.encode(["Q: precision"])[0]
    assert fresh == cached
    assert all(isinstance(x, float) for x in cached)


# ── 坐标系：换模型/换模板必须整体失效 ──────────────────────────────────


def test_coordinate_key_is_part_of_the_file_name(tmp_path) -> None:
    """坐标系进**文件名** ⇒ "整体失效"是结构性的，不靠人去记得清缓存。"""
    a = DiskVectorCache(tmp_path, EmbeddingCoordinate(model="model-a"))
    b = DiskVectorCache(tmp_path, EmbeddingCoordinate(model="model-b"))
    assert a.path != b.path
    assert a.coordinate.key() in a.path.name


def test_model_change_does_not_hit_old_cache(tmp_path) -> None:
    """**换模型后旧缓存不得静默命中。**

    静默命中的后果：拿 A 模型的向量去查 B 模型的集合——维度可能还一样，
    于是**只表现为"检索结果很差"，不报错**（embed/README 的原话）。
    """
    inner = FakeEmbedder()
    coords = EmbeddingCoordinate(model="model-a")
    wrapped, _ = _wrap(inner, tmp_path, coords)
    wrapped.encode(["Q: x"])
    assert inner.call_count == 1

    other = FakeEmbedder()
    wrapped_b, _ = _wrap(other, tmp_path, EmbeddingCoordinate(model="model-b"))
    wrapped_b.encode(["Q: x"])
    assert other.call_count == 1  # ← 没有命中 a 的缓存


def test_template_version_change_does_not_hit_old_cache(tmp_path) -> None:
    """**改渲染模板 = 改 embedding 输入 = 重建索引**（§11.3）。"""
    assert TEMPLATE_VERSION != "v0-test"
    inner = FakeEmbedder()
    wrapped, _ = _wrap(inner, tmp_path, EmbeddingCoordinate(model="m"))
    wrapped.encode(["Q: x"])

    other = FakeEmbedder()
    wrapped2, _ = _wrap(other, tmp_path, EmbeddingCoordinate(model="m", render_template="v0-test"))
    wrapped2.encode(["Q: x"])
    assert other.call_count == 1


def test_cache_file_is_self_describing(tmp_path) -> None:
    """坐标系也写进**文件内容**——万一有人改名或复制文件，第二道锁会响。"""
    cache = DiskVectorCache(tmp_path, COORD)
    row = cache._conn.execute("SELECT v FROM meta WHERE k='coordinate'").fetchone()
    assert row is not None
    assert COORD.describe() in row[0]


def test_corrupt_cache_entry_fails_loudly(tmp_path) -> None:
    """缓存条目自称的维度与 blob 长度不符 ⇒ 响亮失败，不静默用坏数据。"""
    cache = DiskVectorCache(tmp_path, COORD)
    cache.close()
    conn = sqlite3.connect(str(cache.path))
    conn.execute(
        "INSERT OR REPLACE INTO vectors (content_hash, dim, vec, created_at) VALUES (?, ?, ?, 0)",
        ("不可达的哈希", 999, np.asarray([1.0, 2.0], dtype=np.float32).tobytes()),
    )
    conn.commit()
    conn.close()

    reopened = DiskVectorCache(tmp_path, COORD)
    # 直接查那一条（用其真实哈希）
    reopened._conn.execute("UPDATE vectors SET content_hash = ? WHERE dim = 999", (_hash("t"),))
    reopened._conn.commit()
    with pytest.raises(DimensionMismatchError, match="自称 999 维"):
        reopened.get("t")


def test_inner_returning_mixed_dims_fails_loudly(tmp_path) -> None:
    class Mixed:
        @property
        def dim(self) -> int:
            return 3

        def encode(self, texts):  # noqa: ANN001, ANN202
            return [[1.0, 2.0, 3.0], [1.0, 2.0]]

    cache = DiskVectorCache(tmp_path, COORD)
    with pytest.raises(DimensionMismatchError, match="多种维度"):
        CachingEmbedder(Mixed(), cache).encode(["a", "b"])


# ── HTTP 层：OpenAI 兼容的响应形状 ──────────────────────────────────────


def test_http_request_shape() -> None:
    client = _StubClient(_ok_body(3))
    emb = OpenAICompatEmbedder(
        base_url="http://gw.example/v1/", api_key="sk-x", model="BAAI/bge-m3", client=client
    )
    emb.encode(["hello"])

    assert client.calls[0]["url"] == "http://gw.example/v1/embeddings"  # 末尾斜杠被规范化
    assert client.calls[0]["json"] == {"model": "BAAI/bge-m3", "input": ["hello"]}
    assert client.calls[0]["headers"]["Authorization"] == "Bearer sk-x"


def test_http_non_200_raises() -> None:
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient("nope", status=401)
    )
    with pytest.raises(EmbeddingError, match="返回 401"):
        emb.encode(["a"])


def test_http_non_openai_shape_raises() -> None:
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient({"vectors": []})
    )
    with pytest.raises(EmbeddingError, match="响应形状"):
        emb.encode(["a"])


def test_http_count_mismatch_raises() -> None:
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient(_ok_body(3, n=2))
    )
    with pytest.raises(EmbeddingError, match="请求 1 条，返回 2 条"):
        emb.encode(["a"])


def test_http_missing_embedding_raises() -> None:
    body = {"data": [{"index": 0, "embedding": []}]}
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=_StubClient(body)
    )
    with pytest.raises(EmbeddingError, match="缺少 embedding"):
        emb.encode(["a"])


def test_http_respects_index_order_and_batches() -> None:
    """响应可能乱序（服务端并行）——按 `index` 排回来；超过批大小要分批。"""
    body = {
        "data": [
            {"index": 1, "embedding": [9.0, 9.0]},
            {"index": 0, "embedding": [1.0, 1.0]},
        ]
    }
    client = _StubClient(body)
    emb = OpenAICompatEmbedder(
        base_url="http://x", api_key="k", model="m", client=client, batch_size=2
    )
    out = emb.encode(["first", "second", "third", "fourth"])
    assert out[0] == [1.0, 1.0]  # index 0 在前
    assert out[1] == [9.0, 9.0]
    assert len(client.calls) == 2  # 4 条 / 批大小 2 = 两次调用


# ── BGE-M3 的构造入口 ──────────────────────────────────────────────────


def test_bge_m3_from_env_requires_variables() -> None:
    with pytest.raises(ValueError, match="缺少环境变量"):
        BGEM3Embedder.from_env({})


def test_bge_m3_from_env_reads_variables() -> None:
    emb = BGEM3Embedder.from_env({"AML_EMB_BASE_URL": "http://gw/v1", "AML_EMB_API_KEY": "sk-x"})
    assert emb.model == DEFAULT_MODEL == "BAAI/bge-m3"
    assert emb._base_url == "http://gw/v1"  # noqa: SLF001 — 只验构造，不验调用


def test_bge_m3_model_can_be_overridden_by_env() -> None:
    emb = BGEM3Embedder.from_env(
        {"AML_EMB_BASE_URL": "http://gw/v1", "AML_EMB_API_KEY": "k", "AML_EMB_MODEL": "other"}
    )
    assert emb.model == "other"


def test_empty_base_url_or_key_rejected() -> None:
    with pytest.raises(ValueError, match="不得为空"):
        OpenAICompatEmbedder(base_url="", api_key="k", model="m")
    with pytest.raises(ValueError, match="不得为空"):
        OpenAICompatEmbedder(base_url="http://x", api_key="", model="m")


# ── 桩 ─────────────────────────────────────────────────────────────────


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ok_body(dim: int, n: int = 1) -> dict:
    return {"data": [{"index": i, "embedding": [float(j) for j in range(dim)]} for i in range(n)]}


class _Resp:
    def __init__(self, body, status: int) -> None:  # noqa: ANN001
        if isinstance(body, str):
            self._json, self.text, self.status_code = None, body, status
        else:
            import json

            self._json, self.text, self.status_code = body, json.dumps(body), status

    def json(self):  # noqa: ANN201
        if self._json is None:
            raise ValueError("not json")
        return self._json


class _StubClient:
    """只实现 `post` 的桩——**用来把 HTTP 形状钉死，不碰网络**。"""

    def __init__(self, body, status: int = 200) -> None:  # noqa: ANN001
        self.body = body
        self.status = status
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None):  # noqa: ANN001, ANN201
        self.calls.append({"url": url, "json": json, "headers": headers or {}})
        return _Resp(self.body, self.status)

    def close(self) -> None:
        pass
