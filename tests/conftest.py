"""单元测试的共用 fixture。

测试清单的权威来源是 [`../tests/CLAUDE.md`](../tests/CLAUDE.md)：
配对 / 续接 / 幂等 / 契约 / 隔离 / 开关纯度 / 存储。

⚠ 本目录的测试**只用合成的 canonical 消息**（`role` / `content` / 可选 `timestamp`），
**不碰任何数据集文件**——LoCoMo 只作为实现完成后的 fixture 验证数据，
**不得反向影响核心数据模型**（D16）。

⚠ **测试绝不调用远程模型**：embedding 一律用 `FakeEmbedder`（确定性、可计数）。
需要 Qdrant 的用例在 Qdrant 不可达时**明确 skip**，不静默通过。
"""

from __future__ import annotations

import hashlib
import math
import os
import uuid
from collections.abc import Callable, Iterator, Sequence

import pytest

from tianxi_am.pairing.pairing import BatchLimits, Message
from tianxi_am.store.sqlite_store import SqliteStore

# ── ① 的 fixture ────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path) -> Iterator[SqliteStore]:
    """一个建好表、用临时文件的真源（每个用例一份，互不干扰）。"""
    s = SqliteStore.open(tmp_path / "tianxi.db")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def M() -> Callable[..., Message]:
    """消息构造器：`M("user", "你好")`、`M("assistant", "在", ts=1000)`。"""

    def _make(role: str, content: str, ts: int | None = None) -> Message:
        return Message(role=role, content=content, timestamp=ts)

    return _make


@pytest.fixture
def limits_small() -> BatchLimits:
    """小上限，让"命中上限"的用例可读。

    顺带证明一件事：**上限是配置项，不是硬编码**（§15 / D2 对冲 ③）——
    能用别的值跑通，就说明实现里没有把 20 / 2000 写死。
    """
    return BatchLimits(max_messages=3, max_words=1000)


@pytest.fixture
def limits_two() -> BatchLimits:
    """上限 = 2 条消息，用来构造"批次末尾是 pending"的最小场景。"""
    return BatchLimits(max_messages=2, max_words=1000)


# ── ② 的 fixture ────────────────────────────────────────────────────────


class FakeEmbedder:
    """确定性、可计数的假 Embedder。**测试用它，绝不碰网络。**

    * 未 preset 的文本 → 按词的哈希做 bag-of-words 单位向量（同词 ⇒ 高余弦）
    * `preset(text, vec)` → 精确指定某段文本的向量，让 dense 与 bm25 两路
      **可以独立控制**（"只被 dense 召回"这类用例必须能构造）
    * `calls` 记录每一次真实调用，用来断言"缓存命中时没有重复调用"
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._preset: dict[str, list[float]] = {}
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def encoded_texts(self) -> list[str]:
        return [t for batch in self.calls for t in batch]

    def preset(self, text: str, vec: Sequence[float]) -> None:
        if len(vec) != self._dim:
            raise ValueError(f"preset 维度必须是 {self._dim}，收到 {len(vec)}")
        self._preset[text] = list(vec)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        texts = list(texts)
        self.calls.append(texts)
        return [self._vec(t) for t in texts]

    def _vec(self, text: str) -> list[float]:
        if text in self._preset:
            return list(self._preset[text])
        acc = [0.0] * self._dim
        for token in text.lower().split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
            acc[h % self._dim] += 1.0
        norm = math.sqrt(sum(x * x for x in acc)) or 1.0
        return [x / norm for x in acc]


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


def unit_vector(dim: int, index: int) -> list[float]:
    """第 `index` 个基向量——用来构造互不相干的 dense 候选。"""
    return [1.0 if i == index else 0.0 for i in range(dim)]


@pytest.fixture(scope="session")
def qdrant_client():
    """会话级 Qdrant 客户端；**不可达时整组用例 skip**（不静默通过）。"""
    from qdrant_client import QdrantClient

    url = os.environ.get("TIANXI_QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=url, timeout=10.0)
    try:
        client.get_collections()
    except Exception as exc:  # noqa: BLE001 — 任何连接问题都跳过
        client.close()
        pytest.skip(f"Qdrant 不可达（{url}）：{exc}", allow_module_level=True)
    try:
        yield client
    finally:
        client.close()


@pytest.fixture
def qdrant_store(qdrant_client):
    """一个**用完即删**的测试集合。

    ⚠ 刻意**不用**默认的 `memories`：那是开发/线上共用的集合，
    测试往里写点会让"检索结果为什么不对"变成一个无法回答的问题。
    """
    from tianxi_am.store.qdrant_store import QdrantStore

    name = f"memories_test_{uuid.uuid4().hex[:8]}"
    store = QdrantStore(url="unused", collection=name, client=qdrant_client, hybrid=None)
    try:
        yield store
    finally:
        store.drop_collection()
