"""单元测试的共用 fixture。

测试清单的权威来源是 [`../tests/README.md`](../tests/README.md)：
配对 / 续接 / 幂等 / 契约 / 隔离 / 开关纯度 / 存储。

⚠ 本目录的测试**只用合成的 canonical 消息**（`role` / `content` / 可选 `timestamp`），
**不碰任何数据集文件**——LoCoMo 只作为实现完成后的 fixture 验证数据，
**不得反向影响核心数据模型**（D16）。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from tianxi_am.pairing.pairing import BatchLimits, Message
from tianxi_am.store.sqlite_store import SqliteStore


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
