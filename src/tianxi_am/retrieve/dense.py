"""语义那一路：查询侧只做**一次** embedding 调用（§7.2）。

## 三条不可违反的约束

1. **每查询恰好 1 次调用**（§7.2）。索引侧是"每个对一个向量、缓存后趋近 0"，
   查询侧是"每查询恰好 1 次"——次数不对（0 次或 2 次）都说明接线错了。
2. **查询原样送出，不做任何改写**——v1 没有 Query Analyzer（§5），查询改写属 v2；
   而 agent 每轮自产的检索关键词属 agent 循环，**v1 两者都没有**（D13）。
3. **索引侧与查询侧必须用同一个 Embedder 实例**——否则两边的 `dim` 与缓存坐标系
   可能不一致，而**不一致只会表现为"检索结果很差"，不会报错**。

## 关于 `query_instruction.py`（默认关）

Qwen3-Embedding-8B 的模型卡推荐**查询侧**加 `Instruct: …\\nQuery:…`、文档侧不加，
并称不加会让检索掉约 1%–5%。那层适配器在
[`../embed/query_instruction.py`](../embed/query_instruction.py)，
对**查询侧**生效、对文档侧逐字节不动。**它不是"改写"**——是输入格式适配。

本模块用**鸭子类型**认它：持有对象若有 `encode_query()` 就用它，否则退回 `encode()`。
这样 `DenseArm` 不必知道有没有开前缀，而**唯一不可违反的顺序**（前缀必须加在缓存之上）
由构造顺序保证（见该模块的用法示例）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["DenseArm", "DenseArmError"]


class DenseArmError(RuntimeError):
    """查询侧调用不符合约定（次数不对、返回形状不对）。"""


@runtime_checkable
class EmbedderLike(Protocol):
    """本模块对 Embedder 的最小要求（§7.4）。

    结构化的写法而不是 `object` + `getattr`：**接口本身就是文档**，
    而且 `isinstance` 能直接用来做下面的可选能力探测。
    """

    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class SupportsQueryEncoding(Protocol):
    """**可选能力**：`QueryInstructionEmbedder` 用它把查询侧与文档侧分开。

    裸 `Embedder` 没有这个方法——所以它必须是可选的，不能进上面的协议。
    """

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class DenseArm:
    """包住一个 `Embedder`（可能已被 `QueryInstructionEmbedder` 套过一层）。

    ⚠ 传进来的**必须是索引侧用的那一个实例**（或至少共享同一个缓存坐标系），
    否则"检索命中的是什么"与"索引里存的是什么"会漂移。
    """

    embedder: EmbedderLike
    #: 查询侧调用计数——让"每查询恰好 1 次"**可被断言**，而不是靠约定。
    query_calls: list[str] = field(default_factory=list)

    def encode_query(self, query: str) -> list[float]:
        """把一条查询编成 dense 向量。**调用次数累加在这里**，供测试断言。"""
        self.query_calls.append(query)
        inner = self.embedder
        # 可选能力探测：`QueryInstructionEmbedder` 有 `encode_query`，裸 Embedder 没有。
        # 于是本模块**不必知道有没有开指令前缀**——那是构造顺序的事。
        vectors = (
            inner.encode_query([query])
            if isinstance(inner, SupportsQueryEncoding)
            else inner.encode([query])
        )
        if len(vectors) != 1:
            raise DenseArmError(f"查询侧应返回 1 个向量，实际 {len(vectors)} 个")
        vector = vectors[0]
        if not vector:
            raise DenseArmError("查询向量为空——Embedder 返回了空向量")
        return list(vector)

    @property
    def dim(self) -> int:
        """维度来自接口（**不写死**）——集合的向量维度必须与它一致（§7.4 / §2.3）。"""
        return int(self.embedder.dim)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """索引侧：**原样**送（`QueryInstructionEmbedder.encode()` 永不加前缀）。"""
        vectors = self.embedder.encode(list(texts))
        if len(vectors) != len(texts):
            raise DenseArmError(f"索引侧 {len(texts)} 条文本返回 {len(vectors)} 个向量")
        return [list(v) for v in vectors]
