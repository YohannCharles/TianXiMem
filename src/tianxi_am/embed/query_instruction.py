"""查询侧 instruction 前缀 —— **Qwen3-Embedding 系列的输入格式兼容层**。

## 为什么单独一层

Qwen3-Embedding-8B 的官方用法是**两侧输入不一样**（依据：模型卡 `Qwen/Qwen3-Embedding-8B`）：

* **查询侧**：`f'Instruct: {task_description}\\nQuery:{query}'`——任务描述是一句短指令（建议英文）；
* **文档侧**：**不加任何前缀**（模型卡原文："No need to add instruction for retrieval documents"）。

而本项目 §7.2 的规格是"**query 原样送进去，不做任何改写**"。两者冲突，
而且模型卡给了一个量化后果：**查询侧不加 instruction 会让检索性能掉约 1%–5%**。

**因此本模块提供的不是"改写"，而是一层可开关的输入格式适配**：

| `instruction` | 行为 |
| --- | --- |
| `""`（默认） | **与"原样送"逐字节相同**——v1 现状不变，开箱即用 |
| 非空 | 只影响**查询侧**，文档侧那一份渲染**一个字节都不动** |

> **为什么文档侧不能动**：§7.2 要求 embedding 的输入与返回的 `content` 是**同一份渲染**，
> 而文档侧就是索引的输入——**改它等于改索引**（那是"贵"消融项）。

**要不要开、instruction 写什么，是一个待定的规格问题**，不是实现细节——
默认关着，规格怎么定都不会让现有行为漂移。

## ⚠ 一条不可违反的顺序：前缀必须在**缓存之上**

`CachingEmbedder` 的缓存键是**实际送进模型的文本**的哈希。所以：

    QueryInstructionEmbedder(inner=CachingEmbedder(模型))     ✅ 正确
    把前缀写在 CachingEmbedder **之下**（例如塞进具体 embedder 内部）  ❌ 错

后者会出现"同一个键对应两个不同的值"：同一个查询第一次（未命中）拿到带前缀的向量、
第二次（命中）拿到不带前缀的向量 ⇒ **检索结果依赖缓存状态**——
正是 [`base.py`](./base.py) 点名的、本项目最不能接受的失败类型。

**但加前缀本身不需要动 `EmbeddingCoordinate`**：坐标系管的是"模型 + 渲染模板"
（索引侧的身份），而查询侧的两种输入天然落在**不同的缓存键**上。
反过来说——**前缀绝不能瞒着缓存去做**。
"""

from __future__ import annotations

from collections.abc import Sequence

from tianxi_am.embed.base import Embedder

__all__ = [
    "DEFAULT_TASK",
    "DEFAULT_TEMPLATE",
    "QueryInstructionEmbedder",
    "apply_query_instruction",
]

# 模型卡的原文模板（2026-09-24 读取）。
# ⚠ 注意 `Query:` 之后**没有空格**——这是逐字照抄的形状，不要"顺手"加。
DEFAULT_TEMPLATE: str = "Instruct: {task}\nQuery:{query}"

# 模型卡示例里的任务描述。**只是初值**：模型卡建议按场景/语言自己写（英文）。
# 改它 = 改查询侧的输入 ⇒ **换一批缓存键**（不是换坐标系）。
DEFAULT_TASK: str = "Given a web search query, retrieve relevant passages that answer the query"


def apply_query_instruction(query: str, instruction: str, template: str = DEFAULT_TEMPLATE) -> str:
    """给一条查询加 instruction 前缀。`instruction` 为空 ⇒ **原样返回**（v1 现状）。"""
    if not instruction:
        return query
    return template.format(task=instruction, query=query)


class QueryInstructionEmbedder:
    """把 `Embedder` 分成两侧：`encode()`（文档侧）· `encode_query()`（查询侧）。

    用法（**顺序不能反**，理由见模块 docstring）：

        inner = CachingEmbedder(Qwen3EmbeddingEmbedder(base_url=..., api_key=..., model=...),
                                DiskVectorCache(...))
        embedder = QueryInstructionEmbedder(inner, instruction=cfg.retrieval.query_instruction)

    ⚠ 装配点在 [`../service/app.py`](../service/app.py) 的 `build_services()`——
    **参数由它从配置里取**，本模块不读环境变量、也不读配置。

    ⚠ 索引与查询两条路径**必须用同一个实例**——否则两边的 `dim` 与缓存坐标系可能不一致。
    ⚠ `encode()` 永不加前缀。文档侧的前缀会改变索引的输入，**那是"贵"消融项**（改模板 = 重建索引）。
    """

    def __init__(
        self,
        inner: Embedder,
        *,
        instruction: str = "",
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        self._inner = inner
        self._instruction = instruction
        self._template = template

    @property
    def inner(self) -> Embedder:
        return self._inner

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def dim(self) -> int:
        return self._inner.dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """文档侧：**原样**送给内层（不加前缀）。"""
        return self._inner.encode(texts)

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]:
        """查询侧：加前缀**之后**才交给内层——于是缓存键落在带前缀的那份文本上。"""
        return self._inner.encode(
            [apply_query_instruction(text, self._instruction, self._template) for text in texts]
        )
