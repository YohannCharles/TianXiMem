"""`Add` / `Search` 的请求与响应模型——**§2.1 的字面翻译**。

> 本层只做一件事：**把契约形状钉死**。参数合法性在**边界**校验，
> 不依赖下游（`rank.package` / `pairing`）的容错行为。

## 两条刻意的"不校验"

1. **`content` 不校验非空。** §2.1 说"非空 `content`"，但 `pairing.Message` **刻意容忍**
   空/纯空白（多模态消息可能只有图没有字，见 D16 那边的加载层讨论）。
   在边界拒绝会让一整批**被重试 32 次后丢掉**——比容忍它坏得多。
2. **不 `extra="forbid"`。** 未知字段一律忽略：AML 将来加字段不该让我们 422。
   缺字段照旧 422（pydantic 默认行为）。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AddMessage",
    "AddRequest",
    "AddResponse",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
]


# ── Add（§2.1）─────────────────────────────────────────────────────────


class AddMessage(BaseModel):
    """`messages[]` 的一项：**`role` + `content` + 可选 `timestamp`**。

    ⚠ **只有这三个字段**——这就是 canonical 形状，任何数据集专属字段都必须在
    加载层被归一化掉（D16）。
    """

    role: str = Field(min_length=1)
    content: str
    timestamp: int | None = None


class AddRequest(BaseModel):
    """`POST /add` 的请求体（§2.1 逐字）。

    * `request_id` —— 唯一，**须原样回显**
    * `messages` —— **源序**；每次 Add 只喂 ≤20 条消息或 2,000 词（由 AML 切分）
    * `user_id` / `session_id` —— 隔离字段 / 分组字段
    """

    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    #: 空批次在 `pairing` 里是**响亮失败**；在边界先给一个清楚的 422 更省事。
    messages: list[AddMessage] = Field(min_length=1)


class AddResponse(BaseModel):
    """200 响应：`success: true` + **原样回显**三个字段（§2.1）。"""

    success: bool = True
    request_id: str
    user_id: str
    session_id: str


# ── Search（§2.1）──────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """`POST /search` 的请求体。

    ⚠ **`query` 只校验"非空白"，绝不改写它**——§7.2 要求查询**原样送进去**
    （v1 没有 Query Analyzer）。所以这里**不做 `.strip()` 赋值**，
    只用它来判断合法性。
    """

    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    #: ⚠ **边界取 `ge=1`，不是 `ge=0`**：契约没定义 `top_k <= 0`，
    #: 而"请求零条结果"在语义上是空的。宁可在这里 422，**也不要依赖
    #: `rank.package()` 对 `top_k <= 0` 的容错行为**（那是防御分支，不是接口约定）。
    top_k: int = Field(ge=1)

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query 不得为空白")
        return value  # ← 原样返回，不改写


class SearchResultItem(BaseModel):
    """`data[]` 的一项：**§2.1 的四个字段**。

    ⚠ `score` 是 `1/(rank+1)` 这类**单调占位值**，**不是** Qdrant 的融合分数
    （融合分数不是校准量，§8）。它由 `rank/packaging.py` 现算，上游根本没有那个值。
    """

    id: str
    content: str
    created_at: str
    score: float


class SearchResponse(BaseModel):
    """`data` 是**按相关性降序的数组**；**空结果是 `[]` 而不是 `null`**（§2.1）。"""

    data: list[SearchResultItem]
