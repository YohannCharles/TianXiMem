"""`Add` / `Search` 的请求与响应模型。

本层固定 HTTP 契约形状，并在边界校验参数，不依赖下游容错行为。
未知字段会被忽略，以兼容 AML 后续增加字段；缺少必填字段仍返回 422。
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


# ── Add ────────────────────────────────────────────────────────────────


class AddMessage(BaseModel):
    """`messages[]` 的一项：**`role` + `content` + 可选 `timestamp`**。

    ⚠ **只有这三个字段**——这就是 canonical 形状，任何数据集专属字段都必须在
    加载层被归一化掉。
    """

    role: str = Field(min_length=1)
    content: str = Field(min_length=1)
    timestamp: int | None = None

    @field_validator("content")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content 不得为空白")
        return value


class AddRequest(BaseModel):
    """`POST /add` 的请求体。

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
    """200 响应：`success: true` 并原样回显三个请求字段。"""

    success: bool = True
    request_id: str
    user_id: str
    session_id: str


# ── Search ─────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    """`POST /search` 的请求体。

    **`query` 只校验非空白，不改写内容**。校验不使用 `.strip()` 赋值，返回原始字符串。
    空白查询在 HTTP 边界返回 422；AML 不重试 4xx。绕过 HTTP 的进程内调用由
    `SearchPipeline._search` 以 `data: []` 和 `grounded_status = "blank_query"` 处理。
    """

    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    #: 边界要求 `top_k >= 1`；零值请求在 HTTP 层返回 422，不依赖打包层的防御分支。
    top_k: int = Field(ge=1)

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query 不得为空白")
        return value  # ← 原样返回，不改写


class SearchResultItem(BaseModel):
    """`data[]` 的一项，只包含契约规定的四个字段。

    `score` 是 `1/(rank+1)` 形式的单调占位值，不是 Qdrant 融合分数；它由
    `rank/packaging.py` 计算。
    """

    id: str
    content: str
    created_at: str
    score: float


class SearchResponse(BaseModel):
    """`data` 是按相关性降序的数组；空结果使用 `[]`，不使用 `null`。"""

    data: list[SearchResultItem]
