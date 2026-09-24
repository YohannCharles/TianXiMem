"""路由——**薄**。

> `service/CLAUDE.md`：本层只做三件事——**钉死契约形状、串行化 Add、保持可重试**。
> 所以这里只做 `HTTP/schema → 编排` 的映射，**不重新实现** pairing / embedding /
> Qdrant / packaging 的任何逻辑（它们全在 `pipeline.py` 往下调）。

## 两个容易写错的点

**路由写成 `def` 而不是 `async def`。** FastAPI 会把 `def` 路由自动下到线程池，
而下游是**阻塞**的（SQLite 是阻塞调用、远程 embedding 是同步 HTTP）。
写成 `async def` 再接阻塞调用会**堵住事件循环**，让并发请求互相饿死——
而 30 分钟的单请求预算会**掩盖这个问题，直到 Full run 跑 0.5–2 天时才炸**（§15）。
"""

from __future__ import annotations

from fastapi import APIRouter

from tianxi_am.pairing import AddBatch
from tianxi_am.pairing.pairing import Message
from tianxi_am.service.pipeline import AddPipeline, SearchPipeline
from tianxi_am.service.schemas import (
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

__all__ = ["build_router"]


def _to_batch(request: AddRequest) -> AddBatch:
    """§2.1 的请求 → canonical 的 `AddBatch`。

    **只做形状映射**：校验已经在 schema 边界做完，这里不做任何业务判断
    （比如"是不是空批次"——那是 `pairing` 的响亮失败）。
    """
    return AddBatch(
        request_id=request.request_id,
        user_id=request.user_id,
        session_id=request.session_id,
        messages=tuple(
            Message(role=m.role, content=m.content, timestamp=m.timestamp) for m in request.messages
        ),
    )


def build_router(*, add_pipeline: AddPipeline, search_pipeline: SearchPipeline) -> APIRouter:
    """构造全部端点。**依赖是显式传入的**——便于测试注入替身，不靠全局状态。"""
    router = APIRouter()

    @router.get("/health")
    def handle_health() -> dict[str, str]:
        """探活：**无需鉴权的 GET**，任意 2xx 即视为正常
        （[`docs/contract.md`](../../../docs/contract.md) §7.2 / S4）。

        平台**未单独配置 Health 地址时探的是与 Add 同源的 `/health`**，404 可能被判为
        不健康 ⇒ **任务根本跑不起来**——不是掉分，是**没进场**。

        ⚠ **只报"这个进程活着"，不去探下游**（SQLite / Qdrant / 网关一个都不碰）：
        探活做成深检查，任何一个抖动都会让平台判我们不健康；而本端点的**唯一作用**
        就是让平台看得见我们。真挂了，`/add` 自己会响亮失败。
        """
        return {"status": "ok"}

    @router.post("/add", response_model=AddResponse)
    def handle_add(request: AddRequest) -> AddResponse:
        """§2.1：200 + `success: true` + **原样回显**三个字段。

        ⚠ 响应**只有这三个回显字段**：失败时直接抛（5xx），由
        [`errors.py`](./errors.py) 转成明确的可重试错误——**不返回"部分成功"**。
        """
        add_pipeline.apply(_to_batch(request))
        return AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    @router.post("/search", response_model=SearchResponse)
    def handle_search(request: SearchRequest) -> SearchResponse:
        """§2.1：`data` 是**降序数组**，空结果是 `[]`。

        ⚠ **不得生成最终答案，也不得把答案伪装成记忆记录**（§2.1 红线）——
        这里返回的每一项都是**渲染后的证据原文**。
        """
        packed = search_pipeline.run(
            user_id=request.user_id, query=request.query, top_k=request.top_k
        )
        return SearchResponse(
            data=[
                SearchResultItem(
                    id=item.id,
                    content=item.content,
                    created_at=item.created_at,
                    score=item.score,
                )
                for item in packed.items
            ]
        )

    return router
