"""路由——**薄**。

> `service/CLAUDE.md`：本层固定契约形状并保持失败可重试。
> 这里只做 `HTTP/schema → 编排` 的映射，不重新实现 pairing、embedding、
> Qdrant 或 packaging 逻辑。

## 同步路由

路由使用 `def` 而不是 `async def`。FastAPI 会把同步路由放入线程池，而 SQLite 和远程
HTTP 调用都是阻塞操作；在事件循环中直接执行会阻塞并发请求。
"""

from __future__ import annotations

from fastapi import APIRouter

from tianximem.pairing import AddBatch
from tianximem.pairing.pairing import Message
from tianximem.service.pipeline import AddPipeline, SearchPipeline
from tianximem.service.schemas import (
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

__all__ = ["build_router"]


def _to_batch(request: AddRequest) -> AddBatch:
    """把 Add 请求转换成 canonical `AddBatch`。

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
        """无需鉴权的 GET 探活端点；任意 2xx 表示进程存活。

        平台**未单独配置 Health 地址时探的是与 Add 同源的 `/health`**，404 可能被判为
        不健康 ⇒ **任务根本跑不起来**——不是掉分，是**没进场**。

        ⚠ **只报"这个进程活着"，不去探下游**（SQLite / Qdrant / 网关一个都不碰）：
        探活做成深检查，任何一个抖动都会让平台判我们不健康；而本端点的**唯一作用**
        就是让平台看得见我们。真挂了，`/add` 自己会响亮失败。
        """
        return {"status": "ok"}

    @router.post("/add", response_model=AddResponse)
    def handle_add(request: AddRequest) -> AddResponse:
        """返回 200、`success: true`，并原样回显三个请求字段。

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
        """返回按相关性降序的 `data` 数组；无结果时返回 `[]`。

        每个条目都是渲染后的证据文本；服务不生成最终答案，也不把答案伪装成记忆。
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
