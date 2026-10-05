"""异常 → 保持"可重试"的边界处理（§15 / `docs/contract.md` §6）。

## 契约对非 200 的规定是"未定义"

> 契约**只规定了 200 的成功形状**（§2.1），**非 200 的行为未定义，因此必须假设
> AML 会重试**——这正是幂等守卫的动机。

**⇒ 所以本层的纪律是：内部异常让它抛，不要吞、不要返回"部分成功"。**
事务未提交 ⇒ 本批仍可重试；事务已提交但索引没写完 ⇒ 重试会被幂等守卫接住，
并由 `pipeline.py` 的**修复路径**把派生索引补上。

## 为什么不"统一转成 200"

把失败包装成 200 会让 AML 认为本批已成功 ⇒ **那批记忆永远不会重试** ⇒
SQLite 与 Qdrant 永久不一致 ⇒ **某些记忆永远检索不到，且没有任何报错**。
这正是本项目最不能接受的失败类型。

## 唯一一处"重试也没用"的非 200：**409**（D28）

`request_id` 相同、`payload_hash` 不同 ⇒ `PayloadMismatchError` ⇒ **409**。
它**不是**服务端的暂时故障，重试一百次也还是这个结果；但契约里没有比"非 200"
更好的表达方式（§2.1 只定义了 200）。

⇒ 它的价值是**把原因说清楚**（响应体里写着"同 id 不同 payload"），而不是：
* 伪装成 500 —— 那会让排查方向跑到"下游是不是挂了"
* 静默当重放 —— 那会让**两份不同记忆里的一份凭空消失**，而检索侧看不出来
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tianximem.pairing import PayloadMismatchError

__all__ = ["RetryableError", "register_error_handlers"]

logger = logging.getLogger(__name__)


class RetryableError(RuntimeError):
    """暂时性失败——**客户端重试是对的**（例如依赖短暂不可用）。

    来源是 embedding / Qdrant 的短暂故障 ⇒ 这个基类与它的 5xx 映射照留。
    """


def register_error_handlers(app: FastAPI) -> None:
    """把可重试失败与未处理异常映射成明确的 5xx，并**记日志**。

    ⚠ **"非 200 一律按可重试处理"这条契约的落点就是下面两个 handler**。
    """

    @app.exception_handler(RetryableError)
    async def _retryable(request: Request, exc: RetryableError) -> JSONResponse:
        logger.warning("可重试失败：%s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(PayloadMismatchError)
    async def _conflict(request: Request, exc: PayloadMismatchError) -> JSONResponse:
        """同一个 `request_id` 收到了不同的 payload（D28）——**409，不是 500**。

        ⚠ 它**没有写坏任何东西**：冲突是在守卫那一步判出来的，事务整体回滚 ⇒
        真源里那一批仍是**第一次**投进来的那份。这一条有测试钉着
        （`tests/test_idempotency.py::test_conflicting_payload_leaves_the_first_one_intact`）。
        """
        logger.warning("request_id 冲突（同 id 不同 payload）：%s", exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # ⚠ 这里【不】吞异常：照旧返回非 200，让 AML 重试；
        #    同时把完整栈记下来，否则线上只剩一个 500 无从排查。
        logger.exception("未处理异常：%s %s", request.method, request.url.path)
        detail: dict[str, Any] = {"detail": "internal error（本批未被确认，可安全重试）"}
        return JSONResponse(status_code=500, content=detail)
