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
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

__all__ = ["RetryableError", "register_error_handlers"]

logger = logging.getLogger(__name__)


class RetryableError(RuntimeError):
    """暂时性失败——**客户端重试是对的**（例如依赖短暂不可用）。

    ⚠ D25 起不再有"同 session 正忙"这一路来源（`SessionLocks` 已删），
    但 embedding / Qdrant 的短暂故障仍在 ⇒ 这个基类与它的 5xx 映射照留。
    """


def register_error_handlers(app: FastAPI) -> None:
    """把可重试失败与未处理异常映射成明确的 5xx，并**记日志**。

    ⚠ D25 删掉了 `SessionLockTimeout` 那个 handler（连同 `SessionLocks`）——
    **但"非 200 一律按可重试处理"这条契约没变**，下面两个 handler 是它的落点。
    """

    @app.exception_handler(RetryableError)
    async def _retryable(request: Request, exc: RetryableError) -> JSONResponse:
        logger.warning("可重试失败：%s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # ⚠ 这里【不】吞异常：照旧返回非 200，让 AML 重试；
        #    同时把完整栈记下来，否则线上只剩一个 500 无从排查。
        logger.exception("未处理异常：%s %s", request.method, request.url.path)
        detail: dict[str, Any] = {"detail": "internal error（本批未被确认，可安全重试）"}
        return JSONResponse(status_code=500, content=detail)
