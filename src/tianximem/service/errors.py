from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tianximem.pairing import PayloadMismatchError

__all__ = ["RetryableError", "register_error_handlers"]

logger = logging.getLogger(__name__)


class RetryableError(RuntimeError):
    pass


def register_error_handlers(app: FastAPI) -> None:

    @app.exception_handler(RetryableError)
    async def _retryable(request: Request, exc: RetryableError) -> JSONResponse:
        logger.warning("可重试失败：%s", exc)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(PayloadMismatchError)
    async def _conflict(request: Request, exc: PayloadMismatchError) -> JSONResponse:
        logger.warning("request_id 冲突（同 id 不同 payload）：%s", exc)
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("未处理异常：%s %s", request.method, request.url.path)
        detail: dict[str, Any] = {"detail": "internal error（本批未被确认，可安全重试）"}
        return JSONResponse(status_code=500, content=detail)
