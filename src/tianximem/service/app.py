"""FastAPI application lifecycle and HTTP registration.

Dependency construction lives in :mod:`tianximem.service.wiring`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from tianximem.common.config import assert_single_process, load_config
from tianximem.service.capture import CaptureMiddleware
from tianximem.service.errors import register_error_handlers
from tianximem.service.routes import build_router as build_http_router
from tianximem.service.wiring import (
    Services,
    build_agentic,
    build_capture,
    build_embedder,
    build_llm,
    build_metrics_sink,
    build_query_router,
    build_reranker,
    build_services,
)

__all__ = [
    "Services",
    "build_agentic",
    "build_capture",
    "build_embedder",
    "build_llm",
    "build_metrics_sink",
    "build_query_router",
    "build_reranker",
    "build_services",
    "create_app",
    "create_app_from_env",
]


def create_app(services: Services) -> FastAPI:
    """Build an app from already constructed dependencies."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        del app
        yield
        services.close()

    app = FastAPI(title="TianXiMem", lifespan=lifespan)
    register_error_handlers(app)
    app.include_router(
        build_http_router(add_pipeline=services.add, search_pipeline=services.search)
    )
    if services.capture is not None:
        app.add_middleware(CaptureMiddleware, capture=services.capture)
    app.state.services = services
    return app


def create_app_from_env() -> FastAPI:
    """Uvicorn factory entrypoint with the single-process guard."""

    assert_single_process()
    return create_app(build_services(load_config()))
