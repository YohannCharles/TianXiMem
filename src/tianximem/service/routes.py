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
    return AddBatch(
        request_id=request.request_id,
        user_id=request.user_id,
        session_id=request.session_id,
        messages=tuple(
            Message(role=m.role, content=m.content, timestamp=m.timestamp) for m in request.messages
        ),
    )


def build_router(*, add_pipeline: AddPipeline, search_pipeline: SearchPipeline) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def handle_health() -> dict[str, str]:
        return {"status": "ok"}

    @router.post("/add", response_model=AddResponse)
    def handle_add(request: AddRequest) -> AddResponse:
        add_pipeline.apply(_to_batch(request))
        return AddResponse(
            request_id=request.request_id,
            user_id=request.user_id,
            session_id=request.session_id,
        )

    @router.post("/search", response_model=SearchResponse)
    def handle_search(request: SearchRequest) -> SearchResponse:
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
