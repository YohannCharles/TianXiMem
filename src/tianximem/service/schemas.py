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


class AddMessage(BaseModel):
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
    request_id: str = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    messages: list[AddMessage] = Field(min_length=1)


class AddResponse(BaseModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(BaseModel):
    user_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    top_k: int = Field(ge=1)

    @field_validator("query")
    @classmethod
    def _query_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query 不得为空白")
        return value


class SearchResultItem(BaseModel):
    id: str
    content: str
    created_at: str
    score: float


class SearchResponse(BaseModel):
    data: list[SearchResultItem]
