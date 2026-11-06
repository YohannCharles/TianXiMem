from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = ["DenseArm", "DenseArmError"]


class DenseArmError(RuntimeError):
    pass


@runtime_checkable
class EmbedderLike(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


@runtime_checkable
class SupportsQueryEncoding(Protocol):
    def encode_query(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(slots=True)
class DenseArm:
    embedder: EmbedderLike
    query_calls: list[str] = field(default_factory=list)

    def encode_query(self, query: str) -> list[float]:
        self.query_calls.append(query)
        inner = self.embedder
        vectors = (
            inner.encode_query([query])
            if isinstance(inner, SupportsQueryEncoding)
            else inner.encode([query])
        )
        if len(vectors) != 1:
            raise DenseArmError(f"查询侧应返回 1 个向量，实际 {len(vectors)} 个")
        vector = vectors[0]
        if not vector:
            raise DenseArmError("查询向量为空——Embedder 返回了空向量")
        return list(vector)

    @property
    def dim(self) -> int:
        return int(self.embedder.dim)

    def encode_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.embedder.encode(list(texts))
        if len(vectors) != len(texts):
            raise DenseArmError(f"索引侧 {len(texts)} 条文本返回 {len(vectors)} 个向量")
        return [list(v) for v in vectors]
