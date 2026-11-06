from __future__ import annotations

import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Final

from qdrant_client import QdrantClient, models

from tianximem.common.config import DEFAULT_PREFETCH_LIMIT, DEFAULT_WEIGHTS, RRF_K
from tianximem.common.render import render_pair

__all__ = [
    "BM25_MODEL",
    "COLLECTION_DEFAULT",
    "DENSE_VECTOR",
    "SPARSE_VECTOR",
    "HybridParams",
    "MemoryRecord",
    "QdrantStore",
    "ScoredMemoryId",
    "point_id_for",
]

COLLECTION_DEFAULT: Final[str] = "memories"

DENSE_VECTOR: Final[str] = "dense"
SPARSE_VECTOR: Final[str] = "bm25"
BM25_MODEL: Final[str] = "qdrant/bm25"

KEY_MEMORY_ID: Final[str] = "memory_id"
KEY_USER_ID: Final[str] = "user_id"
KEY_SESSION_ID: Final[str] = "session_id"
KEY_LOCAL_INDEX: Final[str] = "local_index"
KEY_EVENT_TIME: Final[str] = "event_time"


def point_id_for(memory_id: str) -> str:
    if len(memory_id) < 32:
        raise ValueError(f"memory_id 太短，无法当 UUID 用：{memory_id!r}")
    return str(uuid.UUID(memory_id[:32]))


def memory_id_from_point_id(point_id: Any) -> str | None:
    try:
        return uuid.UUID(str(point_id)).hex
    except (ValueError, AttributeError, TypeError):
        return None


@dataclass(frozen=True, slots=True)
class HybridParams:
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    weights: tuple[float, float] = DEFAULT_WEIGHTS
    rrf_k: int = RRF_K


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    memory_id: str
    user_id: str
    session_id: str
    local_index: int
    event_time: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ScoredMemoryId:
    memory_id: str
    score: float


class QdrantStore:
    def __init__(
        self,
        *,
        url: str,
        collection: str = COLLECTION_DEFAULT,
        hybrid: HybridParams | None = None,
        api_key: str | None = None,
        timeout: int = 60,
        client: QdrantClient | None = None,
    ) -> None:
        self._collection = collection
        self._hybrid = hybrid or HybridParams()
        self._client = client or QdrantClient(url=url, api_key=api_key, timeout=timeout)

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def hybrid(self) -> HybridParams:
        return self._hybrid

    @property
    def client(self) -> QdrantClient:
        return self._client

    def close(self) -> None:
        self._client.close()

    def exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def collection_dim(self) -> int | None:
        if not self.exists():
            return None
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict):  # pragma: no cover
            raise TypeError(f"集合 {self._collection} 不是命名向量布局")
        dense = vectors[DENSE_VECTOR]
        return None if dense is None else int(dense.size)

    def ensure_collection(self, dim: int) -> bool:
        if dim <= 0:
            raise ValueError(f"维度必须为正：{dim}")

        if self.exists():
            existing = self.collection_dim()
            if existing != dim:
                raise ValueError(
                    f"集合 {self._collection} 的 dense 是 {existing} 维，但 Embedder 给出 "
                    f"{dim} 维。维度变化必须**重建集合**，不能就地混用。"
                )
            return False

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            shard_number=1,
        )
        self._create_payload_indexes()
        return True

    def _create_payload_indexes(self) -> None:
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name=KEY_USER_ID,
            field_schema=models.KeywordIndexParams(
                type=models.KeywordIndexType.KEYWORD, is_tenant=True
            ),
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name=KEY_SESSION_ID,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self._client.create_payload_index(
            collection_name=self._collection,
            field_name=KEY_EVENT_TIME,
            field_schema=models.PayloadSchemaType.INTEGER,
        )

    def payload_index_names(self) -> set[str]:
        info = self._client.get_collection(self._collection)
        return set((info.payload_schema or {}).keys())

    def is_tenant_index(self) -> bool:
        info = self._client.get_collection(self._collection)
        schema = (info.payload_schema or {}).get(KEY_USER_ID)
        params = getattr(schema, "params", None)
        return bool(params is not None and getattr(params, "is_tenant", False))

    def drop_collection(self) -> None:
        if self.exists():
            self._client.delete_collection(self._collection)

    def upsert(
        self,
        records: Sequence[MemoryRecord],
        dense_vectors: Sequence[Sequence[float]],
        *,
        wait: bool = True,
    ) -> int:
        if len(records) != len(dense_vectors):
            raise ValueError(f"记录 {len(records)} 条、向量 {len(dense_vectors)} 个，数量不符")
        if not records:
            return 0

        dim = len(dense_vectors[0])
        for v in dense_vectors:
            if len(v) != dim:
                raise ValueError(f"同批向量维度不一致：{len(v)} vs {dim}")

        points = [
            models.PointStruct(
                id=point_id_for(r.memory_id),
                vector={
                    DENSE_VECTOR: list(v),
                    SPARSE_VECTOR: models.Document(text=r.text, model=BM25_MODEL),
                },
                payload={
                    KEY_MEMORY_ID: r.memory_id,
                    KEY_USER_ID: r.user_id,
                    KEY_SESSION_ID: r.session_id,
                    KEY_LOCAL_INDEX: r.local_index,
                    KEY_EVENT_TIME: r.event_time,
                },
            )
            for r, v in zip(records, dense_vectors, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points, wait=wait)
        return len(points)

    def index_pairs(
        self,
        pairs: Iterable[Any],
        embedder: Any,
        *,
        renderer: Any = render_pair,
        wait: bool = True,
    ) -> int:
        total = 0
        buffer: list[MemoryRecord] = []
        for pair in pairs:
            buffer.append(
                MemoryRecord(
                    memory_id=pair.id,
                    user_id=pair.user_id,
                    session_id=pair.session_id,
                    local_index=pair.local_index,
                    event_time=pair.event_time,
                    text=renderer(pair),
                )
            )
            if len(buffer) >= 64:
                total += self._flush(buffer, embedder, wait=wait)
                buffer = []
        if buffer:
            total += self._flush(buffer, embedder, wait=wait)
        return total

    def _flush(self, records: list[MemoryRecord], embedder: Any, *, wait: bool) -> int:
        vectors = embedder.encode([r.text for r in records])
        if len(vectors) != len(records):  # pragma: no cover
            raise ValueError("Embedder 返回的向量数与文本数不符")
        self.ensure_collection(len(vectors[0]))
        return self.upsert(records, vectors, wait=wait)

    def _user_filter(self, user_id: str) -> models.Filter:
        return models.Filter(
            must=[models.FieldCondition(key=KEY_USER_ID, match=models.MatchValue(value=user_id))]
        )

    def hybrid_search(
        self,
        *,
        user_id: str,
        query_text: str,
        dense_vector: Sequence[float],
    ) -> list[ScoredMemoryId]:
        if not query_text.strip():
            raise ValueError("query_text 不得为空（query 是必填字段）")

        flt = self._user_filter(user_id)
        prefetch = [
            models.Prefetch(
                query=models.Document(text=query_text, model=BM25_MODEL),
                using=SPARSE_VECTOR,
                limit=self._hybrid.prefetch_limit,
                filter=flt,
            ),
            models.Prefetch(
                query=list(dense_vector),
                using=DENSE_VECTOR,
                limit=self._hybrid.prefetch_limit,
                filter=flt,
            ),
        ]
        result = self._client.query_points(
            collection_name=self._collection,
            prefetch=prefetch,
            query=models.RrfQuery(
                rrf=models.Rrf(
                    k=self._hybrid.rrf_k,
                    weights=list(self._hybrid.weights),
                )
            ),
            limit=self._hybrid.prefetch_limit * 2,
            with_payload=[KEY_MEMORY_ID],
        )
        points = result.points
        out: list[ScoredMemoryId] = []
        for p in points:
            payload = p.payload or {}
            memory_id = payload.get(KEY_MEMORY_ID)
            if memory_id is None:  # pragma: no cover
                fallback = memory_id_from_point_id(p.id)
                raise KeyError(
                    f"point {p.id} 缺 {KEY_MEMORY_ID} payload"
                    f"（UUID 反推得到 {fallback!r}，但它不是 canonical id）"
                )
            out.append(ScoredMemoryId(memory_id=str(memory_id), score=float(p.score)))
        return out

    def count(self, *, user_id: str | None = None) -> int:
        if user_id is None:
            return int(self._client.count(self._collection, exact=True).count)
        return int(
            self._client.count(
                self._collection, count_filter=self._user_filter(user_id), exact=True
            ).count
        )

    def fetch_payloads(self, memory_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        if not memory_ids:
            return {}
        found = self._client.retrieve(
            collection_name=self._collection,
            ids=[point_id_for(m) for m in memory_ids],
            with_payload=True,
            with_vectors=False,
        )
        out: dict[str, dict[str, Any]] = {}
        for rec in found:
            payload = dict(rec.payload or {})
            if KEY_MEMORY_ID in payload:
                out[str(payload[KEY_MEMORY_ID])] = payload
        return out
