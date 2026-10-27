"""Qdrant 派生索引。

**Qdrant 是可重建的派生索引，不是真源**——索引损坏时可从 SQLite 的正文全量重建
（`index_pairs()` 就是那条路径）。**正文不进 Qdrant**：`Search` 返回时按检索到的
`memory_id` 回 SQLite 批量取正文。

本模块**不认识任何数据集**：它只接受 canonical 的 `memory_id` / `user_id` /
`session_id` / `local_index` / `event_time` 与**已渲染好的文本**。渲染由
[`../common/render.py`](../common/render.py) 负责，本模块不拼字符串。

## 两路向量的来源

| 向量 | 来源 | 说明 |
| --- | --- | --- |
| `dense` | **外部 Embedder** 算好传进来 | 只认 canonical 的 `list[float]`，不关心哪家模型 |
| `bm25` | **Qdrant 服务端**从文本现算 | **真 BM25**，不用学出来的稀疏权重 |

BM25 分数对分词器高度敏感。Qdrant 负责该模型的分词，不能假定其结果等价于 Lucene
等分析器；改动模型或服务端分词配置会改变检索结果。
"""

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

# 命名向量：dense（外部计算）和 bm25（Qdrant 服务端计算）。
DENSE_VECTOR: Final[str] = "dense"
SPARSE_VECTOR: Final[str] = "bm25"
BM25_MODEL: Final[str] = "qdrant/bm25"

# Payload 键覆盖过滤和溯源需求。`memory_id` 是 SQLite 的 canonical ID，使
# point 与源记录的对应关系显式可读；point ID 是其 UUID 形式。
KEY_MEMORY_ID: Final[str] = "memory_id"
KEY_USER_ID: Final[str] = "user_id"
KEY_SESSION_ID: Final[str] = "session_id"
KEY_LOCAL_INDEX: Final[str] = "local_index"
KEY_EVENT_TIME: Final[str] = "event_time"


def point_id_for(memory_id: str) -> str:
    """SQLite 的 `memory_id`（64 位十六进制）→ Qdrant 接受的 **UUID 字符串**。

    Qdrant point ID 只接受 uint64 或 UUID，而 memory ID 是
    `hash(user_id, session_id, request_id, local_index)` 的十六进制串，因此需要映射。

    取哈希前 128 位作为 UUID。映射是纯函数：同一 memory ID 总是得到同一 point ID，
    重放会覆盖原 point，不会留下孤儿；从 SQLite 全量重建索引也依赖该确定性。
    """
    if len(memory_id) < 32:
        raise ValueError(f"memory_id 太短，无法当 UUID 用：{memory_id!r}")
    return str(uuid.UUID(memory_id[:32]))


def memory_id_from_point_id(point_id: Any) -> str | None:
    """只在 payload 缺失时的兜底：UUID → 32 位十六进制（**不是**完整的 64 位原 id）。

    ⚠ 返回的是截断形式，**不能当 canonical id 用**——它只能用来发现不一致。
    正常路径永远读 payload 里的 `memory_id`。
    """
    try:
        return uuid.UUID(str(point_id)).hex
    except (ValueError, AttributeError, TypeError):
        return None


@dataclass(frozen=True, slots=True)
class HybridParams:
    """混合检索的三个独立参数与融合常量。

    ⚠ **`rrf_k` 是正确性常量，不是调参项。** Qdrant 的默认 `k` 是 **2**
    （官方文档逐字："k is a constant (set to 2 by default)"），不设就会得到一个
    与所有参考实现都不同的融合行为、**且极难排查**。`k = 61` 才等价于 RRF 文献里的 60：
    Qdrant 的秩是 **0-based**，公式 `1/(rank + k)`，首位即 `1/(0+61) = 1/(1+60)`。
    在 Qdrant v1.17.0 中，`rrf:{k:61}` 返回 `0.032786883 = 2/61`，默认配置返回
    `0.5 = 1/2`。

    `prefetch_limit` 是每路进入 RRF 融合的候选池大小，不是扩窗种子数，也不是请求的 `top_k`。
    """

    #: ⚠ 三个默认值**都引用 [`../common/config/constants.py`](../common/config/constants.py) 的单一来源**
    #: ——本目录**不决定**这些值，这里只是给这个 dataclass 一个兜底，
    #: 免得直接构造 `HybridParams()` 时又散出第二份字面量。
    #: **正常路径上它们由 [`../service/app.py`](../service/app.py) 的装配从配置注入。**
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    # 顺序与 `prefetch` 顺序一一对应；这里固定为 (bm25, dense)。
    weights: tuple[float, float] = DEFAULT_WEIGHTS
    rrf_k: int = RRF_K


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """要索引的一条记忆。**文本已经渲染好**（`common/render.py` 的产物）。"""

    memory_id: str
    user_id: str
    session_id: str
    #: 它在**那一次 Add** 里的序号——**只用于溯源与调试**，检索不按它排序或过滤
    #: （排序是 Qdrant 的相似度、筛选只按 `user_id`；"Add 内顺序"是 SQLite 那边的事）。
    local_index: int
    event_time: int | None
    text: str


@dataclass(frozen=True, slots=True)
class ScoredMemoryId:
    """检索结果只返回 `memory_id`；正文回到 SQLite 读取。"""

    memory_id: str
    score: float


class QdrantStore:
    """单一 `memories` collection 的建/写/查。

    **不按用户拆分 collection**。`user_id` 存在 payload 中并建立 tenant 索引，
    RRF 在同一个 collection 内对 BM25 和 dense 两路 `prefetch` 做融合。
    """

    def __init__(
        self,
        *,
        url: str,
        collection: str = COLLECTION_DEFAULT,
        hybrid: HybridParams | None = None,
        api_key: str | None = None,
        #: `QdrantClient` 的 `timeout` 只收 `int`（秒）——标成 `float` 会被 mypy 拦下，
        #: 而"悄悄截断成整数秒"正是本项目要求响亮失败的那一类。
        timeout: int = 60,
        client: QdrantClient | None = None,
    ) -> None:
        self._collection = collection
        self._hybrid = hybrid or HybridParams()
        self._client = client or QdrantClient(url=url, api_key=api_key, timeout=timeout)

    # ── 生命周期 ────────────────────────────────────────────────────────

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

    # ── 建集合（幂等）──────────────────────────────────────────────────

    def exists(self) -> bool:
        return bool(self._client.collection_exists(self._collection))

    def collection_dim(self) -> int | None:
        """集合里 `dense` 向量的维度；集合不存在返回 None。"""
        if not self.exists():
            return None
        info = self._client.get_collection(self._collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, dict):  # pragma: no cover — 本模块只建命名向量
            raise TypeError(f"集合 {self._collection} 不是命名向量布局")
        dense = vectors[DENSE_VECTOR]
        return None if dense is None else int(dense.size)

    def ensure_collection(self, dim: int) -> bool:
        """确保集合存在且**维度正确**。返回 True 表示这次真的建了。

        幂等：已存在则只做校验。**维度不一致直接抛错**——两个不同维度的向量混进
        同一个集合只会表现为“检索结果很差”，而不会报错。

        payload 索引在写入数据前创建；否则过滤优化需要重建 HNSW 索引后才生效。
        这里在建集合的同一个方法里紧接着建索引，就是不让调用方有机会漏掉。
        """
        if dim <= 0:
            raise ValueError(f"维度必须为正：{dim}")

        if self.exists():
            existing = self.collection_dim()
            if existing != dim:
                raise ValueError(
                    f"集合 {self._collection} 的 dense 是 {existing} 维，但 Embedder 给出 "
                    f"{dim} 维。维度变化必须**重建集合**（§2.3 / §16），不能就地混用。"
                )
            return False

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(size=dim, distance=models.Distance.COSINE)
            },
            sparse_vectors_config={
                # BM25 使用 sparse + IDF；距离由 Qdrant 的 BM25 模型确定。
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            # 固定单分片：跨分片融合会改变排名，而这一变化不会报错。
            shard_number=1,
        )
        self._create_payload_indexes()
        return True

    def _create_payload_indexes(self) -> None:
        # user_id：keyword + is_tenant——Qdrant 为租户过滤专门优化的索引类型。
        # is_tenant 只支持 keyword 或 uuid，因此 user_id 使用 keyword 索引。
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
        """已建的 payload 索引字段名（测试用）。"""
        info = self._client.get_collection(self._collection)
        return set((info.payload_schema or {}).keys())

    def is_tenant_index(self) -> bool:
        """`user_id` 是不是 keyword + `is_tenant`（测试用）。"""
        info = self._client.get_collection(self._collection)
        schema = (info.payload_schema or {}).get(KEY_USER_ID)
        params = getattr(schema, "params", None)
        return bool(params is not None and getattr(params, "is_tenant", False))

    def drop_collection(self) -> None:
        """删集合（**只用于测试与"从 SQLite 全量重建"这条路**）。"""
        if self.exists():
            self._client.delete_collection(self._collection)

    # ── 写 ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        records: Sequence[MemoryRecord],
        dense_vectors: Sequence[Sequence[float]],
        *,
        wait: bool = True,
    ) -> int:
        """写入一批记忆。`dense` 由调用方算好；`bm25` 由 Qdrant 从 `text` 现算。

        `wait=True` 是必要行为：Add 只能在索引持久化并立即可搜索后返回。本方法默认启用，
        不应为了吞吐关闭。

        每个 point 同时包含 `dense`、`bm25` 和 payload。
        """
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
                    # 传文本：Qdrant 服务端使用 qdrant/bm25 计算稀疏向量。
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
        """从 SQLite 记录批量写入 Qdrant 派生索引。

        `pairs` 用**鸭子类型**取字段（`id` / `user_id` / `session_id` /
        `local_index` / `event_time` / `question` / `answer`），所以本模块**不必 import `store`**：
        它只要求"像 QA 对一样可读"，不要求那是哪一类对象。

        文本由 `renderer` 生成。同一渲染函数用于 embedding 输入和返回给 AML 的 `content`。

        `renderer` 接收完整的 QA 对象而不是两个字符串，因为渲染还需要 `event_time`。
        默认 `render_pair` 不注入日期；启用绝对日期时，上层传入
        `functools.partial(render_pair, inject_abs_time=True)`。

        维度从 `embedder.dim` 取（**第一次调用之后才有值**，见 `embed/base.py`）。
        """
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
        if len(vectors) != len(records):  # pragma: no cover — embedder 的实现问题
            raise ValueError("Embedder 返回的向量数与文本数不符")
        self.ensure_collection(len(vectors[0]))
        return self.upsert(records, vectors, wait=wait)

    # ── 读 ─────────────────────────────────────────────────────────────

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
        """两路 `prefetch`（bm25 + dense）→ RRF 融合。**只回 `memory_id`。**

        ⚠ **返回【全部】融合结果**（至多 `2 × prefetch_limit`），**不在这里按 `top_k` 截**
        ——截断由调用方在**按 `(-score, memory_id)` 排完之后**做。

        **为什么**：RRF 的分数是离散的，**并列是常态**。而 Qdrant 在 `limit` 处**截在并列
        中间**时，**谁进谁出是随机的**——同一个请求连打 12 次，返回都是 100 条，
        但**第 100 位的 id 有 2 种**，而两条的 `score` 完全相同（都是 0.01）。
        ⇒ 在 Qdrant 里截，等于把"哪 100 条"交给了一个不确定的选择；
        在本地排完再截，**边界也确定**（同分按 `memory_id` 升序，纯确定性）。


        ⚠ **`user_id` 过滤加在【每个 `prefetch` 上】，不是加在融合之后。**
        融合后过滤会让别的用户的候选**先占掉名次**再被丢弃——既污染分数、又浪费名额，
        而且过滤遗漏会造成静默越权。`user_id` 是唯一隔离字段，禁止跨用户检索。

        请求的 `top_k` 由调用方执行；这里返回至多 `2 × prefetch_limit` 条融合结果，
        与请求的 `top_k` 无关。调用方完成确定性排序后按 `top_k` 截断。

        查询文本原样送入，不做改写。
        """
        if not query_text.strip():
            raise ValueError("query_text 不得为空（§2.1：query 是必填字段）")

        flt = self._user_filter(user_id)
        prefetch = [
            # 顺序与 HybridParams.weights 的顺序**一一对应**：(bm25, dense)
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
            # 两路各取 `prefetch_limit` ⇒ 融合结果至多 `2 × prefetch_limit` 条。
            # **要全部**——在 Qdrant 里截会把边界交给不确定的选择（见 docstring）。
            limit=self._hybrid.prefetch_limit * 2,
            with_payload=[KEY_MEMORY_ID],
        )
        points = result.points
        out: list[ScoredMemoryId] = []
        for p in points:
            payload = p.payload or {}
            memory_id = payload.get(KEY_MEMORY_ID)
            if memory_id is None:  # pragma: no cover — 索引写坏才会走到
                fallback = memory_id_from_point_id(p.id)
                raise KeyError(
                    f"point {p.id} 缺 {KEY_MEMORY_ID} payload"
                    f"（UUID 反推得到 {fallback!r}，但它不是 canonical id）"
                )
            out.append(ScoredMemoryId(memory_id=str(memory_id), score=float(p.score)))
        return out

    def count(self, *, user_id: str | None = None) -> int:
        """点数；给 `user_id` 时只数该用户（隔离测试用）。"""
        if user_id is None:
            return int(self._client.count(self._collection, exact=True).count)
        return int(
            self._client.count(
                self._collection, count_filter=self._user_filter(user_id), exact=True
            ).count
        )

    def fetch_payloads(self, memory_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        """按 point id 取 payload（测试用：验证 point ↔ 真源行的映射）。"""
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
