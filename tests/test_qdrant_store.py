"""Qdrant 派生索引：建集合、写点、payload、混合检索、隔离、可重建。

⚠ 需要**在跑的 Qdrant**（`TIANXIMEM_QDRANT_URL`，默认 `http://localhost:6333`）。
不可达时整组 skip——**不静默通过**。集合名每个用例一份、用完即删，
**绝不写进 `memories`**（那是开发/线上共用的集合）。
"""

from __future__ import annotations

import hashlib
import uuid

import pytest
from tests.conftest import FakeEmbedder, rd, unit_vector

from tianximem.common.render import render_pair
from tianximem.store.qdrant_store import (
    BM25_MODEL,
    COLLECTION_DEFAULT,
    DENSE_VECTOR,
    SPARSE_VECTOR,
    HybridParams,
    MemoryRecord,
    QdrantStore,
    memory_id_from_point_id,
    point_id_for,
)

DIM = 8
ORTHOGONAL = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
# 与 unit_vector(0) 的余弦 ≈ 0.9：比完美命中差，但明显好于正交
NEARLY_ALIGNED = [0.9, 0.0, 0.0, 0.44, 0.0, 0.0, 0.0, 0.0]


def _mid(tag: str) -> str:
    """一个**合法 hex** 的 `memory_id`（64 位）——① 的 `make_pair_id` 就是这个形状。

    ⚠ 别用 `"g" * 64` 这类偷懒写法：`point_id_for` 要把它当 UUID 解析，
    而非 hex 字符会直接抛 `ValueError`。
    """
    return hashlib.sha256(tag.encode("utf-8")).hexdigest()


# ── 夹具与帮手 ─────────────────────────────────────────────────────────


@pytest.fixture
def make_store(qdrant_client):
    """按需造若干个临时集合，收尾统一删除。"""
    created: list[QdrantStore] = []

    def _make(**kwargs) -> QdrantStore:
        name = f"memories_test_{uuid.uuid4().hex[:8]}"
        store = QdrantStore(url="unused", collection=name, client=qdrant_client, **kwargs)
        created.append(store)
        return store

    try:
        yield _make
    finally:
        for store in created:
            store.drop_collection()


@pytest.fixture
def emb() -> FakeEmbedder:
    return FakeEmbedder(dim=DIM)


def _rec(memory_id: str, user_id: str, text: str, **over) -> MemoryRecord:
    base = {
        "memory_id": memory_id,
        "user_id": user_id,
        "session_id": "s1",
        "local_index": 0,
        "event_time": 1_700_000_000_000,
        "text": text,
    }
    base.update(over)
    return MemoryRecord(**base)


def _index(store: QdrantStore, embedder: FakeEmbedder, records) -> int:
    vectors = embedder.encode([r.text for r in records])
    store.ensure_collection(len(vectors[0]))
    return store.upsert(records, vectors, wait=True)


def _search(store: QdrantStore, user_id: str, query_text: str, top_k: int = 5):
    return store.hybrid_search(
        user_id=user_id, query_text=query_text, dense_vector=unit_vector(DIM, 0)
    )


# ── collection 结构 ─────────────────────────────────────────────────────


def test_default_collection_name_is_memories() -> None:
    """**单一 `memories` collection**——不允许一用户一个 collection。"""
    assert COLLECTION_DEFAULT == "memories"
    store = QdrantStore(url="http://localhost:1", client=_LazyClient())
    assert store.collection == "memories"
    # 没有"按 user 建集合"这种接口——隔离只能靠 payload 过滤
    public = {n for n in dir(store) if not n.startswith("_")}
    assert not {n for n in public if "per_user" in n or "collection_for_user" in n}


def test_ensure_collection_is_idempotent(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    assert qdrant_store.ensure_collection(emb.dim) is True
    assert qdrant_store.ensure_collection(emb.dim) is False  # 第二次不重建
    assert qdrant_store.collection_dim() == emb.dim


def test_collection_uses_single_shard(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    """分片数 = 1：根级融合跨分片合并，**分片数变化会改变排名且无报错**（§6.3）。"""
    qdrant_store.ensure_collection(emb.dim)
    info = qdrant_store.client.get_collection(qdrant_store.collection)
    assert info.config.params.shard_number == 1


def test_dimension_change_is_rejected_not_mixed(
    qdrant_store: QdrantStore, emb: FakeEmbedder
) -> None:
    """维度不一致**必须响亮失败**——混用只会表现为"检索结果很差"，不会报错。"""
    qdrant_store.ensure_collection(emb.dim)
    with pytest.raises(ValueError, match="重建集合"):
        qdrant_store.ensure_collection(emb.dim + 1)


def test_payload_indexes_created_before_any_write(
    qdrant_store: QdrantStore, emb: FakeEmbedder
) -> None:
    """payload 索引**必须在写入数据前建**（§6.3），且 `user_id` 是 keyword + is_tenant。"""
    qdrant_store.ensure_collection(emb.dim)
    assert {"user_id", "session_id", "event_time"} <= qdrant_store.payload_index_names()
    assert qdrant_store.is_tenant_index() is True
    assert qdrant_store.count() == 0  # 建索引时还没有任何数据


def test_sparse_vector_uses_qdrant_bm25_not_an_embedder() -> None:
    """词法那一路用 Qdrant 原生 `qdrant/bm25`（§7.1）——**不是**学出来的稀疏权重。"""
    assert BM25_MODEL == "qdrant/bm25"
    assert SPARSE_VECTOR == "bm25"
    assert DENSE_VECTOR == "dense"


# ── point ↔ 真源行的映射 ────────────────────────────────────────────────


def test_point_id_is_a_deterministic_uuid() -> None:
    memory_id = "a" * 64
    pid = point_id_for(memory_id)
    assert pid == point_id_for(memory_id)  # 纯函数 ⇒ 补全时 upsert 覆盖同一个 point
    assert uuid.UUID(pid)  # Qdrant 只接受 uint64 / UUID
    assert memory_id_from_point_id(pid) == memory_id[:32]


def test_point_id_rejects_too_short_id() -> None:
    with pytest.raises(ValueError, match="太短"):
        point_id_for("abc")


def test_payload_carries_identity_fields(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    """payload 里 `user_id` / `memory_id` / 位置两半都要正确（`memory_id` 是 canonical id）。"""
    mid = _mid("payload")
    _index(
        qdrant_store,
        emb,
        [
            _rec(
                mid,
                "u1",
                "Q: a\nA: b",
                session_id="s9",
                local_index=3,
                event_time=123,
            )
        ],
    )

    payload = qdrant_store.fetch_payloads([mid])[mid]
    assert payload["memory_id"] == mid
    assert payload["user_id"] == "u1"
    assert payload["session_id"] == "s9"
    # D28：位置里只剩 `local_index`（**只用于溯源**，检索不按它过滤/排序）
    # ⚠ `chunk_ordinal` **已经不在了**——它随 D28 整个删掉（那时它来自解析 request_id）
    assert "chunk_ordinal" not in payload
    assert payload["local_index"] == 3
    assert payload["event_time"] == 123
    # 正文不进 payload（§6.3）
    assert "text" not in payload
    assert "content" not in payload


def test_point_holds_dense_and_sparse_vectors(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    """一个 point 同时带 `dense` 与 `bm25`（§6.3）。"""
    mid = _mid("both-vectors")
    _index(qdrant_store, emb, [_rec(mid, "u1", "alpha beta gamma")])

    points = qdrant_store.client.retrieve(
        collection_name=qdrant_store.collection,
        ids=[point_id_for(mid)],
        with_vectors=True,
    )
    vectors = points[0].vector
    assert set(vectors) == {DENSE_VECTOR, SPARSE_VECTOR}
    assert len(vectors[DENSE_VECTOR]) == DIM
    assert len(vectors[SPARSE_VECTOR].indices) > 0  # bm25 由服务端从文本现算


# ── 写入即可查（wait=true）──────────────────────────────────────────────


def test_wait_true_is_searchable_immediately(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    """**响应前立即可搜**（§2.1）——`upsert` 返回后同一轮就能查到。"""
    _index(qdrant_store, emb, [_rec(_mid("wait"), "u1", "alpha beta")])

    hits = _search(qdrant_store, "u1", "alpha beta")
    assert [h.memory_id for h in hits] == [_mid("wait")]
    assert qdrant_store.count() == 1


# ── 召回与隔离 ─────────────────────────────────────────────────────────


def test_recall_within_user(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    target = _mid("recall-target")
    _index(
        qdrant_store,
        emb,
        [_rec(target, "u1", "火车几点开"), _rec(_mid("recall-other"), "u1", "vacation")],
    )
    hits = _search(qdrant_store, "u1", "火车几点开")
    assert target in {h.memory_id for h in hits}


def test_users_never_recall_each_other(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    """**跨 user 检索被禁止**（§2.2：`user_id` 是唯一的隔离字段）。"""
    a, b = _mid("user-a"), _mid("user-b")
    _index(qdrant_store, emb, [_rec(a, "u1", "alpha"), _rec(b, "u2", "alpha")])

    ids_u1 = {h.memory_id for h in _search(qdrant_store, "u1", "alpha", top_k=10)}
    ids_u2 = {h.memory_id for h in _search(qdrant_store, "u2", "alpha", top_k=10)}

    assert ids_u1 == {a}
    assert ids_u2 == {b}

    assert qdrant_store.count() == 2  # 同一个 collection 里，两个用户
    assert qdrant_store.count(user_id="u1") == 1


def test_user_filter_applies_at_candidate_stage_not_after_fusion(
    make_store, emb: FakeEmbedder
) -> None:
    """**过滤必须在候选检索阶段生效，不能在融合之后。**

    构造：别的用户那条**两路都更强**。若过滤发生在融合之后，它会先占掉两条路的
    rank 0，本用户的条目只能拿到 `1/62 + 1/62`。过滤在候选阶段生效时，
    两条路里只有本用户的条目 ⇒ 它就是 rank 0 ⇒ `2/61`。
    """
    store = make_store(hybrid=HybridParams(prefetch_limit=1, weights=(1.0, 1.0), rrf_k=61))
    mine, theirs = _mid("mine"), _mid("theirs")
    emb.preset("alpha alpha", unit_vector(DIM, 0))
    emb.preset("alpha alpha alpha alpha", unit_vector(DIM, 0))

    _index(
        store,
        emb,
        [
            _rec(mine, "u1", "alpha alpha"),
            _rec(theirs, "u2", "alpha alpha alpha alpha"),  # bm25 更强
        ],
    )

    hits = _search(store, "u1", "alpha", top_k=5)

    assert [h.memory_id for h in hits] == [mine]
    assert hits[0].score == pytest.approx(2 / 61)  # 两路都是 rank 0
    assert hits[0].score != pytest.approx(2 / 62)  # 说明别人没进候选池


# ── 两路都能独立贡献候选 ────────────────────────────────────────────────


def test_dense_only_candidate_enters_fusion(make_store, emb: FakeEmbedder) -> None:
    """**只被 dense 召回**的条目必须能进融合。

    `dense_only`：向量与查询完全一致，但文本与查询**没有一个共同词**。
    两路各自只取 top-1，所以它只会出现在 dense 那一路。
    """
    store = make_store(hybrid=HybridParams(prefetch_limit=1, weights=(1.0, 1.0), rrf_k=61))
    dense_only, sparse_only = _mid("dense-only"), _mid("sparse-only")

    emb.preset("zzz yyy", unit_vector(DIM, 0))  # dense 与查询完全一致
    emb.preset("alpha beta", ORTHOGONAL)  # dense 落选

    _index(
        store,
        emb,
        [
            _rec(dense_only, "u1", "zzz yyy"),  # 无共同词 ⇒ bm25 不进候选
            _rec(sparse_only, "u1", "alpha beta"),  # 有共同词 ⇒ bm25 是候选
        ],
    )

    ids = {h.memory_id for h in _search(store, "u1", "alpha beta")}
    assert dense_only in ids, "dense 那一路的候选没有进融合"
    assert sparse_only in ids, "bm25 那一路的候选没有进融合"


def test_sparse_only_candidate_enters_fusion(make_store, emb: FakeEmbedder) -> None:
    """**只被 bm25 召回**的条目必须能进融合（与上一条互补）。"""
    store = make_store(hybrid=HybridParams(prefetch_limit=1, weights=(1.0, 1.0), rrf_k=61))
    sparse_only, dense_only = _mid("sparse-only-2"), _mid("dense-only-2")

    emb.preset("alpha", ORTHOGONAL)  # 只有 bm25 能召回它
    emb.preset("zzz", unit_vector(DIM, 0))  # 只有 dense 能召回它

    _index(
        store,
        emb,
        [_rec(sparse_only, "u1", "alpha"), _rec(dense_only, "u1", "zzz")],
    )

    ids = {h.memory_id for h in _search(store, "u1", "alpha")}
    assert sparse_only in ids, "bm25 那一路的候选没有进融合"
    # ⚠ 这里**不能**断言 dense_only 不在结果里：`prefetch_limit=1` 时**每一路**都会
    # 贡献它的 top-1，所以两路各送来一条，融合后两条都在（各得 `w/61`）。


def test_both_arms_agreeing_ranks_first(make_store, emb: FakeEmbedder) -> None:
    """两路都排第一的条目必须排在**只会一路**的条目之前。"""
    store = make_store(hybrid=HybridParams(prefetch_limit=10, weights=(0.5, 0.5), rrf_k=61))
    both, dense_second, sparse_second = _mid("both"), _mid("dense2"), _mid("sparse2")

    emb.preset("alpha beta", unit_vector(DIM, 0))  # both：dense 满分
    emb.preset("zzz", NEARLY_ALIGNED)  # dense 次好
    emb.preset("alpha alpha alpha", ORTHOGONAL)  # dense 最差，bm25 很好

    _index(
        store,
        emb,
        [
            _rec(both, "u1", "alpha beta"),
            _rec(dense_second, "u1", "zzz"),
            _rec(sparse_second, "u1", "alpha alpha alpha"),
        ],
    )

    hits = _search(store, "u1", "alpha beta", top_k=3)
    assert [h.memory_id for h in hits][0] == both


# ── 显式验证 rrf k = 61 ─────────────────────────────────────────────────


def test_rrf_k_is_explicitly_61(make_store, emb: FakeEmbedder) -> None:
    """**显式验证 `k=61`。**

    两条路各自只有一条候选、且都排 rank 0 ⇒ 融合分数 = `w/k + w/k`；
    取 `weights=(1,1)` 就是**精确的 `2/61`**。

    ⚠ 这一条同时是**默认值的反证**：Qdrant 的 `k` 默认是 **2**，那会给出 `2/2 = 1.0`。
    两种写法的名次都"看起来正常"，只有分数差一个数量级——**极难排查**。
    """
    store = make_store(hybrid=HybridParams(prefetch_limit=1, weights=(1.0, 1.0), rrf_k=61))
    emb.preset("alpha", unit_vector(DIM, 0))
    _index(store, emb, [_rec(_mid("rrf"), "u1", "alpha")])

    hits = _search(store, "u1", "alpha")

    assert hits[0].score == pytest.approx(2 / 61)
    assert hits[0].score != pytest.approx(2 / 2)  # ← 默认 k=2 会给出 1.0
    assert hits[0].score != pytest.approx(2 / 60)  # ← 文献的 60 直接填也不对


def test_rrf_k_default_of_qdrant_really_is_2(qdrant_client, emb: FakeEmbedder) -> None:
    """反证：不设 `k` 时 Qdrant 给的是 `k=2` 的行为——所以**必须显式设**。

    这里绕过本模块的 `HybridParams`，直接对客户端发一个**不带 `k`** 的融合请求。
    """
    from qdrant_client import models

    store = QdrantStore(
        url="unused", collection=f"memories_test_{uuid.uuid4().hex[:8]}", client=qdrant_client
    )
    try:
        emb.preset("alpha", unit_vector(DIM, 0))
        _index(store, emb, [_rec(_mid("default-k"), "u1", "alpha")])

        flt = models.Filter(
            must=[models.FieldCondition(key="user_id", match=models.MatchValue(value="u1"))]
        )
        result = qdrant_client.query_points(
            collection_name=store.collection,
            prefetch=[
                models.Prefetch(
                    query=models.Document(text="alpha", model=BM25_MODEL),
                    using=SPARSE_VECTOR,
                    limit=1,
                    filter=flt,
                ),
                models.Prefetch(query=unit_vector(DIM, 0), using=DENSE_VECTOR, limit=1, filter=flt),
            ],
            query=models.RrfQuery(rrf=models.Rrf(weights=[1.0, 1.0])),  # ← 不设 k
            limit=5,
        )
        assert result.points[0].score == pytest.approx(2 / 2)  # k=2 的默认行为
    finally:
        store.drop_collection()


# ── 契约：返回数量 ─────────────────────────────────────────────────────


def test_store_returns_every_fused_candidate(make_store, emb: FakeEmbedder) -> None:
    """**store 不按 `top_k` 截**——它回全部融合结果（至多 `2 × prefetch_limit`）。

    契约的 `len(data) <= top_k` 仍然成立，但由 `fusion.search` 在**排序之后**执行。
    ⚠ 为什么搬：RRF 并列是常态，而 Qdrant 在**并列卡住 `limit`** 时选谁是不确定的
    （实测同一请求连打 12 次，第 100 位的 id 有 2 种、两条 `score` 都是 0.01）。
    """
    store = make_store()
    records = [_rec(_mid(f"count-{i}"), "u1", f"alpha beta gamma {i}") for i in range(8)]
    _index(store, emb, records)

    hits = store.hybrid_search(
        user_id="u1", query_text="alpha beta gamma", dense_vector=unit_vector(DIM, 0)
    )
    assert len(hits) == 8, "八条都命中 ⇒ 八条都要回来，由调用方截"
    assert len(hits) <= store._hybrid.prefetch_limit * 2


def test_empty_query_is_rejected(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    _index(qdrant_store, emb, [_rec(_mid("empty-q"), "u1", "alpha")])
    with pytest.raises(ValueError, match="不得为空"):
        _search(qdrant_store, "u1", "   ")


def test_upsert_count_mismatch_rejected(qdrant_store: QdrantStore, emb: FakeEmbedder) -> None:
    qdrant_store.ensure_collection(DIM)
    with pytest.raises(ValueError, match="数量不符"):
        qdrant_store.upsert([_rec(_mid("mismatch"), "u1", "a")], [], wait=True)


# ── 可重建：Qdrant 不是真源 ─────────────────────────────────────────────


def test_rebuild_from_sqlite_reproduces_search(store, make_store, emb: FakeEmbedder) -> None:
    """**Qdrant 数据必须能从 SQLite 完全重建**（§6.3）。

    删掉集合再重建，**每条 `memory_id` 的分数必须相同**——这正是"Qdrant 是派生读存储"的含义。

    ⚠ 断言的是"`memory_id → 分数`的映射"，不是**顺序**：分数相同的条目之间名次是
    未定义的（实测**重建后并列项的顺序会变**），断言顺序会得到一个与实现无关的脆弱测试。

    ⚠ 这里给每条渲染文本**预设互不相同的 dense 向量**。若两路都出现并列，
    名次的抖动会掩盖"重建是否真的复现"这个要测的东西。
    """
    from tianximem.pairing import AddBatch, Message, apply_batch

    msgs = tuple(
        Message(role="user" if i % 2 == 0 else "assistant", content=f"alpha beta {i}")
        for i in range(4)
    )
    apply_batch(store, AddBatch("u1|s1|0", "u1", "s1", msgs))

    pairs = rd(store, store.iter_pairs, user_id="u1")
    assert len(pairs) == 2, "① 落库的块数不是预期的 2"

    # 预设两条渲染文本的 dense 向量：一条与查询完全一致、另一条正交 ⇒ 两路都不并列
    texts = [render_pair(p) for p in pairs]
    emb.preset(texts[0], unit_vector(DIM, 0))
    emb.preset(texts[1], ORTHOGONAL)

    qstore = make_store(hybrid=HybridParams(prefetch_limit=10, weights=(1.0, 1.0), rrf_k=61))

    first = qstore.index_pairs(pairs, emb)
    assert first == len(pairs)
    before = {h.memory_id: h.score for h in _search(qstore, "u1", "alpha beta 0")}
    assert before, "重建前就查不到东西"

    # 全量重建
    qstore.drop_collection()
    assert not qstore.exists()
    again = qstore.index_pairs(pairs, emb)

    after = {h.memory_id: h.score for h in _search(qstore, "u1", "alpha beta 0")}
    assert again == first
    assert after == before  # 逐条分数相同


def test_equal_weights_do_not_change_ranking(make_store, emb: FakeEmbedder) -> None:
    """等权（两路同权重）只是**均匀缩放**，不得改变名次——`top-1` 与结果集都必须一致。

    ⚠ 实测：`weights != 1` 时分数的**绝对尺度**不遵循朴素的 `sum(w/(k+r))`
    （`(0.5,0.5)` 拟合出来是 `sum(0.5/(31+r))`）。所以这里只断言**名次与集合**，
    **不断言分数**——§8 明文规定"RRF 融合后的分数不是校准量"，任何下游都不得依赖它。
    """
    both = _mid("eq-both")
    docs = [
        _rec(both, "u1", "alpha beta"),
        _rec(_mid("eq-dense"), "u1", "zzz"),
        _rec(_mid("eq-sparse"), "u1", "alpha alpha alpha"),
    ]
    emb.preset("alpha beta", unit_vector(DIM, 0))
    emb.preset("zzz", NEARLY_ALIGNED)
    emb.preset("alpha alpha alpha", ORTHOGONAL)

    ids_by_weight = {}
    for weights in ((1.0, 1.0), (0.5, 0.5)):
        store = make_store(hybrid=HybridParams(prefetch_limit=10, weights=weights, rrf_k=61))
        _index(store, emb, docs)
        ids_by_weight[weights] = [h.memory_id for h in _search(store, "u1", "alpha beta", top_k=3)]

    assert ids_by_weight[(1.0, 1.0)][0] == ids_by_weight[(0.5, 0.5)][0] == both
    assert set(ids_by_weight[(1.0, 1.0)]) == set(ids_by_weight[(0.5, 0.5)])


def test_index_pairs_renders_through_common_render(store, make_store, emb: FakeEmbedder) -> None:
    """索引侧的文本必须是渲染后的——**同一个 render 函数**，不是这里另拼一份。"""
    from tianximem.pairing import AddBatch, Message, apply_batch

    apply_batch(
        store,
        AddBatch(
            "u1|s1|0",
            "u1",
            "s1",
            (Message(role="user", content="问题"), Message(role="assistant", content="回答")),
        ),
    )
    qstore = make_store()
    pairs = rd(store, store.iter_pairs, user_id="u1")
    qstore.index_pairs(pairs, emb)

    assert emb.encoded_texts() == ["Q: 问题\nA: [assistant] 回答"]
    assert emb.encoded_texts() == [render_pair(pairs[0])]


class _LazyClient:
    """只为构造 `QdrantStore` 而存在——不发任何请求。"""

    def close(self) -> None:
        pass
