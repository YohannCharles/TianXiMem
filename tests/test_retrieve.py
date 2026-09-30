"""`retrieve/` —— 混合检索的策略与参数所有权（§7.1–§7.3）+ §8 判据。

⚠ 这些用例**不碰 Qdrant**：`store/` 那一侧已有 20+ 个集成用例（含 k=61 的实测）。
本文件测的是**策略层**——参数校验、查询侧调用次数、名次编号、判据逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from tests.conftest import FakeEmbedder

from tianximem.retrieve import (
    RRF_K,
    Candidate,
    DenseArm,
    DenseArmError,
    EvidenceChecker,
    HybridRetriever,
    InMemoryCheckerInstrument,
    RetrievalParamError,
    lexical_query,
    make_hybrid_params,
    rank_consistency,
)
from tianximem.store.qdrant_store import HybridParams, ScoredMemoryId

# ── 桩 ─────────────────────────────────────────────────────────────────


@dataclass
class _StubStore:
    """只记录调用参数的假 `QdrantStore`（本文件不测 Qdrant 行为）。"""

    hits: list[ScoredMemoryId] = field(default_factory=list)
    seen: list[dict] = field(default_factory=list)

    def hybrid_search(self, *, user_id, query_text, dense_vector, top_k):
        self.seen.append(
            {
                "user_id": user_id,
                "query_text": query_text,
                "dense_vector": list(dense_vector),
                "top_k": top_k,
            }
        )
        return list(self.hits[:top_k])


@dataclass
class _InstructionStub:
    """带 `encode_query` 的假 Embedder——验证 `DenseArm` 的鸭子类型分支。"""

    vector: list[float]
    doc_vectors: list[list[float]] = field(default_factory=list)
    query_texts: list[str] = field(default_factory=list)
    doc_texts: list[str] = field(default_factory=list)

    @property
    def dim(self) -> int:
        return len(self.vector)

    def encode_query(self, texts):
        self.query_texts.extend(texts)
        return [list(self.vector) for _ in texts]

    def encode(self, texts):
        self.doc_texts.extend(texts)
        return [list(v) for v in self.doc_vectors] or [[0.0] * len(self.vector) for _ in texts]


def _retriever(store, embedder):
    """只装编排者本身——**融合参数不在这里**（它们注入 `QdrantStore`）。"""
    return HybridRetriever(store=store, dense=DenseArm(embedder))


# ── 参数所有权：rrf_k 不是旋钮 ──────────────────────────────────────────


def test_rrf_k_must_be_exactly_61() -> None:
    """`rrf_k` 是**正确性常量**——填错**必须拒绝**，不是"警告后照用"。

    Qdrant 的默认是 `2`、文献是 `60`，只有 `61` 才等价于文献的 60（秩 0-based，D5）。
    而填错**不会报错**：两种写法的名次都"看起来正常"，只有分数差一个数量级。
    """
    assert make_hybrid_params().rrf_k == RRF_K == 61
    for wrong in (2, 60, 62, 1, 0):
        with pytest.raises(RetrievalParamError, match="rrf_k 必须是 61"):
            make_hybrid_params(rrf_k=wrong)


def test_rrf_k_error_explains_why() -> None:
    """报错要能自解释——否则下一个人会把它当成"太严的校验"绕过。"""
    with pytest.raises(RetrievalParamError) as exc:
        make_hybrid_params(rrf_k=60)
    assert "正确性常量" in str(exc.value)
    assert "D5" in str(exc.value)


def test_prefetch_limit_must_be_positive() -> None:
    for bad in (0, -1):
        with pytest.raises(RetrievalParamError, match="prefetch_limit 必须为正"):
            make_hybrid_params(prefetch_limit=bad)


def test_weights_must_be_two_non_negative_not_both_zero() -> None:
    with pytest.raises(RetrievalParamError, match="必须是两个"):
        make_hybrid_params(weights=[0.5])
    with pytest.raises(RetrievalParamError, match="不得为负"):
        make_hybrid_params(weights=[-0.1, 0.5])
    with pytest.raises(RetrievalParamError, match="同时为 0"):
        make_hybrid_params(weights=[0.0, 0.0])


def test_make_hybrid_params_returns_the_store_type() -> None:
    """校验通过后交给 `store/` 的仍是它认识的那个类型（**执行代码不搬**）。

    ⚠ 这条**不验证"转发"**——它只断言类型与字段值；真正"参数到达 Qdrant"由
    `test_build_services_forwards_retrieval_params_to_the_store` 守。
    """
    params = make_hybrid_params(prefetch_limit=50, weights=[0.7, 0.3], rrf_k=61)
    assert isinstance(params, HybridParams)
    assert (params.prefetch_limit, params.weights, params.rrf_k) == (50, (0.7, 0.3), 61)


# ── 词法那一路：查询侧原样送 ────────────────────────────────────────────


def test_lexical_query_is_identity() -> None:
    """**查询侧不做改写**（§7.2）——把规格变成可断言的对象，而不是文档约定。"""
    for query in ("火车几点开？", "  keep   spaces  ", "MixedCASE", "a\nb"):
        assert lexical_query(query) == query


def test_lexical_query_rejects_blank() -> None:
    for blank in ("", "   ", "\n"):
        with pytest.raises(ValueError, match="不得为空"):
            lexical_query(blank)


# ── 语义那一路：每查询恰好 1 次 ─────────────────────────────────────────


def test_dense_arm_uses_encode_query_when_present() -> None:
    """有 `encode_query`（被 `QueryInstructionEmbedder` 套过）就用它。"""
    stub = _InstructionStub(vector=[1.0, 2.0])
    arm = DenseArm(stub)
    assert arm.encode_query("q") == [1.0, 2.0]
    assert stub.query_texts == ["q"]
    assert stub.doc_texts == []  # 查询侧不碰文档侧的方法


def test_dense_arm_falls_back_to_encode() -> None:
    """没有 `encode_query` 就退回 `encode`——`DenseArm` 不必知道有没有开前缀。"""
    emb = FakeEmbedder(dim=4)
    arm = DenseArm(emb)
    vector = arm.encode_query("hello world")
    assert len(vector) == 4
    assert emb.encoded_texts() == ["hello world"]


def test_dense_arm_counts_query_calls() -> None:
    """**每查询恰好 1 次**要被断言，不能只靠约定。"""
    arm = DenseArm(FakeEmbedder(dim=4))
    arm.encode_query("a")
    arm.encode_query("b")
    assert arm.query_calls == ["a", "b"]


def test_dense_arm_rejects_wrong_vector_count() -> None:
    @dataclass
    class _Two:
        @property
        def dim(self) -> int:
            return 2

        def encode(self, texts):
            return [[1.0, 0.0], [0.0, 1.0]]

    with pytest.raises(DenseArmError, match="应返回 1 个向量"):
        DenseArm(_Two()).encode_query("q")


def test_dense_arm_rejects_empty_vector() -> None:
    @dataclass
    class _Empty:
        @property
        def dim(self) -> int:
            return 0

        def encode(self, texts):
            return [[]]

    with pytest.raises(DenseArmError, match="空向量"):
        DenseArm(_Empty()).encode_query("q")


def test_dense_arm_documents_go_through_encode_unchanged() -> None:
    """索引侧**原样**送——`QueryInstructionEmbedder.encode()` 永不加前缀。"""
    stub = _InstructionStub(vector=[1.0], doc_vectors=[[0.1], [0.2]])
    arm = DenseArm(stub)
    assert arm.encode_documents(["t1", "t2"]) == [[0.1], [0.2]]
    assert stub.doc_texts == ["t1", "t2"]
    assert stub.query_texts == []


def test_dense_arm_dim_comes_from_the_embedder() -> None:
    assert DenseArm(FakeEmbedder(dim=7)).dim == 7


# ── 编排：名次、top_k、调用次数 ─────────────────────────────────────────


def test_search_assigns_zero_based_ranks_in_store_order() -> None:
    store = _StubStore(
        hits=[
            ScoredMemoryId("m0", 0.9),
            ScoredMemoryId("m1", 0.5),
            ScoredMemoryId("m2", 0.1),
        ]
    )
    got = _retriever(store, FakeEmbedder(dim=4)).search(user_id="u1", query="q", top_k=3)
    assert got == [
        Candidate("m0", 0),
        Candidate("m1", 1),
        Candidate("m2", 2),
    ]


def test_candidate_carries_no_score_field() -> None:
    """**`Candidate` 里没有 `score`——这是结构性防泄漏，不是省字段。**

    融合分数不是校准量（§8），而响应里的 `score` 必须是**最终名次的函数**
    （由 `rank/` 现算）。上游手里没有那个值，也就漏不出去。
    Rerank 接入后名次会变，届时 `score` 随之重算——固化在 `Candidate` 里就会过时。
    """
    from dataclasses import fields

    assert {f.name for f in fields(Candidate)} == {"memory_id", "rank"}


def test_search_passes_top_k_from_the_request_not_100() -> None:
    """**`top_k` 来自请求、不写死 100**（§7.3）——写死会在 AML 传更小值时变成契约错误。"""
    store = _StubStore(hits=[ScoredMemoryId(f"m{i}", 1.0 - i * 0.1) for i in range(10)])
    _retriever(store, FakeEmbedder(dim=4)).search(user_id="u1", query="q", top_k=3)
    assert store.seen[0]["top_k"] == 3


def test_search_calls_dense_exactly_once_per_query() -> None:
    store = _StubStore(hits=[ScoredMemoryId("m0", 1.0)])
    retriever = _retriever(store, FakeEmbedder(dim=4))
    retriever.search(user_id="u1", query="q", top_k=5)
    assert len(retriever.dense.query_calls) == 1  # ← 恰好 1 次


def test_search_sends_the_raw_query_to_both_arms() -> None:
    """**查询原样送出**——过滤掉的是"我们偷偷改了查询"这一类 bug。"""
    store = _StubStore(hits=[ScoredMemoryId("m0", 1.0)])
    emb = FakeEmbedder(dim=4)
    raw = "  火车几点开？ "
    _retriever(store, emb).search(user_id="u1", query=raw, top_k=5)

    assert store.seen[0]["query_text"] == raw  # 词法那一路
    assert emb.encoded_texts() == [raw]  # 语义那一路


def test_search_forwards_user_id() -> None:
    """`user_id` 是**唯一**的隔离字段——编排层不得丢它（§2.2）。"""
    store = _StubStore(hits=[ScoredMemoryId("m0", 1.0)])
    _retriever(store, FakeEmbedder(dim=4)).search(user_id="alice", query="q", top_k=5)
    assert store.seen[0]["user_id"] == "alice"


@pytest.mark.parametrize("top_k", [0, -1, -5])
def test_search_with_non_positive_top_k_makes_no_calls(top_k: int) -> None:
    """`top_k <= 0` 直接空手而归——**不查库、不调 embedding**。"""
    store = _StubStore(hits=[ScoredMemoryId("m0", 1.0)])
    emb = FakeEmbedder(dim=4)
    retriever = _retriever(store, emb)
    assert retriever.search(user_id="u1", query="q", top_k=top_k) == []
    assert store.seen == []
    assert retriever.dense.query_calls == []


def test_search_never_returns_more_than_top_k() -> None:
    store = _StubStore(hits=[ScoredMemoryId(f"m{i}", 1.0) for i in range(9)])
    got = _retriever(store, FakeEmbedder(dim=4)).search(user_id="u1", query="q", top_k=4)
    assert len(got) <= 4


def test_build_services_forwards_retrieval_params_to_the_store(tmp_path) -> None:
    """**配置里的检索参数必须真的到达 Qdrant**——`retrieve/CLAUDE.md` 的"参数所有权"。

    ⚠ 只断言"字段被赋值"是**空过的**：`HybridRetriever.search()` 若没把 `params` 传给
    `store.hybrid_search()`、或 `build_services()` 没给 `QdrantStore` 传 `hybrid=`，
    改 yaml 里的 `prefetch_limit` / `weights` 就**静默无效**——只因 store 的默认值
    恰好与 config 默认值相等而看不出来。

    ⇒ 断言落在**装配产物**上：`build_services(config)` 造出的那个 `QdrantStore`
    携带的参数就是 config 给的那些。**故意用非默认值**——用默认值的话，
    改坏了这条用例也照样绿。
    """
    from tianximem.common.config import (
        AppConfig,
        CacheConfig,
        EmbedCacheConfig,
        QdrantConfig,
        RetrievalConfig,
        RrfConfig,
        SqliteConfig,
        StorageConfig,
    )
    from tianximem.service import build_services

    config = AppConfig(
        storage=StorageConfig(
            sqlite=SqliteConfig(path=str(tmp_path / "tianxi.db")),
            qdrant=QdrantConfig(url="http://unused"),
        ),
        cache=CacheConfig(embed=EmbedCacheConfig(dir=str(tmp_path / "cache"))),
        embed_base_url="http://unused/v1",
        embed_api_key="k",
        retrieval=RetrievalConfig(
            prefetch_limit=50,
            rrf=RrfConfig(k=RRF_K, weights=(0.7, 0.3)),
        ),
    )
    services = build_services(config)
    try:
        assert services.qdrant.hybrid.prefetch_limit == 50
        assert services.qdrant.hybrid.weights == (0.7, 0.3)
        assert services.qdrant.hybrid.rrf_k == RRF_K
    finally:
        services.close()


# ── §8 判据 ────────────────────────────────────────────────────────────


def test_rank_consistency_top1_agree() -> None:
    assert rank_consistency(["a", "b"], ["a", "c"]) is True


def test_rank_consistency_top5_overlap() -> None:
    """两路 top-5 重叠 ≥ 3 ⇒ 足够（两路 top-1 不同也算）。"""
    assert rank_consistency(["a", "b", "c", "d"], ["x", "b", "c", "d"]) is True


def test_rank_consistency_bm25_top1_within_dense_top3() -> None:
    assert rank_consistency(["a", "zz"], ["x", "y", "a", "zz"]) is True


def test_rank_consistency_disagreement_is_not_enough() -> None:
    """两路各说各的 ⇒ 不足（"本来会触发多轮搜索"）。"""
    assert rank_consistency(["a", "b"], ["x", "y", "z", "w"]) is False


def test_rank_consistency_handles_empty_arms() -> None:
    assert rank_consistency([], ["a"]) is False
    assert rank_consistency(["a"], []) is False
    assert rank_consistency([], []) is False


def test_rank_consistency_thresholds_are_configurable() -> None:
    """三个初值是**自设阈值**（可调，但要有 ablation 数据支撑）——所以必须可配。"""
    from tianximem.retrieve import CheckerThresholds

    strict = CheckerThresholds(topk_overlap=10, topk_overlap_min=9)
    assert rank_consistency(["a", "b", "c"], ["x", "b", "c"], strict) is False


# ── v1 的 Checker：恒充足，但每轮都记 ───────────────────────────────────


def test_checker_always_says_enough_in_v1() -> None:
    """v1 **不做门控**（D13）——不管判据本来会说什么。"""
    checker = EvidenceChecker()
    assert checker.decide(query="q").enough is True
    assert checker.decide(query="q", arm_rankings=(["a"], ["x", "y", "z"])).enough is True


def test_checker_records_every_decision() -> None:
    """**每轮判定必须记录**——不记，Step 4 之前无法用数据回答 A4。"""
    inst = InMemoryCheckerInstrument()
    checker = EvidenceChecker(instrument=inst)
    checker.decide(query="q1")
    checker.decide(query="q2")
    assert len(inst.decisions) == 2
    assert all(d.enough for d in inst.decisions)


def test_criterion_input_is_none_when_probes_were_not_run() -> None:
    """v1 的调用方不传两路排名 ⇒ 记成 `None`，**不是 `False`**。

    两者对 A4 的结论完全相反：`None` = "没测"，`False` = "不会触发"。
    """
    decision = EvidenceChecker().decide(query="q")
    assert decision.criterion_would_say is None
    assert "未跑两路分离查询" in decision.reason


def test_criterion_would_say_is_recorded_when_probes_ran() -> None:
    decision = EvidenceChecker().decide(query="q", arm_rankings=(["a", "b"], ["x", "y", "z", "w"]))
    assert decision.criterion_would_say is False
    assert "判据本来判定为不足" in decision.reason


def test_would_trigger_rate_is_none_not_zero_without_data() -> None:
    """**没有数据要返回 `None`，不能返回 0。**"""
    inst = InMemoryCheckerInstrument()
    checker = EvidenceChecker(instrument=inst)
    checker.decide(query="q")
    assert inst.would_trigger_rate is None  # ← 不是 0.0

    checker.decide(query="q2", arm_rankings=(["a"], ["a"]))  # 判据：充足
    checker.decide(query="q3", arm_rankings=(["a"], ["x", "y", "z"]))  # 判据：不足
    assert inst.would_trigger_rate == pytest.approx(0.5)


def test_checker_default_instrument_is_a_noop() -> None:
    """不传记录出口时不得炸（`NullCheckerInstrument` 是默认值）。"""
    EvidenceChecker().decide(query="q")


# ── 并列分数的次序（V13，2026-09-26）────────────────────────────────────
def _tied_hits() -> list[ScoredMemoryId]:
    """一组**故意同分**的候选——RRF 的日常形态（`1/(61+名次)` 是离散值）。

    ⚠ 三条的 `score` 完全一样：`dense` 与 `bm25` 里名次相同的两条会落在同一个值上。
    """
    return [
        ScoredMemoryId(memory_id="cc" * 32, score=0.125),
        ScoredMemoryId(memory_id="aa" * 32, score=0.125),
        ScoredMemoryId(memory_id="bb" * 32, score=0.125),
        ScoredMemoryId(memory_id="dd" * 32, score=0.0625),
    ]


def test_tied_scores_get_a_deterministic_order() -> None:
    """**同分项的次序必须与上游给的顺序无关**（V13）。

    为什么这条重要：名次是**位置派生**的（`retrieve()` 里的 `enumerate`），而下游
    （种子选择 → 扩窗 → 打包顺序）全都读名次。上游（Qdrant）对同分项的返回顺序在两次
    相同请求之间都会变 ⇒ 不在这里定死，整条链就跟着抖，表现为"**同配置重跑差 ±3pt**"，
    **而没有任何东西会报错**。

    ⇒ 断言：把同一批同分候选**按不同顺序**喂进去，得到的 `memory_id → rank` 完全一致，
    且并列内部按 `memory_id` 升序。
    """
    orders = [
        _tied_hits(),
        list(reversed(_tied_hits())),
        [_tied_hits()[i] for i in (1, 3, 0, 2)],
    ]
    results = []
    for hits in orders:
        retriever = _retriever(_StubStore(hits=hits), _InstructionStub(vector=[1.0, 0.0]))
        ranked = retriever.search(user_id="u1", query="q", top_k=10)
        results.append([(c.memory_id, c.rank) for c in ranked])

    assert results[0] == results[1] == results[2], "同一批同分候选，次序随上游顺序变了"
    # 并列内部按 `memory_id` 升序；分数高的仍在前（`dd` 分低，排最后）
    assert [mid[:2] for mid, _ in results[0]] == ["aa", "bb", "cc", "dd"]


def test_score_still_wins_over_the_tie_break() -> None:
    """次级键**只在并列时**生效——不能把分数更高的挤到后面去。"""
    hits = [
        ScoredMemoryId(memory_id="zz" * 32, score=0.9),
        ScoredMemoryId(memory_id="aa" * 32, score=0.1),
    ]
    retriever = _retriever(_StubStore(hits=hits), _InstructionStub(vector=[1.0, 0.0]))
    ranked = retriever.search(user_id="u1", query="q", top_k=10)
    assert [c.memory_id[:2] for c in ranked] == ["zz", "aa"]
