"""`Add` 的端到端时序、两层幂等、以及**那个必须专门处理的失败窗口**。

⚠ 用**假 Qdrant**（鸭子类型）而不是真容器：本文件要精确控制"第几次 upsert 失败"。
真 Qdrant 的行为已经由 `tests/test_qdrant_store.py` 的 20+ 集成用例覆盖。

## 连接生命周期与跨线程（D17）

FastAPI 的 `def` 路由跑在**线程池**里，而 `sqlite3.Connection` 只能在**创建它的线程**里用
⇒ `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in that
same thread.`（**线上每个请求都会失败**，不只是测试问题）。

**短生命周期连接**：连接在"一次逻辑操作 / 一个事务"内建立并关闭，**从不跨线程复用**
（见 `store/sqlite_store.py` 的连接模型一节）。本文件那三个并发用例就是这条的回归用例。
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

import pytest
from tests.conftest import FakeEmbedder, rd, run_parallel

from tianxi_am.common.render import render_pair
from tianxi_am.pairing import AddBatch
from tianxi_am.pairing.pairing import Message
from tianxi_am.service.pipeline import AddPipeline
from tianxi_am.store.sqlite_store import SqliteStore

# ── 假 Qdrant ──────────────────────────────────────────────────────────


@dataclass
class _FakeQdrant:
    """只实现 Add 用得到的那两件事：`index_pairs` 与（供 Search 用的）`exists`。

    `points` 按 `memory_id` 去重——这正是真 Qdrant 的语义（point id 由位置派生，
    同一个对永远是同一个 point ⇒ upsert 覆盖而不是新增）。
    """

    points: dict[str, str] = field(default_factory=dict)
    upsert_calls: list[list[str]] = field(default_factory=list)
    fail_next: int = 0
    overlap: _Overlap | None = None
    hold_s: float = 0.0

    def index_pairs(
        self, pairs: Iterable[object], embedder: object, *, renderer=render_pair, wait=True
    ):
        index = list(pairs)
        if self.fail_next > 0:
            self.fail_next -= 1
            raise RuntimeError("模拟 Qdrant/embedding 失败")
        texts = [renderer(p) for p in index]
        embedder.encode(texts)  # type: ignore[attr-defined]
        if self.overlap is not None:
            with self.overlap.enter():
                _sleep(self.hold_s)
        for pair, text in zip(index, texts, strict=True):
            self.points[pair.id] = text  # type: ignore[attr-defined]
        self.upsert_calls.append([p.id for p in index])  # type: ignore[attr-defined]
        return len(index)

    def exists(self) -> bool:
        return True


def _sleep(seconds: float) -> None:
    if seconds > 0:
        threading.Event().wait(seconds)


@dataclass
class _Overlap:
    """记录"同时有几个线程在 index_pairs 里"——用来**证明**串行化。"""

    active: int = 0
    peak: int = 0
    _guard: threading.Lock = field(default_factory=threading.Lock)

    @contextmanager
    def enter(self):
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            yield
        finally:
            with self._guard:
                self.active -= 1


# ── 夹具 ───────────────────────────────────────────────────────────────


@pytest.fixture
def qdrant() -> _FakeQdrant:
    return _FakeQdrant()


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder(dim=8)


@pytest.fixture
def pipeline(store: SqliteStore, qdrant: _FakeQdrant, embedder: FakeEmbedder):
    return AddPipeline(store=store, qdrant=qdrant, embedder=embedder)


def _rid(ordinal: int, *, user_id: str = "u1", session_id: str = "s1") -> str:
    """造一个 **符合 `ingest.chunk_ordinal_pattern`** 的 `request_id`（D25）。

    ⚠ 不再是随便一个字符串：位置由它派生，**取不到 chunk 序号就响亮失败**。
    这里用本仓 harness 的形态（`<user>|<session>|<n>`，见 `eval/harness/batching.py`
    的 `request_id_for`）——默认正则同时认它和平台实发的 `...:chunk-<n>`。
    """
    return f"{user_id}|{session_id}|{ordinal}"


def _batch(
    request_id: str, *contents: str, user_id: str = "u1", session_id: str = "s1"
) -> AddBatch:
    """按 user/assistant 交替造一批消息。"""
    return AddBatch(
        request_id=request_id,
        user_id=user_id,
        session_id=session_id,
        messages=tuple(
            Message(role="user" if i % 2 == 0 else "assistant", content=c)
            for i, c in enumerate(contents)
        ),
    )


def _pairs(store: SqliteStore, user_id: str = "u1") -> list:
    return rd(store, store.iter_pairs, user_id=user_id)


# ── 正常路径 ───────────────────────────────────────────────────────────


def test_add_writes_both_sqlite_and_qdrant(
    pipeline: AddPipeline, store: SqliteStore, qdrant: _FakeQdrant
) -> None:
    """端到端：SQLite 有真源，Qdrant 有对应的 point。"""
    outcome = pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    assert outcome.applied is True
    assert outcome.repaired is False
    assert outcome.new_pair_count == 1

    pairs = _pairs(store)
    assert len(pairs) == 1
    assert qdrant.points == {pairs[0].id: render_pair(pairs[0])}


def test_add_indexes_only_what_the_batch_touched(
    pipeline: AddPipeline, qdrant: _FakeQdrant
) -> None:
    """正常路径**精确到本批触碰的对**，不做全 session 重建。"""
    pipeline.apply(_batch(_rid(1), "Q1", "A1"))
    pipeline.apply(_batch(_rid(2), "Q2", "A2", user_id="u1", session_id="s1"))

    assert len(qdrant.upsert_calls) == 2
    assert all(len(call) == 1 for call in qdrant.upsert_calls)  # 每次只索引 1 条


def test_positions_come_from_the_chunk_ordinal(pipeline: AddPipeline, store: SqliteStore) -> None:
    """**位置由 `request_id` 的 chunk 序号派生**（D25）——不是到达顺序。

    ⚠ 这里刻意**先发 chunk 2 再发 chunk 1**：旧口径（`MAX+1`）下它们会按到达顺序
    拿到 0 与 1，**把会话顺序读反**；现在它们各自拿自己的序号。
    """
    pipeline.apply(_batch(_rid(2), "Q2", "A2"))
    pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    assert sorted(p.chunk_ordinal for p in _pairs(store)) == [1, 2]
    assert [p.question for p in sorted(_pairs(store), key=lambda x: x.chunk_ordinal)] == [
        "Q1",
        "Q2",
    ]
    # 会话顺序（`seq`）由 chunk 序号给出，与到达顺序无关
    assert [p.seq for p in sorted(_pairs(store), key=lambda x: x.seq)] == [0, 1]


# ── 幂等：两层 ─────────────────────────────────────────────────────────


def test_same_request_id_does_not_create_new_rows(
    pipeline: AddPipeline, store: SqliteStore
) -> None:
    """**batch 层幂等**：`applied_batches` 命中 ⇒ 不产生新的行。"""
    pipeline.apply(_batch(_rid(1), "Q1", "A1"))
    before = [(p.id, p.chunk_ordinal, p.local_index) for p in _pairs(store)]

    outcome = pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    assert outcome.applied is False
    assert [(p.id, p.chunk_ordinal, p.local_index) for p in _pairs(store)] == before


def test_same_request_id_does_not_duplicate_points(
    pipeline: AddPipeline, store: SqliteStore, qdrant: _FakeQdrant
) -> None:
    """**point 层幂等**：point id 由位置派生 ⇒ 重放是覆盖，不是新增。"""
    pipeline.apply(_batch(_rid(1), "Q1", "A1"))
    pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    assert len(qdrant.points) == len(_pairs(store)) == 1


# ── 【重点】SQLite 已提交、Qdrant 失败 → 重试必须修复 ─────────────────


def test_qdrant_failure_keeps_truth_source_and_fails_the_request(
    pipeline: AddPipeline, store: SqliteStore, qdrant: _FakeQdrant
) -> None:
    """Qdrant 失败 ⇒ **请求失败**，但 SQLite 的真源已经落地（不该回滚）。

    ⇒ 这是**必须报错的**：若这里返回 200，那批记忆就永远检索不到了。
    """
    qdrant.fail_next = 1

    with pytest.raises(RuntimeError, match="模拟 Qdrant"):
        pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    assert len(_pairs(store)) == 1  # 真源在
    assert qdrant.points == {}  # 派生索引没写成


def test_retry_after_qdrant_failure_repairs_the_index(
    pipeline: AddPipeline, store: SqliteStore, qdrant: _FakeQdrant
) -> None:
    """**【最重要】** SQLite 已提交、Qdrant 失败 → 相同 `request_id` 重试 ⇒ 索引被补齐。

    ⚠ 若重试时因为 `applied_batches` 命中就直接 no-op，这里会留下
    **SQLite 有真源、Qdrant 永久缺索引**——那些记忆永远检索不到，且不报错。
    """
    qdrant.fail_next = 1
    with pytest.raises(RuntimeError):
        pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    outcome = pipeline.apply(_batch(_rid(1), "Q1", "A1"))  # ← 相同 request_id

    assert outcome.applied is False  # 守卫命中（没写新的真源行）
    assert outcome.repaired is True  # 但走了修复路径
    assert len(_pairs(store)) == 1  # 真源没有重复
    assert len(qdrant.points) == 1  # 派生索引被补上了


def test_repair_covers_the_earlier_batches_of_the_same_session_too(
    pipeline: AddPipeline, store: SqliteStore, qdrant: _FakeQdrant
) -> None:
    """修复是**按 session** 的：先前成功批次留下的 point 被原样覆盖，缺失的被补上。

    这顺带说明代价：重放 upsert 的是该 session 的全部对，而不是只有本批那几条
    （换取"不改 ① 的 DDL"，见 `pipeline.py` 的说明）。
    """
    pipeline.apply(_batch(_rid(1), "Q1", "A1"))  # 第一批成功
    qdrant.points.clear()  # 模拟"派生索引丢了"

    pipeline.apply(_batch(_rid(1), "Q1", "A1"))  # 重放

    assert len(qdrant.points) == len(_pairs(store)) == 1


def test_sqlite_failure_does_not_touch_qdrant(
    pipeline: AddPipeline, qdrant: _FakeQdrant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**SQLite 失败时不继续写 Qdrant**——先真源、后派生索引，顺序不可反。"""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("模拟 SQLite 失败")

    monkeypatch.setattr("tianxi_am.service.pipeline.apply_batch", _boom)

    with pytest.raises(RuntimeError, match="模拟 SQLite"):
        pipeline.apply(_batch(_rid(1), "Q1", "A1"))

    assert qdrant.upsert_calls == []
    assert qdrant.points == {}


# ── D25：同 session 的 Add **可以并发** ─────────────────────────────────


def test_same_session_adds_run_concurrently_and_do_not_collide(
    store: SqliteStore, qdrant: _FakeQdrant, embedder: FakeEmbedder
) -> None:
    """**同一个 `(user_id, session_id)` 的 Add 并发**——D25 之后这是**允许**的。

    旧口径（`MAX+1` 分配位置）下这条必须串行，否则位置撞车；现在位置是请求的纯函数
    （`(chunk_ordinal, local_index)`）⇒ 各批触碰**互不相交**的位置。

    断言四件事：
      1. 无异常（**含 `IntegrityError`**——那正是"位置撞车"的信号）
      2. **峰值并发 > 1**：不并发的话这条用例证明不了任何事（旧锁会把它压成 1）
      3. 每个 chunk 的块都落下了，且块序号从 0 连续
      4. `id` 两两不同（位置派生 ⇒ 不同位置必然不同 id）
    """
    qdrant.overlap = _Overlap()
    qdrant.hold_s = 0.05  # 拉长 Qdrant 阶段，让"真的并发重叠"可观测
    pipeline = AddPipeline(store=store, qdrant=qdrant, embedder=embedder)

    errors = run_parallel(
        [(lambda i=i: pipeline.apply(_batch(_rid(i), f"Q{i}", f"A{i}"))) for i in range(4)]
    )

    assert errors == []  # ← 线程里的异常会让用例"空过"，必须显式断言
    assert qdrant.overlap.peak > 1  # ← **真的并发了**（旧锁下恒为 1）

    pairs = _pairs(store)
    assert sorted((p.chunk_ordinal, p.local_index) for p in pairs) == [
        (0, 0), (1, 0), (2, 0), (3, 0)
    ]
    assert len({p.id for p in pairs}) == 4


def test_different_sessions_do_not_block_each_other(
    store: SqliteStore, qdrant: _FakeQdrant, embedder: FakeEmbedder
) -> None:
    """**不同 session 不互相阻塞**（哪怕同一个 user）。

    用 `Barrier(2)` 让两个线程**必须同时在 `index_pairs` 里**——若被串行化，第二个
    永远进不来、barrier 超时 ⇒ 用例失败。这样"没阻塞"是被证明的，不是靠时间赌的。

    ⚠ `_Overlap` 只装在**这一层包装**上，**不要**再设 `qdrant.overlap`：`_FakeQdrant.index_pairs`
    自己也会进一次 `self.overlap`，两处同时开会让计数**翻倍**（peak 变 4 而不是 2）。
    """
    overlap = _Overlap()
    barrier = threading.Barrier(2, timeout=5)
    original = qdrant.index_pairs

    def index_pairs(pairs, emb, *, renderer=render_pair, wait=True):  # noqa: ANN001, ANN202
        with overlap.enter():
            barrier.wait()
            return original(pairs, emb, renderer=renderer, wait=wait)

    qdrant.index_pairs = index_pairs  # type: ignore[method-assign]
    pipeline = AddPipeline(store=store, qdrant=qdrant, embedder=embedder)

    errors = run_parallel(
        [
            (lambda i=i: pipeline.apply(_batch(_rid(i), f"Q{i}", f"A{i}", session_id=f"s{i}")))
            for i in range(2)
        ]
    )

    assert errors == []
    assert overlap.peak == 2  # ← 未串行化


def test_different_users_do_not_block_each_other(
    store: SqliteStore, qdrant: _FakeQdrant, embedder: FakeEmbedder
) -> None:
    """**不同 user 不互相阻塞**（即使 `session_id` 相同）。"""
    barrier = threading.Barrier(2, timeout=5)
    original = qdrant.index_pairs

    def index_pairs(pairs, emb, *, renderer=render_pair, wait=True):  # noqa: ANN001, ANN202
        barrier.wait()
        return original(pairs, emb, renderer=renderer, wait=wait)

    qdrant.index_pairs = index_pairs  # type: ignore[method-assign]
    pipeline = AddPipeline(store=store, qdrant=qdrant, embedder=embedder)

    errors = run_parallel(
        [
            (
                lambda i=i: pipeline.apply(
                    _batch(
                        _rid(i, user_id=f"u{i}", session_id="same"),
                        f"Q{i}",
                        f"A{i}",
                        user_id=f"u{i}",
                        session_id="same",
                    )
                )
            )
            for i in range(2)
        ]
    )

    assert errors == []


# ── 【压力】不同 session 的真实并发写入 ────────────────────────────────


def test_different_sessions_write_to_sqlite_concurrently(
    pipeline: AddPipeline, store: SqliteStore, qdrant: _FakeQdrant
) -> None:
    """**走完整 Service 路径的并发写**：N 个 session 同时真正写 SQLite。

    与 `test_store.py::test_concurrent_write_transactions_across_sessions` 的分工：
    那一个直捣 `store.transaction()`（把写窗口拉长到**必然撞锁**），
    这一个走 `AddPipeline.apply`（真链路、真 SQLite 事务、真 Qdrant 调用），
    验证"不同 session 不被 Service 串行化"**且**"并发下真源一行不差"。

    ⚠ 刻意用**同一个 user 的不同 session**：它们争抢**同一个库文件**，
    所以这同时也在钉死"不同 session 之间不共享任何写入状态"——
    锁若错按 user 分，这里的并发就没了（而结果一样"看起来正常"）。
    """
    n_sessions = 5
    rounds = 3
    start = threading.Barrier(n_sessions, timeout=20)

    def add(i: int) -> None:
        start.wait()  # 同时出发
        for k in range(rounds):
            # ⚠ `request_id` 里要带**这个 session 自己**的键：否则 5 个 session 的
            #   第 k 批会共用同一个 `request_id`，而幂等守卫按它判重 ⇒ 只剩一个 session 落库
            pipeline.apply(
                _batch(
                    _rid(k, session_id=f"s{i}"), f"Q{i}{k}", f"A{i}{k}", session_id=f"s{i}"
                )
            )

    assert run_parallel([(lambda i=i: add(i)) for i in range(n_sessions)]) == []

    pairs = _pairs(store)
    assert len(pairs) == n_sessions * rounds  # 无丢批
    for i in range(n_sessions):
        mine = [p for p in pairs if p.session_id == f"s{i}"]
        # 每个 session 内：chunk 序号恰好是 0..rounds-1，且每批各落一块
        # （两批**互不相交** ⇒ 位置不会撞车，这正是 D25 让并发安全的原因）
        assert sorted(p.chunk_ordinal for p in mine) == list(range(rounds))
        assert [p.local_index for p in mine] == [0] * rounds
    assert len(qdrant.points) == n_sessions * rounds  # 派生索引也齐了
