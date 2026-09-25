"""§6.1 / §6.3 的存储层：DDL、位置派生 id、单向状态、邻域查询、隔离、事务原子性。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 六、存储。

文件末尾另有一组**连接生命周期与跨线程**的用例——它们是 D17 那条跨层缺口的回归用例
（长期持有连接 × FastAPI 的 `def` 路由跑在线程池里），
见 `store/sqlite_store.py` 的连接模型一节。
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from tests.conftest import rd, run_parallel

from tianxi_am.pairing.continuation import AddBatch, apply_batch
from tianxi_am.pairing.pairing import BatchLimits, Message
from tianxi_am.store.sqlite_store import (
    STATUS_COMPLETE,
    STATUS_PENDING,
    SqliteStore,
    make_pair_id,
)


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _one(store: SqliteStore, pair_idx: int):
    """按位置取一行（`u1` / `s1`）。"""
    rows = rd(store, store.fetch_pairs_by_ids, [make_pair_id("u1", "s1", pair_idx)])
    assert rows, f"pair_idx={pair_idx} 不存在"
    return rows[0]


def _seed(store: SqliteStore, contents: list[tuple[int, str, str | None]]) -> None:
    """直接落几行，便于只测存储层（不经过 continuation）。"""
    with store.transaction() as conn:
        for pair_idx, question, answer in contents:
            store.insert_pair(
                conn,
                user_id="u1",
                session_id="s1",
                pair_idx=pair_idx,
                question=question,
                answer=answer,
                status=STATUS_COMPLETE,
                event_time=None,
                request_id="seed",
            )


# ── 不变式 1：id 位置派生 ───────────────────────────────────────────────


def test_pair_id_is_position_derived_not_content_hash() -> None:
    """`id` 只由 (user_id, session_id, pair_idx) 决定——**与内容无关**。

    这是不变式 1：用内容哈希会在**补全**时变 `id`，留下**孤儿 point**。
    """
    a = make_pair_id("u1", "s1", 0)
    assert a == make_pair_id("u1", "s1", 0)  # 确定性
    # 位置不同 ⇒ id 不同
    assert a != make_pair_id("u1", "s1", 1)
    assert a != make_pair_id("u1", "s2", 0)
    assert a != make_pair_id("u2", "s1", 0)
    # 形状上就是"位置派生"：函数签名里根本没有内容参数
    assert len(a) == 64


def test_pair_id_does_not_collide_across_field_boundaries() -> None:
    """分隔符防止字段粘连：("a","b\\x1fc") 与 ("a\\x1fb","c") 不得撞。"""
    assert make_pair_id("a", "b\x1fc", 0) != make_pair_id("a\x1fb", "c", 0)


def test_id_is_stable_across_completion(store: SqliteStore) -> None:
    """**补全前后 `id` 不变**——这正是位置派生的目的。

    用内容哈希时，pending → complete 会改 `answer` ⇒ `id` 变 ⇒
    Qdrant 里旧的 point 成了孤儿（还在，但再也不会被正确更新）。
    """
    limits = BatchLimits(max_messages=2, max_words=1000)
    apply_batch(
        store,
        AddBatch("r1", "u1", "s1", (_msg("user", "Q1"), _msg("assistant", "A1"))),
        limits=limits,
    )
    before = _one(store, 0)
    assert before.status == STATUS_PENDING

    apply_batch(
        store,
        AddBatch("r2", "u1", "s1", (_msg("assistant", "A2"),)),
        limits=limits,
    )
    after = _one(store, 0)

    assert after.id == before.id  # ← 关键
    assert after.answer != before.answer  # 内容确实变了
    assert after.status == STATUS_COMPLETE


# ── 不变式 2：pair_idx 连续（此处只测存储侧的约束）─────────────────────


def test_unique_constraint_rejects_duplicate_pair_idx(store: SqliteStore) -> None:
    """`UNIQUE(user_id, session_id, pair_idx)` 必须真的拦住重复位置。"""
    _seed(store, [(0, "Q", None)])
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        # 故意给一个不同的 id，绕过 PRIMARY KEY，验证 UNIQUE 本身有效
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            pair_idx=0,
            question="Q'",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="seed",
            pair_id="a-different-id",
        )


# ── 不变式 4：UNIQUE 索引正好是邻域查询的键 ─────────────────────────────


def test_neighbor_range_query_uses_an_index(store: SqliteStore) -> None:
    """邻域查询**不得扫全表**——它必须命中 UNIQUE 建出的索引。

    `pair_idx` 有空洞则邻域**静默消失**，所以这条查询的正确性直接决定扩窗是否可靠。
    """
    _seed(store, [(i, f"Q{i}", None) for i in range(5)])

    plan = rd(store, store.explain_idx_range, "u1", "s1", 1, 3)
    assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan
    assert "SCAN qa_pairs" not in plan, plan


def test_neighbor_range_returns_contiguous_window(store: SqliteStore) -> None:
    """±1 扩窗：种子 `pair_idx = 2` ⇒ 返回 1、2、3，按 `pair_idx` 排序。"""
    _seed(store, [(i, f"Q{i}", None) for i in range(5)])

    got = rd(store, store.fetch_pairs_by_idx_range, "u1", "s1", 1, 3)
    assert [p.pair_idx for p in got] == [1, 2, 3]

    # 扩窗从 ±1 改成 ±2 只需改界，不动 schema
    got2 = rd(store, store.fetch_pairs_by_idx_range, "u1", "s1", 0, 4)
    assert [p.pair_idx for p in got2] == [0, 1, 2, 3, 4]


def test_neighbor_range_never_crosses_user_or_session(store: SqliteStore) -> None:
    """邻域是 SQL 查询，**最容易忘记带 `user_id` 条件**——漏了就是跨 user 泄漏。"""
    _seed(store, [(0, "u1-Q", None), (1, "u1-Q1", None)])
    with store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u2",
            session_id="s1",
            pair_idx=0,
            question="u2-Q",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="seed",
        )
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s2",
            pair_idx=0,
            question="u1-s2-Q",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="seed",
        )

    got = rd(store, store.fetch_pairs_by_idx_range, "u1", "s1", 0, 5)
    assert [p.question for p in got] == ["u1-Q", "u1-Q1"]


# ── §6.5 写入规则：填空 + 追加，绝不覆盖 ────────────────────────────────


def test_append_answer_fills_then_appends(store: SqliteStore) -> None:
    """`answer` 原值为空则填入，否则**追加**（append-only）。"""
    _seed(store, [(0, "Q", None)])
    pid = make_pair_id("u1", "s1", 0)

    with store.transaction() as conn:
        assert store.append_answer(conn, pid, "[assistant] a") is True
    assert _one(store, 0).answer == "[assistant] a"

    with store.transaction() as conn:
        assert store.append_answer(conn, pid, "[assistant] b") is True
    # 追加用单个换行连接 ⇒ 渲染成 A: 块时就是 §11.3 示例的形状
    assert _one(store, 0).answer == "[assistant] a\n[assistant] b"


def test_append_answer_ignores_empty_text(store: SqliteStore) -> None:
    _seed(store, [(0, "Q", None)])
    pid = make_pair_id("u1", "s1", 0)
    with store.transaction() as conn:
        assert store.append_answer(conn, pid, "") is False
    assert _one(store, 0).answer is None


def test_append_question_fills_then_appends(store: SqliteStore) -> None:
    """`question` 是 **填空 + 追加**（append-only）——与 `answer` 同一个写模式。

    ⚠ 不能只在 NULL 时写入："一个 `question` 恰好来自一条 user 消息"这个前提被 D20 修正了
    ——连续 user 消息（AML 拆超长 message 的产物）并进**同一个** `question`，
    而它们可能落在不同批次里 ⇒ 必须允许追加。
    """
    _seed(store, [(0, None, None)])  # 无问的对
    pid = make_pair_id("u1", "s1", 0)

    with store.transaction() as conn:
        assert store.append_question(conn, pid, "第一段") is True
    assert _one(store, 0).question == "第一段"

    with store.transaction() as conn:
        assert store.append_question(conn, pid, "第二段") is True
    # 与 append_answer 一样用单个换行连接（§11.3：AML 只做 "\n".join，不插分隔符）
    assert _one(store, 0).question == "第一段\n第二段"


def test_append_question_ignores_empty_text(store: SqliteStore) -> None:
    _seed(store, [(0, None, None)])
    pid = make_pair_id("u1", "s1", 0)
    with store.transaction() as conn:
        assert store.append_question(conn, pid, "") is False
    assert _one(store, 0).question is None


# ── 单向状态 ───────────────────────────────────────────────────────────


def test_mark_complete_is_one_way(store: SqliteStore) -> None:
    """`status` **只允许 pending → complete**；对已 complete 的行调用是空操作。

    反向是**做不到**的：`mark_complete` 带 `WHERE status = 'pending'`，
    而且**没有任何方法**能把状态写回 pending。
    """
    _seed(store, [(0, "Q", "[assistant] A")])
    pid = make_pair_id("u1", "s1", 0)

    with store.transaction() as conn:
        assert store.mark_complete(conn, pid) is False  # 已经是 complete，空操作
    assert _one(store, 0).status == STATUS_COMPLETE

    # 反向不可达：store 的公开方法里没有 set_status/set_pending
    public = {n for n in dir(store) if not n.startswith("_")}
    assert "mark_pending" not in public
    assert "set_status" not in public


def test_insert_rejects_illegal_status(store: SqliteStore) -> None:
    with pytest.raises(ValueError, match="非法 status"), store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            pair_idx=0,
            question="Q",
            answer=None,
            status="half-done",  # type: ignore[arg-type]
            event_time=None,
            request_id="seed",
        )


# ── 事务：整批原子 ─────────────────────────────────────────────────────


def test_batch_write_rolls_back_entirely_on_error(
    store: SqliteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**一个批次是一个事务**：中途异常 ⇒ 整批回滚，不留半个批次。

    contract.md §6：内部异常应让本批保持"可重试"（事务未提交），
    而不是返回一个"部分成功"。
    """
    limits = BatchLimits(max_messages=99, max_words=1000)
    original = store.insert_pair
    calls = {"n": 0}

    def flaky(conn, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # 第二个对落库时炸
            raise RuntimeError("模拟中途失败")
        return original(conn, **kwargs)

    monkeypatch.setattr(store, "insert_pair", flaky)

    with pytest.raises(RuntimeError, match="模拟中途失败"):
        apply_batch(
            store,
            AddBatch(
                "r1",
                "u1",
                "s1",
                # ⚠ 必须产出**两个**对，否则 `insert_pair` 只被调一次、第二轮永远不炸
                #    （连续 user 消息会并成一个 question，所以这里刻意用交替形状）
                (
                    _msg("user", "Q1"),
                    _msg("assistant", "A1"),
                    _msg("user", "Q2"),
                    _msg("assistant", "A2"),
                ),
            ),
            limits=limits,
        )

    # 一个对都没留下，守卫也没记 —— 整批仍然"可重试"
    assert rd(store, store.assert_isolation, "u1") == []
    assert rd(store, store.is_batch_applied, "r1") is False


# ── 读：按 id 批量取正文 / 隔离 ────────────────────────────────────────


def test_fetch_pairs_by_ids_preserves_caller_order(store: SqliteStore) -> None:
    """按 id 批量取正文时**保持调用方给来的顺序**（那是检索名次顺序）。"""
    _seed(store, [(i, f"Q{i}", None) for i in range(4)])
    ids = [make_pair_id("u1", "s1", i) for i in (3, 0, 2)]

    got = rd(store, store.fetch_pairs_by_ids, ids)
    assert [p.pair_idx for p in got] == [3, 0, 2]
    assert rd(store, store.fetch_pairs_by_ids, []) == []
    assert rd(store, store.fetch_pairs_by_ids, ["不存在"]) == []


def test_isolation_by_user_id(store: SqliteStore) -> None:
    """`user_id` 是**唯一**的检索隔离字段；`session_id` 只是分组字段。"""
    _seed(store, [(0, "u1-Q", None)])
    with store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u2",
            session_id="s1",
            pair_idx=0,
            question="u2-Q",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="seed",
        )

    assert [p.question for p in rd(store, store.assert_isolation, "u1")] == ["u1-Q"]
    assert [p.question for p in rd(store, store.assert_isolation, "u2")] == ["u2-Q"]


# ── 连接生命周期与跨线程（D17 的回归用例）──────────────────────────────


def test_each_operation_gets_its_own_connection(store: SqliteStore) -> None:
    """**每个操作各拿一个连接**——连接不长期持有，也不跨操作复用。

    这是短生命周期模型最直接的可观测性质：两次 `read()` 拿到的**不是同一个对象**。
    """
    with store.read() as a, store.read() as b:
        assert a is not b


def test_store_is_usable_from_another_thread(store: SqliteStore) -> None:
    """**连接不跨线程复用**——在另一个线程里读、写都必须正常。

    ⚠ 这是那个跨层缺口的**直接回归用例**：若连接在构造它的线程里建、之后长期复用，
    这里会抛 `sqlite3.ProgrammingError: SQLite objects created in a thread can only be
    used in that same thread.`——而 FastAPI 的 `def` 路由**就跑在线程池里**。
    """
    _seed(store, [(0, "Q0", "A0")])
    seen: list[int] = []

    def worker() -> None:
        seen.append(len(rd(store, store.assert_isolation, "u1")))  # 读（另一个线程）
        with store.transaction() as conn:  # 写（另一个线程）
            store.insert_pair(
                conn,
                user_id="u1",
                session_id="s2",
                pair_idx=0,
                question="Q-s2",
                answer="A-s2",
                status=STATUS_COMPLETE,
                event_time=None,
                request_id="r-thread",
            )
        seen.append(len(rd(store, store.assert_isolation, "u1")))

    assert run_parallel([worker]) == []
    assert seen == [1, 2]


def test_transaction_rolls_back_and_closes_on_exception(store: SqliteStore) -> None:
    """异常路径：**先回滚**（半批不落库），**再关连接**（不泄漏）。

    两件事都要断言——只断言回滚会漏掉连接泄漏，只断言关闭会漏掉脏数据。
    """
    captured: dict[str, sqlite3.Connection] = {}

    with (
        pytest.raises(RuntimeError, match="模拟中途失败"),
        store.transaction() as conn,
    ):
        captured["conn"] = conn
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            pair_idx=0,
            question="半批",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="r-boom",
        )
        raise RuntimeError("模拟中途失败")

    assert rd(store, store.assert_isolation, "u1") == []  # ① 回滚：半批一个字都没落
    assert rd(store, store.is_batch_applied, "r-boom") is False
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        captured["conn"].execute("SELECT 1")  # ② 已关：拿它再用会响亮失败


def test_concurrent_write_transactions_across_sessions(store: SqliteStore) -> None:
    """**【压力】多个 session 真正并发地跑 `BEGIN IMMEDIATE` + 写事务。**

    不只验证"它们能同时进入 Service"——这里让 N 个线程**同时**发 `BEGIN IMMEDIATE`，
    并在**持有写锁期间**停 10ms，把写窗口拉长到必然互相排队（谁也没法"恰好错开"）。

    这验证的是 SQLite 自己的 writer 串行化（WAL + `busy_timeout`）：
    输的那一方**等待**，而不是抛 `SQLITE_BUSY`。

    断言四件事：无异常（含 `database is locked` / `cannot start a transaction within
    a transaction`）· 无数据丢失 · **每个 session 内 `pair_idx` 连续** · session 之间无污染。
    """
    n_sessions = 6
    rounds = 3
    window_s = 0.01  # 持写锁的时间——够长，足以让其他线程在 BEGIN IMMEDIATE 上排队
    start = threading.Barrier(n_sessions, timeout=20)

    def writer(session: str) -> None:
        start.wait()  # 所有线程同时出发，最大化撞锁概率
        for i in range(rounds):
            with store.transaction() as conn:
                idx = store.next_pair_idx(conn, "u1", session)
                store.insert_pair(
                    conn,
                    user_id="u1",
                    session_id=session,
                    pair_idx=idx,
                    question=f"{session}-Q{i}",
                    answer=f"{session}-A{i}",
                    status=STATUS_COMPLETE,
                    event_time=None,
                    request_id=f"{session}-{i}",
                )
                time.sleep(window_s)

    errors = run_parallel([(lambda s=f"s{j}": writer(s)) for j in range(n_sessions)])
    assert errors == []  # ← 无 SQLITE_BUSY、无事务嵌套错误

    all_pairs = rd(store, store.iter_pairs, user_id="u1")
    assert len(all_pairs) == n_sessions * rounds  # 无数据丢失

    for j in range(n_sessions):
        session = f"s{j}"
        mine = [p for p in all_pairs if p.session_id == session]
        # 该 session 内连续、从 0 起、无重复——读-改-写没有被任何别的线程插进来
        assert [p.pair_idx for p in mine] == list(range(rounds))
        # 无跨 session 污染：每一行的内容都属于它自己的 session
        assert sorted(p.question for p in mine) == sorted(f"{session}-Q{i}" for i in range(rounds))


def test_same_session_concurrent_writers_never_take_the_same_pair_idx(store: SqliteStore) -> None:
    """**同一个 `(user_id, session_id)`** 的多个写事务并发 ⇒ `pair_idx` 不得撞车。

    上一条用例的线程**各写各的 session**（UNIQUE 键各不相同），所以它**证明不了**
    `BEGIN IMMEDIATE` 的必要性——把 IMMEDIATE 降成默认的 deferred `BEGIN`，它照样会过。
    这一条才是那个决定的**必要性用例**：

    * `next_pair_idx` 是**读-改-写**。deferred `BEGIN` 下两个事务会读到**同一个**
      `MAX(pair_idx)` ⇒ 插入同一个位置 ⇒ 撞 `UNIQUE(user_id, session_id, pair_idx)`；
    * 或者：读锁升级写锁时对方正持写锁 ⇒ **立刻** `SQLITE_BUSY`（SQLite **不对锁升级
      应用 `busy_timeout`**，它宁可立刻报错也不冒死锁的险）。

    两条路都会让 `errors` 非空。`BEGIN IMMEDIATE` 在事务开头就拿写锁，把读-改-写整体串行化。
    """
    n_writers = 6
    rounds = 3
    start = threading.Barrier(n_writers, timeout=20)

    def writer(tag: str) -> None:
        start.wait()
        for i in range(rounds):
            with store.transaction() as conn:
                idx = store.next_pair_idx(conn, "u1", "shared")
                store.insert_pair(
                    conn,
                    user_id="u1",
                    session_id="shared",
                    pair_idx=idx,
                    question=f"{tag}-Q{i}",
                    answer=None,
                    status=STATUS_COMPLETE,
                    event_time=None,
                    request_id=f"{tag}-{i}",
                )
                time.sleep(0.01)  # 拉长读-改-写的窗口

    assert run_parallel([(lambda t=f"w{j}": writer(t)) for j in range(n_writers)]) == []

    idxs = [p.pair_idx for p in rd(store, store.assert_isolation, "u1")]
    assert idxs == list(range(n_writers * rounds))  # 连续、无重复、无空洞
