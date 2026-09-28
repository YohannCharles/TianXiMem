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

from tianxi_am.pairing import AddBatch, Message, apply_batch
from tianxi_am.store.sqlite_store import (
    STATUS_COMPLETE,
    SqliteStore,
    make_pair_id,
)


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _one(store: SqliteStore, ordinal: int):
    """按位置取一行（`u1` / `s1`）。"""
    rows = rd(store, store.fetch_pairs_by_ids, [make_pair_id("u1", "s1", ordinal, 0)])
    assert rows, f"chunk_ordinal={ordinal} 不存在"
    return rows[0]


def _seed(store: SqliteStore, contents: list[tuple[int, str, str | None]]) -> None:
    """直接落几行，便于只测存储层（不经过 continuation）。"""
    with store.transaction() as conn:
        for ordinal, question, answer in contents:
            store.insert_pair(
                conn,
                user_id="u1",
                session_id="s1",
                chunk_ordinal=ordinal,
                local_index=0,
                question=question,
                answer=answer,
                status=STATUS_COMPLETE,
                event_time=None,
                request_id="seed",
            )


# ── 不变式 1：id 位置派生 ───────────────────────────────────────────────


def test_pair_id_is_position_derived_not_content_hash() -> None:
    """`id` 只由位置 `(user_id, session_id, chunk_ordinal, local_index)` 决定——**与内容无关**。

    这是不变式 1：用内容哈希会在**补全**时变 `id`，留下**孤儿 point**。
    """
    a = make_pair_id("u1", "s1", 0, 0)
    assert a == make_pair_id("u1", "s1", 0, 0)  # 确定性
    # 位置不同 ⇒ id 不同
    assert a != make_pair_id("u1", "s1", 1, 0)
    assert a != make_pair_id("u1", "s2", 0, 0)
    assert a != make_pair_id("u2", "s1", 0, 0)
    # 形状上就是"位置派生"：函数签名里根本没有内容参数
    assert len(a) == 64


def test_pair_id_does_not_collide_across_field_boundaries() -> None:
    """分隔符防止字段粘连：("a","b\\x1fc") 与 ("a\\x1fb","c") 不得撞。"""
    assert make_pair_id("a", "b\x1fc", 0, 0) != make_pair_id("a\x1fb", "c", 0, 0)


def test_id_is_position_derived_not_content_derived(store: SqliteStore) -> None:
    """**`id` 只由位置 `(user_id, session_id, chunk_ordinal, local_index)` 决定，与内容无关。**

    用内容哈希时，两段**内容相同**的记忆会拿到同一个 `id`（这里的两行就是），
    而且**任何一次内容改写**都会让 `id` 变 ⇒ Qdrant 里旧 point 成孤儿
    （还在，但再也不会被正确更新）。

    ⚠ D24 之后没有"补全时改写内容"的路径了，但**这条不变式仍然不能松**：
    重建索引（`tools/reindex.py`）靠的正是"同一行永远是同一个 point"。
    """
    _seed(store, [(0, "同一段问题", "[assistant] 同一段回答")])
    _seed(store, [(1, "同一段问题", "[assistant] 同一段回答")])
    first, second = _one(store, 0), _one(store, 1)

    assert first.question == second.question  # 内容逐字相同…
    assert first.id != second.id  # …而 id 不同（内容哈希在这里就会撞）
    assert first.id == make_pair_id("u1", "s1", 0, 0)
    assert second.id == make_pair_id("u1", "s1", 1, 0)


# ── 不变式 2（D25 重塑）：位置 = (chunk_ordinal, local_index)，唯一 ──────


def test_unique_constraint_rejects_a_duplicate_position(store: SqliteStore) -> None:
    """`UNIQUE(user_id, session_id, chunk_ordinal, local_index)` 必须真的拦住重复位置。

    ⚠ D25 之后**并发**不再靠应用层锁来避免撞车——位置由请求派生，各批写各自的
    `(chunk, local)`。所以这条约束守的是"**同一批被写两次**"（守卫该拦住的），
    以及调用方算错位置的情形。"""
    _seed(store, [(0, "Q", None)])
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        # 故意给一个不同的 id，绕过 PRIMARY KEY，验证 UNIQUE 本身有效
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            chunk_ordinal=0,
            local_index=0,
            question="Q'",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="seed",
            pair_id="a-different-id",
        )


# ── 不变式 4（D25 重塑）：UNIQUE 索引正好是会话有序查询的键 ─────────────


def test_session_ordered_query_uses_an_index(store: SqliteStore) -> None:
    """会话有序查询**不得扫全表**，也不该额外排序——它必须命中 UNIQUE 建出的索引。

    这才是 §10 扩窗的数据来源（D25 起按会话取整段，不再按整数窗口取）。
    """
    _seed(store, [(i, f"Q{i}", None) for i in range(5)])

    plan = rd(store, store.explain_session_ordered, "u1", "s1")
    assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan
    assert "SCAN qa_pairs" not in plan, plan
    assert "TEMP B-TREE" not in plan, plan  # ← 不该额外排序


def test_session_ordered_returns_the_whole_session_in_order(store: SqliteStore) -> None:
    """整段取回：**按 `(chunk_ordinal, local_index)` 排**，且每行带上稠密序 `seq`。"""
    _seed(store, [(i, f"Q{i}", None) for i in range(5)])

    got = rd(store, store.fetch_session_ordered, "u1", "s1")
    assert [p.chunk_ordinal for p in got] == [0, 1, 2, 3, 4]
    assert [p.seq for p in got] == [0, 1, 2, 3, 4]


def test_session_ordered_seq_is_dense_even_when_chunk_ordinals_have_holes(
    store: SqliteStore,
) -> None:
    """**chunk 序号跳号不破坏 `seq` 的稠密性**——这是"缺号不断相邻"成立的地方。

    空洞只出现在 `chunk_ordinal` 上，而相邻性看的是现算的 `seq`（必然 0..n-1 连续）。
    """
    _seed(store, [(i, f"Q{i}", None) for i in (0, 1, 7, 8)])

    got = rd(store, store.fetch_session_ordered, "u1", "s1")
    assert [p.chunk_ordinal for p in got] == [0, 1, 7, 8]
    assert [p.seq for p in got] == [0, 1, 2, 3]  # ← 稠密


def test_session_ordered_never_crosses_user_or_session(store: SqliteStore) -> None:
    """会话有序查询是 SQL，**最容易忘记带 `user_id` 条件**——漏了就是跨 user 泄漏。"""
    _seed(store, [(0, "u1-Q", None), (1, "u1-Q1", None)])
    with store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u2",
            session_id="s1",
            chunk_ordinal=0,
            local_index=0,
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
            chunk_ordinal=0,
            local_index=0,
            question="u1-s2-Q",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            request_id="seed",
        )

    got = rd(store, store.fetch_session_ordered, "u1", "s1")
    assert [p.question for p in got] == ["u1-Q", "u1-Q1"]


def test_insert_rejects_illegal_status(store: SqliteStore) -> None:
    with pytest.raises(ValueError, match="非法 status"), store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            chunk_ordinal=0,
            local_index=0,
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
    original = store.insert_pair
    calls = {"n": 0}

    def flaky(conn, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # 第二个块落库时炸
            raise RuntimeError("模拟中途失败")
        return original(conn, **kwargs)

    monkeypatch.setattr(store, "insert_pair", flaky)

    with pytest.raises(RuntimeError, match="模拟中途失败"):
        apply_batch(
            store,
            AddBatch(
                # ⚠ request_id 必须**取得出 chunk 序号**（D25）——这一条用的是
                #   本仓 harness 的形态（`<user>|<session>|<n>`）。
                "u1|s1|0",
                "u1",
                "s1",
                # ⚠ 必须产出**两个**块，否则 `insert_pair` 只被调一次、第二轮永远不炸
                #    （连续 user 消息会并成一个 question，所以这里刻意用交替形状）
                (
                    _msg("user", "Q1"),
                    _msg("assistant", "A1"),
                    _msg("user", "Q2"),
                    _msg("assistant", "A2"),
                ),
            ),
        )

    # 一个块都没留下，守卫也没记 —— 整批仍然"可重试"
    assert rd(store, store.assert_isolation, "u1") == []
    assert rd(store, store.is_batch_applied, "u1|s1|0") is False


# ── 读：按 id 批量取正文 / 隔离 ────────────────────────────────────────


def test_fetch_pairs_by_ids_preserves_caller_order(store: SqliteStore) -> None:
    """按 id 批量取正文时**保持调用方给来的顺序**（那是检索名次顺序）。

    ⚠ 回来的行 `seq` 是 `-1`（按主键取没有会话上下文）——`rank/neighbor.py` 会补上它。
    """
    _seed(store, [(i, f"Q{i}", None) for i in range(4)])
    ids = [make_pair_id("u1", "s1", i, 0) for i in (3, 0, 2)]

    got = rd(store, store.fetch_pairs_by_ids, ids)
    assert [p.chunk_ordinal for p in got] == [3, 0, 2]
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
            chunk_ordinal=0,
            local_index=0,
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
                chunk_ordinal=0,
                local_index=0,
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
            chunk_ordinal=0,
            local_index=0,
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
    a transaction`）· 无数据丢失 · **每个 session 内位置与写的内容一一对上** · session 之间无污染。

    ⚠ **位置由调用方给定**（这里用 `i` 当 `chunk_ordinal`），所以这条用例测的是
    "并发写事务在 SQLite 处排队而不是失败"——**不测"读-改-写有没有被插进来"**
    （那需要一个"分配位置"的步骤，而它不存在）。
    """
    n_sessions = 6
    rounds = 3
    window_s = 0.01  # 持写锁的时间——够长，足以让其他线程在 BEGIN IMMEDIATE 上排队
    start = threading.Barrier(n_sessions, timeout=20)

    def writer(session: str) -> None:
        start.wait()  # 所有线程同时出发，最大化撞锁概率
        for i in range(rounds):
            with store.transaction() as conn:
                store.insert_pair(
                    conn,
                    user_id="u1",
                    session_id=session,
                    chunk_ordinal=i,
                    local_index=0,
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
        # 该 session 内的位置恰好是 0..rounds-1，且每一号都对上它自己的内容
        assert sorted(p.chunk_ordinal for p in mine) == list(range(rounds))
        assert {p.chunk_ordinal: p.question for p in mine} == {
            i: f"{session}-Q{i}" for i in range(rounds)
        }
        # 无跨 session 污染：每一行的内容都属于它自己的 session
        assert sorted(p.question for p in mine) == sorted(f"{session}-Q{i}" for i in range(rounds))


def test_same_session_concurrent_writers_do_not_collide(store: SqliteStore) -> None:
    """**同一个 `(user_id, session_id)`** 的多个写事务并发 ⇒ 各行落位**互不相交**。

    与上一条的分工：那一条的线程**各写各的 session**（UNIQUE 键各不相同），
    它证明不了"同 session 并发是安全的"。这一条才是。

    **位置由调用方给定**（`(chunk, local)`），所以：

    * 每一批写**自己的**位置 ⇒ 不撞 `UNIQUE`；（若位置来自"读当前最大值再 +1"，
      这里就必须靠应用层串行化——而那个步骤不存在。）
    * 也不再需要 `BEGIN IMMEDIATE` 来挡"两个事务读到同一个 MAX"——那是**位置**的问题，
      而位置的读-改-写已经没有了。

    ⚠ **但 `BEGIN IMMEDIATE` 仍然必须保留**，理由是另一条：写事务**先读后写**
    （守卫 `SELECT` → `INSERT`），deferred `BEGIN` 下两方都要把读锁**升级**成写锁，
    SQLite 会**立刻**返回 `SQLITE_BUSY`（不对锁升级应用 `busy_timeout`）。
    `test_batch_write_rolls_back_entirely_on_error` 与上面那条跨 session 用例一起钉着它。
    """
    n_writers = 6
    rounds = 3
    start = threading.Barrier(n_writers, timeout=20)

    def writer(tag: str, base: int) -> None:
        start.wait()
        for i in range(rounds):
            with store.transaction() as conn:
                # 每个 writer 占**自己那一段** chunk 序号 —— 位置互不相交
                store.insert_pair(
                    conn,
                    user_id="u1",
                    session_id="shared",
                    chunk_ordinal=base + i,
                    local_index=0,
                    question=f"{tag}-Q{i}",
                    answer=None,
                    status=STATUS_COMPLETE,
                    event_time=None,
                    request_id=f"{tag}-{i}",
                )
                time.sleep(0.01)  # 拉长写窗口，逼出排队/升级争用

    errors = run_parallel(
        [(lambda t=f"w{j}", b=j * rounds: writer(t, b)) for j in range(n_writers)]
    )
    assert errors == []  # ← 无 SQLITE_BUSY、无 UNIQUE 冲突

    pairs = rd(store, store.assert_isolation, "u1")
    # 一共 n_writers*rounds 行，chunk 序号恰好是 0..n-1（无重复、无丢失）
    assert sorted(p.chunk_ordinal for p in pairs) == list(range(n_writers * rounds))
