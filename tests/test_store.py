"""§6.1 / §6.3 的存储层：DDL、位置派生 id、Add 内邻接、隔离、事务原子性。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 六、存储。

⚠ **D28 起位置 = `(request_id, local_index)`**：本文件里"落几行"的写法统一走
`conftest.seed_in_add`（**一次 Add 内**几个块，自动连好 `prev` / `next`）与
`seed_pair_in`（每个 `index` 是**一次独立的 Add**，彼此不相邻）。

文件末尾另有一组**连接生命周期与跨线程**的用例——它们是 D17 那条跨层缺口的回归用例
（长期持有连接 × FastAPI 的 `def` 路由跑在线程池里），
见 `store/sqlite_store.py` 的连接模型一节。
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from tests.conftest import rd, run_parallel, seed_in_add, seed_pair_in

from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.store.sqlite_store import (
    STATUS_COMPLETE,
    SqliteStore,
    make_pair_id,
)


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _one(store: SqliteStore, index: int):
    """按位置取一行（`u1` / `s1`，`seed_pair_in` 造的那种"一次 Add 一块"）。"""
    rows = rd(store, store.fetch_pairs_by_ids, [make_pair_id("u1", "s1", f"seed:{index}", 0)])
    assert rows, f"seed:{index} 不存在"
    return rows[0]


def _seed(store: SqliteStore, contents: list[tuple[int, str, str | None]]) -> None:
    """落几行，**每个 `index` 一次独立的 Add**（便于只测存储层）。"""
    for index, question, answer in contents:
        seed_pair_in(store, index, question, answer)


# ── 不变式 1：id 位置派生 ───────────────────────────────────────────────


def test_pair_id_is_position_derived_not_content_hash() -> None:
    """`id` 只由位置 `(user_id, session_id, request_id, local_index)` 决定——**与内容无关**。

    这是不变式 1：用内容哈希会在**重试/重建**时留下**孤儿 point**。
    ⚠ `request_id` **作为一个整体**参与哈希（D28），**不解析**它的内容。
    """
    a = make_pair_id("u1", "s1", "req-1", 0)
    assert a == make_pair_id("u1", "s1", "req-1", 0)  # 确定性
    # 位置不同 ⇒ id 不同
    assert a != make_pair_id("u1", "s1", "req-1", 1)
    assert a != make_pair_id("u1", "s1", "req-2", 0)
    assert a != make_pair_id("u1", "s2", "req-1", 0)
    assert a != make_pair_id("u2", "s1", "req-1", 0)
    # 形状上就是"位置派生"：函数签名里根本没有内容参数
    assert len(a) == 64
    # ⚠ **任意形状的 request_id 都等价可用**（D28 的核心）：这一段不解析它
    for opaque in ("abc", "req-001", "foo:bar", "xxx:chunk-0", "xxx:chunk-0-extra"):
        assert len(make_pair_id("u1", "s1", opaque, 0)) == 64


def test_pair_id_does_not_collide_across_field_boundaries() -> None:
    """分隔符防止字段粘连：("a","b\\x1fc") 与 ("a\\x1fb","c") 不得撞。"""
    assert make_pair_id("a", "b\x1fc", "r", 0) != make_pair_id("a\x1fb", "c", "r", 0)


def test_id_is_position_derived_not_content_derived(store: SqliteStore) -> None:
    """**`id` 只由位置决定，与内容无关。**

    用内容哈希时，两段**内容相同**的记忆会拿到同一个 `id`（这里的两行就是），
    而且**任何一次内容改写**都会让 `id` 变 ⇒ Qdrant 里旧 point 成孤儿。

    ⚠ 重建索引（`tools/reindex.py`）靠的正是"同一行永远是同一个 point"。
    """
    _seed(store, [(0, "同一段问题", "[assistant] 同一段回答")])
    _seed(store, [(1, "同一段问题", "[assistant] 同一段回答")])
    first, second = _one(store, 0), _one(store, 1)

    assert first.question == second.question  # 内容逐字相同…
    assert first.id != second.id  # …而 id 不同（内容哈希在这里就会撞）
    assert first.id == make_pair_id("u1", "s1", "seed:0", 0)
    assert second.id == make_pair_id("u1", "s1", "seed:1", 0)


def test_position_is_derived_from_the_request_alone() -> None:
    """**位置是请求的纯函数**（D28）：同一个 `(user, session, request_id, local_index)`
    无论谁来算、算几次，永远是同一个 `id`；而**内容变了 `id` 也不变**。

    ⇒ 同一批重试 ⇒ 必然算出同一个位置 ⇒ 撞的是 `UNIQUE`（响亮）而不是静默重复。
    """
    assert make_pair_id("u1", "s1", "r", 3) == make_pair_id("u1", "s1", "r", 3)
    # 形近的 id **不共享**位置（不是"取尾部数字"那一套）
    assert make_pair_id("u1", "s1", "r3", 0) != make_pair_id("u1", "s1", "r", 3)


# ── 不变式 2（D28）：位置 = (request_id, local_index)，唯一 ──────────────


def test_unique_constraint_rejects_a_duplicate_position(store: SqliteStore) -> None:
    """`UNIQUE(user_id, session_id, request_id, local_index)` 必须真的拦住重复位置。

    ⚠ 并发**不靠应用层锁**来避免撞车——位置由请求派生，各批写各自的
    `(request_id, local)`。所以这条约束守的是"**同一批被写两次**"（守卫该拦住的），
    以及调用方算错位置的情形。
    """
    _seed(store, [(0, "Q", None)])
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        # 故意给一个不同的 id，绕过 PRIMARY KEY，验证 UNIQUE 本身有效
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            request_id="seed:0",
            local_index=0,
            prev_memory_id=None,
            next_memory_id=None,
            question="Q'",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
            pair_id="a-different-id",
        )


# ── 不变式 4（D28）：UNIQUE 索引正好是"取一次 Add"的键 ─────────────────


def test_by_request_query_uses_an_index(store: SqliteStore) -> None:
    """"取一次 Add 的全部块"**不得扫全表**，也不该额外排序——它必须命中 UNIQUE 建出的索引。

    这才是 §10 扩窗的数据来源（D28 起按 `request_id` 取整批）。
    """
    seed_in_add(store, [(f"Q{i}", None) for i in range(5)])

    plan = rd(store, store.explain_by_request, "u1", "s1", "seed:one-add")
    assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan
    assert "SCAN qa_pairs" not in plan, plan
    assert "TEMP B-TREE" not in plan, plan  # ← 不该额外排序


def test_by_request_returns_that_add_in_local_index_order(store: SqliteStore) -> None:
    """取回的**恰好是那一次 Add**，按 `local_index` 升序，并带着显式邻接。"""
    ids = seed_in_add(store, [(f"Q{i}", f"A{i}") for i in range(5)])

    got = rd(store, store.fetch_by_request, "u1", "s1", "seed:one-add")
    assert [p.local_index for p in got] == [0, 1, 2, 3, 4]
    assert [p.id for p in got] == ids
    # 显式邻接：链首的 prev 与链尾的 next 都是 None
    assert got[0].prev_memory_id is None
    assert got[0].next_memory_id == ids[1]
    assert got[-1].next_memory_id is None
    assert got[-1].prev_memory_id == ids[3]
    # ⚠ 每个块都完整 ⇒ 每个都在链上
    assert all(p.is_complete for p in got)


def test_incomplete_blocks_stay_out_of_the_chain(store: SqliteStore) -> None:
    """**A-only / Q-only 不参与邻接**（D28）：它们照样落库，但两侧指针都是 `None`。

    形状取 `A Q A Q`：开头的 A-only 配不上 user、后面的 QA 正常成链。
    """
    ids = seed_in_add(store, [(None, "A0"), ("Q1", "A1"), ("Q2", "A2"), ("Q3", None)])

    got = rd(store, store.fetch_by_request, "u1", "s1", "seed:one-add")
    assert [p.local_index for p in got] == [0, 1, 2, 3]  # 一条都没丢
    # 两端的不完整块：不进链
    assert (got[0].prev_memory_id, got[0].next_memory_id) == (None, None)
    assert (got[3].prev_memory_id, got[3].next_memory_id) == (None, None)
    # 中间两个完整 QA 互连
    assert got[1].next_memory_id == ids[2]
    assert got[2].prev_memory_id == ids[1]


def test_by_request_never_crosses_user_session_or_request(store: SqliteStore) -> None:
    """按 Add 取数是 SQL，**最容易忘记带条件**——漏了就是跨 user 泄漏、或跨 Add 串味。"""
    seed_in_add(store, [("u1-Q", None)])
    seed_in_add(store, [("u2-Q", None)], user_id="u2")
    seed_in_add(store, [("u1-s2-Q", None)], session_id="s2")
    seed_in_add(store, [("u1-s1-other-Q", None)], request_id="seed:another-add")

    got = rd(store, store.fetch_by_request, "u1", "s1", "seed:one-add")
    assert [p.question for p in got] == ["u1-Q"]


def test_missing_middle_row_breaks_the_chain_but_keeps_the_rows(store: SqliteStore) -> None:
    """**链上缺一行**（中间那块从没写下）⇒ 指针仍然指着它，而它不在库里。

    ⚠ 这正是"段合并必须读指针、不能数下标"的理由：第 0 块的 `next` 指着第 1 块，
    而第 1 块不存在 ⇒ 拿 `local_index` 硬算会把 0 和 2 拼成一段。
    """
    ids = seed_in_add(store, [("Q0", "A0"), ("Q1", "A1"), ("Q2", "A2")])
    with store.transaction() as conn:
        conn.execute("DELETE FROM qa_pairs WHERE id = ?", (ids[1],))

    got = rd(store, store.fetch_by_request, "u1", "s1", "seed:one-add")
    assert [p.local_index for p in got] == [0, 2]
    assert got[0].next_memory_id == ids[1]  # ← 指着那个已经不存在的 id
    assert got[1].prev_memory_id == ids[1]


def test_insert_rejects_illegal_status(store: SqliteStore) -> None:
    with pytest.raises(ValueError, match="非法 status"), store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            request_id="r",
            local_index=0,
            prev_memory_id=None,
            next_memory_id=None,
            question="Q",
            answer=None,
            status="half-done",  # type: ignore[arg-type]
            event_time=None,
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
                # ⚠ `request_id` 是 opaque string（D28）——**不解析它**，
                #    所以这里随便写什么都行（这一条用的是个正常形状）。
                "r-rollback",
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
    assert rd(store, store.is_batch_applied, "r-rollback") is False


# ── 读：按 id 批量取正文 / 隔离 ────────────────────────────────────────


def test_fetch_pairs_by_ids_preserves_caller_order(store: SqliteStore) -> None:
    """按 id 批量取正文时**保持调用方给来的顺序**（那是检索名次顺序）。"""
    _seed(store, [(i, f"Q{i}", None) for i in range(4)])
    ids = [make_pair_id("u1", "s1", f"seed:{i}", 0) for i in (3, 0, 2)]

    got = rd(store, store.fetch_pairs_by_ids, ids)
    assert [p.question for p in got] == ["Q3", "Q0", "Q2"]
    assert rd(store, store.fetch_pairs_by_ids, []) == []
    assert rd(store, store.fetch_pairs_by_ids, ["不存在"]) == []


def test_isolation_by_user_id(store: SqliteStore) -> None:
    """`user_id` 是**唯一**的检索隔离字段；`session_id` 只是分组字段。"""
    _seed(store, [(0, "u1-Q", None)])
    seed_pair_in(store, 0, "u2-Q", None, user_id="u2")

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
        seed_pair_in(store, 0, "Q-s2", "A-s2", session_id="s2")  # 写（另一个线程）
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
            request_id="r-boom",
            local_index=0,
            prev_memory_id=None,
            next_memory_id=None,
            question="半批",
            answer=None,
            status=STATUS_COMPLETE,
            event_time=None,
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

    ⚠ **位置由调用方给定**（这里每个 writer 用自己的 `request_id`），所以这条用例测的是
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
                    request_id=f"{session}-r{i}",
                    local_index=0,
                    prev_memory_id=None,
                    next_memory_id=None,
                    question=f"{session}-Q{i}",
                    answer=f"{session}-A{i}",
                    status=STATUS_COMPLETE,
                    event_time=None,
                )
                time.sleep(window_s)

    errors = run_parallel([(lambda s=f"s{j}": writer(s)) for j in range(n_sessions)])
    assert errors == []  # ← 无 SQLITE_BUSY、无事务嵌套错误

    all_pairs = rd(store, store.iter_pairs, user_id="u1")
    assert len(all_pairs) == n_sessions * rounds  # 无数据丢失

    for j in range(n_sessions):
        session = f"s{j}"
        mine = [p for p in all_pairs if p.session_id == session]
        # 该 session 内的批号恰好是 r0..r{rounds-1}，且每一批都对上它自己的内容
        assert sorted(p.request_id for p in mine) == sorted(
            f"{session}-r{i}" for i in range(rounds)
        )
        assert {p.request_id: p.question for p in mine} == {
            f"{session}-r{i}": f"{session}-Q{i}" for i in range(rounds)
        }
        # 无跨 session 污染：每一行的内容都属于它自己的 session
        assert sorted(p.question for p in mine) == sorted(f"{session}-Q{i}" for i in range(rounds))


def test_same_session_concurrent_writers_do_not_collide(store: SqliteStore) -> None:
    """**同一个 `(user_id, session_id)`** 的多个写事务并发 ⇒ 各行落位**互不相交**。

    与上一条的分工：那一条的线程**各写各的 session**（键各不相同），
    它证明不了"同 session 并发是安全的"。这一条才是。

    **位置由调用方给定**（`(request_id, local_index)`），所以：

    * 每一批写**自己的** `request_id` ⇒ 不撞 `UNIQUE`；（若位置来自"读当前最大值再 +1"，
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

    def writer(tag: str) -> None:
        start.wait()
        for i in range(rounds):
            with store.transaction() as conn:
                # 每个 writer 占**自己的** request_id —— 位置互不相交
                store.insert_pair(
                    conn,
                    user_id="u1",
                    session_id="shared",
                    request_id=f"{tag}-r{i}",
                    local_index=0,
                    prev_memory_id=None,
                    next_memory_id=None,
                    question=f"{tag}-Q{i}",
                    answer=None,
                    status=STATUS_COMPLETE,
                    event_time=None,
                )
                time.sleep(0.01)  # 拉长写窗口，逼出排队/升级争用

    errors = run_parallel([(lambda t=f"w{j}": writer(t)) for j in range(n_writers)])
    assert errors == []  # ← 无 SQLITE_BUSY、无 UNIQUE 冲突

    pairs = rd(store, store.assert_isolation, "u1")
    # 一共 n_writers*rounds 行，且每个 writer 的每一轮都在（无重复、无丢失）
    assert sorted(p.request_id for p in pairs) == sorted(
        f"w{j}-r{i}" for j in range(n_writers) for i in range(rounds)
    )


# ── 旧库迁移（D28）：一次性的、在事务里、保正文 ─────────────────────────


_OLD_SCHEMA = """
CREATE TABLE qa_pairs (
    id            TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    chunk_ordinal INTEGER NOT NULL,
    local_index   INTEGER NOT NULL,
    question      TEXT,
    answer        TEXT,
    status        TEXT NOT NULL,
    event_time    INTEGER,
    request_id    TEXT NOT NULL,
    UNIQUE(user_id, session_id, chunk_ordinal, local_index)
);
CREATE TABLE applied_batches (
    request_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    applied_at  INTEGER NOT NULL
);
"""


def _old_db(tmp_path) -> Path:  # noqa: F821 — 只在本文件用一次
    """造一个**按 D25 规则写过**的库（旧 schema + 旧 id 公式）。"""
    import hashlib
    from pathlib import Path

    path = Path(tmp_path) / "legacy.db"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_OLD_SCHEMA)
        # 两批：第一批 3 块（含一块 A-only）、第二批 2 块；旧 id = hash(u, s, chunk, local)
        rows = [
            # (chunk, local, question, answer, status, event_time, request_id)
            (0, 0, None, "[assistant] A0", "complete", None, "old-0"),  # A-only
            (0, 1, "Q1", "[assistant] A1", "complete", 111, "old-0"),
            (0, 2, "Q2", "[assistant] A2", "complete", 222, "old-0"),
            (7, 0, "Q3", "[assistant] A3", "complete", 333, "old-7"),  # 跳号
            (7, 1, "Q4", "[assistant] A4", "pending", 444, "old-7"),  # 旧库里可能有的 pending
        ]
        for chunk, local, question, answer, status, event_time, request_id in rows:
            old_id = hashlib.sha256(
                f"{len('u1')}:u1{len('s1')}:s1{len(str(chunk))}:{chunk}{len(str(local))}:{local}"
                .encode()
            ).hexdigest()
            conn.execute(
                "INSERT INTO qa_pairs (id, user_id, session_id, chunk_ordinal, local_index,"
                " question, answer, status, event_time, request_id)"
                " VALUES (?, 'u1', 's1', ?, ?, ?, ?, ?, ?, ?)",
                (old_id, chunk, local, question, answer, status, event_time, request_id),
            )
        conn.execute(
            "INSERT INTO applied_batches (request_id, user_id, session_id, applied_at)"
            " VALUES ('old-0', 'u1', 's1', 1)"
        )
        conn.commit()
    finally:
        conn.close()
    return path


def test_legacy_db_is_migrated_in_place(tmp_path) -> None:
    """**旧库（D25）搬到 D28**：正文一字不丢、`id` 重算、链重连、旧批次的指纹留空。

    ⚠ 这是**唯一**会碰旧数据的代码路径（`open()` 里自动跑一次），所以它必须有测试：
    没测过的迁移比没有迁移更危险——它会在**真实数据**上第一次运行。
    """
    path = _old_db(tmp_path)
    store = SqliteStore.open(path)

    first = rd(store, store.fetch_by_request, "u1", "s1", "old-0")
    second = rd(store, store.fetch_by_request, "u1", "s1", "old-7")

    # ① 正文一字不丢（含旧库里可能存在的 'pending'）
    assert [(p.question, p.answer) for p in first] == [
        (None, "[assistant] A0"),
        ("Q1", "[assistant] A1"),
        ("Q2", "[assistant] A2"),
    ]
    assert [(p.event_time, p.status) for p in second] == [(333, "complete"), (444, "pending")]

    # ② `id` 按**新公式**重算（旧 id 不是新公式的产物）
    assert first[0].id == make_pair_id("u1", "s1", "old-0", 0)
    assert second[0].id == make_pair_id("u1", "s1", "old-7", 0)

    # ③ 链重连：A-only 不进链，后两个完整 QA 互连；两批**互不相邻**
    assert (first[0].prev_memory_id, first[0].next_memory_id) == (None, None)
    assert first[1].next_memory_id == first[2].id
    assert first[2].prev_memory_id == first[1].id
    assert first[2].next_memory_id is None
    assert second[0].prev_memory_id is None

    # ④ 旧批次的指纹**留空**（不知道 payload）⇒ 重放只能放行，不能判冲突
    assert rd(store, store.applied_batch_payload_hash, "old-0") is None
    assert rd(store, store.is_batch_applied, "old-0") is True


def test_migration_is_idempotent(tmp_path) -> None:
    """**再开一次不会又搬一遍**（判据是列的形状，不是"跑过没有"）。"""
    path = _old_db(tmp_path)
    first_open = SqliteStore.open(path)
    before = [(p.id, p.request_id, p.local_index) for p in rd(first_open, first_open.iter_pairs)]

    second_open = SqliteStore.open(path)  # 同一个库，再开一次

    after = [(p.id, p.request_id, p.local_index) for p in rd(second_open, second_open.iter_pairs)]
    assert after == before
