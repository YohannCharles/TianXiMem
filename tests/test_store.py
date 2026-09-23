"""§6.1 / §6.3 的存储层：DDL、位置派生 id、单向状态、邻域查询、隔离、事务原子性。

对应 [`../tests/README.md`](../tests/README.md) 六、存储。
"""

from __future__ import annotations

import sqlite3

import pytest

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
    rows = store.fetch_pairs_by_ids(store.connection, [make_pair_id("u1", "s1", pair_idx)])
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

    plan = store.explain_idx_range(store.connection, "u1", "s1", 1, 3)
    assert "USING INDEX" in plan or "USING COVERING INDEX" in plan, plan
    assert "SCAN qa_pairs" not in plan, plan


def test_neighbor_range_returns_contiguous_window(store: SqliteStore) -> None:
    """±1 扩窗：种子 `pair_idx = 2` ⇒ 返回 1、2、3，按 `pair_idx` 排序。"""
    _seed(store, [(i, f"Q{i}", None) for i in range(5)])

    got = store.fetch_pairs_by_idx_range(store.connection, "u1", "s1", 1, 3)
    assert [p.pair_idx for p in got] == [1, 2, 3]

    # 扩窗从 ±1 改成 ±2 只需改界，不动 schema
    got2 = store.fetch_pairs_by_idx_range(store.connection, "u1", "s1", 0, 4)
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

    got = store.fetch_pairs_by_idx_range(store.connection, "u1", "s1", 0, 5)
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


def test_fill_question_if_null_never_overwrites(store: SqliteStore) -> None:
    """`question` **只在原值为 NULL 时**写入。"""
    _seed(store, [(0, "原始问题", None), (1, None, None)])

    with store.transaction() as conn:
        assert store.fill_question_if_null(conn, make_pair_id("u1", "s1", 0), "改掉它") is False
        assert store.fill_question_if_null(conn, make_pair_id("u1", "s1", 1), "补上的问题") is True

    assert _one(store, 0).question == "原始问题"
    assert _one(store, 1).question == "补上的问题"


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
            AddBatch("r1", "u1", "s1", (_msg("user", "Q1"), _msg("user", "Q2"))),
            limits=limits,
        )

    # 一个对都没留下，守卫也没记 —— 整批仍然"可重试"
    assert store.assert_isolation(store.connection, "u1") == []
    assert store.is_batch_applied(store.connection, "r1") is False


# ── 读：按 id 批量取正文 / 隔离 ────────────────────────────────────────


def test_fetch_pairs_by_ids_preserves_caller_order(store: SqliteStore) -> None:
    """按 id 批量取正文时**保持调用方给来的顺序**（那是检索名次顺序）。"""
    _seed(store, [(i, f"Q{i}", None) for i in range(4)])
    ids = [make_pair_id("u1", "s1", i) for i in (3, 0, 2)]

    got = store.fetch_pairs_by_ids(store.connection, ids)
    assert [p.pair_idx for p in got] == [3, 0, 2]
    assert store.fetch_pairs_by_ids(store.connection, []) == []
    assert store.fetch_pairs_by_ids(store.connection, ["不存在"]) == []


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

    assert [p.question for p in store.assert_isolation(store.connection, "u1")] == ["u1-Q"]
    assert [p.question for p in store.assert_isolation(store.connection, "u2")] == ["u2-Q"]
