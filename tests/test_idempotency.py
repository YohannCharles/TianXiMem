"""§6.5 / §15 的批次级幂等——**唯一能抓到"守卫被绕开"的一组用例**。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 二、幂等。**Test 10 就是这里的
`test_same_request_id_applied_once` + `test_retry_after_lost_response_*`。**

核心事实（D4）：**内容幂等与"同一批被写了两次"解决的是两个不同的问题。**
块是"写下即最终"的（D24 之后更没有回头改写的路径）——**但"写下的内容不会变"
管不住"同一批被写了两次"**。

> ⚠ **D25（2026-09-27）改变了这一层的失败形状，但守卫照样必须留**：
> 位置成为**请求的纯函数** ⇒ 重放必然算出**同一位置**，于是重放会撞 `UNIQUE`
> （**响亮**），而不再静默落成重复记录。
> ⇒ 守卫的职责从"防静默重复"变成**"别让 AML 的正常重试变成 500"**。
> ⚠ **D24 到 D25 之间这一层一个字都没改**：跨 Add 合并的有无与幂等正交。
"""

from __future__ import annotations

import sqlite3

import pytest
from tests.conftest import rd

from tianxi_am.pairing import AddBatch, Message, apply_batch
from tianxi_am.store.sqlite_store import STATUS_COMPLETE, SqliteStore


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _rid(ordinal: int) -> str:
    """**必须能取得出 chunk 序号**（D25）：位置由它派生，取不到就响亮失败。

    用本仓 harness 的形态（`<user>|<session>|<n>`）——默认正则同时认它和平台实发的
    `...:chunk-<n>`。
    """
    return f"u1|s1|{ordinal}"


def _batch(request_id: str) -> AddBatch:
    return AddBatch(
        request_id,
        "u1",
        "s1",
        (_msg("user", "Q1"), _msg("assistant", "A1")),
    )


def _apply(store: SqliteStore, batch: AddBatch):
    return apply_batch(store, batch)


def _pairs(store: SqliteStore):
    return sorted(rd(store, store.assert_isolation, "u1"), key=lambda p: p.seq)


# ── Test 10：同一 `request_id` 重试 ⇒ 不重复落库 ────────────────────────


def test_same_request_id_applied_once(store: SqliteStore) -> None:
    """Test 10：同一个 Add 用同一个 `request_id` 重试两次 ⇒

    * 不重复 `memory row`
    * 不重复 `vector`（`indexed` 由 `AddPipeline` 从真源重取，见 `test_service_add.py`）
    * 不重复 `embedding record`（块的内容不变 ⇒ 缓存键不变）
    """
    first = _apply(store, _batch(_rid(0)))
    assert first.applied is True
    assert first.new_pair_count == 1

    second = _apply(store, _batch(_rid(0)))
    assert second.applied is False
    assert second.new_pair_count == 0
    assert second.blocks == ()  # 守卫命中时**不产出任何块**
    assert second.new_pair_ids == ()

    assert len(_pairs(store)) == 1  # 没有新增行


def test_different_request_id_with_same_payload_is_a_different_batch(
    store: SqliteStore,
) -> None:
    """判重看的是 `request_id`，不是 payload——不同批同内容必须各自落库。"""
    _apply(store, _batch(_rid(0)))
    _apply(store, _batch(_rid(1)))

    assert [p.seq for p in _pairs(store)] == [0, 1]


# ── 核心用例：事务已提交、响应未发出 ⇒ AML 重试 ─────────────────────────


def test_retry_after_lost_response_does_not_reallocate_positions(
    store: SqliteStore,
) -> None:
    """模拟"事务已提交、响应【未发出】"的中间态，然后重放同一批。

    断言两件事：
      1. 库里**没有新增行**
      2. 既有行的**位置与内容不变**

    ⚠ **重放必须走一个全新的 store 实例**（模拟进程崩溃后重启）——
    否则一个内存里的去重（缓存、上一次的 `request_id` 变量）就能让测试通过，
    而线上真正崩溃时那个内存状态已经没了。守卫必须在**库里**。
    """
    applied = _apply(store, _batch(_rid(0)))
    assert applied.applied is True
    before = [(p.id, p.seq, p.answer) for p in _pairs(store)]
    # 此刻事务已提交（apply_batch 内部已 COMMIT），但"响应还未发出"

    # 换一个**全新的 store 实例**（新对象、无内存状态）＝模拟进程重启后重连同一个库
    reopened = SqliteStore.open(store.db_path)
    replayed = apply_batch(reopened, _batch(_rid(0)))

    assert replayed.applied is False
    after = [(p.id, p.seq, p.answer) for p in _pairs(reopened)]
    assert after == before  # 行数、位置、内容都不变


def test_retry_after_a_later_batch_still_dedupes(store: SqliteStore) -> None:
    """**守卫查的是 `applied_batches`，不是 `qa_pairs.request_id`。**

    D4 的场景：批次 A 的指纹可能在任何一行上都读不到了，但 `applied_batches` 里还在
    ⇒ 重放 A 必须被拦住。

    ⚠ D24/D25 之后每一行只被**一个** Add 写过（没有 `touch_request_id` 那种改写路径），
    所以 `qa_pairs.request_id` 今天**看起来**也够用。守卫**仍然只查旁表**，理由是它
    **不依赖"这些行此后不会被改写"这条前提**——旁表是唯一记录"这批被应用过"的地方，
    还带着 `user_id` / `session_id` / `applied_at` 三样运维信息。
    """
    _apply(store, _batch(_rid(0)))
    assert rd(store, store.is_batch_applied, _rid(0)) is True

    # 后来的批次照旧落库（同一 session、新的行）
    _apply(store, AddBatch(_rid(1), "u1", "s1", (_msg("user", "Q2"), _msg("assistant", "A2"))))
    assert [p.seq for p in _pairs(store)] == [0, 1]

    rows_before = len(_pairs(store))
    replayed = _apply(store, _batch(_rid(0)))
    assert replayed.applied is False
    assert len(_pairs(store)) == rows_before


# ── 反证：没有守卫时位置会被重分配（证明上面的用例不是空的）────────────


def test_without_a_guard_the_same_chunk_would_collide(store: SqliteStore) -> None:
    """**反证用例**：守卫缺失时，同一批重放会撞 `UNIQUE`——**响亮**，不是静默重复。

    位置是请求的纯函数 ⇒ 同一批必然算出**同一位置** ⇒
    重放撞 `UNIQUE(user_id, session_id, chunk_ordinal, local_index)`。

    ⚠ 但守卫仍然必须留：AML 重试是**正常行为**（最多 32 次），
    每次都靠撞 UNIQUE 来失败会让整批重试 32 次、最终判定失败。
    """
    _apply(store, _batch(_rid(0)))
    assert [(p.chunk_ordinal, p.local_index) for p in _pairs(store)] == [(0, 0)]

    # 无守卫路径：绕过 `applied_batches` 直接再写一遍同一批
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            chunk_ordinal=0,
            local_index=0,
            question="Q1",
            answer="[assistant] A1",
            status=STATUS_COMPLETE,
            event_time=None,
            request_id=_rid(0),
            pair_id=store.make_pair_id("u1", "s1", 0, 0) if False else None,
        )

    assert len(_pairs(store)) == 1  # 一行都没多


# ── applied_batches 只增不改 ───────────────────────────────────────────


def test_applied_batches_is_append_only(store: SqliteStore) -> None:
    """旁表**只增不改**——它没有任何 UPDATE 路径，重复写同一主键会直接抛错。"""
    import sqlite3

    import pytest

    _apply(store, _batch(_rid(0)))

    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        store.record_batch(conn, _rid(0), "u1", "s1")

    with store.read() as conn:
        rows = conn.execute(
            "SELECT request_id, user_id, session_id FROM applied_batches"
        ).fetchall()
    assert [r["request_id"] for r in rows] == [_rid(0)]

    # store 的公开方法里没有能改写旁表的手段
    public = {n for n in dir(store) if not n.startswith("_")}
    assert not {n for n in public if n.startswith("update_") or n.startswith("delete_")}


def test_guard_is_per_request_id_not_per_session(store: SqliteStore) -> None:
    """守卫的键是 `request_id`——同一 session 的不同批次互不影响。"""
    _apply(store, _batch(_rid(0)))
    _apply(store, _batch(_rid(1)))

    assert rd(store, store.is_batch_applied, _rid(0)) is True
    assert rd(store, store.is_batch_applied, _rid(1)) is True
    assert rd(store, store.is_batch_applied, _rid(2)) is False
