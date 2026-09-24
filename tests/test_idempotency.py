"""§6.5 / §15 的批次级幂等——**唯一能抓到"位置重分配"的一组用例**。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 二、幂等。

核心事实（D4）：**内容幂等与位置幂等解决的是两个不同的问题。**
"只填空不覆盖"管得住内容，**管不住位置**——`pair_idx` 的分配是读-改-写。
"""

from __future__ import annotations

from tests.conftest import rd

from tianxi_am.pairing.continuation import AddBatch, apply_batch
from tianxi_am.pairing.pairing import BatchLimits, Message
from tianxi_am.store.sqlite_store import SqliteStore, make_pair_id

# 上限 = 2 条消息：让第一批恰好留下一个 pending，方便构造续接后的重放
_LIMITS = BatchLimits(max_messages=2, max_words=1000)


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _batch(request_id: str) -> AddBatch:
    return AddBatch(
        request_id,
        "u1",
        "s1",
        (_msg("user", "Q1"), _msg("assistant", "A1")),
    )


def _apply(store: SqliteStore, batch: AddBatch):
    return apply_batch(store, batch, limits=_LIMITS)


def _pairs(store: SqliteStore):
    return sorted(rd(store, store.assert_isolation, "u1"), key=lambda p: p.pair_idx)


# ── 基本幂等：同一 request_id 至多应用一次 ──────────────────────────────


def test_same_request_id_applied_once(store: SqliteStore) -> None:
    first = _apply(store, _batch("A"))
    assert first.applied is True
    assert first.new_pair_count == 1

    second = _apply(store, _batch("A"))
    assert second.applied is False
    assert second.new_pair_count == 0
    assert second.plan is None

    assert len(_pairs(store)) == 1  # 没有新增行


def test_different_request_id_with_same_payload_is_a_different_batch(
    store: SqliteStore,
) -> None:
    """判重看的是 `request_id`，不是 payload——不同批同内容必须各自落库。"""
    _apply(store, _batch("A"))
    _apply(store, _batch("B"))

    assert [p.pair_idx for p in _pairs(store)] == [0, 1]
    assert rd(store, store.open_pair, "u1", "s1") is not None


# ── 核心用例：事务已提交、响应未发出 ⇒ AML 重试 ─────────────────────────


def test_retry_after_lost_response_does_not_reallocate_positions(
    store: SqliteStore,
) -> None:
    """模拟"事务已提交、响应【未发出】"的中间态，然后重放同一批。

    断言两件事：
      1. 库里**没有新增行**
      2. 既有行的 `pair_idx` **不变**

    ⚠ **重放必须走一个全新的 store 实例**（模拟进程崩溃后重启）——
    否则一个内存里的去重（缓存、上一次的 `request_id` 变量）就能让测试通过，
    而线上真正崩溃时那个内存状态已经没了。守卫必须在**库里**。
    """
    applied = _apply(store, _batch("A"))
    assert applied.applied is True
    before = [(p.id, p.pair_idx, p.answer) for p in _pairs(store)]
    # 此刻事务已提交（apply_batch 内部已 COMMIT），但"响应还未发出"

    # 换一个**全新的 store 实例**（新对象、无内存状态）＝模拟进程重启后重连同一个库
    reopened = SqliteStore.open(store.db_path)
    replayed = apply_batch(reopened, _batch("A"), limits=_LIMITS)

    assert replayed.applied is False
    after = [(p.id, p.pair_idx, p.answer) for p in _pairs(reopened)]
    assert after == before  # 行数、位置、内容都不变


def test_retry_after_a_later_batch_still_dedupes(store: SqliteStore) -> None:
    """**守卫查的是 `applied_batches`，不是 `qa_pairs.request_id`。**

    这是 D4 的场景：批次 A 的行被**批次 B 覆盖了 `request_id`**，
    A 的指纹在 `qa_pairs` 里已经丢了。若守卫复用了那一列，就会查不到 ⇒
    重放 A 会**重复应用**（落到新的 `pair_idx` 上）。
    """
    _apply(store, _batch("A"))
    pending = rd(store, store.open_pair, "u1", "s1")
    assert pending is not None
    assert pending.request_id == "A"

    # B 批续接并关闭 A 留下的那个 pending ⇒ 这一行被 B "触碰"
    _apply(
        store,
        AddBatch("B", "u1", "s1", (_msg("assistant", "A1b"),)),
    )

    touched = rd(store, store.fetch_pairs_by_ids, [pending.id])[0]
    assert touched.status == "complete"
    assert touched.request_id == "B"  # ← A 的指纹在 qa_pairs 里已经没了

    # 但 applied_batches 里还在 ⇒ 重放 A 必须被拦住
    assert rd(store, store.is_batch_applied, "A") is True

    rows_before = len(_pairs(store))
    replayed = _apply(store, _batch("A"))
    assert replayed.applied is False
    assert len(_pairs(store)) == rows_before


# ── 反证：没有守卫时位置会被重分配（证明上面的用例不是空的）────────────


def test_without_guard_positions_are_reallocated(store: SqliteStore) -> None:
    """**反证用例。**

    不走守卫、直接把同一批消息再应用一次（模拟"守卫缺失"的实现），
    位置会被重新分配到**新的** `pair_idx` 上 —— 落成重复记录，**且不会报错**。

    ⚠ 注意这里必须先让 `MAX(pair_idx)` 前移，这与"简单重复 POST 两次"不同：
    重复 POST 时状态相同，抓不到这个 bug（tests/CLAUDE.md）。
    """
    _apply(store, _batch("A"))
    assert [p.pair_idx for p in _pairs(store)] == [0]

    # 无守卫路径：恢复位置 → 直接把整批再写一遍（这里用一行代表那一批）
    with store.transaction() as conn:
        next_idx = store.next_pair_idx(conn, "u1", "s1")
        assert next_idx == 1  # ← MAX(pair_idx) 已经前移
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            pair_idx=next_idx,
            question="Q1",
            answer="[assistant] A1",
            status="pending",
            event_time=None,
            request_id="A",
        )

    # 同一批内容现在占了两行、两个不同的 id（位置派生 ⇒ 不同位置 = 不同 id）
    pairs = _pairs(store)
    assert [p.pair_idx for p in pairs] == [0, 1]
    assert pairs[0].id == make_pair_id("u1", "s1", 0)
    assert pairs[1].id == make_pair_id("u1", "s1", 1)
    assert pairs[0].id != pairs[1].id


# ── applied_batches 只增不改 ───────────────────────────────────────────


def test_applied_batches_is_append_only(store: SqliteStore) -> None:
    """旁表**只增不改**——它没有任何 UPDATE 路径，重复写同一主键会直接抛错。"""
    import sqlite3

    import pytest

    _apply(store, _batch("A"))

    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        store.record_batch(conn, "A", "u1", "s1")

    with store.read() as conn:
        rows = conn.execute(
            "SELECT request_id, user_id, session_id FROM applied_batches"
        ).fetchall()
    assert [r["request_id"] for r in rows] == ["A"]

    # store 的公开方法里没有能改写旁表的手段
    public = {n for n in dir(store) if not n.startswith("_")}
    assert not {n for n in public if n.startswith("update_") or n.startswith("delete_")}


def test_guard_is_per_request_id_not_per_session(store: SqliteStore) -> None:
    """守卫的键是 `request_id`——同一 session 的不同批次互不影响。"""
    _apply(store, _batch("A"))
    _apply(store, _batch("B"))

    assert rd(store, store.is_batch_applied, "A") is True
    assert rd(store, store.is_batch_applied, "B") is True
    assert rd(store, store.is_batch_applied, "C") is False
