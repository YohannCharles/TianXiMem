"""§6.5 / §15 的批次级幂等——**唯一能抓到"守卫被绕开"的一组用例**。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 二、幂等。**Test 10 就是这里的
`test_same_request_id_applied_once` + `test_retry_after_lost_response_*`。**

核心事实（D4）：**内容幂等与"同一批被写了两次"解决的是两个不同的问题。**
块是"写下即最终"的（D24 之后更没有回头改写的路径）——**但"写下的内容不会变"
管不住"同一批被写了两次"**。

> ⚠ **D28（2026-09-29）给守卫加了一半新职责：看 payload**。
> 位置是**请求的纯函数** ⇒ 重放必然算出**同一位置**，于是重放会撞 `UNIQUE`（**响亮**），
> 而不再静默落成重复记录。⇒ 守卫的职责是**"别让 AML 的正常重试变成 500"**。
> 而"同一个 `request_id`、**不同的** payload"是另一件事：它**不是**重放，
> 静默挑一份落库会让另一份记忆凭空消失（检索侧看不出来）⇒ 现在**响亮冲突**（409）。
"""

from __future__ import annotations

import sqlite3

import pytest
from tests.conftest import rd

from tianximem.pairing import AddBatch, Message, PayloadMismatchError, apply_batch
from tianximem.store.sqlite_store import STATUS_COMPLETE, SqliteStore


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


def _rid(ordinal: int) -> str:
    """**不透明**的 `request_id`（D28）——形状刻意做得像官方实发的那种。

    ⚠ 位置不再从它里面解析任何东西：它只是幂等键 + 位置的一半（整体参与哈希）。
    """
    return f"r_{ordinal:04d}"


def _batch(request_id: str, *, question: str = "Q1", answer: str = "A1") -> AddBatch:
    return AddBatch(
        request_id,
        "u1",
        "s1",
        (_msg("user", question), _msg("assistant", answer)),
    )


def _apply(store: SqliteStore, batch: AddBatch):
    return apply_batch(store, batch)


def _pairs(store: SqliteStore):
    return sorted(
        rd(store, store.assert_isolation, "u1"),
        key=lambda p: (p.request_id, p.local_index),
    )


def _snapshot(store: SqliteStore) -> list[tuple[str, str, int, str | None, str | None]]:
    return [(p.id, p.request_id, p.local_index, p.question, p.answer) for p in _pairs(store)]


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

    assert [(p.request_id, p.local_index) for p in _pairs(store)] == [("r_0000", 0), ("r_0001", 0)]
    assert len({p.id for p in _pairs(store)}) == 2


# ── D28 新增：同一个 request_id、**不同的** payload ─────────────────────


def test_same_request_id_with_a_different_payload_is_a_conflict(store: SqliteStore) -> None:
    """**409，不是 500、更不是静默重放**（D28）。

    静默挑一份落库的代价是：**另一份记忆凭空消失**，而检索侧完全看不出来
    ——它只会表现为"某些问题答不上来"。
    """
    _apply(store, _batch(_rid(0), question="原来的问题", answer="原来的回答"))

    with pytest.raises(PayloadMismatchError, match="不同的 payload"):
        _apply(store, _batch(_rid(0), question="换了个问题", answer="换了个回答"))


def test_conflicting_payload_leaves_the_first_one_intact(store: SqliteStore) -> None:
    """冲突**不改动真源**：第一次投进来的那份一个字节都没变（事务整体回滚）。"""
    _apply(store, _batch(_rid(0), question="原来的问题", answer="原来的回答"))
    before = _snapshot(store)

    with pytest.raises(PayloadMismatchError):
        _apply(store, _batch(_rid(0), question="换了个问题"))

    assert _snapshot(store) == before
    assert len(rd(store, store.assert_isolation, "u1")) == 1


def test_payload_fingerprint_ignores_only_whitespace_noise(store: SqliteStore) -> None:
    """指纹算在 **canonical 形状**上：只差首尾空白的两次投递**不算冲突**。

    ⚠ `Message.__post_init__` 已经 strip 过 ⇒ "重试时内容一模一样、只是边上多了空格"
    不会把一个正常的重试打成 409。
    """
    _apply(store, _batch(_rid(0), question="Q1", answer="A1"))
    replayed = _apply(store, _batch(_rid(0), question="  Q1  ", answer="A1\n"))

    assert replayed.applied is False  # ← 正常重放，不是冲突


def test_payload_fingerprint_changes_with_any_real_content_change(store: SqliteStore) -> None:
    """**只要内容真的变了**（哪怕一个字、或 `timestamp` 变了）⇒ 冲突。"""

    def batch_at(ts: int) -> AddBatch:
        return AddBatch("r_ts", "u1", "s1", (Message(role="user", content="Q1", timestamp=ts),))

    _apply(store, batch_at(1))

    with pytest.raises(PayloadMismatchError):
        _apply(store, batch_at(2))


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
    before = _snapshot(store)
    # 此刻事务已提交（apply_batch 内部已 COMMIT），但"响应还未发出"

    # 换一个**全新的 store 实例**（新对象、无内存状态）＝模拟进程重启后重连同一个库
    reopened = SqliteStore.open(store.db_path)
    replayed = apply_batch(reopened, _batch(_rid(0)))

    assert replayed.applied is False
    assert _snapshot(reopened) == before  # 行数、位置、内容都不变


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
    assert [p.request_id for p in _pairs(store)] == ["r_0000", "r_0001"]

    rows_before = len(_pairs(store))
    replayed = _apply(store, _batch(_rid(0)))
    assert replayed.applied is False
    assert len(_pairs(store)) == rows_before


# ── 反证：没有守卫时同一位置会被写两次（证明上面的用例不是空的）────────


def test_without_a_guard_the_same_position_would_collide(store: SqliteStore) -> None:
    """**反证用例**：守卫缺失时，同一批重放会撞 `UNIQUE`——**响亮**，不是静默重复。

    位置是请求的纯函数 ⇒ 同一批必然算出**同一位置** ⇒
    重放撞 `UNIQUE(user_id, session_id, request_id, local_index)`。

    ⚠ 但守卫仍然必须留：AML 重试是**正常行为**（最多 32 次），
    每次都靠撞 UNIQUE 来失败会让整批重试 32 次、最终判定失败。
    """
    _apply(store, _batch(_rid(0)))
    assert [(p.request_id, p.local_index) for p in _pairs(store)] == [("r_0000", 0)]

    # 无守卫路径：绕过 `applied_batches` 直接再写一遍同一批
    with pytest.raises(sqlite3.IntegrityError), store.transaction() as conn:
        store.insert_pair(
            conn,
            user_id="u1",
            session_id="s1",
            request_id=_rid(0),
            local_index=0,
            prev_memory_id=None,
            next_memory_id=None,
            question="Q1",
            answer="[assistant] A1",
            status=STATUS_COMPLETE,
            event_time=None,
        )

    assert len(_pairs(store)) == 1  # 一行都没多


# ── applied_batches 只增不改 ───────────────────────────────────────────


def test_applied_batches_is_append_only(store: SqliteStore) -> None:
    """旁表**只增不改**——它没有任何 UPDATE 路径，重复写同一主键会直接抛错。"""
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


def test_payload_hash_is_recorded_and_readable(store: SqliteStore) -> None:
    """旁表里真的存下了指纹，而且**重放时读得回来**（守卫要靠它判冲突）。"""
    from tianximem.pairing import payload_fingerprint

    batch = _batch(_rid(0))
    _apply(store, batch)

    stored = rd(store, store.applied_batch_payload_hash, _rid(0))
    assert stored == payload_fingerprint(batch)
    assert stored is not None and len(stored) == 64
