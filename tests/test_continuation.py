"""§6.5 的批次续接三步（3a / 3b / 3c / 3d）——接真源一起测。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 一、配对与续接。
**这是唯一能同时验证"配对逻辑"与"落库结果"的一层。**
"""

from __future__ import annotations

from tests.conftest import rd

from tianxi_am.pairing.continuation import AddBatch, apply_batch
from tianxi_am.pairing.instrument import InMemoryPendingInstrument
from tianxi_am.pairing.pairing import BatchLimits, Message
from tianxi_am.store.sqlite_store import STATUS_COMPLETE, STATUS_PENDING, SqliteStore


def _msg(role: str, content: str, ts: int | None = None) -> Message:
    return Message(role=role, content=content, timestamp=ts)


def _apply(
    store: SqliteStore,
    request_id: str,
    messages: tuple[Message, ...],
    *,
    limits: BatchLimits,
    user_id: str = "u1",
    session_id: str = "s1",
    instrument=None,
):
    return apply_batch(
        store,
        AddBatch(request_id, user_id, session_id, messages),
        limits=limits,
        instrument=instrument,
    )


def _all(store: SqliteStore, user_id: str = "u1"):
    return sorted(rd(store, store.assert_isolation, user_id), key=lambda p: p.pair_idx)


# ── 跨批续接：3a 追加 + 3b 关闭 ────────────────────────────────────────


def test_cross_batch_continuation_appends_then_closes(store: SqliteStore) -> None:
    """完整走一遍：A 批留下 pending → B 批开头续写（3a）→ B 批首个 user 关闭它（3b）。"""
    limits = BatchLimits(max_messages=3, max_words=1000)

    # 批次 A：3 条 = 命中上限 ⇒ 最后一对是 pending
    res_a = _apply(
        store,
        "A",
        (
            _msg("user", "火车几点开？", 100),
            _msg("assistant", "我查一下。", 110),
            _msg("tool_result", '{"t":"09:42"}', 120),
        ),
        limits=limits,
    )
    assert res_a.new_pair_count == 1
    p0 = _all(store)[0]
    assert p0.status == STATUS_PENDING
    assert p0.answer == '[assistant] 我查一下。\n[tool_result] {"t":"09:42"}'
    assert p0.event_time == 100  # 该对【首条消息】的 timestamp

    # 批次 B：开头是续写（3a），随后首个 user 消息把它关掉（3b）
    _apply(
        store,
        "B",
        (_msg("assistant", "09:42。", 130), _msg("user", "下一班呢？", 200)),
        limits=limits,
    )

    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0, 1]

    # 3a：续写被**追加**进同一个对，且带 role 标记
    assert pairs[0].answer == (
        '[assistant] 我查一下。\n[tool_result] {"t":"09:42"}\n[assistant] 09:42。'
    )
    # 3b：被关闭 ⇒ 不再挂在 pending
    assert pairs[0].status == STATUS_COMPLETE
    # 这是它 11.3 渲染成 A: 块时的形状（标记已在 answer 里，渲染器无需重建边界）
    assert pairs[0].answer.count("\n") == 2

    # 3c：新对接着数，pair_idx = 1
    assert pairs[1].pair_idx == 1
    assert pairs[1].question == "下一班呢？"
    assert pairs[1].status == STATUS_COMPLETE  # B 批 2 条 < 3，未命中上限 ⇒ session 结束


def test_3b_leaves_no_pending_behind(store: SqliteStore) -> None:
    """3b 漏掉的后果是"永久挂在 pending"。**断言关掉之后查不到 pending。**

    漏 3b 时现象会**伪装成 AML 的切分行为**，让人跑去改配对规则——方向完全错了。
    """
    limits = BatchLimits(max_messages=2, max_words=1000)
    _apply(store, "A", (_msg("user", "Q1"), _msg("assistant", "A1")), limits=limits)
    assert rd(store, store.pending_pair, "u1", "s1") is not None

    # B 批直接以 user 开头 ⇒ 只有 3b 这一条路能关掉它
    _apply(store, "B", (_msg("user", "Q2"),), limits=limits)

    assert rd(store, store.pending_pair, "u1", "s1") is None
    assert _all(store)[0].status == STATUS_COMPLETE


# ── 纯接续批（零条 user 消息）：3d 的"也走这一步" ────────────────────────


def test_pure_continuation_batch_closes_pending(store: SqliteStore) -> None:
    """零条 user 消息的批次：被追加的那个 pending 就是最后一带，同样要标状态。

    ⚠ 不标的话它会**一直挂到 session 结束**（pairing/CLAUDE.md 的 ※）。
    """
    limits = BatchLimits(max_messages=2, max_words=1000)
    _apply(store, "A", (_msg("user", "Q1"), _msg("assistant", "A1")), limits=limits)
    assert _all(store)[0].status == STATUS_PENDING

    res = _apply(store, "B", (_msg("assistant", "A2"),), limits=limits)

    assert res.new_pair_count == 0  # 这一步不建新对
    p0 = _all(store)[0]
    assert p0.answer == "[assistant] A1\n[assistant] A2"
    assert p0.status == STATUS_COMPLETE


def test_pure_continuation_batch_that_hits_limit_stays_pending(
    store: SqliteStore,
) -> None:
    """纯接续批**命中上限**时不能标 complete——AML 还会继续喂。"""
    _apply(
        store,
        "A",
        (_msg("user", "Q1"),),
        limits=BatchLimits(max_messages=1, max_words=1000),
    )
    assert _all(store)[0].status == STATUS_PENDING

    _apply(
        store,
        "B",
        (_msg("assistant", "A1"),),
        limits=BatchLimits(max_messages=1, max_words=1000),
    )

    p0 = _all(store)[0]
    assert p0.answer == "[assistant] A1"
    assert p0.status == STATUS_PENDING  # 仍然开着


# ── 无问的对 ───────────────────────────────────────────────────────────


def test_batch_starting_with_non_user_makes_questionless_pair(
    store: SqliteStore,
) -> None:
    """批次以非 user 消息开头且**不存在 pending** ⇒ 建一个 `question` 为空的对。"""
    _apply(
        store,
        "A",
        (_msg("assistant", "先交代一下背景。"), _msg("user", "Q1")),
        limits=BatchLimits(max_messages=9, max_words=1000),
    )

    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0, 1]
    assert pairs[0].question is None
    assert pairs[0].answer == "[assistant] 先交代一下背景。"
    assert pairs[0].status == STATUS_COMPLETE  # 被 Q1 关闭
    assert pairs[1].question == "Q1"


# ── pair_idx 跨批连续，且从不重置 ──────────────────────────────────────


def test_pair_idx_continues_and_never_resets(store: SqliteStore) -> None:
    """迭代若干批后 `pair_idx` 仍连续、无空洞、且**不回到 0**。

    有空洞则 ±1 邻域**静默消失**；从 0 重开会撞 `id` 并**静默覆盖上一批的数据**。
    """
    limits = BatchLimits(max_messages=2, max_words=1000)
    _apply(store, "A", (_msg("user", "Q0"), _msg("assistant", "A0")), limits=limits)
    _apply(store, "B", (_msg("assistant", "A0b"), _msg("user", "Q1")), limits=limits)
    _apply(store, "C", (_msg("assistant", "A1"), _msg("assistant", "A1b")), limits=limits)

    idxs = [p.pair_idx for p in _all(store)]
    assert idxs == list(range(len(idxs)))  # 连续、从 0 起、无空洞
    assert rd(store, store.next_pair_idx, "u1", "s1") == len(idxs)


def test_mid_batch_pairs_are_complete(store: SqliteStore) -> None:
    """中间的对一律 complete——只有最后一对可能是 pending。"""
    _apply(
        store,
        "A",
        (_msg("user", "Q1"), _msg("user", "Q2"), _msg("assistant", "A2")),
        limits=BatchLimits(max_messages=3, max_words=1000),
    )

    pairs = _all(store)
    assert [p.status for p in pairs] == [STATUS_COMPLETE, STATUS_PENDING]


def test_sessions_are_independent(store: SqliteStore) -> None:
    """`pair_idx` 是 **session 内**的序号：不同 session 各自从 0 起。"""
    limits = BatchLimits(max_messages=9, max_words=1000)
    _apply(store, "A", (_msg("user", "Q"),), limits=limits, session_id="s1")
    _apply(store, "B", (_msg("user", "Q"),), limits=limits, session_id="s2")

    assert rd(store, store.next_pair_idx, "u1", "s1") == 1
    assert rd(store, store.next_pair_idx, "u1", "s2") == 1


# ── 三个计数器 ─────────────────────────────────────────────────────────


def test_counters_track_created_completed_orphaned(store: SqliteStore) -> None:
    """三个埋点：创建 / 被后续批次补全 / 活到 session 末。"""
    inst = InMemoryPendingInstrument()
    limits = BatchLimits(max_messages=2, max_words=1000)

    # A：命中上限 ⇒ 一个 pending 被创建
    _apply(
        store, "A", (_msg("user", "Q1"), _msg("assistant", "A1")), limits=limits, instrument=inst
    )
    assert inst.counters.pending_created == 1
    assert inst.counters.pending_completed == 0
    assert inst.counters.pending_orphaned == 0

    # B：以 user 开头 ⇒ 3b 把它关掉 = completed
    _apply(store, "B", (_msg("user", "Q2"),), limits=limits, instrument=inst)
    assert inst.counters.pending_completed == 1
    assert inst.counters.pending_orphaned == 0

    # C：再制造一个 pending（命中上限）
    _apply(
        store, "C", (_msg("user", "Q3"), _msg("assistant", "A3")), limits=limits, instrument=inst
    )
    assert inst.counters.pending_created == 2

    # D：纯接续批 + 未命中上限 ⇒ session 结束判定时它仍 pending ⇒ orphaned
    _apply(store, "D", (_msg("assistant", "A3b"),), limits=limits, instrument=inst)
    assert inst.counters.pending_orphaned == 1
    assert _all(store)[2].status == STATUS_COMPLETE


def test_default_instrument_is_a_noop(store: SqliteStore) -> None:
    """不传 instrument 时不得炸（NullPendingInstrument）。"""
    _apply(
        store,
        "A",
        (_msg("user", "Q"),),
        limits=BatchLimits(max_messages=1, max_words=1000),
    )


# ── 空批次 ─────────────────────────────────────────────────────────────


def test_empty_batch_is_rejected_loudly(store: SqliteStore) -> None:
    """空批次抛错，**不静默当成 no-op**。

    §2.1 的 `messages` 就是批次的全部内容；空批次是调用方（harness/加载层）的 bug。
    静默 no-op 会让"整个 session 一条都没写进去"伪装成成功——本层最怕的失败模式。
    """
    import pytest

    with pytest.raises(ValueError, match="不得为空"):
        _apply(store, "A", (), limits=BatchLimits())
