"""§6.5 的批次续接三步（3a′ / 3a / 3b / 3c / 3d）——接真源一起测。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 一、配对与续接。
**这是唯一能同时验证"配对逻辑"与"落库结果"的一层。**

⚠ **本文件里最重要的两条是"碎片跨批"**（`test_cross_batch_fragments_*`）：
它们钉住"连续 user 消息并入同一个 `question`"这条规则**在跨 Add 时也成立**，
而且**不依赖词数计数**——那是 S2 里我们复现不了的那个量。
"""

from __future__ import annotations

import random
from collections import Counter

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
    """3b 漏掉的后果是"永久挂在 pending"。**断言关掉之后库里不再有 pending 状态的行。**

    漏 3b 时现象会**伪装成 AML 的切分行为**，让人跑去改配对规则——方向完全错了。

    ⚠ 判据用 `status` 而**不是** `open_pair()`：B 批新建的那个 `(Q2, NULL)` 对
    `answer` 为空，按内容判据它"question 还在写"（那是**设计如此**，见
    `test_question_only_pair_absorbs_the_next_batchs_user_message`）——
    它并不意味着 3b 漏了。3b 的效果恰恰只体现在 `status` 上。
    """
    limits = BatchLimits(max_messages=2, max_words=1000)
    _apply(store, "A", (_msg("user", "Q1"), _msg("assistant", "A1")), limits=limits)
    assert [p.pair_idx for p in _all(store) if p.status == STATUS_PENDING] == [0]

    # B 批直接以 user 开头 ⇒ 只有 3b 这一条路能关掉它
    _apply(store, "B", (_msg("user", "Q2"),), limits=limits)

    assert _all(store)[0].status == STATUS_COMPLETE
    assert [p.pair_idx for p in _all(store) if p.status == STATUS_PENDING] == []


# ── 碎片跨批：连续 user 消息并成一个 question（D20）─────────────────────


def test_cross_batch_fragments_merge_into_one_question(store: SqliteStore) -> None:
    """**核心用例**：一条超长 user 消息被切成几段、每段各占一个 Add ⇒ 仍是一问一答。

    真实形状（AML 按句边界拆超长 message 时）：`[user frag1] / [user frag2] / [assistant A]`
    分三次 Add 到达。期望落库是**一个**对 `(frag1\\nfrag2, A)`，
    而不是三个对（其中两个还有问无答）。
    """
    limits = BatchLimits(max_messages=3, max_words=2)  # 每段碎片都恰好命中词数上限

    _apply(store, "A", (_msg("user", "fragment one"),), limits=limits)
    assert _all(store)[0].status == STATUS_PENDING  # 命中上限 ⇒ AML 还会继续喂

    _apply(store, "B", (_msg("user", "fragment two"),), limits=limits)

    # 碎片被并进同一个 question——**没有**新建对、**没有**把前一段关成有问无答
    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0]
    assert pairs[0].question == "fragment one\nfragment two"
    assert pairs[0].answer is None

    _apply(store, "C", (_msg("assistant", "answer"),), limits=limits)

    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0]  # 始终只有一个对
    assert pairs[0].question == "fragment one\nfragment two"
    assert pairs[0].answer == "[assistant] answer"
    assert pairs[0].status == STATUS_COMPLETE  # C 批未命中上限 ⇒ session 结束


def test_fragments_merge_even_when_the_previous_batch_marked_the_pair_complete(
    store: SqliteStore,
) -> None:
    """**跨批合并不依赖词数计数**——这是它比"只认 pending 状态"更硬的地方。

    A 批**两限都没命中**（我们这一侧认为 session 结束了）⇒ 那一对拿到 `status = complete`。
    B 批照样必须把碎片并进去：判据是 `answer IS NULL` 这个**内容事实**，
    不是 `status`。

    为什么重要：`pending` 是从"本批是否命中上限"推出来的，而"Adapter 计的词"官方从未
    定义（S2）。若判据只看状态，AML 的计数口径一旦与我们不同，这里就会**静默**退化成
    "有问无答 + 无问的对"。
    """
    limits = BatchLimits(max_messages=9, max_words=100)  # 任何一批都命中不了上限

    _apply(store, "A", (_msg("user", "第一段"),), limits=limits)
    assert _all(store)[0].status == STATUS_COMPLETE  # ← 状态已经是 complete 了

    _apply(store, "B", (_msg("user", "第二段"),), limits=limits)

    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0]
    assert pairs[0].question == "第一段\n第二段"


def test_answer_is_attached_to_a_question_only_pair(store: SqliteStore) -> None:
    """**顺带修掉的一类静默退化**：回复落在下一批时，必须接回它的 question。

    A 批判成 complete（两限未命中）但 `answer` 还空着；B 批带来助手回复。
    旧行为会建一个 `question` 为空的"无问的对"——**回复与问题彻底脱钩**，
    而库里看起来完全正常。新判据下它被追加回原对。
    """
    limits = BatchLimits(max_messages=9, max_words=100)

    _apply(store, "A", (_msg("user", "问题"),), limits=limits)
    _apply(store, "B", (_msg("assistant", "答复"),), limits=limits)

    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0]  # 不是两个对
    assert pairs[0].question == "问题"
    assert pairs[0].answer == "[assistant] 答复"
    assert all(p.question is not None for p in pairs)  # 没有"无问的对"


def test_question_only_pair_absorbs_the_next_batchs_user_message(
    store: SqliteStore,
) -> None:
    """**设计如此的边界**：一个"有问无答"的尾对会吸收下一批开头的 user 消息。

    这是同一条规则的必然推论，**不是 bug**：判据只看"这个对的 answer 空不空"，
    而它空着——我们**无法**区分"AML 还会继续喂碎片"与"这个 session 就在这里结束了"。

    触发条件：某一批两限都没命中（⇒ 我们判定 session 结束、把它标 complete），
    但**之后同一 session 又来了一批**。线上正常时序下这不会发生；它发生意味着
    我们与 AML 的词数计数不一致（S2），或同一个 session 被喂了两轮。
    两种情况下"并进同一个 question"都比"留一个孤儿问题 + 一个无问的对"更接近原意。
    """
    limits = BatchLimits(max_messages=2, max_words=1000)

    _apply(store, "A", (_msg("user", "Q1"), _msg("assistant", "A1")), limits=limits)
    _apply(store, "B", (_msg("user", "Q2"),), limits=limits)  # 1 条 ⇒ 两限未命中

    assert _all(store)[1].answer is None  # Q2 那个对：有问无答

    _apply(store, "C", (_msg("user", "Q3"), _msg("assistant", "A3")), limits=limits)

    pairs = _all(store)
    assert [p.pair_idx for p in pairs] == [0, 1]  # 没新建第三个对
    assert pairs[1].question == "Q2\nQ3"
    assert pairs[1].answer == "[assistant] A3"


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
        (
            _msg("user", "Q1"),
            _msg("assistant", "A1"),
            _msg("user", "Q2"),
            _msg("assistant", "A2"),
        ),
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
    """三个埋点：创建 / 被后续批次补全 / 活到 session 末。

    ⚠ 每批都让**同一个对**既被关闭又留下新的一带，是为了让三种事件各自只发生一次——
    混在一起数不清。判据见 `apply_batch` 的发射段。
    """
    inst = InMemoryPendingInstrument()
    limits = BatchLimits(max_messages=2, max_words=1000)

    # A：命中上限 ⇒ 一个 pending 被创建
    _apply(
        store, "A", (_msg("user", "Q1"), _msg("assistant", "A1")), limits=limits, instrument=inst
    )
    assert inst.counters.pending_created == 1
    assert inst.counters.pending_completed == 0
    assert inst.counters.pending_orphaned == 0

    # B：以 user 开头、**且那个对已有 answer** ⇒ 3b 关掉它（completed）；
    #    B 自己命中上限 ⇒ 又创建一个 pending
    _apply(
        store, "B", (_msg("user", "Q2"), _msg("assistant", "A2")), limits=limits, instrument=inst
    )
    assert inst.counters.pending_completed == 1
    assert inst.counters.pending_created == 2
    assert inst.counters.pending_orphaned == 0

    # C：纯接续批 + 未命中上限 ⇒ session 结束判定时它仍 pending ⇒ orphaned
    _apply(store, "C", (_msg("assistant", "A2b"),), limits=limits, instrument=inst)
    assert inst.counters.pending_orphaned == 1
    assert _all(store)[1].status == STATUS_COMPLETE


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


# ── 随机暴力：三条不变式 ────────────────────────────────────────────────


def test_random_batches_keep_three_invariants(store: SqliteStore) -> None:
    """随机会话 + **随机切批** + 随机上限，逐 trial 断言三条不变式。

    这一条取代不了上面那些具名用例（它不告诉你**为什么**错），但它能盖住具名用例
    想不到的组合——尤其是 D20 之后**规则是"跨批有状态"的**，手写的形状容易只覆盖
    相邻的两批。种子固定 ⇒ 失败可复现。

    | 不变式 | 为什么它是要命的 |
    | --- | --- |
    | **内容无损**：每条发出的消息恰好出现一次 | 丢一条是"静默丢消息"；多一条是重复写入 |
    | **`answer` 为空的对必在末尾** | `open_pair()` 靠 `LIMIT 1` 取它——不成立时会取错对 |
    | **`pair_idx` 连续** | 有空洞则 ±1 邻域扩张**静默消失**（§6.1） |
    """
    rng = random.Random(20260924)
    roles = ("user", "assistant", "tool_result", "system")

    for trial in range(60):
        session_id = f"s{trial}"
        contents = [f"m{trial}-{i}" for i in range(rng.randint(1, 18))]
        messages = tuple(
            _msg("user" if rng.random() < 0.45 else rng.choice(roles), c) for c in contents
        )
        limits = BatchLimits(
            max_messages=rng.choice((1, 2, 3, 5, 20)), max_words=rng.choice((1, 2, 1000))
        )

        batch_index = 0
        start = 0
        while start < len(messages):
            size = rng.randint(1, 5)
            _apply(
                store,
                f"r{trial}-{batch_index}",
                messages[start : start + size],
                limits=limits,
                session_id=session_id,
            )
            batch_index += 1
            start += size

        pairs = _all(store)
        pairs = [p for p in pairs if p.session_id == session_id]
        pairs.sort(key=lambda p: p.pair_idx)

        # (1) 内容无损：把 question / answer 都拆回行，去掉 answer 的 role 标记后逐行比对
        seen: Counter[str] = Counter()
        for pair in pairs:
            for blob in (pair.question or "", pair.answer or ""):
                for line in blob.splitlines():
                    if not line:
                        continue
                    seen[line.split("] ", 1)[-1] if line.startswith("[") else line] += 1
        assert seen == Counter(contents), f"trial {trial}：内容不无损（{seen - Counter(contents)}）"

        # (2) `answer` 为空的对必在末尾
        no_answer = [p.pair_idx for p in pairs if p.answer is None]
        assert no_answer in ([], [pairs[-1].pair_idx]), f"trial {trial}：有问无答的对不在末尾"

        # (3) `pair_idx` 连续
        assert [p.pair_idx for p in pairs] == list(range(len(pairs))), f"trial {trial}：有空洞"
