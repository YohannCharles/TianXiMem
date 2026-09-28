"""一次 `Add` 的落库边界（D24）——组合 + 真源一起测。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 一、配对与落库。
**这是唯一能同时验证"组合逻辑"与"落库结果"的一层。**

⚠ **本文件里最重要的是跨 Add 的三条**（Test 7 / 8 / 9）：
它们钉住"**一次 Add 就是组合的唯一边界**"——不同 Add 永不拼接、乱序到达不影响结果。
纯函数侧的规则在 [`test_pairing.py`](./test_pairing.py)。

> 旧版（D20/§6.5）这里是"批次续接三步 3a′/3a/3b/3d + `pending` 判定"。
> 那些用例已随跨批合并一起删除——**别把它们改回来**，那正是 D24 推翻的东西。
"""

from __future__ import annotations

import random

from tests.conftest import rd

from tianxi_am.pairing import AddBatch, Message, apply_batch
from tianxi_am.store.sqlite_store import STATUS_COMPLETE, SqliteStore


def _msg(role: str, content: str, ts: int | None = None) -> Message:
    return Message(role=role, content=content, timestamp=ts)


def _rid(ordinal: int, *, user_id: str = "u1", session_id: str = "s1") -> str:
    """造一个 **符合 `ingest.chunk_ordinal_pattern`** 的 `request_id`（D25）。

    ⚠ 不再是随便一个字符串：位置由 `(chunk_ordinal, local_index)` 派生，
    而 chunk 序号**只能**从这个 id 里取 ⇒ 取不到就响亮失败（没有回退）。
    """
    return f"{user_id}|{session_id}|{ordinal}"


def _apply(
    store: SqliteStore,
    request_id: str,
    messages: tuple[Message, ...],
    *,
    user_id: str = "u1",
    session_id: str = "s1",
):
    return apply_batch(store, AddBatch(request_id, user_id, session_id, messages))


def _all(store: SqliteStore, user_id: str = "u1"):
    return sorted(rd(store, store.assert_isolation, user_id), key=lambda p: p.seq)


def _shape(store: SqliteStore, user_id: str = "u1") -> list[str]:
    """该 user 的全部行，摊成 `Q: … / A: …` 的列表——断言里比这个比逐个字段可读。"""
    out = []
    for p in _all(store, user_id):
        lines = []
        if p.question:
            lines.append(f"Q: {p.question}")
        if p.answer:
            lines.append(f"A: {p.answer}")
        out.append("\n".join(lines))
    return out


# ── Test 7：跨 Add 不拼接 ───────────────────────────────────────────────


def test_7_no_pairing_across_adds(store: SqliteStore) -> None:
    """Test 7：`Add0 = U A U` / `Add1 = A U A` ⇒ **不出现 U(Add0)+A(Add1)**。

    这是整个改动要买的那件事：`Q1` 与 `A1` 看起来明显是一对，**依然不许拼**。
    """
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("assistant", "A0"), _msg("user", "Q1")))
    _apply(store, _rid(1), (_msg("assistant", "A1"), _msg("user", "Q2"), _msg("assistant", "A2")))

    assert _shape(store) == [
        "Q: Q0\nA: [assistant] A0",  # Add0
        "Q: Q1",  # Add0 末尾配不上 ⇒ 独立（**没有被 A1 补上**）
        "A: [assistant] A1",  # Add1 开头配不上 ⇒ 独立
        "Q: Q2\nA: [assistant] A2",  # Add1
    ]
    assert [p.seq for p in _all(store)] == [0, 1, 2, 3]


# ── Test 8：连续 assistant 跨 Add 不合并 ────────────────────────────────


def test_8_consecutive_assistants_do_not_merge_across_adds(store: SqliteStore) -> None:
    """Test 8：`Add0 = U A A` / `Add1 = A U A` ⇒ 不合并成 `AAA`。"""
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("assistant", "A0"), _msg("assistant", "A1")))
    _apply(store, _rid(1), (_msg("assistant", "A2"), _msg("user", "Q1"), _msg("assistant", "A3")))

    assert _shape(store) == [
        "Q: Q0\nA: [assistant] A0\n[assistant] A1",
        "A: [assistant] A2",  # ← 必须独立，**不许**并进上一块的 answer
        "Q: Q1\nA: [assistant] A3",
    ]


def test_consecutive_assistants_still_merge_within_one_add(store: SqliteStore) -> None:
    """同一条规则的另一半：**同一个 Add 内**的连续 assistant 照旧合并（不是一刀切成单条）。"""
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("assistant", "A0"), _msg("assistant", "A1")))
    assert _shape(store) == ["Q: Q0\nA: [assistant] A0\n[assistant] A1"]


# ── Test 9：乱序 Add 与顺序 Add 结果一致 ────────────────────────────────


def test_9_out_of_order_arrival_matches_in_order(tmp_path) -> None:
    """Test 9：按 `Add2 / Add0 / Add1` 调用，**每个 Add 内产出的块与顺序调用时一致**。

    ⚠ 这一条**只看组合**（D24）。**更强的那条在下面**：D25 之后位置也由请求决定
    ⇒ 乱序到达的**最终真源**应当与顺序到达**逐字一致**（位置集合、`id` 集合、`seq` 序）。
    """
    adds = {
        _rid(0): (_msg("user", "Q0"), _msg("assistant", "A0"), _msg("user", "Q1")),
        _rid(1): (_msg("assistant", "A1"), _msg("user", "Q2")),
        _rid(2): (_msg("user", "Q3"), _msg("assistant", "A2"), _msg("assistant", "A3")),
    }
    expected = {
        _rid(0): ["Q: Q0\nA: [assistant] A0", "Q: Q1"],
        _rid(1): ["A: [assistant] A1", "Q: Q2"],
        _rid(2): ["Q: Q3\nA: [assistant] A2\n[assistant] A3"],
    }

    orders = [
        (_rid(0), _rid(1), _rid(2)),
        (_rid(2), _rid(0), _rid(1)),
        (_rid(1), _rid(2), _rid(0)),
    ]
    for order in orders:
        fresh = SqliteStore.open(tmp_path / f"{'_'.join(order)}.db")
        per_add: dict[str, list[str]] = {}
        for rid in order:
            result = _apply(fresh, rid, adds[rid])
            per_add[rid] = [
                "\n".join(
                    line
                    for line in (
                        f"Q: {b.question}" if b.question else None,
                        f"A: {b.answer}" if b.answer else None,
                    )
                    if line
                )
                for b in result.blocks
            ]
        # 每个 Add 的块只由它自己的消息决定 ⇒ 与到达顺序无关
        assert per_add == expected
        # 而且**真的落进了库**（不是只有返回值对）：r0 两块、r1 两块、r2 一块
        assert len(_all(fresh)) == 5
        assert [p.seq for p in _all(fresh)] == list(range(5))


# ── 位置：由 `request_id` 的 chunk 序号派生（**与组合边界无关**，D25）─────────


def test_position_is_derived_from_request_id_not_arrival_order(store: SqliteStore) -> None:
    """位置是 **session** 的属性（chunk 序号在请求里），不是 Add 的——组合按 Add 切，位置不按。

    `id` 由位置派生（不变式 1），所以按 Add 从 0 重开会让不同 Add 的行**撞 id**，
    而写入是 upsert ⇒ **静默覆盖**。本用例走的是 harness 形态的 `request_id`
    （`<user>|<session>|<n>`），序号 0/1/2 逐批递增。
    """
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("assistant", "A0")))
    _apply(store, _rid(1), (_msg("user", "Q1"), _msg("assistant", "A1")))
    _apply(store, _rid(2), (_msg("user", "Q2"),))

    assert [p.seq for p in _all(store)] == [0, 1, 2]
    assert len({p.id for p in _all(store)}) == 3


def test_sessions_are_independent(store: SqliteStore) -> None:
    """两个 session 各从 0 起编号，互不影响。"""
    _apply(store, _rid(0), (_msg("user", "Q0"),), session_id="s1")
    _apply(store, _rid(1), (_msg("user", "Q1"),), session_id="s2")
    assert [p.seq for p in _all(store)] == [0, 0]
    assert {(p.session_id, p.seq) for p in _all(store)} == {("s1", 0), ("s2", 0)}


# ── 状态：写下的那一刻就是最终形状 ──────────────────────────────────────


def test_every_row_is_complete_at_write_time(store: SqliteStore) -> None:
    """D24：没有 pending、没有 repair —— **没有任何后台任务会回头改这些行**。"""
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("user", "Q1"), _msg("assistant", "A0")))
    _apply(store, _rid(1), (_msg("assistant", "A1"),))
    assert [p.status for p in _all(store)] == [STATUS_COMPLETE, STATUS_COMPLETE]


def test_a_lone_question_is_stored_and_stays_lone(store: SqliteStore) -> None:
    """配不上的块**照样落库**，且**不会**被后来的 Add 补上（Test 7 的持久化形态）。"""
    _apply(store, _rid(0), (_msg("user", "孤零零的问题"),))
    first = _all(store)[0]
    _apply(store, _rid(1), (_msg("assistant", "迟到的回答"),))

    after = _all(store)[0]
    assert (after.question, after.answer) == ("孤零零的问题", None)
    assert after.id == first.id  # 位置派生 ⇒ 同一行没被改写
    assert len(_all(store)) == 2


# ── 空批次仍要响亮失败 ──────────────────────────────────────────────────


def test_empty_batch_is_rejected_loudly(store: SqliteStore) -> None:
    """空批次是**调用方的 bug**（§2.1 的 `messages` 就是批次的全部内容）。

    静默当成 no-op 会让"整个 session 一条都没写进去"伪装成成功。
    """
    import pytest

    with pytest.raises(ValueError):
        _apply(store, _rid(0), ())
    assert _all(store) == []


# ── 随机暴力：三条不变式 ────────────────────────────────────────────────


def test_random_batches_keep_three_invariants(store: SqliteStore) -> None:
    """随机批次序列之后仍成立：

    1. `seq` 在 session 内**连续、无空洞**（扩窗与段合并依赖它）
    2. 每一行**都来自某一次 Add**，且 `id` 位置派生（无孤儿、无覆盖）
    3. **没有任何一条消息丢失**：把全部行的正文摊平，与喂进去的消息逐条对得上
    4. **chunk 序号各自唯一**（同一批不会被写两次）
    """
    rng = random.Random(20260927)
    fed: list[str] = []
    for i in range(40):
        size = rng.randint(1, 5)
        messages = tuple(
            _msg(rng.choice(["user", "assistant", "user"]), f"m{i}-{j}") for j in range(size)
        )
        fed.extend(m.content for m in messages)
        _apply(store, _rid(i), messages)

    pairs = _all(store)
    assert [p.seq for p in pairs] == list(range(len(pairs)))
    assert len({p.chunk_ordinal for p in pairs}) == 40  # 每批各占一个 chunk 序号

    seen: set[str] = set()
    for p in pairs:
        assert p.id not in seen
        seen.add(p.id)
        assert p.status == STATUS_COMPLETE

    # 每条消息恰好落进一个块一次（question 侧不带标记、answer 侧带 `[role] `）
    landed = []
    for p in pairs:
        landed.extend((p.question or "").split("\n") if p.question else [])
        if p.answer:
            landed.extend(
                line.split("] ", 1)[1] if "] " in line else "" for line in p.answer.split("\n")
            )
    assert sorted(x for x in landed if x) == sorted(fed)
