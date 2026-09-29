"""一次 `Add` 的落库边界（D24 / **D28**）——组合 + 落库 + Add 内邻接一起测。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 一、配对与落库。
**这是唯一能同时验证"组合逻辑"与"落库结果"的一层。**

⚠ **本文件的头号主题是"边界只在一次 Add 内"**：

* **组合**（D24）：不同 Add 永不拼接（Test 7 / 8）
* **邻接**（D28）：`prev` / `next` 只在同一次 Add 内相连——**即使 user/session 都一样**

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
    """一个**不透明**的 `request_id`（D28）——形状刻意做成**像官方实发的那种**。

    ⚠ 它长什么样**完全不重要**（这正是要钉住的那件事）：不透明、不解析、原样参与哈希。
    本函数保留 `user_id` / `session_id` 两个参数只是为了让调用点在两种 session 之间切换时
    读起来自然——**它们不再进入 id 的字符串**。
    """
    return f"r_{ordinal:04d}{user_id[-1] if user_id != 'u1' else ''}"


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
    """该 user 的全部行，按 `(request_id, local_index)` 排——**不是"会话顺序"**（D28）。"""
    return sorted(
        rd(store, store.assert_isolation, user_id),
        key=lambda p: (p.request_id, p.local_index),
    )


def _positions(store: SqliteStore, user_id: str = "u1") -> list[tuple[str, int]]:
    return [(p.request_id, p.local_index) for p in _all(store, user_id)]


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
    # 位置：两次 Add 各自的 local_index 都从 0 起，**不接续**
    assert _positions(store) == [("r_0000", 0), ("r_0000", 1), ("r_0001", 0), ("r_0001", 1)]


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


# ── `request_id` 是 opaque string（D28）────────────────────────────────


def test_consecutive_roles_merge_into_two_qa_blocks(store: SqliteStore) -> None:
    """`Q Q A A Q A` ⇒ 逻辑上是 `Q A Q A` ⇒ **2 个完整 QA**，且两个互连（D28）。

    ⚠ 合并**不丢内容**：第二段的 `question` 是两条 user 消息的拼接、`answer` 每条带 `[role]`。
    """
    _apply(
        store,
        _rid(0),
        (
            _msg("user", "Q0a"),
            _msg("user", "Q0b"),
            _msg("assistant", "A0a"),
            _msg("assistant", "A0b"),
            _msg("user", "Q1"),
            _msg("assistant", "A1"),
        ),
    )

    rows = _all(store)
    assert len(rows) == 2
    assert rows[0].question == "Q0a" + chr(10) + "Q0b"
    assert rows[0].answer == "[assistant] A0a" + chr(10) + "[assistant] A0b"
    assert rows[1].question == "Q1"
    assert rows[0].next_memory_id == rows[1].id
    assert rows[1].prev_memory_id == rows[0].id


def test_request_id_is_opaque_and_any_shape_works(store: SqliteStore) -> None:
    """**任意形状的 `request_id` 都必须能正常 Add**（D28）。

    尤其这两类——它们在 D25 的正则下会**响亮失败**（500），正是 2026-09-29 那次
    "Add 全挂"的形状：

    * 根本没有 chunk 序号：平台实发的 `r_3115…`、`abc`
    * 尾部多一段：`xxx:chunk-0-extra`
    """
    opaque_ids = [
        "abc",
        "req-001",
        "foo:bar",
        "xxx:chunk-0",
        "xxx:chunk-0-extra",
        "r_31156f4174b24abe83ad2c09a486cc5f398ddc819b4ca9e33b9c94bb1e2caab8",
        "11111111-2222-3333-4444-555555555555",
        " 前后有空格  ",
    ]
    for request_id in opaque_ids:
        result = _apply(store, request_id, (_msg("user", f"Q-{request_id}"),))
        assert result.applied is True, request_id

    assert len(_all(store)) == len(opaque_ids)  # 一条都没丢
    assert {p.request_id for p in _all(store)} == set(opaque_ids)  # 原样落库（不被改写）
    # 每一条都拿到了自己的位置（不会互相覆盖）
    assert len({p.id for p in _all(store)}) == len(opaque_ids)


# ── 邻接：只在一次 Add 内（D28 的核心）─────────────────────────────────


def test_full_qa_blocks_in_one_add_form_a_chain(store: SqliteStore) -> None:
    """`Q A Q A Q A` ⇒ 三个完整 QA，`M1 <-> M2 <-> M3`（链只在这一批内）。"""
    _apply(
        store,
        _rid(0),
        (
            _msg("user", "Q0"),
            _msg("assistant", "A0"),
            _msg("user", "Q1"),
            _msg("assistant", "A1"),
            _msg("user", "Q2"),
            _msg("assistant", "A2"),
        ),
    )

    rows = _all(store)
    ids = [p.id for p in rows]
    assert [p.question for p in rows] == ["Q0", "Q1", "Q2"]
    assert [(p.prev_memory_id, p.next_memory_id) for p in rows] == [
        (None, ids[1]),
        (ids[0], ids[2]),
        (ids[1], None),
    ]
    assert all(p.is_complete for p in rows)


def test_incomplete_blocks_are_stored_indexed_and_outside_the_chain(store: SqliteStore) -> None:
    """`A Q A Q` ⇒ `A-only / QA / QA`：**三块都落库**，而 A-only 两侧都是 `None`。

    ⚠ "不完整"不等于"丢弃"：它照样有 `id`、照样会被索引（`AddPipeline` 索引本批**全部**
    行）、照样能被检索到——只是**不参与邻接**。
    """
    _apply(
        store,
        _rid(0),
        (
            _msg("assistant", "A0"),
            _msg("user", "Q1"),
            _msg("assistant", "A1"),
            _msg("user", "Q2"),
            _msg("assistant", "A2"),
        ),
    )

    rows = _all(store)
    assert len(rows) == 3
    assert [(p.question, p.answer) for p in rows] == [
        (None, "[assistant] A0"),  # A-only
        ("Q1", "[assistant] A1"),
        ("Q2", "[assistant] A2"),
    ]
    assert (rows[0].prev_memory_id, rows[0].next_memory_id) == (None, None)  # 不进链
    assert rows[1].prev_memory_id is None  # 链首是它（前一条不完整）
    assert rows[1].next_memory_id == rows[2].id
    assert rows[2].prev_memory_id == rows[1].id


def test_trailing_lone_question_is_stored_and_stays_lone(store: SqliteStore) -> None:
    """`Q A Q` ⇒ `QA` + `Q-only`；Q-only **不进链**，而且**不会被后来的 Add 补上**。"""
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("assistant", "A0"), _msg("user", "孤零零")))
    rows = _all(store)
    assert (rows[1].question, rows[1].answer) == ("孤零零", None)
    assert (rows[1].prev_memory_id, rows[1].next_memory_id) == (None, None)
    assert rows[0].next_memory_id is None  # 链到 QA 就断了（后面那条不完整）

    _apply(store, _rid(1), (_msg("assistant", "迟到的回答"),))

    after = _all(store)
    assert len(after) == 3  # 又落了一块，**没有**并进上面那条
    assert (after[1].question, after[1].answer) == ("孤零零", None)  # 一个字都没被改写
    assert after[0].id == rows[0].id  # 位置派生 ⇒ 同一行还是同一行


def test_neighbours_never_cross_adds_even_in_the_same_session(store: SqliteStore) -> None:
    """**同一 `(user_id, session_id)` 的两次 Add，绝不互为邻居**（D28）。

    ```text
    Add1: Q A Q A  →  M1 <-> M2
    Add2: Q A Q A  →  M3 <-> M4
    必须：M2.next is None、M3.prev is None
    ```
    """
    batch = (
        _msg("user", "Q0"),
        _msg("assistant", "A0"),
        _msg("user", "Q1"),
        _msg("assistant", "A1"),
    )
    _apply(store, _rid(0), batch)
    _apply(store, _rid(1), batch)

    rows = _all(store)
    assert len(rows) == 4
    m1, m2, m3, m4 = rows
    assert (m1.prev_memory_id, m1.next_memory_id) == (None, m2.id)
    assert (m2.prev_memory_id, m2.next_memory_id) == (m1.id, None)  # ← 链在此断开
    assert (m3.prev_memory_id, m3.next_memory_id) == (None, m4.id)  # ← 新链从零开始
    assert (m4.prev_memory_id, m4.next_memory_id) == (m3.id, None)
    # 两条链的 request_id 不同
    assert m1.request_id != m3.request_id
    # 而且 local_index 各自从 0 起（**不是** 2 和 3）
    assert [p.local_index for p in rows] == [0, 1, 0, 1]


def test_all_questions_or_all_answers_are_all_isolated(store: SqliteStore) -> None:
    """全 Q / 全 A 的批次：**全部内容都存下来**，且**没有一个进链**。"""
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("user", "Q1"), _msg("user", "Q2")))
    _apply(store, _rid(1), (_msg("assistant", "A0"), _msg("assistant", "A1")))

    rows = _all(store)
    assert [(p.question, p.answer) for p in rows] == [
        ("Q0\nQ1\nQ2", None),  # 连续同 role 合并成一块
        (None, "[assistant] A0\n[assistant] A1"),
    ]
    assert all(p.prev_memory_id is None and p.next_memory_id is None for p in rows)


# ── 位置与到达顺序无关（D28 的强形式）───────────────────────────────────


def test_arrival_order_does_not_change_the_truth_source(tmp_path) -> None:
    """**按任意顺序投递，真源逐字相同**（位置 = 请求的纯函数）。

    ⚠ 与 D25 那版的区别：这里比的**不是"会话顺序"**（D28 起它不存在），
    而是 `(id, request_id, local_index, question, answer, prev, next)` 的**多重集合**——
    它才是"这一批写出了什么"的完整描述。
    """
    adds = {
        "r_0000": (_msg("user", "Q0"), _msg("assistant", "A0"), _msg("user", "Q1")),
        "r_0001": (_msg("assistant", "A1"), _msg("user", "Q2")),
        "r_0002": (_msg("user", "Q3"), _msg("assistant", "A2"), _msg("assistant", "A3")),
    }
    expected_per_add = {
        "r_0000": ["Q: Q0\nA: [assistant] A0", "Q: Q1"],
        "r_0001": ["A: [assistant] A1", "Q: Q2"],
        "r_0002": ["Q: Q3\nA: [assistant] A2\n[assistant] A3"],
    }

    snapshots = []
    for order in (
        ("r_0000", "r_0001", "r_0002"),
        ("r_0002", "r_0000", "r_0001"),
        ("r_0001", "r_0002", "r_0000"),
    ):
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
        assert per_add == expected_per_add
        snapshots.append(
            sorted(
                (p.id, p.request_id, p.local_index, p.question, p.answer, p.prev_memory_id,
                 p.next_memory_id)
                for p in _all(fresh)
            )
        )
    # 三种到达顺序 ⇒ 真源一模一样（5 块，位置与邻接都不变）
    assert snapshots[0] == snapshots[1] == snapshots[2]
    assert len(snapshots[0]) == 5


def test_sessions_are_independent(store: SqliteStore) -> None:
    """两个 session 各写各的，互不影响（位置里带着 session）。"""
    _apply(store, _rid(0), (_msg("user", "Q0"),), session_id="s1")
    _apply(store, _rid(1), (_msg("user", "Q1"),), session_id="s2")

    assert [(p.session_id, p.local_index) for p in _all(store)] == [("s1", 0), ("s2", 0)]
    assert len({p.id for p in _all(store)}) == 2


# ── 状态：写下的那一刻就是最终形状 ──────────────────────────────────────


def test_every_row_is_complete_at_write_time(store: SqliteStore) -> None:
    """D24：没有 pending、没有 repair —— **没有任何后台任务会回头改这些行**。"""
    _apply(store, _rid(0), (_msg("user", "Q0"), _msg("user", "Q1"), _msg("assistant", "A0")))
    _apply(store, _rid(1), (_msg("assistant", "A1"),))
    assert [p.status for p in _all(store)] == [STATUS_COMPLETE, STATUS_COMPLETE]


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

    1. **一次 Add 内的 `local_index` 连续、无空洞、从 0 起**
    2. 每一行**都来自某一次 Add**，且 `id` 位置派生（无孤儿、无覆盖）
    3. **没有任何一条消息丢失**：把全部行的正文摊平，与喂进去的消息逐条对得上
    4. **链的完整性**：每个完整 QA 的 `prev` / `next` 都在**同一次 Add 内**，
       且首尾分别为 `None`
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
    by_request: dict[str, list] = {}
    for pair in pairs:
        by_request.setdefault(pair.request_id, []).append(pair)

    seen: set[str] = set()
    for request_id, rows in by_request.items():
        # 1 —— 批内位置连续、从 0 起
        assert [p.local_index for p in rows] == list(range(len(rows))), request_id
        ids = {p.id for p in rows}
        for p in rows:
            # 2 —— 无重复 id、状态恒为 complete
            assert p.id not in seen
            seen.add(p.id)
            assert p.status == STATUS_COMPLETE
            # 4 —— 邻接只在批内（指针要么是 None，要么指向同一批的行）
            for neighbour in (p.prev_memory_id, p.next_memory_id):
                assert neighbour is None or neighbour in ids, (request_id, p.local_index)

    # 每条消息恰好落进一个块一次（question 侧不带标记、answer 侧带 `[role] `）
    landed = []
    for p in pairs:
        landed.extend((p.question or "").split("\n") if p.question else [])
        if p.answer:
            landed.extend(
                line.split("] ", 1)[1] if "] " in line else "" for line in p.answer.split("\n")
            )
    assert sorted(x for x in landed if x) == sorted(fed)
