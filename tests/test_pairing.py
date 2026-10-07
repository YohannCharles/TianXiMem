"""记忆块组合规则（D24）——**纯函数**侧。

```text
一次 Add 的消息
  → ① 一段连续 user + 紧随的非 user 才配对（合并只是这一步）
  → ② 相邻 UserBlock + 非UserBlock 配成一个 MemoryBlock
  → ③ 配不上的 RoleBlock 独立成块
```

**规则一处声明在 [`../src/tianximem/pairing/CLAUDE.md`](../src/tianximem/pairing/CLAUDE.md)**，
本文件只覆盖它。跨 Add 的那一半（不拼接、乱序无关、幂等）在
[`test_apply.py`](./test_apply.py)。

> ⚠ **下面 Test 1–6 的编号是需求里给定的**，改它们先回去改需求。
> Test 7–10 是跨 Add 的，住在 `test_apply.py` / `test_idempotency.py`。
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from tianximem.pairing.pairing import (
    MemoryBlock,
    Message,
    compose_memory_blocks,
    encode_answer,
    is_user,
    join_question,
)


def _roles(block: MemoryBlock) -> tuple[str, ...]:
    return block.roles


def _text(block: MemoryBlock) -> str:
    """把块摊成 `Q:` / `A:` 的渲染形状——断言里比这个比比两个字段更贴近检索单元。"""
    lines = []
    if block.question:
        lines.append(f"Q: {block.question}")
    if block.answer:
        lines.append(f"A: {block.answer}")
    return "\n".join(lines)


# ── Test 1–6：需求里给定的六个形状 ───────────────────────────────────────


def test_1_user_assistant_user_assistant(M: Callable[..., Message]) -> None:
    """Test 1：`U A U A` ⇒ `[U+A] [U+A]`。"""
    blocks = compose_memory_blocks(
        [M("user", "Q0"), M("assistant", "A0"), M("user", "Q1"), M("assistant", "A1")]
    )
    assert [_text(b) for b in blocks] == [
        "Q: Q0\nA: [assistant] A0",
        "Q: Q1\nA: [assistant] A1",
    ]
    assert all(b.is_paired for b in blocks)


def test_2_runs_merge_within_one_memory(M: Callable[..., Message]) -> None:
    """Test 2：`U U A A A U A` ⇒ **3 块**（**D32**）——一段 user 只把**最后一条**配进去。

    ⚠ 形状由 **D32**（2026-10-03）定（一段 user 只配最后一条）：一段长 user 全进 question
    会到 38,270 token（窗口 8,192）⇒ 线上静默截断、本地 400。代价是前面的 `Q0`
    变成**有问无答**。
    """
    blocks = compose_memory_blocks(
        [
            M("user", "Q0"),
            M("user", "Q1"),
            M("assistant", "A0"),
            M("assistant", "A1"),
            M("assistant", "A2"),
            M("user", "Q2"),
            M("assistant", "A3"),
        ]
    )
    assert [_text(b) for b in blocks] == [
        "Q: Q0",
        "Q: Q1\nA: [assistant] A0\n[assistant] A1\n[assistant] A2",
        "Q: Q2\nA: [assistant] A3",
    ]
    assert len(blocks) == 3  # 7 条消息 ⇒ 3 个块（多出来的那个是配不上的 Q0）


def test_3_leading_assistant_stands_alone(M: Callable[..., Message]) -> None:
    """Test 3：`A U A` ⇒ `[A] [U+A]`——首个 A 配不上，**独立成块**。"""
    blocks = compose_memory_blocks([M("assistant", "A0"), M("user", "Q1"), M("assistant", "A1")])
    assert [_text(b) for b in blocks] == ["A: [assistant] A0", "Q: Q1\nA: [assistant] A1"]
    assert blocks[0].question is None
    assert blocks[0].is_paired is False


def test_4_trailing_user_stands_alone(M: Callable[..., Message]) -> None:
    """Test 4：`U A U` ⇒ `[U+A] [U]`——末尾的 U 配不上，**独立成块**。"""
    blocks = compose_memory_blocks([M("user", "Q0"), M("assistant", "A0"), M("user", "Q1")])
    assert [_text(b) for b in blocks] == ["Q: Q0\nA: [assistant] A0", "Q: Q1"]
    assert blocks[1].answer is None


def test_5_lone_assistant(M: Callable[..., Message]) -> None:
    """Test 5：`A` ⇒ `[A]`。"""
    blocks = compose_memory_blocks([M("assistant", "A0")])
    assert [_text(b) for b in blocks] == ["A: [assistant] A0"]


def test_6_lone_user(M: Callable[..., Message]) -> None:
    """Test 6：`U` ⇒ `[U]`。"""
    blocks = compose_memory_blocks([M("user", "Q0")])
    assert [_text(b) for b in blocks] == ["Q: Q0"]
    assert blocks[0].answer is None


# ── 组合的边界（与 Test 1–6 同一批规则，补上"容易漏"的形状）────────────────


def test_adjacent_same_role_runs_alternate_strictly(M: Callable[..., Message]) -> None:
    """⚠ **相邻的 user-only 块是正常的**，不是"漏合并"。

    **D32** 只把一段 user 的**最后一条**配进 question ⇒ 一段 `U U A` 产出 `[U] [U+A]`，
    那两个 `U` **是独立的两条消息**。形状变了要有人知道，所以这条钉住**新的**形状。
    """
    blocks = compose_memory_blocks(
        [
            M("user", "Q0"),
            M("user", "Q1"),
            M("assistant", "A0"),
            M("user", "Q2"),
            M("user", "Q3"),
            M("user", "Q4"),
            M("assistant", "A1"),
        ]
    )
    assert [_roles(b) for b in blocks] == [
        ("user",),  # Q0 配不上（它后面还有 user）
        ("user", "assistant"),  # Q1 + A0 —— 这段 user 的**最后一条**
        ("user",),  # Q2 配不上
        ("user",),  # Q3 配不上
        ("user", "assistant"),  # Q4 + A1
    ]
    # 只有**完整的对**能连成链（`link_blocks` 的口径），落单的 Q0/Q2/Q3 不进链
    assert [b.is_paired for b in blocks] == [False, True, False, False, True]


def test_wholly_non_user_batch_splits_per_message(M: Callable[..., Message]) -> None:
    """整批没有 user ⇒ **每条各自一块**（2026-10-01：合并只在"配得上对"时发生）。

    ⚠ 旧规则这里是**一个**块。改的理由见模块 docstring——判分池那种全 `user` 的批
    在旧规则下会整个塌成一块，而这里同一条规则的另一半。
    """
    blocks = compose_memory_blocks([M("assistant", "A0"), M("assistant", "A1")])
    assert len(blocks) == 2
    assert all(b.question is None for b in blocks)
    assert [b.answer for b in blocks] == ["[assistant] A0", "[assistant] A1"]


def test_a_user_run_without_a_reply_does_not_merge(M: Callable[..., Message]) -> None:
    """**判分池那种形状**：`U U U U` ⇒ **4 个独立的 Q-only 块**，不是一个巨块。

    这正是这次改规则要解决的问题：官方**判分池**的 Add 实测是 100% `role: user`，
    旧规则把一次 Add（≤20 条）整个塌成 1 块，而且**一个配对的块都没有**
    ⇒ `link_blocks` 只连配对块 ⇒ 邻接链全空 ⇒ 扩窗与段合并**一行都不执行**。
    """
    blocks = compose_memory_blocks([M("user", f"Q{i}") for i in range(4)])
    assert [b.question for b in blocks] == ["Q0", "Q1", "Q2", "Q3"]
    assert all(b.answer is None for b in blocks)
    assert all(not b.is_paired for b in blocks)


def test_a_user_run_pairs_only_its_last_message(M: Callable[..., Message]) -> None:
    """**D32**：`U U U A` ⇒ `[U] [U] [U+A]`——只有**最后一条**进对。

    ⚠ 这正是 **S5** 那个形状（平台把一条超长 user 消息切成多片，2026-10-01 实测：
    会切、硬上限 ≈8,000 字符）。**D20 当初把整段并进一个 question**，正是为了让被切碎的
    文档"问题完整"；**D32 把它反过来了**——因为合起来会让块到 38,270 token，
    超过 `text-embedding-v4` 的 8,192 窗口。

    ⇒ **两条路都错，只是错的地方不同**：合并 ⇒ 块嵌不进去（线上静默截断）；
    不合并 ⇒ 前面那些 `q` **有问无答**。D32 选了后者，这条用例钉住这个取舍。
    """
    blocks = compose_memory_blocks([M("user", f"Q{i}") for i in range(3)] + [M("assistant", "A")])
    assert len(blocks) == 3
    assert [b.question for b in blocks] == ["Q0", "Q1", "Q2"]
    assert [b.answer for b in blocks] == [None, None, "[assistant] A"]
    assert [b.is_paired for b in blocks] == [False, False, True]


def test_alternating_roles_never_pair_across_a_run(M: Callable[..., Message]) -> None:
    """`A U A U` ⇒ 3 个块（首个 A 落单，末尾 U 落单）——**不会**把 U1 回头配给 A0。"""
    blocks = compose_memory_blocks(
        [
            M("assistant", "A0"),
            M("user", "Q1"),
            M("assistant", "A1"),
            M("user", "Q2"),
        ]
    )
    assert [_text(b) for b in blocks] == [
        "A: [assistant] A0",
        "Q: Q1\nA: [assistant] A1",
        "Q: Q2",
    ]


# ── 未知 role：不丢消息、不枚举白名单 ────────────────────────────────────


def test_unknown_role_is_not_dropped(M: Callable[..., Message]) -> None:
    """`system` / 工具输出 / 没见过的 role **一律按非 user 处理**，且**带 role 标记**。

    ⚠ AML 传入的取值域**没有文档**——枚举白名单会在遇到没见过的 role 时**静默丢消息**。
    """
    blocks = compose_memory_blocks(
        [
            M("user", "Q0"),
            M("system", "sys"),
            M("tool_result", "42"),
        ]
    )
    assert len(blocks) == 1
    assert blocks[0].answer == "[system] sys\n[tool_result] 42"


def test_tool_call_between_user_and_assistant_stays_in_one_block(
    M: Callable[..., Message],
) -> None:
    """一条 user 后跟多条非 user（工具调用等）⇒ **全部归入同一块**。"""
    blocks = compose_memory_blocks(
        [
            M("user", "Q0"),
            M("assistant", "A0"),
            M("tool", "T0"),
            M("assistant", "A1"),
        ]
    )
    assert len(blocks) == 1
    assert blocks[0].answer == "[assistant] A0\n[tool] T0\n[assistant] A1"


def test_empty_content_is_kept_as_marker_only(M: Callable[..., Message]) -> None:
    """空 content **不丢弃**（§2.1 保证非空，这里是防御）——渲染成 `[role]`，否则是静默丢消息。"""
    blocks = compose_memory_blocks([M("user", "Q0"), M("assistant", "")])
    assert blocks[0].answer == "[assistant]"


# ── 来源元数据：合并**不丢**任何一条消息 ────────────────────────────────


def test_source_indexes_cover_every_message_exactly_once(M: Callable[..., Message]) -> None:
    """每个块都带回来源下标，**合起来恰好是 0..n-1 的一个排列**（无丢失、无重复）。"""
    messages = [
        M("user", "Q0"),
        M("user", "Q1"),
        M("assistant", "A0"),
        M("user", "Q2"),
        M("assistant", "A1"),
        M("assistant", "A2"),
    ]
    blocks = compose_memory_blocks(messages)
    seen = [i for b in blocks for i in b.source_idxs]
    assert seen == list(range(len(messages)))


def test_source_metadata_is_kept_verbatim(M: Callable[..., Message]) -> None:
    """`messages` / `roles` / `timestamps` 与原始输入**逐条对应**。

    ⚠ D32 之后 `U U A` 产出**两块**（`[Q0]` 与 `[Q1+A0]`）⇒ 断言要跨两块收集，
    不能只取一个块。`source_idxs` 合起来仍必须是 `0..n-1` 的排列（不丢不重）。
    """
    m0, m1, m2 = M("user", "Q0", 100), M("user", "Q1", 200), M("assistant", "A0", 300)
    blocks = compose_memory_blocks([m0, m1, m2])
    assert [b.messages for b in blocks] == [(m0,), (m1, m2)]
    assert [b.roles for b in blocks] == [("user",), ("user", "assistant")]
    assert [b.timestamps for b in blocks] == [(100,), (200, 300)]
    assert sorted(i for b in blocks for i in b.source_idxs) == [0, 1, 2]


def test_event_time_is_the_first_message_of_the_block(M: Callable[..., Message]) -> None:
    """`event_time` = **该块首条消息**的 timestamp（§6.1 的口径）。"""
    blocks = compose_memory_blocks(
        [
            M("assistant", "A0", 111),
            M("user", "Q1", 222),
            M("assistant", "A1", 333),
        ]
    )
    assert [b.event_time for b in blocks] == [111, 222]


def test_event_time_is_none_when_missing(M: Callable[..., Message]) -> None:
    """没有 timestamp 时是 `None`——**绝不拿 Add 的到达时间兜底**（那是"何时写入"）。"""
    blocks = compose_memory_blocks([M("user", "Q0"), M("assistant", "A0")])
    assert blocks[0].event_time is None


# ── 纯函数：确定性、无状态、不看别的 Add ────────────────────────────────


def test_composition_is_deterministic(M: Callable[..., Message]) -> None:
    """同一批消息反复调用，产出**逐字相同**（⇒ 乱序 Add 不影响各自的内容）。"""
    messages = [M("user", "Q0"), M("user", "Q1"), M("assistant", "A0"), M("user", "Q2")]
    first = compose_memory_blocks(messages)
    for _ in range(5):
        assert compose_memory_blocks(messages) == first


def test_prefix_blocks_are_independent_of_what_follows(M: Callable[..., Message]) -> None:
    """**前面的块的形状不受后面消息影响**——这是"不跨 Add 等下一个 chunk"的可断言形式。

    旧规则（D20/§6.5）里，末尾那个只有 question 的对会被下一个 Add 的答案补上；
    现在 `Q1` **在下一次调用里仍然是同一个独立的块**（`Q1+A1` 只存在于第二次调用里）。
    """
    first_block = compose_memory_blocks([M("user", "Q0"), M("assistant", "A0")])
    with_tail = compose_memory_blocks([M("user", "Q0"), M("assistant", "A0"), M("user", "Q1")])
    assert with_tail[0] == first_block[0]

    # 末尾只有 question 的块**不会**被后续的 assistant 补上：那次补全发生在**另一次调用**里。
    # （比 `_text` 而不是整个块：`source_idxs` 是 **Add 内**的下标，两次调用的下标本来就不同）
    assert _text(with_tail[1]) == _text(compose_memory_blocks([M("user", "Q1")])[0]) == "Q: Q1"


def test_empty_input_yields_no_blocks() -> None:
    """空输入 ⇒ 空元组。**调用方（`apply_batch`）在此之前就该抛错**——见那里的 docstring。"""
    assert compose_memory_blocks([]) == ()


# ── 判据与编码器的单点纪律（原有断言，跟着规则一起搬过来）──────────────


def test_is_user_only_judges_role(M: Callable[..., Message]) -> None:
    """判据只做一种判断：`role` 是不是 `user`。**大小写敏感、不做归一化**。"""
    assert is_user(M("user", "x")) is True
    assert is_user(M("User", "x")) is False  # 契约里就是小写；不猜
    assert is_user(M("assistant", "x")) is False


def test_join_question_has_no_role_markers(M: Callable[..., Message]) -> None:
    """`question` 侧**不加 role 标记**：role 均一，而拼接要能把被切开的一条原消息拼回去。"""
    assert join_question([M("user", "第一段"), M("user", "第二段")]) == "第一段\n第二段"


def test_encode_answer_always_marks_role(M: Callable[..., Message]) -> None:
    """`answer` 侧**无条件加** `[role]` 标记——单个 TEXT 无法在渲染时重建消息边界。

    ⚠ 无条件（而不是"多条时才加"）是刻意的：形状统一，读的人不必分情况。
    """
    assert encode_answer([M("assistant", "只有一条")]) == "[assistant] 只有一条"


def test_message_strips_surrounding_whitespace() -> None:
    """首尾空白在**进入核心之前**去掉：AML 只做 `"\\n".join(...)`、不插分隔符（§11.3）。"""
    assert Message(role="user", content="  两侧有空白  ").content == "两侧有空白"


def test_message_rejects_empty_role() -> None:
    with pytest.raises(ValueError):
        Message(role="", content="x")
