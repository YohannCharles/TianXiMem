"""§6.2 的配对判据（三种真实情况 + 未知 role）+ §6.5 第 3 步的纯逻辑。

对应 [`../tests/CLAUDE.md`](../tests/CLAUDE.md) 一、配对与续接。
全部是纯函数测试：**不碰数据库**，因此失败时一定是配对逻辑本身的问题。

⚠ **配对判据与 §6.2 的字面有一处有意差异**：连续 user 消息**并入同一个 `question`**，
而不是把前一个对关成"有问无答"（理由与影响面见 D20）。本文件里凡涉及"连续 user"的
用例都在钉这条差异。
"""

from __future__ import annotations

from collections.abc import Callable

from tianxi_am.pairing.pairing import (
    STATUS_COMPLETE,
    STATUS_PENDING,
    BatchLimits,
    Message,
    encode_answer,
    is_user,
    join_question,
    plan_batch,
)


def _plan(messages, *, next_idx=0, open_pair_id=None, open_pair_answer=None, limits=None):
    """`open_pair_answer=None` = 那个既有对的 `answer` 为空 ⇒ 它的 `question` 还在写。

    想构造 §6.5 的"pending 对（已有答话、等后续）"，就**必须**显式给一个非 None 的
    `open_pair_answer`——否则这条 user 消息会被当成 question 的续写而不是关闭信号。
    """
    return plan_batch(
        messages,
        next_idx=next_idx,
        open_pair_id=open_pair_id,
        open_pair_answer=open_pair_answer,
        limits=limits,
    )


# ── 判据本身 ────────────────────────────────────────────────────────────


def test_is_user_only_judges_role(M: Callable[..., Message]) -> None:
    """唯一判据：`role == "user"`。**不枚举白名单。**"""
    assert is_user(M("user", "x"))
    assert not is_user(M("assistant", "x"))
    assert not is_user(M("system", "x"))
    assert not is_user(M("tool_result", "x"))
    # 大小写/空白不改变判据（role 原样比较——AML 的取值域没有文档）
    assert not is_user(M("User", "x"))


def test_status_literals_match_ddl() -> None:
    """status 的两个字面量与 §6.1 的 DDL 注释一致（防拼写漂移）。"""
    assert STATUS_COMPLETE == "complete"
    assert STATUS_PENDING == "pending"


# ── 三种真实情况（pairing/CLAUDE.md 的表）───────────────────────────────


def test_consecutive_user_messages_merge_into_one_question(
    M: Callable[..., Message],
) -> None:
    """连续多条 user 消息 ⇒ **并成同一个 `question`**（D20 对 §6.2 字面的修正）。

    这条形状的真实来源：AML 可能把**一条超长 user 消息**按句边界物理切成多条，
    于是 `q1 q2 q3 a` 会被当成"连续 user"。它逻辑上仍是一问一答。
    """
    plan = _plan([M("user", "Q1", 100), M("user", "Q2", 200), M("user", "Q3", 300)])

    assert len(plan.new_pairs) == 1
    pair = plan.new_pairs[0]
    assert pair.question == "Q1\nQ2\nQ3"
    assert pair.answer is None
    assert pair.event_time == 100  # 取该对【首条消息】的 timestamp
    assert pair.status == STATUS_COMPLETE  # 两限未命中 ⇒ session 结束


def test_consecutive_users_then_assistant_is_one_pair(M: Callable[..., Message]) -> None:
    """`q1 q2 q3 a` ⇒ **一个** `(Q1\\nQ2\\nQ3, A)`，而不是三个对。"""
    plan = _plan([M("user", "Q1"), M("user", "Q2"), M("user", "Q3"), M("assistant", "A")])

    assert len(plan.new_pairs) == 1
    pair = plan.new_pairs[0]
    assert pair.question == "Q1\nQ2\nQ3"
    assert pair.answer == "[assistant] A"


def test_user_after_answer_still_opens_a_new_pair(M: Callable[..., Message]) -> None:
    """**防过度合并**：前一个对已经有 `answer` ⇒ 下一条 user 必须开**新**对。

    这是"合并"与"吞掉整个 session"的分界线——少了这条判据，一个 session 会被并成一对。
    """
    plan = _plan(
        [
            M("user", "Q1"),
            M("assistant", "A1"),
            M("user", "Q2"),
            M("assistant", "A2"),
        ]
    )

    assert len(plan.new_pairs) == 2
    assert [(p.question, p.answer) for p in plan.new_pairs] == [
        ("Q1", "[assistant] A1"),
        ("Q2", "[assistant] A2"),
    ]


def test_join_question_has_no_role_markers(M: Callable[..., Message]) -> None:
    """`question` 的拼接**不加 role 标记**——与 `encode_answer` 相反。

    标记会一并进 embedding（§7.2 同一份渲染），而这段文本是用户的原始发言；
    要让"被切开的一条原消息"能逐字拼回去，就不能插入源文本里没有的 token。
    """
    assert join_question([M("user", "第一句。"), M("user", "第二句。")]) == "第一句。\n第二句。"
    # 空 content 也不丢（与 encode_answer 的防御口径一致：宁可留一个空行）
    assert join_question([M("user", "a"), M("user", "   ")]) == "a\n"


def test_user_then_multiple_assistants_all_in_one_pair(
    M: Callable[..., Message],
) -> None:
    """一条 user 后跟多条 assistant（工具调用等）⇒ 全部归入该对，且每条带 role 标记。"""
    plan = _plan(
        [
            M("user", "火车几点开？", 100),
            M("assistant", "我查一下。", 110),
            M("tool_result", '{"train": "09:42"}', 120),
            M("assistant", "09:42。", 130),
        ]
    )

    assert len(plan.new_pairs) == 1
    pair = plan.new_pairs[0]
    assert pair.question == "火车几点开？"
    assert pair.answer == (
        '[assistant] 我查一下。\n[tool_result] {"train": "09:42"}\n[assistant] 09:42。'
    )
    # event_time 取【该对首条消息】的 timestamp
    assert pair.event_time == 100


def test_session_starts_with_assistant_makes_questionless_pair(
    M: Callable[..., Message],
) -> None:
    """session 以非 user 消息开头 ⇒ `question` 为空的对（"无问的对"）。"""
    plan = _plan([M("assistant", "先说明一下背景。", 50), M("user", "Q1", 100)])

    assert len(plan.new_pairs) == 2
    q0, q1 = plan.new_pairs
    assert q0.question is None
    assert q0.answer == "[assistant] 先说明一下背景。"
    assert q0.event_time == 50
    assert q0.status == STATUS_COMPLETE  # 被 Q1 关闭
    assert q1.question == "Q1"


def test_whole_batch_without_user_is_one_questionless_pair(
    M: Callable[..., Message],
) -> None:
    """整批都没有 user 消息，且不存在 pending ⇒ 仍然只建一个无问的对。"""
    plan = _plan([M("assistant", "a"), M("assistant", "b")])

    assert len(plan.new_pairs) == 1
    assert plan.new_pairs[0].question is None
    assert plan.new_pairs[0].answer == "[assistant] a\n[assistant] b"


# ── 未知 role 不得丢消息 ────────────────────────────────────────────────


def test_unknown_role_is_not_dropped(M: Callable[..., Message]) -> None:
    """**不得丢消息**：判据只依赖"是不是 user"，未知 role 一律归入当前对。

    这条是"枚举 role 白名单"那个 bug 的守卫——白名单会在这里静默丢消息。
    """
    plan = _plan(
        [
            M("user", "Q"),
            M("weird_role_from_aml", "x"),
            M("又一种没见过的", "y"),
        ]
    )

    answer = plan.new_pairs[0].answer
    assert answer is not None
    assert "x" in answer
    assert "y" in answer
    assert answer.startswith("[weird_role_from_aml] x")
    # 两条非 user 消息一条不少
    assert len(answer.splitlines()) == 2


def test_empty_content_is_kept_as_marker_only(M: Callable[..., Message]) -> None:
    """空 content **不丢弃**——丢它是"静默丢消息"。

    §2.1 保证 content 非空，这里是防御分支（多模态消息可能只有图没有字）。
    """
    assert encode_answer([M("assistant", "   ")]) == "[assistant]"
    # 首尾空白在进入核心之前就被去掉（AML 只做 "\\n".join，不插分隔符）
    assert encode_answer([M("assistant", "  hi  ")]) == "[assistant] hi"


# ── pair_idx：连续 + 不重置 ─────────────────────────────────────────────


def test_pair_idx_starts_from_next_idx_not_zero(M: Callable[..., Message]) -> None:
    """**不重置 `pair_idx`**：新批次接着上一个批次数。

    从 0 重开会与既有行撞 `id`（`id` 位置派生），而写入是 upsert ⇒
    **静默覆盖上一批的数据**。
    """
    plan = _plan(
        [M("user", "Q"), M("assistant", "A"), M("user", "Q2"), M("assistant", "A2")],
        next_idx=7,
    )

    assert [p.pair_idx for p in plan.new_pairs] == [7, 8]


def test_pair_idx_is_contiguous_across_a_batch(M: Callable[..., Message]) -> None:
    plan = _plan(
        [
            M("user", "a"),
            M("assistant", "a"),
            M("user", "b"),
            M("assistant", "b"),
            M("user", "c"),
            M("assistant", "c"),
            M("user", "d"),
            M("assistant", "d"),
        ],
        next_idx=3,
    )
    idxs = [p.pair_idx for p in plan.new_pairs]
    assert idxs == [3, 4, 5, 6]
    assert idxs == list(range(idxs[0], idxs[0] + len(idxs)))  # 无空洞


# ── §6.5：pending 判定（上限规则）───────────────────────────────────────


def test_defaults_match_spec() -> None:
    """默认上限必须与 §6.5 一致：**20 条消息 或 2,000 词**。"""
    limits = BatchLimits()
    assert limits.max_messages == 20
    assert limits.max_words == 2000


def test_last_pair_is_pending_when_message_limit_hit(
    M: Callable[..., Message], limits_two: BatchLimits
) -> None:
    """命中消息上限 ⇒ 本批最后一对是 `pending`（AML 还会继续喂）。"""
    plan = _plan([M("user", "Q1"), M("assistant", "A1")], limits=limits_two)

    assert plan.hit_message_limit
    assert not plan.session_ended
    assert plan.new_pairs[-1].status == STATUS_PENDING


def test_last_pair_is_complete_when_no_limit_hit(
    M: Callable[..., Message], limits_two: BatchLimits
) -> None:
    """两限都未命中 ⇒ session 已结束 ⇒ 最后一对是 `complete`。"""
    plan = _plan([M("user", "Q1")], limits=limits_two)

    assert not plan.hit_message_limit
    assert plan.session_ended
    assert plan.new_pairs[-1].status == STATUS_COMPLETE


def test_last_pair_is_pending_when_word_limit_hit(M: Callable[..., Message]) -> None:
    """词数上限同样会让最后一对变 pending。"""
    limits = BatchLimits(max_messages=99, max_words=3)
    plan = _plan([M("user", "one two three four")], limits=limits)

    assert plan.hit_word_limit
    assert not plan.hit_message_limit
    assert plan.new_pairs[-1].status == STATUS_PENDING


def test_only_the_last_pair_can_be_pending(M: Callable[..., Message]) -> None:
    """中间的对一律 `complete`——只有最后一对可能是 pending。"""
    plan = _plan(
        [
            M("user", "a"),
            M("assistant", "b"),
            M("user", "c"),
            M("assistant", "d"),
        ],
        limits=BatchLimits(max_messages=3, max_words=1000),
    )

    assert len(plan.new_pairs) == 2
    assert plan.new_pairs[0].status == STATUS_COMPLETE
    assert plan.new_pairs[1].status == STATUS_PENDING


# ── §6.5 的 3a′ / 3a / 3b / 3d（纯逻辑侧）──────────────────────────────
#
# 下面凡是构造 §6.5 那个"pending 对"的用例，都显式给 `open_pair_answer`——
# 它表示"这个对已经有答话了"，而这正是 3b 关闭它、3a 追加答话的前提。


def test_3a_appends_leading_messages_to_existing_pending(
    M: Callable[..., Message], limits_small: BatchLimits
) -> None:
    """3a：本批开头、首个 user 消息之前的消息 ⇒ 追加到既有 pending。"""
    plan = _plan(
        [M("assistant", "续写的回复"), M("user", "Q2")],
        next_idx=1,
        open_pair_id="P0",
        open_pair_answer="[assistant] 前面那句",
        limits=limits_small,
    )

    assert plan.resume is not None
    assert plan.resume.append_answer == "[assistant] 续写的回复"
    assert plan.resume.append_question is None
    assert plan.resume.close is True  # 3b：首个 user 消息把它关掉
    assert [p.question for p in plan.new_pairs] == ["Q2"]


def test_3a_prime_merges_leading_users_into_the_open_question(
    M: Callable[..., Message], limits_small: BatchLimits
) -> None:
    """**3a′（本轮新增）**：那个对的 `answer` 还是空的 ⇒ 开头这些 user 是它 question 的续写。

    这是"AML 把一条超长 user 消息切成几段、每段各占一个 Add"的那条路径。
    **不关对、不建新对**——这正是旧规则做不到的地方。
    """
    plan = _plan(
        [M("user", "Q2"), M("user", "Q3"), M("assistant", "A")],
        next_idx=1,
        open_pair_id="P0",
        open_pair_answer=None,  # ← 空 answer：question 还在写
        limits=limits_small,
    )

    assert plan.resume is not None
    assert plan.resume.append_question == "Q2\nQ3"
    assert plan.resume.append_answer == "[assistant] A"
    assert plan.resume.close is False  # 没有关对——它们都还在同一个 question 里
    assert plan.new_pairs == ()


def test_3b_closes_pending_with_nothing_to_fill(
    M: Callable[..., Message], limits_small: BatchLimits
) -> None:
    """3b：本批直接以 user 开头、**且那个对已有 answer** ⇒ 它被关掉，没有内容可填。

    ⚠ 这是最容易漏的一步。漏了它会永久挂在 pending，而现象会伪装成
    AML 的切分行为，让人跑去改配对规则——方向完全错了。
    """
    plan = _plan(
        [M("user", "Q2")],
        next_idx=1,
        open_pair_id="P0",
        open_pair_answer="[assistant] 前面那句",
        limits=limits_small,
    )

    assert plan.resume is not None
    assert plan.resume.append_question is None
    assert plan.resume.append_answer is None
    assert plan.resume.close is True
    # 被 3b 关闭的对不是"最后一带" ⇒ 不能在这里再给它定状态
    assert plan.resume.final_status is None


def test_user_does_not_close_a_pair_that_has_no_answer_yet(
    M: Callable[..., Message], limits_small: BatchLimits
) -> None:
    """**3b 的反面**：`answer` 还空着时，user 消息**不得**关闭那个对。

    少了这条判据，一条被切开的 user 消息会把前一段关成"有问无答"的对——
    正是本轮要消除的失败形态。
    """
    plan = _plan(
        [M("user", "Q2")],
        next_idx=1,
        open_pair_id="P0",
        open_pair_answer=None,
        limits=limits_small,
    )

    assert plan.resume is not None
    assert plan.resume.close is False
    assert plan.resume.append_question == "Q2"
    # 它是最后一带 ⇒ 仍然由 3d 定状态
    assert plan.resume.final_status == STATUS_COMPLETE


def test_pure_continuation_batch_applies_3d_to_pending(
    M: Callable[..., Message], limits_small: BatchLimits
) -> None:
    """纯接续批（零条 user 消息）也要走 3d：被追加的那个 pending 就是最后一带。"""
    plan = _plan(
        [M("assistant", "补上的一句")],
        next_idx=1,
        open_pair_id="P0",
        open_pair_answer="[assistant] 前面那句",
        limits=limits_small,
    )

    assert plan.new_pairs == ()
    assert plan.resume is not None
    assert plan.resume.append_answer == "[assistant] 补上的一句"
    assert plan.resume.close is False
    assert plan.resume.final_status == STATUS_COMPLETE  # 本批未命中上限 ⇒ session 结束


def test_pure_continuation_batch_that_hits_limit_stays_pending(
    M: Callable[..., Message],
) -> None:
    """纯接续批**命中上限**时不能标 complete——AML 还会继续喂。"""
    plan = _plan(
        [M("assistant", "补上的一句")],
        next_idx=1,
        open_pair_id="P0",
        open_pair_answer=None,  # 一个还没答过的对，等来第一条答话
        limits=BatchLimits(max_messages=1, max_words=1000),
    )

    assert plan.resume is not None
    assert plan.resume.append_answer == "[assistant] 补上的一句"
    assert plan.resume.final_status == STATUS_PENDING
