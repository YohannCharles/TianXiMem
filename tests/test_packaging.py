"""`rank/packaging.py` —— **段级打包 + 双预算里的 token 那一半**（§11.3 / §6.4）。

⚠ 这里测的是"**段进来之后，响应怎么生成**"。段的构造（扩窗、合并、段内/段间顺序）
在 [`test_neighbor.py`](./test_neighbor.py)——本文件**直接构造 `ContextSegment`**，
好把 `top_k`、`score`、`created_at`、预算这几件事**孤立**出来测。

> **与"取正文"那一层的分界**（`src/tianxi_am/rank/CLAUDE.md`）：正文由段带进来，
> `packaging.py` **一个数据库查询都不做**——`_segment()` 这个 helper 就是那个边界的体现。
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tianxi_am.common.render import render, render_segment
from tianxi_am.common.tokens import TokenCounter
from tianxi_am.rank import (
    ContextSegment,
    ResponseItem,
    day_granularity,
    package,
    placeholder_score,
)

#: 东八区——**只用来演示"如果按本地时区算，这个时间点会落到哪一天"**，
#: 证明跨日的时间点确实能暴露口径（**不再传给任何被调函数**，见那条用例）。
CST = timezone(timedelta(hours=8))

#: 2023-05-08 23:30 UTC —— 一个**跨日**的时间点，用来暴露时区口径。
#: UTC 口径 ⇒ `2023-05-08`；东八区口径 ⇒ `2023-05-09`。
NEAR_MIDNIGHT_UTC = int(datetime(2023, 5, 8, 23, 30, tzinfo=UTC).timestamp() * 1000)


def _segment(
    counter: TokenCounter,
    *,
    anchor: str,
    content: str,
    best_rank: int = 0,
    event_time: int | None = None,
    source: Sequence[str] | None = None,
) -> ContextSegment:
    """手搓一个段。**`token_count` 走真的计数器**（那正是被测对象的一部分）。"""
    return ContextSegment(
        source_memory_ids=tuple(source) if source is not None else (anchor,),
        user_id="u1",
        session_id="s1",
        start_seq=0,
        end_seq=0,
        anchor_memory_id=anchor,
        best_rank=best_rank,
        anchor_event_time=event_time,
        content=content,
        token_count=counter.count(content),
        rerank_member_count=1,
    )


def _line(counter: TokenCounter, count: int, *, size: int = 1) -> list[ContextSegment]:
    """`count` 个**互不相干**的段，内容依次为 `"c0"`、`"c1"`…（`best_rank` 同步递增）。

    `size` 控制每段正文长度，用来把预算卡在想要的位置上。
    """
    return [
        _segment(counter, anchor=f"m{i}", content=f"c{i}" * size, best_rank=i) for i in range(count)
    ]


# ── content：段带进来，packaging 不自己拼 ───────────────────────────────


def test_content_is_taken_from_the_segment_verbatim(counter: TokenCounter) -> None:
    """**同一份渲染**——它既是 embedding 的输入，也是模型读到的东西（不变式 I1）。

    ⚠ 正文是在 `neighbor._build_segment` 里用 `common/render` 拼好、**由段带进来的**；
    打包层再拼一遍就会让"检索命中的是什么"与"模型读到的是什么"漂移，**而且不报错**。
    所以这里断言的是"原样搬过去"，不是"packaging 拼得对"。
    """
    content = render_segment([render("火车几点开？", "[assistant] 09:42。")])
    got = package(
        [_segment(counter, anchor="m0", content=content)],
        top_k=5,
        counter=counter,
        max_tokens=1000,
    )

    assert got.items[0].content == content
    assert got.items[0].content == "Q: 火车几点开？\nA: [assistant] 09:42。"


def test_multiple_pairs_in_one_segment_stay_one_item(counter: TokenCounter) -> None:
    """**段是原子单位**：段里几对，输出里就是**一项**（§11.2）。

    这一条同时是"没有在打包阶段把段拆回单条"的结构性证据。
    """
    content = render_segment([render("q0", "a0"), render("q1", "a1")])
    seg = _segment(counter, anchor="m1", content=content, source=["m0", "m1"])

    got = package([seg], top_k=5, counter=counter, max_tokens=1000)

    assert got.count == 1  # 不是 2
    assert got.items[0].content == "Q: q0\nA: a0\nQ: q1\nA: a1"
    assert got.items[0].content.count("\n") == 3  # 段内分隔符在，段没被拆


def test_packaging_does_not_touch_the_source() -> None:
    """**打包层一个数据库查询都不做**——正文随段进来。

    ⚠ 这是签名上的守门：`package()` 一旦重新长出 `store=` 参数，
    就说明有人把"取正文"又搬回来了（那会让 `content` 有两处实现，
    而两处不一致**不会报错**）。
    """
    params = inspect.signature(package).parameters
    assert "store" not in params
    # 同理：渲染器**不能**是参数——`common/render` 是唯一实现，不是可替换件
    assert "renderer" not in params and "render" not in params


# ── created_at：日粒度、NULL → ""、时区口径显式 ─────────────────────────


def test_created_at_is_day_granularity(counter: TokenCounter) -> None:
    """**只给到日粒度**——粒度会被模型看见，秒级会诱发它按秒级回答（§11.3）。"""
    seg = _segment(counter, anchor="m0", content="c", event_time=NEAR_MIDNIGHT_UTC)
    created_at = package([seg], top_k=5, counter=counter, max_tokens=1000).items[0].created_at

    assert created_at == "2023-05-08"
    assert len(created_at) == 10  # 没有时间部分
    assert ":" not in created_at
    assert created_at != str(NEAR_MIDNIGHT_UTC)  # 不是裸毫秒


def test_created_at_is_empty_when_event_time_is_null(counter: TokenCounter) -> None:
    """`event_time` 为 NULL ⇒ **空串**（有定义的降级路径，§11.3）。"""
    seg = _segment(counter, anchor="m0", content="c", event_time=None)
    assert package([seg], top_k=5, counter=counter, max_tokens=1000).items[0].created_at == ""


def test_created_at_does_not_fall_back_to_write_time(counter: TokenCounter) -> None:
    """**绝不拿 Add 的到达时间兜底**——那是"何时写入"而不是"何时发生"。

    兜底会给模型**错误信息**，而且比空串更坏：空串会退化成 `- {text}`，
    错误日期则会变成一条看起来可信的假时间。
    """
    seg = _segment(counter, anchor="m0", content="c", event_time=None)
    created_at = package([seg], top_k=5, counter=counter, max_tokens=1000).items[0].created_at

    assert created_at == ""
    # 今天（= 写入时间）的日期绝不能出现
    assert datetime.now(UTC).strftime("%Y-%m-%d") not in created_at


def test_created_at_is_utc_regardless_of_the_machine_timezone(counter: TokenCounter) -> None:
    """时区口径**固定 UTC，没有参数**——同一毫秒在任何机器上都得到同一个日期。

    ⚠ 契约没规定时区（`docs/contract.md` §3 只说日粒度）。固定 UTC 的理由是
    **与机器无关**：用本地时区会让同一份数据在不同机器上差一天，而本项目最怕的就是
    不可复现。

    ⚠ **刻意不做成配置项**，见 `packaging.py` 的 `UTC_ONLY` 注释：
    一个"可以随手改的 `created_at_tz`"会与**加载层**（合成 `event_time` 的地方）脱钩——
    改了它日期整体偏一天，而**没有任何东西会报错**。
    要改口径就两边一起改，那是一次需要重新验证的决定，不是一次配置调整。

    ⚠ **它反过来也钉住了一件事**：`package()` / `day_granularity()` 的签名里
    **不该再出现任何时区参数**——本用例是那件事的守门人。
    """
    seg = _segment(counter, anchor="m0", content="c", event_time=NEAR_MIDNIGHT_UTC)

    # NEAR_MIDNIGHT_UTC 是 UTC 当天的 23:30 ⇒ UTC 口径下是 05-08（在东八区会是 05-09）
    got = package([seg], top_k=5, counter=counter, max_tokens=1000)
    assert got.items[0].created_at == "2023-05-08"
    assert day_granularity(NEAR_MIDNIGHT_UTC) == "2023-05-08"

    # 证明这个时间点**真的**能暴露口径（否则上面那条断言是空过的）
    east8 = datetime.fromtimestamp(NEAR_MIDNIGHT_UTC / 1000, tz=CST).strftime("%Y-%m-%d")
    assert east8 == "2023-05-09"

    # 签名守门：不接受 tz / timezone 之类的参数
    params = inspect.signature(day_granularity).parameters
    assert "tz" not in params and "timezone" not in params
    assert "tz" not in inspect.signature(package).parameters


def test_created_at_comes_from_the_anchor_not_the_first_member(counter: TokenCounter) -> None:
    """`created_at` 跟着**锚点**走——段里第一条可能是扩窗补进来的邻居（§11.2）。

    照第一条取会得到一个"这段对话里其实没被选中"的时间。
    """
    seg = _segment(
        counter,
        anchor="m2",
        content="c",
        event_time=NEAR_MIDNIGHT_UTC,
        source=["m1", "m2", "m3"],
    )
    assert package([seg], top_k=5, counter=counter, max_tokens=1000).items[0].created_at == (
        "2023-05-08"
    )


# ── score：1/(rank+1)，且**不是**融合分数 ───────────────────────────────


def test_score_is_reciprocal_rank(counter: TokenCounter) -> None:
    """`score = 1/(rank+1)`——单调递减的占位值（§11.3）。

    它无论 AML 是否按 `score` 重排，**顺序都不变**，所以是最安全的一侧。
    """
    got = package(_line(counter, 4), top_k=5, counter=counter, max_tokens=1000)

    assert [item.score for item in got.items] == [1.0, 0.5, 1 / 3, 0.25]


def test_score_is_strictly_decreasing(counter: TokenCounter) -> None:
    scores = [
        i.score for i in package(_line(counter, 5), top_k=9, counter=counter, max_tokens=1000).items
    ]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_score_follows_output_position_not_segment_rank(counter: TokenCounter) -> None:
    """`score` 按**输出位置**重算，不是照抄段的名次。

    本用例里段的 `best_rank` 是 5、9、12（模拟"前面的段被 `top_k` 或预算砍掉了"），
    输出里的分数仍必须是 `1.0, 0.5, 1/3`——照抄名次会让分数出现空洞，
    而它必须单调递减、且第一位就是 `1.0`。
    """
    segments = [
        _segment(counter, anchor="m5", content="a", best_rank=5),
        _segment(counter, anchor="m9", content="b", best_rank=9),
        _segment(counter, anchor="m12", content="c", best_rank=12),
    ]
    got = package(segments, top_k=5, counter=counter, max_tokens=1000)

    assert [item.score for item in got.items] == [1.0, 0.5, 1 / 3]
    assert [item.id for item in got.items] == ["m5", "m9", "m12"]


def test_placeholder_score_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="rank 不得为负"):
        placeholder_score(-1)


def test_response_item_has_exactly_the_contract_fields(counter: TokenCounter) -> None:
    """`data[]` 的一项**就是 §2.1 的四个字段，一个不多**。

    ⚠ 这条同时是"**没有泄漏 RRF 分数**"的结构性证据：`ResponseItem` 里那个 `score`
    是 `1/(rank+1)` 现算的，而上游 `Candidate` **根本没有**融合分数字段
    （见 `tests/test_retrieve.py::test_candidate_carries_no_score_field`）。
    """
    assert {f.name for f in fields(ResponseItem)} == {"id", "content", "created_at", "score"}


def test_no_internal_fields_leak_into_the_item(counter: TokenCounter) -> None:
    """`session_id` / 位置 / `status` / `best_rank` 都不该出现在响应项里。

    ⚠ 位置自 **D25** 起是**两个**字段（`chunk_ordinal` + `local_index`）⇒ 两个都要挡；
    只挡旧的 `pair_idx` 会**静默放行**它们（那个名字已经不存在了，断言永远为真）。
    """
    seg = _segment(counter, anchor="m0", content="c")
    item = package([seg], top_k=5, counter=counter, max_tokens=1000).items[0]
    dumped = item.__dict__ if hasattr(item, "__dict__") else {}
    assert "session_id" not in dumped
    assert "pair_idx" not in dumped  # 旧名，D25 后不该再有人加回来
    assert "chunk_ordinal" not in dumped
    assert "local_index" not in dumped
    assert "seq" not in dumped
    assert "status" not in dumped
    assert "best_rank" not in dumped


def test_id_is_the_anchor_of_the_segment(counter: TokenCounter) -> None:
    """`id` = **段的锚点**，不是段里第一条（那条可能只是扩窗补的）。"""
    seg = _segment(counter, anchor="m2", content="c", source=["m1", "m2", "m3"])
    assert package([seg], top_k=5, counter=counter, max_tokens=1000).items[0].id == "m2"


# ── top_k：本函数是最终数量的守门人 ─────────────────────────────────────


def test_top_k_zero_returns_empty(counter: TokenCounter) -> None:
    """`top_k = 0` ⇒ 空响应，**且不数 token**（结果必然是空的）。"""
    got = package(_line(counter, 3), top_k=0, counter=counter, max_tokens=1000)
    assert got.items == ()
    assert got.count == 0
    assert got.output_tokens == 0


def test_top_k_one(counter: TokenCounter) -> None:
    got = package(_line(counter, 3), top_k=1, counter=counter, max_tokens=1000)
    assert got.count == 1
    assert got.items[0].id == "m0"


def test_upstream_giving_more_than_top_k_is_truncated(counter: TokenCounter) -> None:
    """**即使上游给出更多段，也必须在这里保证 ≤ `top_k`**（§2.2 精确计数）。"""
    segments = _line(counter, 6)
    for top_k in (1, 2, 3, 5):
        got = package(segments, top_k=top_k, counter=counter, max_tokens=1000)
        assert got.count == top_k
        assert got.count <= top_k
        assert got.truncated_by_top_k is True


def test_top_k_larger_than_available_does_not_pad(counter: TokenCounter) -> None:
    """**绝不为凑满 `top_k` 复制或补造结果。**"""
    got = package(_line(counter, 3), top_k=10, counter=counter, max_tokens=1000)

    assert got.count == 3  # 不是 10
    assert len({item.id for item in got.items}) == 3  # 没有重复
    assert [item.id for item in got.items] == ["m0", "m1", "m2"]
    assert got.truncated_by_top_k is False


def test_negative_top_k_returns_empty(counter: TokenCounter) -> None:
    assert package(_line(counter, 1), top_k=-1, counter=counter, max_tokens=1000).count == 0


def test_empty_segments_returns_empty_response(counter: TokenCounter) -> None:
    got = package([], top_k=5, counter=counter, max_tokens=1000)
    assert got.items == ()
    assert got.dropped_missing == 0
    assert got.considered_segments == 0
    assert got.output_tokens == 0


# ── 顺序：跟段给的顺序，不重排 ─────────────────────────────────────────


def test_order_follows_the_given_segment_order(counter: TokenCounter) -> None:
    """输出的顺序 = **段给来的顺序**（那是 `merge_segments` 按 `best_rank` 排好的）。

    ⚠ 打包层**不重新排序**：段的先后由 `neighbor.merge_segments` 拥有（§11.2 的组间顺序）。
    这里反过来给一批乱序的段，就是为了证明打包层**照单全收**——
    如果它偷偷按 `best_rank` 再排一次，两处排序就会各自演化。
    """
    segments = [
        _segment(counter, anchor="mC", content="c", best_rank=2),
        _segment(counter, anchor="mA", content="a", best_rank=0),
        _segment(counter, anchor="mB", content="b", best_rank=1),
    ]
    got = package(segments, top_k=5, counter=counter, max_tokens=1000)

    assert [item.id for item in got.items] == ["mC", "mA", "mB"]
    assert [item.content for item in got.items] == ["c", "a", "b"]


# ── 真源缺行：不静默（计数由扩窗阶段给出，这里只负责带出去）──────────────


def test_dropped_missing_is_passed_through(counter: TokenCounter) -> None:
    """真源缺行的条数**从扩窗阶段传进来、原样带出去**。

    正常情况下不该发生（Qdrant 是派生索引、`id` 与 SQLite 一一对应）。
    **非 0 说明索引与真源脱钩了**——所以这个数要能被调用方看见，而不是静默少一项。
    """
    got = package(_line(counter, 2), top_k=5, counter=counter, max_tokens=1000, dropped_missing=3)
    assert got.dropped_missing == 3
    assert got.count == 2  # 缺行**不占**输出位置


# ── token 预算（§6.4）：段是原子单位，装不下就停 ───────────────────────


def test_budget_stops_at_the_first_segment_that_does_not_fit(counter: TokenCounter) -> None:
    """**装不下就停**——不跳过它去塞后面的（那会让 `score` 与名次脱钩）。"""
    segments = _line(counter, 4, size=10)  # 每段 20 字符
    # 20 + 1(分隔) + 20 = 41 ≤ 45，再加第三段就是 62 > 45 ⇒ 停在第 2 段
    got = package(segments, top_k=10, counter=counter, max_tokens=45)

    assert got.count == 2
    assert [item.id for item in got.items] == ["m0", "m1"]
    assert got.truncated_by_budget is True
    assert got.truncated_by_top_k is False
    assert got.considered_segments == 4


def test_budget_never_produces_half_a_segment(counter: TokenCounter) -> None:
    """**超预算不产生半个段**：段要么整段进去，要么整段不进。

    截半个段会让模型读到一段**缺了一环的连续对话**，而 `content` 里看不出来（§11.2）。
    """
    segments = _line(counter, 3, size=30)  # 每段 60 字符
    # 只装得下第一段（60）+ 分隔（1）；第二段要 62，放不下
    got = package(segments, top_k=10, counter=counter, max_tokens=70)

    assert got.count == 1
    assert got.items[0].content == segments[0].content  # 一个字都没被裁
    assert got.items[0].content == "c0" * 30
    assert got.output_tokens == 60


def test_budget_uses_the_real_joined_string(counter: TokenCounter) -> None:
    """预算对**最终拼接好的真实字符串**成立——不是各段数字的和（§6.4）。

    ⚠ 分隔符**也要算**。漏掉它会让实际输出比预算多出 `(段数 - 1)` 个 token，
    而 100 个段时那就是 99 个——**正好把前缀截断点推到一段证据的中间**。
    """
    segments = _line(counter, 3, size=5)  # 每段 10 字符 + 1 分隔 = 11
    got = package(segments, top_k=10, counter=counter, max_tokens=100)

    joined = "\n".join(s.content for s in segments)
    assert got.output_tokens == counter.count(joined) == 32  # 10+1+10+1+10
    assert got.output_tokens == len(joined)
    assert got.output_tokens <= 100

    # 反过来：如果谁把分隔符漏了，得到的会是 30 —— 本断言就是那道防线
    assert got.output_tokens != sum(s.token_count for s in segments)


def test_budget_boundary_is_exact_without_a_separator(counter: TokenCounter) -> None:
    """只有一个段时，分隔符**不该**被计入（否则第一段就白扣一个 token）。"""
    segments = _line(counter, 1, size=10)  # 20 字符
    got = package(segments, top_k=10, counter=counter, max_tokens=20)

    assert got.count == 1
    assert got.truncated_by_budget is False
    assert got.output_tokens == 20


def test_budget_zero_yields_nothing_but_still_counts(counter: TokenCounter) -> None:
    """预算为 0 ⇒ **一个段都不装**，但"有几段候选"这件事仍要报出来。"""
    got = package(_line(counter, 3), top_k=10, counter=counter, max_tokens=0)

    assert got.count == 0
    assert got.truncated_by_budget is True
    assert got.considered_segments == 3


def test_top_k_is_applied_before_the_budget(counter: TokenCounter) -> None:
    """**`top_k` 先于预算**：被 `top_k` 切掉的段不该再占预算的判断。

    顺序反过来的话，预算会先在一批**根本不会被返回**的段上做判断——
    一旦它们在预算里把位置占满，实际返回的段数就会**少于该有的**。
    """
    segments = _line(counter, 4, size=10)  # 每段 20 字符
    # top_k=2 ⇒ 只看前两段：20 + 1 + 20 = 41 ≤ 41 ⇒ 两个都进
    got = package(segments, top_k=2, counter=counter, max_tokens=41)

    assert got.count == 2
    assert got.truncated_by_top_k is True
    assert got.truncated_by_budget is False


def test_budget_counts_only_what_is_actually_returned(counter: TokenCounter) -> None:
    """`output_tokens` 数的是**真正返回的那几个段**，不是全部候选段。"""
    segments = _line(counter, 5, size=4)  # 每段 8 字符
    got = package(segments, top_k=2, counter=counter, max_tokens=1000)

    assert got.count == 2
    assert got.output_tokens == len("c0c0c0c0\nc1c1c1c1") == 17


def test_output_tokens_and_count_never_exceed_the_two_budgets(counter: TokenCounter) -> None:
    """两条上界同时成立：`count <= top_k` 且 `output_tokens <= max_tokens`。

    这条是 §2.2 + §6.4 合起来的守门断言——**任何一条被破都是契约错误**。
    """
    segments = _line(counter, 12, size=7)
    for top_k in (1, 3, 100):
        for max_tokens in (0, 5, 40, 200, 10**6):
            got = package(segments, top_k=top_k, counter=counter, max_tokens=max_tokens)
            assert got.count <= top_k
            assert got.output_tokens <= max_tokens
            assert got.count <= got.considered_segments
