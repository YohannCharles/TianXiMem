"""`rank/packaging.py` —— **Step 1 的最小打包切片**（§11.3）。

⚠ 这里测的是"响应怎么生成"，**不是**最终 pipeline：Rerank / Neighbor Expansion /
双预算截断都还没接（见 `src/tianxi_am/rank/CLAUDE.md` 的两个阶段对照）。
"""

from __future__ import annotations

import inspect
from dataclasses import fields
from datetime import UTC, datetime, timedelta, timezone

import pytest

from tianxi_am.common.render import render
from tianxi_am.rank import (
    ResponseItem,
    day_granularity,
    package,
    placeholder_score,
)
from tianxi_am.retrieve import Candidate
from tianxi_am.store.sqlite_store import SqliteStore

#: 东八区——**只用来演示"如果按本地时区算，这个时间点会落到哪一天"**，
#: 证明跨日的时间点确实能暴露口径（**不再传给任何被调函数**，见那条用例）。
CST = timezone(timedelta(hours=8))

#: 2023-05-08 23:30 UTC —— 一个**跨日**的时间点，用来暴露时区口径。
#: UTC 口径 ⇒ `2023-05-08`；东八区口径 ⇒ `2023-05-09`。
NEAR_MIDNIGHT_UTC = int(datetime(2023, 5, 8, 23, 30, tzinfo=UTC).timestamp() * 1000)


def _seed(
    store: SqliteStore,
    pair_idx: int,
    question: str | None,
    answer: str | None,
    *,
    event_time: int | None = None,
    user_id: str = "u1",
    session_id: str = "s1",
) -> str:
    """落一个对，返回它的 canonical `memory_id`。"""
    with store.transaction() as conn:
        pair = store.insert_pair(
            conn,
            user_id=user_id,
            session_id=session_id,
            pair_idx=pair_idx,
            question=question,
            answer=answer,
            status="complete",
            event_time=event_time,
            request_id="seed",
        )
    return pair.id


def _candidates(*memory_ids: str) -> list[Candidate]:
    return [Candidate(mid, rank) for rank, mid in enumerate(memory_ids)]


# ── content：必须走 common/render（不变式 I1）──────────────────────────


def test_content_comes_from_common_render(store: SqliteStore) -> None:
    """**同一份渲染**——它既是 embedding 的输入，也是模型读到的东西。

    打包层自己拼一遍就会让"检索命中的是什么"与"模型读到的是什么"漂移，
    **而且不报错**。
    """
    mid = _seed(store, 0, "火车几点开？", "[assistant] 09:42。")
    got = package(_candidates(mid), store=store, top_k=5)

    assert got.items[0].content == render("火车几点开？", "[assistant] 09:42。")
    assert got.items[0].content == "Q: 火车几点开？\nA: [assistant] 09:42。"


def test_pending_pair_renders_question_line_only(store: SqliteStore) -> None:
    """`answer` 暂缺的 `pending` 对只输出 `Q:` 那一行（§11.3 的有定义降级）。"""
    mid = _seed(store, 0, "问题", None)
    assert package(_candidates(mid), store=store, top_k=5).items[0].content == "Q: 问题"


def test_questionless_pair_renders_answer_line_only(store: SqliteStore) -> None:
    """`question` 为空的对（§6.2 的"无问的对"）只输出 `A:` 那一行。"""
    mid = _seed(store, 0, None, "[assistant] 先交代背景")
    got = package(_candidates(mid), store=store, top_k=5)
    assert got.items[0].content == "A: [assistant] 先交代背景"


def test_content_has_no_surrounding_whitespace(store: SqliteStore) -> None:
    """AML 只做 `"\\n".join(...)` 拼接——首尾留白会让拼接处粘连（§11.3）。"""
    mid = _seed(store, 0, "  q  ", "  [assistant] a  ")
    content = package(_candidates(mid), store=store, top_k=5).items[0].content
    assert content == content.strip()


# ── created_at：日粒度、NULL → ""、时区口径显式 ─────────────────────────


def test_created_at_is_day_granularity(store: SqliteStore) -> None:
    """**只给到日粒度**——粒度会被模型看见，秒级会诱发它按秒级回答（§11.3）。"""
    mid = _seed(store, 0, "q", "a", event_time=NEAR_MIDNIGHT_UTC)
    created_at = package(_candidates(mid), store=store, top_k=5).items[0].created_at

    assert created_at == "2023-05-08"
    assert len(created_at) == 10  # 没有时间部分
    assert ":" not in created_at
    assert created_at != str(NEAR_MIDNIGHT_UTC)  # 不是裸毫秒


def test_created_at_is_empty_when_event_time_is_null(store: SqliteStore) -> None:
    """`event_time` 为 NULL ⇒ **空串**（有定义的降级路径，§11.3）。"""
    mid = _seed(store, 0, "q", "a", event_time=None)
    assert package(_candidates(mid), store=store, top_k=5).items[0].created_at == ""


def test_created_at_does_not_fall_back_to_write_time(store: SqliteStore) -> None:
    """**绝不拿 Add 的到达时间兜底**——那是"何时写入"而不是"何时发生"。

    兜底会给模型**错误信息**，而且比空串更坏：空串会退化成 `- {text}`，
    错误日期则会变成一条看起来可信的假时间。
    """
    mid = _seed(store, 0, "q", "a", event_time=None)
    created_at = package(_candidates(mid), store=store, top_k=5).items[0].created_at

    assert created_at == ""
    # 今天（= 写入时间）的日期绝不能出现
    assert datetime.now(UTC).strftime("%Y-%m-%d") not in created_at


def test_created_at_is_utc_regardless_of_the_machine_timezone(store: SqliteStore) -> None:
    """时区口径**固定 UTC，没有参数**——同一毫秒在任何机器上都得到同一个日期。

    ⚠ 契约没规定时区（`docs/contract.md` §3 只说日粒度）。固定 UTC 的理由是
    **与机器无关**：用本地时区会让同一份数据在不同机器上差一天，而本项目最怕的就是
    不可复现。

    ⚠ **刻意不做成配置项**（2026-09-24），见 `packaging.py` 的 `UTC_ONLY` 注释：
    一个"可以随手改的 `created_at_tz`"会与**加载层**（合成 `event_time` 的地方）脱钩——
    改了它日期整体偏一天，而**没有任何东西会报错**。
    要改口径就两边一起改，那是一次需要重新验证的决定，不是一次配置调整。

    ⚠ **它反过来也钉住了一件事**：`package()` / `day_granularity()` 的签名里
    **不该再出现任何时区参数**——本用例是那件事的守门人。
    """
    mid = _seed(store, 0, "q", "a", event_time=NEAR_MIDNIGHT_UTC)

    # NEAR_MIDNIGHT_UTC 是 UTC 当天的 23:30 ⇒ UTC 口径下是 05-08（在东八区会是 05-09）
    assert package(_candidates(mid), store=store, top_k=5).items[0].created_at == "2023-05-08"
    assert day_granularity(NEAR_MIDNIGHT_UTC) == "2023-05-08"

    # 证明这个时间点**真的**能暴露口径（否则上面那条断言是空过的）
    east8 = datetime.fromtimestamp(NEAR_MIDNIGHT_UTC / 1000, tz=CST).strftime("%Y-%m-%d")
    assert east8 == "2023-05-09"

    # 签名守门：不接受 tz / timezone 之类的参数
    params = inspect.signature(day_granularity).parameters
    assert "tz" not in params and "timezone" not in params
    assert "tz" not in inspect.signature(package).parameters


# ── score：1/(rank+1)，且**不是**融合分数 ───────────────────────────────


def test_score_is_reciprocal_rank(store: SqliteStore) -> None:
    """`score = 1/(rank+1)`——单调递减的占位值（§11.3）。

    它无论 AML 是否按 `score` 重排，**顺序都不变**，所以是最安全的一侧。
    """
    ids = [_seed(store, i, f"q{i}", f"a{i}") for i in range(4)]
    got = package(_candidates(*ids), store=store, top_k=5)

    assert [item.score for item in got.items] == [1.0, 0.5, 1 / 3, 0.25]


def test_score_is_strictly_decreasing(store: SqliteStore) -> None:
    ids = [_seed(store, i, f"q{i}", f"a{i}") for i in range(5)]
    scores = [i.score for i in package(_candidates(*ids), store=store, top_k=9).items]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) == len(scores)


def test_placeholder_score_rejects_negative_rank() -> None:
    with pytest.raises(ValueError, match="rank 不得为负"):
        placeholder_score(-1)


def test_response_item_has_exactly_the_contract_fields(store: SqliteStore) -> None:
    """`data[]` 的一项**就是 §2.1 的四个字段，一个不多**。

    ⚠ 这条同时是"**没有泄漏 RRF 分数**"的结构性证据：`ResponseItem` 里那个 `score`
    是 `1/(rank+1)` 现算的，而上游 `Candidate` **根本没有**融合分数字段
    （见 `tests/test_retrieve.py::test_candidate_carries_no_score_field`）。
    """
    assert {f.name for f in fields(ResponseItem)} == {"id", "content", "created_at", "score"}


def test_no_internal_fields_leak_into_the_item(store: SqliteStore) -> None:
    """`session_id` / `pair_idx` / `status` 都不该出现在响应项里。"""
    mid = _seed(store, 0, "q", "a")
    item = package(_candidates(mid), store=store, top_k=5).items[0]
    dumped = item.__dict__ if hasattr(item, "__dict__") else {}
    assert "session_id" not in dumped
    assert "pair_idx" not in dumped
    assert "status" not in dumped


# ── top_k：本函数是最终数量的守门人 ─────────────────────────────────────


def test_top_k_zero_returns_empty_without_touching_the_source(
    store: SqliteStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`top_k = 0` ⇒ 空响应，**且不查库**。"""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("top_k=0 不该访问真源")

    monkeypatch.setattr(store, "fetch_pairs_by_ids", _boom)
    mid = _seed(store, 0, "q", "a")
    got = package(_candidates(mid), store=store, top_k=0)
    assert got.items == ()
    assert got.count == 0


def test_top_k_one(store: SqliteStore) -> None:
    ids = [_seed(store, i, f"q{i}", f"a{i}") for i in range(3)]
    got = package(_candidates(*ids), store=store, top_k=1)
    assert got.count == 1
    assert got.items[0].id == ids[0]


def test_upstream_giving_more_than_top_k_is_truncated(store: SqliteStore) -> None:
    """**即使上游给出更多候选，也必须在这里保证 ≤ `top_k`**（§2.2 精确计数）。"""
    ids = [_seed(store, i, f"q{i}", f"a{i}") for i in range(6)]
    for top_k in (1, 2, 3, 5):
        got = package(_candidates(*ids), store=store, top_k=top_k)
        assert got.count == top_k
        assert got.count <= top_k


def test_top_k_larger_than_available_does_not_pad(store: SqliteStore) -> None:
    """**绝不为凑满 `top_k` 复制或补造结果。**"""
    ids = [_seed(store, i, f"q{i}", f"a{i}") for i in range(3)]
    got = package(_candidates(*ids), store=store, top_k=10)

    assert got.count == 3  # 不是 10
    assert len({item.id for item in got.items}) == 3  # 没有重复
    assert [item.id for item in got.items] == ids


def test_negative_top_k_returns_empty(store: SqliteStore) -> None:
    mid = _seed(store, 0, "q", "a")
    assert package(_candidates(mid), store=store, top_k=-1).count == 0


def test_empty_candidates_returns_empty_response(store: SqliteStore) -> None:
    got = package([], store=store, top_k=5)
    assert got.items == ()
    assert got.dropped_missing == 0


# ── 顺序：跟检索名次，不跟库里的顺序 ───────────────────────────────────


def test_order_follows_the_retrieval_rank_not_the_database_order(
    store: SqliteStore,
) -> None:
    """输出的顺序 = **候选给来的顺序**（那是检索名次），不是 `pair_idx` 或主键顺序。"""
    a = _seed(store, 0, "q0", "a0")
    b = _seed(store, 1, "q1", "a1")
    c = _seed(store, 2, "q2", "a2")

    got = package(_candidates(c, a, b), store=store, top_k=5)
    assert [item.id for item in got.items] == [c, a, b]
    assert [item.content for item in got.items] == [
        "Q: q2\nA: a2",
        "Q: q0\nA: a0",
        "Q: q1\nA: a1",
    ]


# ── 真源缺行：不静默 ───────────────────────────────────────────────────


def test_missing_source_row_is_dropped_and_counted(store: SqliteStore) -> None:
    """真源里查不到正文 ⇒ 丢掉**并计数**。

    正常情况下不该发生（Qdrant 是派生索引、`id` 与 SQLite 一一对应）。
    **非 0 说明索引与真源脱钩了**（例如 SQLite 被回滚到旧状态）——
    所以这个数要能被调用方看见，而不是静默少一项。
    """
    a = _seed(store, 0, "q0", "a0")
    c = _seed(store, 2, "q2", "a2")
    ghost = "f" * 64  # 库里没有这一行

    got = package(_candidates(a, ghost, c), store=store, top_k=5)

    assert [item.id for item in got.items] == [a, c]
    assert got.dropped_missing == 1


def test_scores_are_renumbered_by_output_position(store: SqliteStore) -> None:
    """缺行被跳过后，`score` 按**输出位置**重算——不是照抄输入名次。

    照抄会让分数出现空洞（`1.0, 1/3`），而它必须单调递减、且首位就是 `1.0`。
    """
    a = _seed(store, 0, "q0", "a0")
    c = _seed(store, 2, "q2", "a2")
    ghost = "e" * 64

    got = package(_candidates(a, ghost, c), store=store, top_k=5)

    assert [item.score for item in got.items] == [1.0, 0.5]
