"""**契约层**：`Search` / `Add` 的响应形状与计数（§2.1 / §2.2）。

* 前半部分走 `pipeline` 层：断言 `SearchPipeline.run()` 的产出 + **真正的 pydantic
  响应模型**（形状与序列化钉死在这里，便于精确控制"返回哪些 id、分数是多少"）。
* 文件末尾三个用例走**真的 HTTP**（`TestClient`）：覆盖路由、状态码、
  以及"边界校验在真 HTTP 上是 422"。

> **跨层：连接生命周期与线程**（见 `store/sqlite_store.py` 的连接模型一节 + D17）：
> FastAPI 的 `def` 路由跑在**线程池**里，而 `sqlite3.Connection` 只能在**创建它的线程**里用
> ⇒ 必须**短生命周期连接**；长期持有一个连接会让**任何 HTTP 往返**都抛
> `sqlite3.ProgrammingError: SQLite objects created in a thread can only be used in
> that same thread.`

⚠ 用假 Qdrant（鸭子类型）而**不是**真容器：本文件要精确控制"返回哪些 id、分数是多少"。
真 Qdrant 的行为由 `tests/test_qdrant_store.py` 覆盖。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest
from pydantic import ValidationError
from tests.conftest import FUSED_SCORE, Wired, seed_line, seed_pair_in

from tianximem.common.config import DEFAULT_RADIUS
from tianximem.common.render import render
from tianximem.service.schemas import (
    AddMessage,
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)


def _as_wire(packed) -> dict:  # noqa: ANN001
    """把打包结果**过一遍真的响应模型**——形状与序列化都钉死在这里。"""
    return SearchResponse(
        data=[
            SearchResultItem(id=i.id, content=i.content, created_at=i.created_at, score=i.score)
            for i in packed.items
        ]
    ).model_dump()


#: 候选之间的间隔，**由扩窗半径推出**：必须 **> 2 × radius + 1**。
#:
#: ⚠ 三个量要同时满足，少一个都会让"N 个候选 ⇒ N 个返回项"塌成一段：
#:
#: 1. **间隔 ≥ 2**：相邻两个块会被合并成同一个 Context Segment；
#: 2. **间隔 > 2 × radius**：否则两个种子的窗口**重叠**，把候选互相拉进对方的段；
#: 3. **间隔 > 2 × radius + 1**：否则两个窗口**首尾相接**（如 radius=2、间隔 5 时
#:    `[0,2]` 与 `[3,7]` 紧挨着）——中间那块被前一个种子扩出来、又被后一个接上，
#:    于是整条链子连成一段。**这一条最容易漏**：它不报错，只是段数少了。
#:
#: ⇒ 写成 `2 × DEFAULT_RADIUS + 3`（而不是硬编码一个数）：默认半径若改，这里跟着对。
_SPACING: Final[int] = 2 * DEFAULT_RADIUS + 3


def _idx(i: int) -> int:
    """第 `i` 个候选的**会话位置**：**隔 `_SPACING` 格**（默认半径下是 0, 7, 14, …）。

    ⚠ **为什么必须隔开**：位置相邻的两个块会被合并成**同一个** Context Segment，
    而扩窗（±radius）也会把它们互相拉进来。本文件测的是**契约形状与计数**，需要
    "N 个候选 ⇒ N 个返回项"这种最干净的情形；隔开之后中间那些块**存在但没被选中**
    ⇒ 既没有相邻可合并、也没有可扩的邻居 ⇒ 一个候选恰好一段。

    ⚠ 光落 0、7、14 这几行是不够的——`next` 是**写下时按整批**算好的，
    缺了中间那些行反而造不出"洞"。所以落库要走 `seed_line`（理由见它的 docstring），
    它只**返回**这几个位置的 `id`。

    **合并与扩窗本身的行为在 [`test_neighbor.py`](./test_neighbor.py) 里测**
    （那里才是它们的主场），两处刻意不重叠。
    """
    return i * _SPACING


# ── Search 响应形状（§2.1）──────────────────────────────────────────────


def test_search_response_has_exactly_the_contract_fields(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    ids = seed_line(
        wired.store, [_idx(i) for i in range(3)], qa=lambda i: (f"q{i}", f"[assistant] a{i}")
    )
    wired.qdrant.by_user["u1"] = ids

    body = _as_wire(wired.search(top_k=5))

    assert set(body) == {"data"}
    assert isinstance(body["data"], list)
    for item in body["data"]:
        assert set(item) == {"id", "content", "created_at", "score"}


def test_search_preserves_retrieval_order(wired: Wired, seed_pair: Callable[..., str]) -> None:
    ids = seed_line(wired.store, [_idx(i) for i in range(3)], qa=lambda i: (f"q{i}", f"a{i}"))
    wired.qdrant.by_user["u1"] = ids

    assert [i["id"] for i in _as_wire(wired.search(top_k=3))["data"]] == ids


@pytest.mark.parametrize("top_k", [1, 2, 3, 5])
def test_search_never_exceeds_top_k(
    wired: Wired, seed_pair: Callable[..., str], top_k: int
) -> None:
    """**精确计数**（§2.2 第一条）——返回超过 `top_k` 是契约错误，不会被静默截断。"""
    ids = seed_line(wired.store, [_idx(i) for i in range(5)], qa=lambda i: (f"q{i}", f"a{i}"))
    wired.qdrant.by_user["u1"] = ids

    data = _as_wire(wired.search(top_k=top_k))["data"]
    assert len(data) <= top_k
    assert len(data) == min(top_k, len(ids))


def test_empty_result_is_a_list_not_null(wired: Wired) -> None:
    """**空结果是 `[]` 而不是 `null`**（§2.1）。"""
    wired.qdrant.exists_flag = False  # 集合还不存在 ⇒ 没有数据可检索

    assert _as_wire(wired.search()) == {"data": []}


def test_result_shorter_than_top_k_when_fewer_pairs(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    ids = seed_line(wired.store, [_idx(i) for i in range(2)], qa=lambda i: (f"q{i}", f"a{i}"))
    wired.qdrant.by_user["u1"] = ids

    assert len(_as_wire(wired.search(top_k=10))["data"]) == 2  # 不补造、不复制


# ── 各字段的来源 ───────────────────────────────────────────────────────


def test_content_comes_from_sqlite_through_common_render(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """**正文只从 SQLite 取**，且走 `common/render` 的同一份渲染（不变式 I1）。"""
    mid = seed_line(wired.store, [0], qa=lambda _i: ("火车几点开？", "[assistant] 09:42。"))[0]
    wired.qdrant.by_user["u1"] = [mid]

    item = _as_wire(wired.search(top_k=1))["data"][0]

    assert item["content"] == render("火车几点开？", "[assistant] 09:42。")


def test_created_at_is_day_granularity_or_empty(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """`created_at` **始终存在**：日粒度或 `""`（§11.3）。"""
    with_time = seed_pair_in(wired.store, 0, "q0", "a0", event_time=1683525360000)
    # ⚠ 两条的间隔走 `_SPACING`（由扩窗半径推出，见那里的说明）：位置挨着会被合并成一段
    # ⚠ **中间那些位置要落满**（`seed_line` 的同一道理，见它的 docstring）
    for pos in range(1, _SPACING):
        seed_pair_in(wired.store, pos)
    without = seed_pair_in(wired.store, _SPACING, "q1", "a1", event_time=None)
    wired.qdrant.by_user["u1"] = [with_time, without]

    data = _as_wire(wired.search(top_k=2))["data"]

    assert data[0]["created_at"] == "2023-05-08"
    assert data[1]["created_at"] == ""


def test_score_is_monotonic_placeholder(wired: Wired, seed_pair: Callable[..., str]) -> None:
    """`score` = `1/(rank+1)`，严格递减。"""
    ids = seed_line(wired.store, [_idx(i) for i in range(3)], qa=lambda i: (f"q{i}", f"a{i}"))
    wired.qdrant.by_user["u1"] = ids

    scores = [i["score"] for i in _as_wire(wired.search(top_k=3))["data"]]
    assert scores == [1.0, 0.5, 1 / 3]


def test_fused_score_never_leaks_into_the_response(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """**Qdrant 的 RRF 分数不得进入 API。**

    假 Qdrant 返回一个刻意刺眼的 `0.987654321`；它出现在任何一项里都说明有人透传了融合分数。
    而 `Candidate` 里**根本没有**该字段（见
    `tests/test_retrieve.py::test_candidate_carries_no_score_field`）⇒ 结构上漏不出来。
    """
    ids = seed_line(wired.store, [_idx(i) for i in range(3)], qa=lambda i: (f"q{i}", f"a{i}"))
    wired.qdrant.by_user["u1"] = ids

    body = _as_wire(wired.search(top_k=3))

    assert FUSED_SCORE not in [item["score"] for item in body["data"]]
    assert (
        str(FUSED_SCORE)
        not in SearchResponse(data=[SearchResultItem(**i) for i in body["data"]]).model_dump_json()
    )


# ── 调用次数与顺序 ─────────────────────────────────────────────────────


def test_dense_embedding_called_exactly_once_per_query(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """**每查询恰好 1 次** embedding 调用（§7.2）。"""
    wired.qdrant.by_user["u1"] = seed_line(
        wired.store, [_idx(i) for i in range(2)], qa=lambda i: (f"q{i}", f"a{i}")
    )

    wired.search(query="火车几点开？", top_k=2)

    assert wired.retriever.dense.query_calls == ["火车几点开？"]


def test_query_is_passed_through_unchanged(wired: Wired, seed_pair: Callable[..., str]) -> None:
    """**查询原样送**——不改写、不 strip（§7.2 的 v1 规格）。"""
    wired.qdrant.by_user["u1"] = [seed_pair(wired.store, 0, "q", "a")]
    raw = "  火车几点开？ "
    wired.search(query=raw, top_k=1)

    assert wired.qdrant.calls[0]["query_text"] == raw
    assert wired.retriever.dense.query_calls == [raw]


def test_pipeline_order_hybrid_then_checker_then_packaging(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """顺序：`hybrid → checker → packaging`。

    checker 与 packaging 都不是 `hybrid` 能触发的——两者都跑到就说明顺序没被跳过。
    """
    wired.qdrant.by_user["u1"] = seed_line(
        wired.store, [_idx(i) for i in range(2)], qa=lambda i: (f"q{i}", f"a{i}")
    )

    data = _as_wire(wired.search(top_k=2))["data"]

    assert len(wired.qdrant.calls) == 1  # hybrid 跑了
    assert len(wired.instrument.decisions) == 1  # checker 跑了，且每轮记账（D13）
    assert len(data) == 2  # packaging 跑了


def test_checker_is_passthrough_in_v1(wired: Wired, seed_pair: Callable[..., str]) -> None:
    """v1 的 Checker **不做门控**：哪怕候选很少也照样返回（D13）。"""
    wired.qdrant.by_user["u1"] = [seed_pair(wired.store, 0, "q", "a")]

    assert len(_as_wire(wired.search())["data"]) == 1
    assert wired.instrument.decisions[0].enough is True


def test_top_k_from_request_is_forwarded_not_hardcoded(
    wired: Wired, seed_pair: Callable[..., str]
) -> None:
    """**`top_k` 来自请求、不写死 100**（§7.3）。"""
    wired.qdrant.by_user["u1"] = [
        seed_pair(wired.store, _idx(i), f"q{i}", f"a{i}") for i in range(5)
    ]

    wired.search(top_k=3)

    assert wired.qdrant.calls[0]["top_k"] == 3


# ── 边界校验（**schema 层**，不需要 HTTP）──────────────────────────────


@pytest.mark.parametrize(
    "body",
    [
        {"user_id": "u1", "query": "q", "top_k": 0},
        {"user_id": "u1", "query": "q", "top_k": -1},
        {"user_id": "u1", "query": "   ", "top_k": 5},
        {"user_id": "", "query": "q", "top_k": 5},
        {"user_id": "u1", "query": "q"},
    ],
)
def test_search_schema_rejects_illegal_requests(body: dict) -> None:
    """**参数合法性在边界校验**——不依赖下游（`rank.package`）的容错行为。"""
    with pytest.raises(ValidationError):
        SearchRequest(**body)


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": "r", "user_id": "u", "session_id": "s", "messages": []},
        {
            "request_id": "",
            "user_id": "u",
            "session_id": "s",
            "messages": [{"role": "user", "content": "x"}],
        },
        {
            "request_id": "r",
            "user_id": "u",
            "session_id": "s",
            "messages": [{"role": "", "content": "x"}],
        },
        {"request_id": "r", "user_id": "u", "session_id": "s"},
    ],
)
def test_add_schema_rejects_illegal_requests(body: dict) -> None:
    with pytest.raises(ValidationError):
        AddRequest(**body)


def test_unknown_extra_fields_are_ignored() -> None:
    """未知字段一律忽略——AML 将来加字段不该让我们 422。"""
    req = SearchRequest(**{"user_id": "u1", "query": "q", "top_k": 1, "future_field": 1})
    assert req.top_k == 1


def test_add_response_echoes_exactly_three_fields() -> None:
    """`Add` 的 200 形状：`success: true` + **原样回显**三个字段（§2.1）。"""
    body = AddResponse(request_id="r1", user_id="u1", session_id="s1").model_dump()
    assert body == {
        "success": True,
        "request_id": "r1",
        "user_id": "u1",
        "session_id": "s1",
    }


def test_add_message_schema_shape() -> None:
    """`messages[]` **只有三个字段**——这就是 canonical 形状（D16）。"""
    msg = AddMessage(role="user", content="hi")
    assert msg.model_dump() == {"role": "user", "content": "hi", "timestamp": None}


# ── HTTP 往返（§2.1 的最后一跳）────────────────────────────────────────


def test_http_search_round_trip(wired: Wired, seed_pair: Callable[..., str]) -> None:
    """**HTTP 那一跳**：路由 + 状态码 + 响应项的字段集合。"""
    from fastapi.testclient import TestClient

    from tianximem.service import create_app

    wired.qdrant.by_user["u1"] = [seed_pair(wired.store, 0, "q", "a")]
    with TestClient(create_app(wired.services)) as client:
        resp = client.post("/search", json={"user_id": "u1", "query": "q", "top_k": 1})
    assert resp.status_code == 200
    assert set(resp.json()["data"][0]) == {"id", "content", "created_at", "score"}


def test_http_add_round_trip_echoes_fields(wired: Wired) -> None:
    from fastapi.testclient import TestClient

    from tianximem.service import create_app

    payload = {
        # ⚠ `request_id` 是 **opaque string**（D28）——服务端**不解析它**，任意形状都行。
        #    这里用本仓 harness 的形态。
        "request_id": "u1|s1|0",
        "user_id": "u1",
        "session_id": "s1",
        "messages": [{"role": "user", "content": "Q1"}],
    }
    with TestClient(create_app(wired.services)) as client:
        resp = client.post("/add", json=payload)
    assert resp.status_code == 200
    assert resp.json()["request_id"] == "u1|s1|0"  # 原样回显（§2.1）


def test_http_rejects_illegal_request_at_the_boundary(wired: Wired) -> None:
    """边界校验在**真的 HTTP 上**是 422。"""
    from fastapi.testclient import TestClient

    from tianximem.service import create_app

    with TestClient(create_app(wired.services)) as client:
        resp = client.post("/search", json={"user_id": "u1", "query": "q", "top_k": 0})
    assert resp.status_code == 422


def test_http_health_is_an_unauthenticated_2xx(wired: Wired) -> None:
    """S4：平台探的是**与 Add 同源的 `/health`**，无需鉴权，任意 2xx 即正常。

    这一条的失败模式不是掉分，是**没进场**——所以它必须有测试钉住（`contract.md` §7.2）。
    """
    from fastapi.testclient import TestClient

    from tianximem.service import create_app

    with TestClient(create_app(wired.services)) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
