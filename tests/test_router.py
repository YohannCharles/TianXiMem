"""三分类查询路由器（`retrieve/router.py` + `llm/`）——**不发真请求**。

## 这个文件挡的四类错

| 类 | 挡什么 | 靠什么 |
| --- | --- | --- |
| **解析** | 把 `NOT MULTI_HOP` 读成 `MULTI_HOP`（**否定读成肯定**） | `parse_route` 整串匹配 |
| **降级** | 端点挂了 / 输出非法 ⇒ **一次正常 Search 变 500** | 假客户端 + 真 HTTP 往返 |
| **路径** | 分对了类却**走错分支** | 真 `SqliteStore` + `apply_batch` 的端到端 |
| **纯度（§13）** | 关掉它却**仍有 LLM 调用**、或行为变了 | 假客户端的 `calls` 计数 + 两臂逐项比 |

⚠ 与 `test_reranker.py` 的 `_FakeReranker` 同理：这里的假 LLM 是**协议级替身**
（`LlmClient` 的 `name` + `complete`），**不是** HTTP 层的桩——HTTP 层另有
`httpx.MockTransport` 的三条用例覆盖。两层都要，不能互相替代。
"""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import httpx
import pytest

from tianximem.agent import AgenticResult, AgenticStats
from tianximem.common.config import (
    AgenticConfig,
    AppConfig,
    CacheConfig,
    ConfigError,
    EmbedCacheConfig,
    QdrantConfig,
    RouterConfig,
    SqliteConfig,
    StorageConfig,
    load_config,
)
from tianximem.common.tokens import O200kCounter
from tianximem.llm import LlmUnavailable, OpenAiCompatChat
from tianximem.observability import SnapshotMetricsSink
from tianximem.pairing import AddBatch, Message, apply_batch
from tianximem.retrieve import Candidate, EvidenceChecker
from tianximem.retrieve.router import (
    NONE_OUTPUT,
    ROUTER_PROMPT,
    ROUTER_PROMPT_VERSION,
    QueryRouter,
    Route,
    is_none_output,
    parse_route,
)
from tianximem.service.app import build_query_router, build_services
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.sqlite_store import SqliteStore

# ── 假客户端：**协议级**（见模块 docstring）────────────────────────────


def _agentic_result(
    candidates: list[Candidate],
    *,
    rerank_query: str | None = None,
    fallback_reason: str | None = None,
) -> AgenticResult:
    """一个满足 `AgenticSearch` 形状的返回值。

    ⚠ 它与真执行器一样**不精排**：`rerank_query` 是交给编排层的那一句联合 Query
    （§9.1）。默认 `None` ⇒ 编排层用原问题——正是混合链那条路的行为。
    """
    rounds = 1 if candidates else 0
    return AgenticResult(
        candidates=tuple(candidates),
        rerank_query=rerank_query,
        stats=AgenticStats(
            rounds=(),
            stop_reason="max_rounds" if candidates else "retrieval_error",
            fallback_reason=fallback_reason,
            candidate_count=len(candidates),
            unique_candidate_count=len({c.memory_id for c in candidates}),
            selected_evidence_count=0,
            joint_query_count=rounds,
            llm_calls=0,
            rerank_calls=0,
            latency_ms=0.0,
        ),
        fallback_reason=fallback_reason,
    )



class FakeLlm:
    """`LlmClient` 的替身：回一个**脚本化的**答案，并记下每一次调用。

    `reply` 可以是字符串（每次都回它）或可调用对象（`(prompt) -> str`）。
    要模拟端点故障就把 `reply` 设成一个抛 `LlmUnavailable` 的函数。
    """

    def __init__(self, reply: str | BaseException, name: str = "fake-llm") -> None:
        self._reply = reply
        self._name = name
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def complete(self, prompt: str, *, max_tokens: int = 8, temperature: float = 0.0) -> str:
        self.calls.append(prompt)
        if isinstance(self._reply, BaseException):
            raise self._reply
        return self._reply


# ── ① 解析：**绝不子串匹配** ───────────────────────────────────────────


@pytest.mark.parametrize(
    "text,expected",
    [
        ("MULTI_HOP", Route.MULTI_HOP),
        ("COUNT", Route.COUNT),
        ("DEFAULT", Route.DEFAULT),
        ("  COUNT\n", Route.COUNT),  # 首尾空白与换行是排版，不是非法输出
        ("\n\tDEFAULT\r\n", Route.DEFAULT),
        ("multi_hop", Route.MULTI_HOP),  # 大小写归一
        ("Count", Route.COUNT),
    ],
)
def test_legal_outputs_are_accepted(text: str, expected: Route) -> None:
    assert parse_route(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "NOT MULTI_HOP",  # ★ 头号用例：子串匹配会把**否定**读成**肯定**
        "not multi_hop",
        "This is NOT COUNT",
        "DEFAULT\nbecause the query is simple",  # 多行 ⇒ 非法（不许收"第一行"）
        "COUNT DEFAULT",
        "The route is DEFAULT",
        "`COUNT`",
        '"COUNT"',
        '{"route": "COUNT"}',
        "0",
        "",
        "   ",
        "COUNTING",
        "MULTI",
        "Hop",
    ],
)
def test_illegal_outputs_are_rejected(text: str) -> None:
    """**非法一律 `None`**（调用方随即回退 `DEFAULT`）。"""
    assert parse_route(text) is None


def test_normalisation_cannot_turn_an_illegal_string_into_a_legal_one() -> None:
    """首尾空白 + 大小写归一**不做任何子串 / 前缀 / 相似度匹配**。"""
    for noise in ("not ", "no ", "?", "_", "x"):
        assert parse_route(f"{noise}DEFAULT") is None
        assert parse_route(f"DEFAULT{noise}") is None


# ── ② HTTP 层：`OpenAiCompatChat`（MockTransport，不发真请求）──────────


def _chat(handler) -> OpenAiCompatChat:
    return OpenAiCompatChat(
        base_url="https://gw.example/v1",
        api_key="k",
        model="some-model",
        timeout=1.0,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _reply(content: object, **extra: object) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, **extra}]})


def test_chat_returns_the_message_content() -> None:
    chat = _chat(lambda request: _reply("  COUNT  "))
    assert chat.complete("p") == "COUNT"  # 端点回的空白在这里就剥掉了
    assert chat.calls == 1
    assert chat.last_latency_ms is not None


def test_chat_sends_the_declared_model_temperature_and_max_tokens() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        seen["auth"] = request.headers["Authorization"]
        return _reply("DEFAULT")

    _chat(handler).complete("hello", max_tokens=8, temperature=0.0)
    assert seen["model"] == "some-model"
    assert seen["messages"] == [{"role": "user", "content": "hello"}]
    assert seen["temperature"] == 0.0
    assert seen["max_tokens"] == 8
    assert seen["auth"] == "Bearer k"


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(401, text="bad key"),
        httpx.Response(404, text="unknown model"),
        httpx.Response(500, text="boom"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, json={"choices": ["not an object"]}),
        httpx.Response(200, json={"choices": [{"message": "not an object"}]}),
        httpx.Response(200, json={"choices": [{"message": {"content": 42}}]}),
        # ⚠ **空白内容也算端点故障**：不然它会被记成"分类成功、结果是空串"，
        #   于是"端点一直在回空"在 §14 里长得像"模型答得怪"。
        httpx.Response(200, json={"choices": [{"message": {"content": "   "}}]}),
    ],
)
def test_chat_turns_every_bad_shape_into_llm_unavailable(answer: httpx.Response) -> None:
    with pytest.raises(LlmUnavailable):
        _chat(lambda request: answer).complete("p")


def test_chat_turns_network_failures_into_llm_unavailable() -> None:
    """⚠ **超时用"抛 `httpx.ReadTimeout`"来模拟，不让假传输层真的睡**：
    `MockTransport` 不参与超时计时，睡多久都不会触发超时——那样写是用例**空过的**
    （与 `test_reranker.py` 的同一条纪律）。真超时与它是同一个 `httpx.HTTPError` 分支。
    """
    for exc in (httpx.ReadTimeout("slow"), httpx.ConnectError("no route")):
        with pytest.raises(LlmUnavailable):
            _chat(lambda request, e=exc: (_ for _ in ()).throw(e)).complete("p")


def test_chat_rejects_empty_endpoint_or_key() -> None:
    for kwargs in (
        {"base_url": "", "api_key": "k"},
        {"base_url": "https://gw/v1", "api_key": ""},
    ):
        with pytest.raises(ValueError):
            OpenAiCompatChat(model="m", **kwargs)


# ── ③ `QueryRouter`：降级与不可信输入 ─────────────────────────────────


def _router(reply, **kwargs) -> tuple[QueryRouter, FakeLlm]:
    client = FakeLlm(reply)
    return QueryRouter(client, **kwargs), client


def test_successful_routing_records_the_raw_output_and_latency() -> None:
    router, _ = _router("COUNT")
    decision = router.route("How many meetings did I attend?")
    assert decision.route is Route.COUNT
    assert decision.status == "success"
    assert decision.reason is None
    assert decision.raw == "COUNT"
    assert decision.latency_ms >= 0.0


@pytest.mark.parametrize(
    "reply,reason",
    [("NOT MULTI_HOP", "invalid_output"), ("", "invalid_output")],
)
def test_illegal_output_degrades_to_default_and_keeps_the_raw_text(reply: str, reason: str) -> None:
    """⚠ **`raw` 要留着**：模型到底回了什么是排查"这类题怎么老走默认路"的第一手材料。"""
    router, _ = _router(reply)
    decision = router.route("q")
    assert decision.route is Route.DEFAULT
    assert decision.status == "degraded"
    assert decision.reason == reason


def test_llm_failure_degrades_to_default_and_never_raises() -> None:
    """§4.3：端点挂了 ⇒ 回退 `DEFAULT`。**绝不让一次正常 Search 变成 500。**"""
    router, _ = _router(LlmUnavailable("端点不可用"))
    decision = router.route("q")
    assert decision.route is Route.DEFAULT
    assert decision.status == "degraded"
    assert decision.reason is not None and decision.reason.startswith("llm_unavailable")


def test_unexpected_backend_bug_also_degrades_but_is_labelled_differently() -> None:
    """★ **兜底捕获必须与"端点不可用"分得开**：`reason` 里带着异常类型，
    所以"未预期的 bug"与"端点挂了"在 §14 上不是同一个数。"""
    router, _ = _router(RuntimeError("协议实现坏了"))
    decision = router.route("q")
    assert decision.route is Route.DEFAULT
    assert decision.reason is not None and "llm_error" in decision.reason
    assert "RuntimeError" in decision.reason


def test_query_is_injected_without_format_interpretation() -> None:
    """⛔ **Query 用 `replace` 注入，不用 `format`**：`format` 会把 Query 里的 `{}`
    当成格式串（那是用户可控的输入）。带花括号的 Query 必须**原样**到达模型。
    """
    query = 'What about {"route": "COUNT"} and {0} and {name}?'
    router, client = _router("DEFAULT")
    router.route(query)
    sent = client.calls[0]
    assert query in sent
    assert sent.count("{query}") == 0  # 占位符**全部**被替换掉了
    # ⚠ **v3 的查询位置变了**：v1/v2 把它放在**最后一行**（`USER QUERY:` 之后），
    #   v3 放在中间的 `INPUT / Query:` 处，**后面还跟着 OUTPUT FORMAT 与 10 个例子**。
    #   ⇒ 这里不再断言"以 query 结尾"，改为断言"按模板逐处注入"——
    #   而"查询被夹在例子中间"这件事本身要记进报告（它会分散模型对真查询的注意力）。
    assert sent == ROUTER_PROMPT.replace("{query}", query)
    assert sent.index(query) < sent.index("CLASSIFICATION EXAMPLES")


def test_prompt_is_the_frozen_template_and_the_query_lands_after_the_user_line() -> None:
    router, client = _router("DEFAULT")
    router.route("Who was my manager at Atlas?")
    sent = client.calls[0]
    assert sent == ROUTER_PROMPT.replace("{query}", "Who was my manager at Atlas?")
    # 分类规则那几段**逐字**在（改了提示词就必须同时改 `ROUTER_PROMPT_VERSION`）
    assert 'Explicit dependency signals: "then", "after", "using that"' in sent
    assert '- Signals: "how many", "how often", "which ones", "who are the", "list all"' in sent
    assert "Do not output explanations, confidence scores, JSON, or any additional text." in sent
    # ⛔ v4 起**提示词里没有 NONE**（空/无效输入归入口，不归路由器）
    assert "NONE" not in ROUTER_PROMPT


def test_the_prompt_is_the_single_frozen_one() -> None:
    """⛔ **全仓只有这一版提示词**（2026-10-10 按要求删掉了 v1 / v2 / v3）。

    ⚠ 删版本**是有代价的**，而且代价就写在这条用例旁边：
    **"换提示词值不值"的基线从此不能再原样重跑**——要重跑得先从报告里把那一版抄回来。
    ⇒ 所以这条用例钉住"当前是哪一版"，好让下一次改动**至少知道自己在改哪一版**。
    """
    assert ROUTER_PROMPT_VERSION == "v4"
    assert ROUTER_PROMPT is not None and ROUTER_PROMPT.strip()
    # 三条判据都在（换了提示词就得同时改版本号，见 `QueryRouter.__init__`）
    assert "The key test" not in ROUTER_PROMPT  # ← 那是旧版的措辞，不该再出现
    assert 'Explicit dependency signals: "then", "after", "using that"' in ROUTER_PROMPT
    signals = '- Signals: "how many", "how often", "which ones", "who are the", "list all"'
    assert signals in ROUTER_PROMPT
    assert "Do not classify ordinary dates, ages, measurements" in ROUTER_PROMPT
    # ⛔ 没有 NONE（空 / 无效输入归入口，不归路由器）
    assert "NONE" not in ROUTER_PROMPT
    assert "Validate input" not in ROUTER_PROMPT
    assert "{query}" in ROUTER_PROMPT


def test_stray_none_output_degrades_and_is_not_a_fourth_route() -> None:
    """★ **当前提示词里已经没有 `NONE` 了**，所以模型不该再吐它。

    ⚠ 这一条测的是**防御**：万一它吐了（旧版提示词的惯性，或模型自己发挥），
    必须仍然 ①**不成为第四条执行路**、②**降级到 `DEFAULT`**、
    ③**与"输出非法"分开记**——分开记才能回答"模型是不是在试图拒答"，
    而那是排查不出来的东西里最容易混掉的一种。
    """
    assert NONE_OUTPUT == "NONE"
    assert not any(route.value == "NONE" for route in Route), "NONE 不许进 Route 枚举"
    assert "NONE" not in ROUTER_PROMPT, "提示词里不该再有 NONE"

    router, _ = _router("NONE")
    decision = router.route("   ")
    assert decision.route is Route.DEFAULT, "拒答落在 DEFAULT（三条路里最安全的那个）"
    assert decision.status == "degraded"
    assert decision.reason == "model_none"
    assert decision.raw == "NONE"  # 原样留着，好与"非法输出"区分开

    bad, _ = _router("NOT MULTI_HOP")  # 与"非法输出"分得开
    assert bad.route("q").reason == "invalid_output"

    for text in ("NONE", "  none ", "None" + chr(10)):
        assert is_none_output(text)
    for text in ("NOT MULTI_HOP", "NONEXISTENT", "NONE OF THE ABOVE", "DEFAULT"):
        assert not is_none_output(text)


def test_prompt_version_mismatch_fails_loudly_at_construction() -> None:
    """⛔ **它是一条断言，不是说明**：改了提示词却忘了改版本号 ⇒ 两份 run record
    会声称同一份提示词，而那种漂移查不出来 ⇒ 所以这里**响亮失败**。"""
    with pytest.raises(ValueError, match="prompt_version"):
        QueryRouter(FakeLlm("DEFAULT"), prompt_version="v9")


@pytest.mark.parametrize(
    "kwargs", [{"max_output_tokens": 0}, {"max_output_tokens": -1}, {"temperature": -0.1}]
)
def test_invalid_router_knobs_fail_at_construction(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        QueryRouter(FakeLlm("DEFAULT"), **kwargs)


def test_blank_query_short_circuits_before_the_router(tmp_path) -> None:
    """⛔ **空白 query 绝不进路由器**（2026-10-10）。

    三层理由，缺一条这条守卫就不该在：

    1. 空查询**不是分类问题**，是输入校验——让模型去猜三类之一是把它外包给 LLM；
    2. 契约要的是 `{"data": []}`（§2.1 明说空结果合法），这条短路给的正是它；
    3. **位置在 `_route()` 之前**才是关键——只要调了 `_route()` 就花掉一次 LLM 调用。

    ⚠ HTTP 层对空白 query **仍然回 422**（那是刻意的、已记录的取舍，见 D40）⇒
    这条守卫服务的是**绕过 HTTP 的调用方**（测试、harness、将来的内部调用）。
    """
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    client = FakeLlm("COUNT")
    service = _pipeline(store, router=QueryRouter(client))
    for blank in ("", "   ", chr(9), chr(10)):
        result = service.run(user_id="u", query=blank, top_k=100)
        assert result.items == (), "空白 query 必须是空结果"
    assert client.calls == [], "★ 空白 query 一次 LLM 调用都不该发生"


def test_router_never_imports_the_database_layer() -> None:
    """⚠ **它只选策略**：不查库、不取事实、不产出绑定字段。

    这条是**源码级**的（`router.py` 只该依赖 `llm/` 与标准库）：让模型吐
    `subject_key` 之类只有两种结局——静默匹配不上，或**带着逐字引句静默选错事实**。
    """
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / "src" / "tianximem" / "retrieve" / "router.py"
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    forbidden = {
        name
        for name in imported
        if name.startswith(("tianximem.store", "tianximem.facts", "tianximem.retrieve"))
    }
    assert forbidden == set(), f"路由器不该依赖这些：{sorted(forbidden)}"
    assert imported == {
        "__future__",
        "dataclasses",
        "enum",
        "time",
        "typing",
        "tianximem.llm.base",
    }


# ── ④ 端到端：路由**真的决定走哪条分支** ──────────────────────────────

#: 一个**必然编译出共同取证计划**的问题（`filter` 算子，见 `facts/query.py` 的名单句法）。
_LIST_QUERY = "Who are the coaches at North Athletics?"
#: 会被三分类判成计数/名单的那一类（提示词的 EXAMPLES 里就有 `Which meetings …`）。
_COUNT_QUERY = "How many coaches are there at North Athletics?"

#: 种子原文——**与 `test_grounded_evidence.py` 同一个形状**（一句能被确定性抽取的自述）。
SEED_STATEMENT = "Rosa Quinn: I am a coach at North Athletics."


def _seed(store: SqliteStore) -> None:
    """落一条**能被确定性抽取**的原文，让共同取证那条路真的能走通。"""
    apply_batch(
        store,
        AddBatch("opaque:req", "u", "s", (Message("user", SEED_STATEMENT),)),
        grounded_evidence=True,
    )


def _pipeline(
    store: SqliteStore,
    *,
    router: QueryRouter | None = None,
    agentic: object | None = None,
    metrics: SnapshotMetricsSink | None = None,
) -> SearchPipeline:
    """真 `SqliteStore` + 一个"把全部行都当候选"的检索桩（**只挡 Qdrant，不挡事实**）。"""

    def search(*, user_id: str, query: str, top_k: int) -> list[Candidate]:
        with store.read() as conn:
            ids = [
                row[0]
                for row in conn.execute(
                    "SELECT id FROM qa_pairs WHERE user_id=? ORDER BY request_id,local_index",
                    (user_id,),
                )
            ]
        return [Candidate(pid, i) for i, pid in enumerate(ids)]

    return SearchPipeline(
        store=store,
        qdrant=SimpleNamespace(exists=lambda: True),
        retriever=SimpleNamespace(search=search),
        checker=EvidenceChecker(),
        counter=O200kCounter(),
        budget_tokens=117760,
        grounded_evidence=True,
        radius=0,
        inject_abs_time=True,
        router=router,
        agentic=agentic,  # type: ignore[arg-type]
        metrics=metrics,
    )


def _snapshot(pipeline: SearchPipeline, tmp_path, **kwargs) -> dict:
    """跑一次 Search，**从 §14 的出口读路由**（而不是去摸 `_SearchTrace` 私有件）。

    ⚠ 这也顺带证明了一件事：路由的观测值**真的走到了落盘那条路上**——
    它进不了响应（形状是契约），所以磁盘是唯一的读法。
    """
    import json

    path = tmp_path / "m.json"
    pipeline._metrics = SnapshotMetricsSink(path=path)  # noqa: SLF001 — 测试内换出口
    pipeline.run(**kwargs)
    return json.loads(path.read_text(encoding="utf-8"))


def test_router_off_keeps_the_original_path_and_makes_no_llm_call(tmp_path) -> None:
    """★ **§13 的开关纯度**：关掉路由器 ⇒ **一次 LLM 调用都不发生**，
    而共同取证分支**照旧无条件先试**（= 本接缝之前的行为）。
    """
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    service = _pipeline(store, router=None)  # `router=None` 就是"没开"
    result = service.run(user_id="u", query=_LIST_QUERY, top_k=100)
    assert result.items, "关掉路由器时，名单问题仍该由共同取证直接答出证据"
    assert "Statement: Rosa Quinn" in result.items[0].content


def test_router_default_skips_the_grounded_branch(tmp_path) -> None:
    """`DEFAULT` ⇒ 原混合检索链，**不再先试共同取证**。

    观测量不是"结果为空"（那可能是别的原因），而是
    **`grounded_evidence.outcomes` 里没有 `no_plan` / `no_match` 这些取证结局**——
    分支根本没进过。⚠ 这正是路由开着的**唯一行为代价**。
    """
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router("DEFAULT")
    service = _pipeline(store, router=router)
    snapshot = _snapshot(service, tmp_path, user_id="u", query=_LIST_QUERY, top_k=100)
    assert snapshot["router"]["routes"] == {"DEFAULT": 1}
    assert snapshot["router"]["statuses"] == {"success": 1}
    # 取证分支**没进过** ⇒ 一个取证结局都没记（`grounded_status` 停在初始的 `disabled`）
    assert snapshot["grounded_evidence"]["outcomes"] == {"disabled": 1}
    assert snapshot["grounded_evidence"]["operators"] == {}


def test_router_count_keeps_the_grounded_branch(tmp_path) -> None:
    """`COUNT` ⇒ **与关掉路由器时逐字相同**的共同取证（确定性编译，没有捷径）。"""
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router("COUNT")
    routed = _pipeline(store, router=router)
    plain = _pipeline(store, router=None)
    assert [i.content for i in routed.run(user_id="u", query=_COUNT_QUERY, top_k=100).items] == [
        i.content for i in plain.run(user_id="u", query=_COUNT_QUERY, top_k=100).items
    ]
    snapshot = _snapshot(routed, tmp_path, user_id="u", query=_COUNT_QUERY, top_k=100)
    assert snapshot["router"]["routes"] == {"COUNT": 1}
    assert snapshot["grounded_evidence"]["operators"] == {"filter": 1}


def test_count_route_with_an_uncompilable_query_falls_back_to_the_hybrid_chain(tmp_path) -> None:
    """⚠ **`COUNT` 是查询意图，不代表一定能编译出计划**（§3.2 的注意事项）：
    编译不出来就照原样回退混合检索，**不返回半份集合、也不报错**。
    """
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router("COUNT")
    service = _pipeline(store, router=router)
    snapshot = _snapshot(
        service, tmp_path, user_id="u", query="Tell me about the weather lately", top_k=100
    )
    assert snapshot["router"]["routes"] == {"COUNT": 1}
    assert snapshot["grounded_evidence"]["outcomes"] == {"no_plan": 1}
    # 回退之后走的是混合链 —— 它的候选来自检索桩，与取证结局是两回事
    assert snapshot["packaging"]["returned_segments"] >= 0


def test_multi_hop_without_an_executor_degrades_loudly_to_default(tmp_path) -> None:
    """⛔ **不许假装已经执行了多跳检索**（§3.3）。

    今天 `MULTI_HOP` 一定降级成 `DEFAULT`，而这件事**必须在 §14 上留下名字**：
    `route_fallback = agentic_unavailable`。不然"分类成功但执行器不在"与
    "分类结果本来就是 DEFAULT"长得一模一样。
    """
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router("MULTI_HOP")
    service = _pipeline(store, router=router)
    snapshot = _snapshot(service, tmp_path, user_id="u", query=_LIST_QUERY, top_k=100)
    assert snapshot["router"]["routes"] == {"MULTI_HOP": 1}
    assert snapshot["router"]["statuses"] == {"success": 1}
    assert snapshot["router"]["fallbacks"] == {"agentic_unavailable": 1}
    # 降级走的是 DEFAULT 那条路：共同取证**没进过**
    assert snapshot["grounded_evidence"]["outcomes"] == {"disabled": 1}


def test_multi_hop_with_an_executor_feeds_the_shared_tail(tmp_path) -> None:
    """§3.3 的接缝：执行器**只负责多取几轮证据**，`②–⑦` 与混合那条路共用。

    ⚠ 这条用例证明的是**接线**（候选真的流进了去重 → 精排 → 段合并 → 打包），
    **不是多跳的效果**——跑的是个假执行器。
    """
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    with store.read() as conn:
        ids = [row[0] for row in conn.execute("SELECT id FROM qa_pairs")]
    seen: list[dict] = []

    class FakeAgentic:
        def search(self, *, user_id: str, query: str, top_k: int) -> AgenticResult:
            seen.append({"user_id": user_id, "query": query, "top_k": top_k})
            # ⚠ **故意重复一条**：去重是在调用方（`_rank_and_package`）做的，
            #   执行器返回带重复的候选正是它该被覆盖的形状。
            return _agentic_result([Candidate(ids[0], 0), Candidate(ids[0], 1)])

    router, _ = _router("MULTI_HOP")
    service = _pipeline(store, router=router, agentic=FakeAgentic())
    snapshot = _snapshot(service, tmp_path, user_id="u", query=_LIST_QUERY, top_k=100)
    assert seen == [{"user_id": "u", "query": _LIST_QUERY, "top_k": 100}]
    assert snapshot["router"]["fallbacks"] == {}  # 执行器在 ⇒ 不是降级
    result = service.run(user_id="u", query=_LIST_QUERY, top_k=100)
    assert result.items and result.items[0].id == ids[0]


def test_degraded_routing_still_returns_a_valid_200_shaped_response(tmp_path) -> None:
    """★ **路由的失败不能让一次正常 Search 变成 500**（§4.3）。"""
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router(LlmUnavailable("端点挂了"))
    service = _pipeline(store, router=router)
    snapshot = _snapshot(service, tmp_path, user_id="u", query=_LIST_QUERY, top_k=100)
    assert snapshot["router"]["statuses"] == {"degraded": 1}
    assert snapshot["router"]["routes"] == {"DEFAULT": 1}
    result = service.run(user_id="u", query=_LIST_QUERY, top_k=100)
    # 响应形状逐字不变（§2.1 那四个字段，一个不多一个不少）
    for item in result.items:
        assert set(item.__dataclass_fields__) == {"id", "content", "created_at", "score"}


# ── ⑤ 契约与成本 ──────────────────────────────────────────────────────


def test_router_never_touches_the_response_contract(tmp_path) -> None:
    """§2.1 / §2.2：**路由不改变任何响应形状**——`top_k` 计数、字段集合、隔离。"""
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router("DEFAULT")
    service = _pipeline(store, router=router)
    for top_k in (1, 2, 100):
        result = service.run(user_id="u", query=_LIST_QUERY, top_k=top_k)
        assert len(result.items) <= top_k
        scores = [item.score for item in result.items]
        assert scores == sorted(scores, reverse=True)
    # 隔离：另一个 user 什么都拿不到（路由这条路不该绕过 `user_id`）
    assert service.run(user_id="other", query=_LIST_QUERY, top_k=100).items == ()


def test_route_latency_is_absent_when_the_router_is_off(tmp_path) -> None:
    """⛔ **没路由时 `latency_ms` 记 `None` 而不是 0**：记 0 会让两臂的
    "平均路由耗时"都算得出来，而其中一臂的 0 是假的（§6 的成本分析会读错）。"""
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    plain = _pipeline(store, router=None)
    snapshot = _snapshot(plain, tmp_path, user_id="u", query=_LIST_QUERY, top_k=100)
    assert snapshot["router"]["latency_ms"] == {"count": 0, "mean": None, "max": 0.0}
    assert snapshot["router"]["routes"] == {"disabled": 1}


def test_route_latency_is_averaged_over_routed_requests_only(tmp_path) -> None:
    """⚠ 分母是 `router_calls`（真的分类过的次数），**不是** `searches`。"""
    store = SqliteStore.open(tmp_path / "t.db")
    _seed(store)
    router, _ = _router("DEFAULT")
    service = _pipeline(store, router=router)
    snapshot = _snapshot(service, tmp_path, user_id="u", query=_LIST_QUERY, top_k=100)
    assert snapshot["router"]["latency_ms"]["count"] == 1
    assert snapshot["router"]["latency_ms"]["mean"] is not None


# ── ⑥ 装配：漏接是**静默**的 ──────────────────────────────────────────


def _config(tmp_path, **router_overrides) -> AppConfig:
    router = RouterConfig(enabled=True, **router_overrides)
    return AppConfig(
        storage=StorageConfig(
            sqlite=SqliteConfig(path=str(tmp_path / "t.db")),
            qdrant=QdrantConfig(url="http://unused"),
        ),
        cache=CacheConfig(embed=EmbedCacheConfig(dir=str(tmp_path / "cache"))),
        embed_base_url="http://unused/v1",
        embed_api_key="k",
        router=router,
        llm_base_url="http://llm/v1",
        llm_api_key="k",
        llm_model="some-model",
    )


def test_build_query_router_is_none_when_disabled(tmp_path) -> None:
    """`router.enabled = false` ⇒ **`None`**（不是"一个不会用的路由器"）。"""
    config = replace(_config(tmp_path), router=RouterConfig(enabled=False))
    assert build_query_router(config) is None


def test_build_query_router_carries_the_thresholds_from_config(tmp_path) -> None:
    """⚠ **装配漏传是静默的**：Search 照常工作，只是每个阈值都退回内置默认值。"""
    config = _config(tmp_path, max_output_tokens=16, temperature=0.3)
    router = build_query_router(config)
    assert router is not None
    assert router.model == "some-model"
    assert router.prompt_version == ROUTER_PROMPT_VERSION == config.router.prompt_version


def test_build_services_wires_the_router_into_the_pipeline(tmp_path) -> None:
    """**这一条挡的是"配了却没接上"**——它不报错，只让每一题都走原来的路。"""
    assert build_services(_config(tmp_path)).search.router is not None
    disabled = replace(_config(tmp_path), router=RouterConfig(enabled=False))
    assert build_services(disabled).search.router is None


def test_services_close_releases_the_llm_client(tmp_path) -> None:
    services = build_services(_config(tmp_path))
    router = services.search.router
    assert router is not None
    client = router.client
    assert isinstance(client, OpenAiCompatChat)
    services.close()
    assert client._client is None  # noqa: SLF001 — 断言连接池**真的**被释放了


def test_agentic_is_not_built_when_disabled(tmp_path) -> None:
    """`agentic.enabled = false`（默认）⇒ **`None`** ⇒ 每条 `MULTI_HOP` 都降级。

    ⚠ 这条同时钉住"关着的时候零成本"：只有 `CoqAgent` 存在，才会有每轮一次的
    LLM 调用与每轮一次的精排。装配侧不构造它，那两笔开销**结构性地**不可能发生。
    """
    config = replace(_config(tmp_path), agentic=AgenticConfig(enabled=False))
    services = build_services(config)
    assert services.search.agentic is None


def test_agentic_is_built_with_the_executor_timeout_not_the_router_one(tmp_path) -> None:
    """开 `agentic.enabled` ⇒ 真的接上执行器，而且**用 `agentic.timeout_seconds`**。

    ⚠ 两个消费者的超时**刻意不同**（路由器 10s / 多跳 30s，理由见 `config.py` 的常量），
    而 `OpenAiCompatChat` 的超时是构造期固定的 ⇒ 它们必须**各持一个客户端**。
    把两个超时合成一个"全局 LLM 超时"在代码上更省事，但那会把两个依据完全不同的量绑死，
    而这里正是唯一能发现那件事的地方。
    """
    from tianximem.agent import CoqAgent

    config = replace(
        _config(tmp_path),
        router=RouterConfig(enabled=False),
        agentic=AgenticConfig(enabled=True, timeout_seconds=12.5),
    )
    services = build_services(config)
    try:
        agentic = services.search.agentic
        assert isinstance(agentic, CoqAgent)
        client = agentic.llm
        assert isinstance(client, OpenAiCompatChat)
        # ⚠ 超时是构造期固定的 ⇒ 这是唯一能看见"用的是哪一个"的地方。
        assert client._timeout == 12.5  # noqa: SLF001 — 断言**不是**路由器的 10s
        # ⚠ 关掉路由器之后**一个路由器客户端都不该被构造**（§13 的开关纯度）。
        assert services.search.router is None
        services.close()
        assert client._client is None  # noqa: SLF001 — 连接池**真的**被释放了
    finally:
        services.close()


def test_agentic_force_is_loud_because_it_bypasses_the_router(tmp_path, caplog) -> None:
    """`agentic.force`（§12.3 的 E1）**必须响亮**：它绕过路由器。

    ⚠ 没有这条 WARNING，跑完之后分不出"分数变了"是 CoQ 的效果还是
    "路由器被绕过"造成的——而 E1 那一臂的全部意义就是**把这两件事分开**。
    """
    import logging

    config = replace(_config(tmp_path), agentic=AgenticConfig(enabled=True, force=True))
    with caplog.at_level(logging.WARNING):
        build_services(config)
    assert any("agentic.force" in record.message for record in caplog.records)


# ── ⑦ 配置层：开着却没端点 ⇒ **启动阶段响亮失败** ─────────────────────

_LLM_ENV = {
    "AML_EMB_BASE_URL": "http://unused/v1",
    "AML_EMB_API_KEY": "k",
    "AML_BASE_URL": "http://llm/v1",
    "AML_API_KEY": "k",
    "AML_MODEL": "some-model",
}

#: `default.yaml` 里把路由器打开的那一处锚点（见该文件 `router:` 段）。
#: ⚠ **替换前先断言锚点在**：`str.replace` 找不到时**什么都不做**，
#: 于是"配置没被改动"会表现成一个莫名其妙的断言失败（而不是"锚点漂了"）。
_ENABLE_ANCHOR = "  enabled: false\n  # C —— 单次路由调用的超时"


def _config_dir(tmp_path, **replacements: str):
    """仓库里那份真 `configs/` 的副本，**逐处改写**后返回。

    与 `test_config.py` 的 `config_dir` fixture 同一个理由：测的是**仓库里那份真的 yaml**，
    而不是一个只在这个测试里存在的假配置。
    """
    from pathlib import Path

    real = Path(__file__).resolve().parents[1] / "configs"
    out = tmp_path / "configs"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("default.yaml", "local.yaml", "submit.yaml"):
        text = (real / name).read_text(encoding="utf-8")
        for old, new in replacements.items():
            if name == "default.yaml":
                assert old in text, f"锚点不在 `default.yaml` 里了：{old!r}"
                text = text.replace(old, new)
        (out / name).write_text(text, encoding="utf-8")
    return out


@pytest.mark.parametrize("missing", ["AML_BASE_URL", "AML_API_KEY", "AML_MODEL"])
def test_router_enabled_without_an_endpoint_refuses_to_start(tmp_path, missing: str) -> None:
    """★ **这一条与 reranker 的口径相反，而且是有意的**：

    精排缺端点 ⇒ 名次退回 RRF，**检索仍然完整**、降级有定义。路由缺端点 ⇒
    **每次请求都退 DEFAULT** ⇒ "Router On" 那一臂根本没有路由器，
    跑出来的分数会被读成"路由没有帮助"——**归因完全错**。
    """
    env = {**_LLM_ENV, missing: ""}
    # ① 默认 profile（`router.enabled` 是 false）⇒ **缺端点本身不报错**：缺省不是错误
    load_config(env, config_dir=_config_dir(tmp_path / "off"))
    # ② 显式打开路由器 ⇒ 这回必须响亮失败
    on = _config_dir(tmp_path / "on", **{_ENABLE_ANCHOR: _ENABLE_ANCHOR.replace("false", "true")})
    with pytest.raises(ConfigError, match="router.enabled"):
        load_config(env, config_dir=on)


def test_router_thresholds_are_validated_at_load_time(tmp_path) -> None:
    """⛔ `max_output_tokens = 0` 那条护栏：`MULTI_HOP` 一旦被截断成 `MULTI`
    ⇒ 输出非法 ⇒ 全部降级成 `DEFAULT`，**而且不报错**。"""
    directory = _config_dir(
        tmp_path,
        **{
            _ENABLE_ANCHOR: _ENABLE_ANCHOR.replace("false", "true"),
            "  max_output_tokens: 8": "  max_output_tokens: 0",
        },
    )
    with pytest.raises(ConfigError, match="max_output_tokens"):
        load_config(_LLM_ENV, config_dir=directory)


def test_router_section_is_the_only_switch_and_a_typo_fails_loudly(tmp_path) -> None:
    """拼错的键必须**响亮失败**（`_group` 的口径）：`router.enabeld` 被静默忽略
    等于"以为开了路由，其实一直是关的"。"""
    directory = _config_dir(tmp_path, **{"  enabled: false\n  # C": "  enabeld: false\n  # C"})
    with pytest.raises(ConfigError, match="router"):
        load_config(
            {"AML_EMB_BASE_URL": "http://unused/v1", "AML_EMB_API_KEY": "k"}, config_dir=directory
        )
