"""`rank/reranker.py` —— 远端精排的接入（§11.2 / D12）。

```text
Hybrid → RRF → 去重 → 【rerank（恰好一次）】 → 扩窗 → 合并 → 打包
                            ↑ 本文件测这一环
```

**三层分开测**，因为它们坏的方式完全不同：

| 层 | 测什么 | 坏了会怎样 |
| --- | --- | --- |
| `RemoteReranker` | 线格式、四条集合校验、异常一律转 `RerankUnavailable` | 请求/响应形状漂了 |
| `SearchPipeline._maybe_rerank` | 恰好一次、降级、两个计数器、不换集合 | 名次错了，或 Search 挂 |
| `build_reranker` | 装配：什么情况下接、什么情况下 `None` | 配了却没接上（表现为"静默不精排"） |

⚠ 本文件**不发真请求**：HTTP 层用 `httpx.MockTransport`（它让"超时/5xx/坏 JSON"
变成可精确构造的输入），pipeline 层用 `FakeReranker`。
真端点的探针是 [`tools/probe_reranker.py`](../tools/probe_reranker.py)——**那一个**打真网关。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import httpx
import pytest
from tests.conftest import Wired
from tests.conftest import seed_pair_in as _seed

from tianxi_am.common.render import render
from tianxi_am.rank import RemoteReranker, RerankUnavailable
from tianxi_am.retrieve import EvidenceChecker
from tianxi_am.service.pipeline import SearchPipeline
from tianxi_am.store.sqlite_store import SqliteStore

#: 仓库里的 `configs/`。**用绝对路径**：相对路径会让用例依赖 cwd
#: （从别的目录跑 pytest 时 `configs/` 找不到，而那是**报错**，不是静默——但没必要）。
_CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# ── 脚手架 ──────────────────────────────────────────────────────────────


class FakeReranker:
    """确定性的假 reranker —— **协议级的**（只收文本、只回分数）。

    `want` 是期望的**输出顺序**（输入位置的排列）：`[2, 0, 1]` = "第三条排最前"。
    `None` = 保持原顺序（这时它仍然是一次真调用，只是没改变名次）。

    `exc` 用来模拟"端点不可用"——**协议承诺的就是抛 `RerankUnavailable`**。
    """

    def __init__(self, *, want: list[int] | None = None, exc: Exception | None = None) -> None:
        self._want = want
        self._exc = exc
        self.calls = 0
        self.queries: list[str] = []
        self.documents: list[list[str]] = []

    @property
    def name(self) -> str:
        return "fake-reranker"

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        self.calls += 1
        self.queries.append(query)
        self.documents.append(list(documents))
        if self._exc is not None:
            raise self._exc
        order = self._want if self._want is not None else list(range(len(documents)))
        # 分数按 `want` 递减：`want[0]` 拿最高分 ⇒ 排序后它就是第一条
        rank_of = {position: place for place, position in enumerate(order)}
        return [float(len(documents) - rank_of[i]) for i in range(len(documents))]


def _ids(store: SqliteStore, count: int, *, step: int = 2, session_id: str = "s1") -> list[str]:
    """`count` 个**互不相邻**的候选（`pair_idx` 隔一个）⇒ 一候选一段、彼此不扩窗。"""
    return [
        _seed(store, i * step, session_id=session_id, question=f"q{i}", answer=f"a{i}")
        for i in range(count)
    ]


def _wire(
    wired: Wired,
    reranker: object,
    *,
    seed_limit: int | None = None,
    radius: int | None = None,
) -> SearchPipeline:
    """把 `wired` 的 Search 换成**接了指定 reranker** 的那一条，其余依赖原样复用。"""
    kwargs: dict[str, object] = {}
    if seed_limit is not None:
        kwargs["seed_limit"] = seed_limit
    if radius is not None:
        kwargs["radius"] = radius
    pipeline = SearchPipeline(
        store=wired.store,
        qdrant=wired.qdrant,
        retriever=wired.retriever,
        checker=EvidenceChecker(),
        counter=wired.counter,
        budget_tokens=wired.budget_tokens,
        reranker=reranker,  # type: ignore[arg-type]
        **kwargs,
    )
    wired.services.search = pipeline
    return pipeline


def _content(wired: Wired, **kwargs) -> str:  # noqa: ANN003
    """一次 Search 的**全部正文拼起来**（跨段看内容时比逐项断言省事）。"""
    return "\n".join(item.content for item in wired.search(**kwargs).items)


# ══ 一、RemoteReranker：线格式与四条集合校验 ═════════════════════════════


def _remote(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    timeout: float = 5.0,
) -> RemoteReranker:
    """一个打假传输层的 `RemoteReranker`（**不发真请求**）。"""
    client = httpx.Client(
        transport=httpx.MockTransport(handler), timeout=timeout, base_url="https://gw.test"
    )
    return RemoteReranker(
        base_url="https://gw.test/v1",
        api_key="k",
        model="Qwen3-Reranker-4B",
        timeout=timeout,
        client=client,
    )


def _ok(*results: dict) -> Callable[[httpx.Request], httpx.Response]:
    """一个总是回 200 + 给定 `results` 的假端点。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"model": "/srv/Qwen3-Reranker-4B", "results": list(results)}
        )

    return handler


def _result(index: int, score: float) -> dict:
    """真实响应的一项。**刻意带上 `text`**——真端点就有它，多一个键不该被拒。"""
    return {"index": index, "score": score, "text": f"doc {index}"}


def test_request_shape_is_the_one_the_endpoint_actually_accepts() -> None:
    """请求体就是**实测出来的那个形状**：`{model, query, documents}`。

    ⚠ **断言里没有 `top_n`**——它是个静默截断的旋钮：传了就只回前 k 条，
    于是 `_parse` 的集合校验会失败 ⇒ 每次 Search 都降级。
    **不传它就等于"全都要"**（实测）。
    """
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["url"] = str(request.url)
        seen["method"] = request.method
        seen["auth"] = request.headers.get("Authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [_result(0, 1.0), _result(1, 2.0)]})

    reranker = _remote(handler)
    scores = reranker.score(query="几点开", documents=["A", "B"])

    assert seen["method"] == "POST"
    assert seen["url"] == "https://gw.test/v1/rerank"
    assert seen["auth"] == "Bearer k"
    body = seen["body"]
    assert isinstance(body, dict)
    assert body["query"] == "几点开"
    assert body["documents"] == ["A", "B"]
    assert body["model"] == "Qwen3-Reranker-4B"
    assert "top_n" not in body, "传 top_n 会静默截断候选（见模块 docstring）"
    # 分数**按输入位置**对齐（index 0 → 1.0，index 1 → 2.0），不是按响应顺序
    assert scores == [1.0, 2.0]


def test_scores_are_aligned_to_input_positions_even_when_results_are_sorted() -> None:
    """响应是**按分数降序**的，但 `score()` 吐出来的是**按输入位置**的分数。

    ⚠ 这一条是"候选集合不可能被改动"的技术基础：调用方拿到的是"第 i 篇得几分"，
    于是它只能对**同一个列表**重排。若把"排好序的结果"返回出去，
    "reranker 少回了一条"就会变成**候选真的少了一条**，而且**看起来完全正常**。
    """
    # index 1 得分最高 ⇒ 响应顺序是 [1, 0]，与输入顺序相反
    reranker = _remote(_ok(_result(1, 0.9), _result(0, 0.1)))

    assert reranker.score(query="q", documents=["first", "second"]) == [0.1, 0.9]


def test_empty_documents_short_circuits_without_calling() -> None:
    """空输入 ⇒ 空输出，**不打网络**（端点对空 `documents` 返回 400，而"没有候选"不是错误）。"""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"results": []})

    reranker = _remote(handler)
    assert reranker.score(query="q", documents=[]) == []
    assert calls["n"] == 0
    assert reranker.calls == 0


# ── 四条集合校验：reranker 只能改顺序，不能改集合 ─────────────────────────


def test_unknown_index_is_rejected() -> None:
    """**未知候选**：`index` 越界 ⇒ 降级（我们会拿着一个不存在的位次去取候选）。"""
    reranker = _remote(_ok(_result(0, 1.0), _result(7, 0.5)))
    with pytest.raises(RerankUnavailable, match="越界"):
        reranker.score(query="q", documents=["A", "B"])


def test_duplicate_index_is_rejected() -> None:
    """**重复候选**：同一个 `index` 出现两次 ⇒ 降级（重复项会被排两次、占两个名额）。"""
    reranker = _remote(_ok(_result(0, 1.0), _result(0, 0.5), _result(1, 0.2)))
    with pytest.raises(RerankUnavailable, match="重复候选"):
        reranker.score(query="q", documents=["A", "B"])


def test_missing_index_is_rejected() -> None:
    """**静默丢候选**：有 `index` 没出现 ⇒ 降级。

    ⚠ 这是四条里**最危险**的一条：少一条证据的响应**看起来完全合法**，
    只是答案阶段能用的材料少了一份——而没有任何地方会报错。
    """
    reranker = _remote(_ok(_result(0, 1.0)))
    with pytest.raises(RerankUnavailable, match="静默丢了候选"):
        reranker.score(query="q", documents=["A", "B"])


@pytest.mark.parametrize(
    ("body", "match"),
    [
        ([], "形状"),
        ({"results": {}}, "形状"),
        ({"results": ["not-a-dict"]}, "非对象"),
        ({"results": [{"index": "0", "score": 1.0}]}, "必须是整数"),
        ({"results": [{"index": True, "score": 1.0}]}, "必须是整数"),  # bool 是 int 的子类！
        ({"results": [{"index": 0, "score": "high"}]}, "必须是数字"),
        ({"results": [{"index": 0}]}, "必须是数字"),
    ],
)
def test_illegal_shapes_are_rejected(body: object, match: str) -> None:
    """格式非法 ⇒ 降级。

    ⚠ `index: true` 那一条不是凑数：`True` **是** `int` 的实例，
    不显式挡掉就会被当成 `index=1` 用下去——**错位的名次不会报错**。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=body)

    reranker = _remote(handler)
    with pytest.raises(RerankUnavailable, match=match):
        reranker.score(query="q", documents=["A", "B"])


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_scores_are_rejected(literal: str) -> None:
    """`NaN` / `±Infinity` ⇒ 降级。

    ⚠ 三个都得**手工构造原始文本**才能送进来（`httpx` 的 `json=` 走
    `allow_nan=False`，会在构造响应时就 `ValueError`）。这不是绕路：
    **非标准 JSON 字面量恰恰是坏端点最可能吐出来的东西**，而 Python 的 `json.loads`
    **默认接受它们**——所以一道"字段存在且是数字"的检查挡不住。

    ⚠ `NaN` 参与排序时比较**恒为假** ⇒ 名次会悄悄退化成"输入顺序"，
    而那个结果**看起来完全正常**——正是本项目最怕的那类失败。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=f'{{"results": [{{"index": 0, "score": {literal}}}]}}')

    with pytest.raises(RerankUnavailable, match="有限数"):
        _remote(handler).score(query="q", documents=["A"])


def test_extra_keys_are_ignored() -> None:
    """**多余的键不算格式非法**——真响应就带 `text` 与 `model`。

    把额外字段当错误，会让实现**对着真端点反而失败**："严格"不等于"拒绝用不到的字段"。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "/srv/Qwen3-Reranker-4B",
                "usage": {"total_tokens": 12},
                "results": [
                    {"index": 0, "score": 0.7, "text": "A", "document": "A", "extra": [1, 2]},
                    {"index": 1, "score": 0.2, "text": "B"},
                ],
            },
        )

    assert _remote(handler).score(query="q", documents=["A", "B"]) == [0.7, 0.2]


# ── 失败一律转 RerankUnavailable（D12：降级而不是 5xx）─────────────────


@pytest.mark.parametrize(
    ("handler_factory", "match"),
    [
        (lambda: _raise(httpx.ReadTimeout("timed out")), "调用失败"),
        (lambda: _raise(httpx.ConnectError("no route")), "调用失败"),
        (lambda: _status(500), "返回 500"),
        (lambda: _status(403), "返回 403"),
        (lambda: _status(422), "返回 422"),
        (lambda: _bad_json(), "不是合法 JSON"),
    ],
)
def test_transport_failures_become_rerank_unavailable(
    handler_factory: Callable[[], Callable[[httpx.Request], httpx.Response]], match: str
) -> None:
    """超时 / 连接失败 / 非 2xx / 坏 JSON ⇒ **一律 `RerankUnavailable`**。

    ⚠ 超时用**抛异常**来模拟（`httpx.ReadTimeout`），不是让假传输层真的睡：
    `MockTransport` 不参与超时计时，睡多久都不会触发超时——那样写出来的用例是**空过的**。
    真超时路径属于同一个 `httpx.HTTPError` 分支，由这一条覆盖。
    """
    reranker = _remote(handler_factory())
    with pytest.raises(RerankUnavailable, match=match):
        reranker.score(query="q", documents=["A"])
    assert reranker.last_error is not None  # ★ 降级必须留下痕迹


def _raise(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


def _status(code: int) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(code, json={"detail": "boom"})

    return handler


def _bad_json() -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    return handler


def test_unexpected_exception_also_degrades_but_is_labelled() -> None:
    """**未预期的异常也降级**（D12：Search 绝不因 reranker 失败），但**标签不同**。

    ⚠ 兜底捕获不是偷懒：一次没排上序 vs 整个 Search 500——后者的代价是 Full 的一次真机会。
    但它**必须留下痕迹**：`last_error` 里带着异常类型，
    所以"适配器里有 bug"与"端点不可用"在诊断上分得开。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise RuntimeError("adapter bug")

    reranker = _remote(handler)
    with pytest.raises(RerankUnavailable, match="未预期的失败（RuntimeError）"):
        reranker.score(query="q", documents=["A"])
    assert "RuntimeError" in (reranker.last_error or "")


def test_constructor_rejects_incomplete_endpoint() -> None:
    """端点或密钥为空 ⇒ **构造时就拒绝**（不要让一个必然失败的客户端进装配）。"""
    for kwargs in ({"base_url": "", "api_key": "k"}, {"base_url": "https://g/v1", "api_key": ""}):
        with pytest.raises(ValueError, match="不得为空"):
            RemoteReranker(model="m", **kwargs)  # type: ignore[arg-type]


# ══ 二、装配：什么时候接、什么时候 None ═════════════════════════════════


def test_build_reranker_returns_a_client_when_fully_configured(tmp_path) -> None:
    """项 1：配置齐备 ⇒ **真的构造 `RemoteReranker`**，不是又传了 `None`。"""
    from tianxi_am.service.app import build_reranker

    config = _app_config(tmp_path, reranker=True)
    reranker = build_reranker(config)

    assert isinstance(reranker, RemoteReranker)
    assert reranker.name == "Qwen3-Reranker-4B"
    assert reranker.url == "https://gw.test/v1/rerank"
    reranker.close()


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({}, "端点与密钥都没填"),
        ({"TIANXI_RERANKER_BASE_URL": "https://gw.test/v1"}, "只填了端点"),
        ({"TIANXI_RERANKER_API_KEY": "test-key"}, "只填了密钥"),
    ],
)
def test_build_reranker_is_none_without_endpoint(tmp_path, env: dict, reason: str) -> None:
    """项 2 的一半：**没配全 ⇒ `None`**（服务照常起，走 `rerank_disabled`）。

    ⚠ "没配"不报错是刻意的（D12：reranker 是唯一不被规则保证可用的组件），
    但它会打一条 WARNING——"想用却没配全"与"明确关掉"是两件事，
    否则它会表现成"每次检索都静默不精排"。
    """
    from tianxi_am.service.app import build_reranker

    config = _app_config(tmp_path, reranker=False, rerank_env=env)
    assert build_reranker(config) is None, reason


def test_build_reranker_is_none_when_explicitly_disabled(tmp_path) -> None:
    """显式 `rerank.enabled = false` ⇒ `None`（**消融开关**，不是漏配）。"""
    import dataclasses

    from tianxi_am.common.config import RerankConfig
    from tianxi_am.service.app import build_reranker

    config = _app_config(tmp_path, reranker=True)
    off = dataclasses.replace(config, rerank=RerankConfig(enabled=False))
    assert build_reranker(off) is None


def _app_config(tmp_path, *, reranker: bool, rerank_env: dict | None = None):  # noqa: ANN001, ANN202
    """造一份 `AppConfig`。`reranker=True` ⇒ 带上完整的 `TIANXI_RERANKER_*`。

    ⚠ **`rerank.enabled` 一律强制为 `true`**：本组用例测的是 `build_reranker`
    的三分支表，而出厂默认值是 `false`（`configs/default.yaml`——A3 未跑之前不默认付
    精排的算力）。不强制打开的话，"齐备 ⇒ 有客户端"会静默变成命中"明确关掉"，
    `test_build_reranker_is_none_without_endpoint` 想验的**缺端点**分支也就走不到了
    ——那就是一条**永远不会 FAIL 的检查**。
    """
    from tianxi_am.common.config import load_config

    env = {
        "AML_EMB_BASE_URL": "http://unused/v1",
        "AML_EMB_API_KEY": "k",
        "TIANXI_SQLITE_PATH": str(tmp_path / "tianxi.db"),
        "TIANXI_QDRANT_URL": "http://unused",
        "TIANXI_EMBED_CACHE_DIR": str(tmp_path / "cache"),
    }
    if reranker:
        env.update(
            {
                "TIANXI_RERANKER_BASE_URL": "https://gw.test/v1",
                "TIANXI_RERANKER_API_KEY": "test-key",
                "TIANXI_RERANKER_MODEL": "Qwen3-Reranker-4B",
            }
        )
    env.update(rerank_env or {})
    config = load_config(env, config_dir=_CONFIGS)
    return replace(config, rerank=replace(config.rerank, enabled=True))


def test_build_services_wires_the_reranker_into_the_search_pipeline(tmp_path) -> None:
    """项 1：**装配点真的接上了**——`services.search.reranker` 是那个客户端。

    ⚠ 这条用例的价值在于它测的是 `build_services`：装配点若传硬编码的 `None`，
    **没有任何别处会发现**（Search 照常工作，只是名次没被精排）。
    """
    from tianxi_am.service.app import build_services

    services = build_services(_app_config(tmp_path, reranker=True))
    try:
        assert isinstance(services.search.reranker, RemoteReranker)
    finally:
        services.close()


def test_close_releases_the_reranker_client(tmp_path) -> None:
    """`Services.close()` 要**连 reranker 的 HTTP 客户端一起**释放。

    ⚠ 漏掉它的表现是"连接池在退出时没关"——平时完全看不出来。
    """
    from tianxi_am.service.app import build_services

    services = build_services(_app_config(tmp_path, reranker=True))
    reranker = services.search.reranker
    assert isinstance(reranker, RemoteReranker)
    services.close()
    assert reranker._client is None  # noqa: SLF001 — 这条断言只能看私有状态


# ══ 三、SearchPipeline：恰好一次、重排、降级、两个计数器 ════════════════


def test_rerank_runs_exactly_once_per_search(wired: Wired) -> None:
    """项 3：**每个 Search query 恰好一次**远端调用（§7.2 同款纪律）。

    ⚠ 用**真的 HTTP 客户端 + 假传输层**（不是 `FakeReranker`）：这样数的是
    **真实请求次数**，而不是"我们以为自己调了几次"。
    """
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        import json

        docs = json.loads(request.content)["documents"]
        return httpx.Response(
            200, json={"results": [_result(i, 1.0 - i * 0.1) for i in range(len(docs))]}
        )

    ids = _ids(wired.store, 3)
    wired.qdrant.by_user["u1"] = ids
    reranker = _remote(handler)
    pipeline = _wire(wired, reranker)

    wired.search(top_k=3)

    assert len(requests) == 1, f"一次 Search 打了 {len(requests)} 次远端"
    assert reranker.calls == 1
    assert pipeline.rerank_calls == 1


def test_rerank_input_is_the_query_plus_every_candidate_text(wired: Wired) -> None:
    """项 4：送出去的是 **query + 每一条候选的正文**，且用的是 `common/render` 的同一份渲染。"""
    ids = _ids(wired.store, 3)
    wired.qdrant.by_user["u1"] = ids
    fake = FakeReranker(want=[2, 0, 1])
    _wire(wired, fake)

    wired.search(query="火车几点开", top_k=3)

    assert fake.queries == ["火车几点开"]
    assert fake.documents == [[render(f"q{i}", f"a{i}") for i in range(3)]]
    # 正文确实从真源来（不是从候选里带的）——渲染里带着库里的事实
    assert fake.documents[0][0] == "Q: q0\nA: a0"


def test_candidate_truth_is_fetched_in_one_batch(wired: Wired, monkeypatch) -> None:  # noqa: ANN001
    """项 5：**一次批量读**，绝不逐 candidate 查询。

    判据：`fetch_pairs_by_ids` 的每次调用都带着**整批 id**（没有单条 id 的调用）。
    逐条查在短生命周期连接模型下要付 N 次 `connect`（D17），
    而 N=100 时那是 100 次连接——**慢，但完全不影响正确性**，所以只有这种断言抓得住。
    """
    ids = _ids(wired.store, 4)
    wired.qdrant.by_user["u1"] = ids
    original = wired.store.fetch_pairs_by_ids
    calls: list[list[str]] = []

    def spy(conn, pair_ids):  # noqa: ANN001
        calls.append(list(pair_ids))
        return original(conn, pair_ids)

    monkeypatch.setattr(wired.store, "fetch_pairs_by_ids", spy)
    _wire(wired, FakeReranker())

    wired.search(top_k=4)

    assert calls, "根本没取正文"
    assert [len(c) for c in calls] == [4, 4], f"出现了非批量的取正文：{calls}"
    assert sorted(calls[0]) == sorted(ids)


def test_successful_rerank_reorders_the_candidates(wired: Wired) -> None:
    """项 6：**顺序真的变了**（不是"调用了但没用结果"）。"""
    ids = _ids(wired.store, 3)  # 三个互不相邻的对 ⇒ 三段，一段一项
    wired.qdrant.by_user["u1"] = ids  # RRF 顺序 = ids
    fake = FakeReranker(want=[2, 0, 1])
    _wire(wired, fake)

    got = wired.search(top_k=3)

    assert [item.id for item in got.items] == [ids[2], ids[0], ids[1]]


def test_candidate_set_is_unchanged_by_rerank(wired: Wired) -> None:
    """**reranker 只能改顺序，不能改集合**（§13 的开关纯度）。

    ⚠ 这条不是"再断言一次顺序"：它断言的是**同一批 id、一条不多一条不少**——
    如果将来有人让 rerank 返回"排好序的 id 列表"，这条会在集合层面炸。
    """
    ids = _ids(wired.store, 5)
    wired.qdrant.by_user["u1"] = ids
    _wire(wired, FakeReranker(want=[4, 3, 2, 1, 0]))

    got = wired.search(top_k=5)

    assert {item.id for item in got.items} == set(ids)
    assert len(got.items) == len(ids)


def test_rank_zero_after_rerank_is_the_expansion_seed(wired: Wired) -> None:
    """项 7 / 8：`seed_limit` 卡的是 **rerank 之后**的名次，不是 RRF 名次。

    **怎么在响应里看见它**：两个 session 各有一条候选 + 一个**不是候选**的邻居。
    谁的名次是 0，谁就扩窗、把它自己的邻居带进来——
    所以"响应里有谁的邻居"直接说明了名次 0 是谁。

    ```text
    s1: pair0 = 邻居"n-s1"      pair1 = 候选 A
    s2: pair0 = 候选 B          pair1 = 邻居"n-s2"
    RRF 顺序 [A, B] ⇒ 不精排时种子是 A（扩出 n-s1）
    精排 want=[1,0] ⇒ 种子变成 B（扩出 n-s2）
    ```

    ⚠ `seed_limit=1` 是**必须的**：默认 30 会让 A 和 B **都**是种子，
    于是两边的邻居都会进来，本用例就什么也证明不了（那是"两条都过"的假绿）。
    """
    a = _seed(wired.store, 1, session_id="s1", question="A", answer="a")
    _seed(wired.store, 0, session_id="s1", question="n-s1", answer="x")
    b = _seed(wired.store, 0, session_id="s2", question="B", answer="b")
    _seed(wired.store, 1, session_id="s2", question="n-s2", answer="y")
    wired.qdrant.by_user["u1"] = [a, b]  # RRF：A 在前

    # ① 对照组：不接 reranker（`None`）⇒ 种子 = A（RRF 名次 0）⇒ 进来的是 s1 的邻居
    _wire(wired, None, seed_limit=1)
    without = _content(wired, top_k=5)
    assert "n-s1" in without
    assert "n-s2" not in without

    # ② 精排把 B 顶到名次 0 ⇒ 进来的是 s2 的邻居（**换了人**）
    _wire(wired, FakeReranker(want=[1, 0]), seed_limit=1)
    with_rerank = _content(wired, top_k=5)
    assert "n-s2" in with_rerank
    assert "n-s1" not in with_rerank


def test_candidates_past_the_seed_limit_are_kept_but_not_expanded(wired: Wired) -> None:
    """项 9：`rank31+`（这里用 `seed_limit=1` 缩小到 `rank 2+`）**保留但不扩邻居**。

    构造：一个 session 5 条连续候选。精排把**最后一条**顶到名次 0，
    `seed_limit=1` ⇒ 只有它能扩窗。
    """
    ids = [_seed(wired.store, i, question=f"q{i}", answer=f"a{i}") for i in range(5)]
    wired.qdrant.by_user["u1"] = ids
    # want=[4,0,1,2,3]：ids[4] 名次 0，其余保持相对顺序
    _wire(wired, FakeReranker(want=[4, 0, 1, 2, 3]), seed_limit=1, radius=1)

    got = wired.search(top_k=10)

    # 5 条都还在（一条不少），只是没被拆开
    assert got.count == 1
    content = got.items[0].content
    for i in range(5):
        assert f"Q: q{i}" in content
    # 名次 0 = ids[4]（pair 4）⇒ 锚点是它
    assert got.items[0].id == ids[4]


@pytest.mark.parametrize(
    ("label", "exc"),
    [
        ("超时/网络", RerankUnavailable("ReadTimeout")),
        ("5xx", RerankUnavailable("返回 500")),
        ("坏 JSON", RerankUnavailable("不是合法 JSON")),
        ("重复候选", RerankUnavailable("重复候选")),
        ("缺候选", RerankUnavailable("静默丢了候选")),
        ("未知候选", RerankUnavailable("越界")),
    ],
)
def test_unavailable_reranker_falls_back_to_rrf_order(
    wired: Wired, label: str, exc: Exception
) -> None:
    """项 10–15：**每一条失败路径都退回 RRF 顺序**，而不是让 Search 失败。

    ⚠ 适配器把六种远端问题**统一**成 `RerankUnavailable`（它的单元测试在上面），
    所以这里可以对六者用同一个断言：流水线**不需要知道**失败细节，
    它只需要知道"这次没排上序"。
    """
    ids = _ids(wired.store, 3)
    wired.qdrant.by_user["u1"] = ids
    pipeline = _wire(wired, FakeReranker(exc=exc))

    got = wired.search(top_k=3)

    assert [item.id for item in got.items] == ids, f"{label} 时没有退回 RRF 顺序"
    assert pipeline.rerank_degraded == 1
    assert pipeline.rerank_disabled == 0
    assert pipeline.rerank_calls == 1  # 打算调了，也确实调了


def test_none_reranker_counts_as_disabled_not_degraded(wired: Wired) -> None:
    """项 17 的一半：**没配 / 关掉 ⇒ `disabled`**，不是 `degraded`。

    ⚠ 两者产出的名次**一模一样**。混为一个计数会让"reranker 一直失败"
    看起来像"我们没打算用它"——那是两种完全不同的处置。
    """
    ids = _ids(wired.store, 2)
    wired.qdrant.by_user["u1"] = ids
    pipeline = _wire(wired, None)

    wired.search(top_k=2)

    assert pipeline.rerank_disabled == 1
    assert pipeline.rerank_degraded == 0
    assert pipeline.rerank_calls == 0


def test_degraded_and_disabled_are_counted_separately_across_searches(wired: Wired) -> None:
    """项 17 的另一半：**两种计数在同一条流水线上各记各的**。

    先跑一次"没配"的，再跑一次"配了但端点挂了"的，最后跑一次"配了且好的"——
    三个计数器必须各自落在该落的地方。
    """
    ids = _ids(wired.store, 2)
    wired.qdrant.by_user["u1"] = ids

    off = _wire(wired, None)
    wired.search(top_k=2)
    assert (off.rerank_disabled, off.rerank_degraded, off.rerank_calls) == (1, 0, 0)

    broken = _wire(wired, FakeReranker(exc=RerankUnavailable("boom")))
    wired.search(top_k=2)
    assert (broken.rerank_disabled, broken.rerank_degraded, broken.rerank_calls) == (0, 1, 1)

    healthy = _wire(wired, FakeReranker(want=[1, 0]))
    wired.search(top_k=2)
    assert (healthy.rerank_disabled, healthy.rerank_degraded, healthy.rerank_calls) == (0, 0, 1)


def test_missing_source_row_degrades_rather_than_disables(wired: Wired, monkeypatch) -> None:  # noqa: ANN001
    """真源缺行 ⇒ **`degraded`**：我们**本来是要调的**，只是这一步没能兑现。

    ⚠ 这条区分很容易写反：判据是**"有没有打算调用"**，不是"有没有调用成功"。
    """
    ids = _ids(wired.store, 2)
    ghost = "f" * 64  # Qdrant 里有、真源里没有
    wired.qdrant.by_user["u1"] = [*ids, ghost]
    pipeline = _wire(wired, FakeReranker())

    wired.search(top_k=3)

    assert pipeline.rerank_degraded == 1
    assert pipeline.rerank_disabled == 0
    assert pipeline.rerank_calls == 0


def test_empty_candidate_set_does_not_call_the_reranker(wired: Wired) -> None:
    """没有候选 ⇒ **不调用**（`disabled`）：没有可排序的东西，调用没有意义。"""
    wired.qdrant.by_user["u1"] = []
    pipeline = _wire(wired, FakeReranker())

    wired.search(top_k=5)

    assert pipeline.rerank_disabled == 1
    assert pipeline.rerank_calls == 0


def test_ties_keep_the_incoming_order(wired: Wired) -> None:
    """同分 ⇒ **按进来的名次**（稳定排序）。

    ⚠ 不稳定的话，同分项的先后会随排序实现漂移——那是"消融不可复现"的经典来源，
    而**每次跑出来的结果都还是合法的**。
    """

    class AllEqual:
        name = "all-equal"

        def score(self, *, query: str, documents: list[str]) -> list[float]:
            return [0.5] * len(documents)

    ids = _ids(wired.store, 4)
    wired.qdrant.by_user["u1"] = ids
    _wire(wired, AllEqual())

    got = wired.search(top_k=4)

    assert [item.id for item in got.items] == ids  # 原样不动


# ══ 四、raw score 绝不外泄（§11.3）══════════════════════════════════════


def test_raw_rerank_score_never_reaches_the_api(wired: Wired) -> None:
    """项 18：reranker 的原始分**只活在适配器内部**，响应里的 `score` 仍是 `1/(rank+1)`。

    ⚠ 这一条同时钉住三件事：①原始分没透传、②`score` 不是融合分数、③**它是按输出位置
    重算的**——若谁改成"照抄 rerank 名次"，第一位就不再是 `1.0`。
    """
    from tianxi_am.service.schemas import SearchResponse, SearchResultItem

    marker = 0.987654321  # 刻意刺眼：出现在响应里一眼就能看见

    class Marked:
        name = "marked"

        def score(self, *, query: str, documents: list[str]) -> list[float]:
            return [marker - i * 1e-6 for i in range(len(documents))]

    ids = _ids(wired.store, 3)
    wired.qdrant.by_user["u1"] = ids
    _wire(wired, Marked())

    got = wired.search(top_k=3)

    assert [item.score for item in got.items] == [1.0, 0.5, 1 / 3]
    body = SearchResponse(
        data=[
            SearchResultItem(id=i.id, content=i.content, created_at=i.created_at, score=i.score)
            for i in got.items
        ]
    ).model_dump_json()
    assert str(marker) not in body
    assert "0.987" not in body


def test_rerank_does_not_change_content_or_created_at(wired: Wired) -> None:
    """精排只动**顺序**：`content` / `created_at` 仍是同一批真源数据。

    ⚠ 如果 reranker 的结果被拿去反查正文，就会出现"名次对了但正文串了"——
    而那种错**在 id 层面完全看不出来**。
    """
    import datetime as dt

    ids = _ids(wired.store, 3)
    a = _seed(wired.store, 100, question="Q100", answer="A100")
    wired.qdrant.by_user["u1"] = [*ids, a]
    stamp = int(dt.datetime(2023, 5, 8, 23, 30, tzinfo=dt.UTC).timestamp() * 1000)
    with wired.store.transaction() as conn:
        conn.execute("UPDATE qa_pairs SET event_time = ? WHERE id = ?", (stamp, a))
    _wire(wired, FakeReranker(want=[3, 0, 1, 2]))

    got = wired.search(top_k=4)
    by_id = {item.id: item for item in got.items}

    assert got.items[0].id == a
    assert by_id[a].content == "Q: Q100\nA: A100"
    assert by_id[a].created_at == "2023-05-08"
    assert by_id[ids[0]].content == "Q: q0\nA: a0"


# ══ 五、HTTP 那一跳：降级之后**仍然是 200**（D12）══════════════════════


@pytest.mark.parametrize(
    "exc",
    [
        RerankUnavailable("超时"),
        RerankUnavailable("返回 503"),
        RerankUnavailable("不是合法 JSON"),
    ],
)
def test_http_search_returns_200_when_the_reranker_degrades(wired: Wired, exc: Exception) -> None:
    """项 16：reranker 挂了 ⇒ **整个 Search 仍然 200**、`data` 依然合法。

    这是 D12 的核心要求：Full 只有 2 次，不能因为一个不被规则保证可用的组件
    把一次真机会变成 5xx。**降级后的响应必须与"没开 rerank"逐字等价地合法。**
    """
    from fastapi.testclient import TestClient

    from tianxi_am.service import create_app

    ids = _ids(wired.store, 3)
    wired.qdrant.by_user["u1"] = ids
    _wire(wired, FakeReranker(exc=exc))

    with TestClient(create_app(wired.services)) as client:
        resp = client.post("/search", json={"user_id": "u1", "query": "q", "top_k": 3})

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 3
    assert [item["id"] for item in data] == ids  # RRF 顺序
    assert set(data[0]) == {"id", "content", "created_at", "score"}


# ══ 六、渲染与计数 ══════════════════════════════════════════════════════


def test_rerank_uses_the_same_render_as_indexing_and_output(wired: Wired) -> None:
    """不变式 I1 在精排这一步同样成立：送进 reranker 的正文 = 返回的 `content` 的原料。

    ⚠ 精排吃的是"检索命中的文本"；如果它自己拼一遍，
    "重排的是哪一段"与"模型读到的是哪一段"就会漂移——**而两边都合法**。
    """
    ids = _ids(wired.store, 2)
    wired.qdrant.by_user["u1"] = ids
    fake = FakeReranker()
    _wire(wired, fake)

    got = wired.search(top_k=2)

    # 送出去的两篇 = 两段 content（候选互不相邻 ⇒ 一段一对，所以能逐字对上）
    assert fake.documents == [[item.content for item in got.items]]


def test_rerank_does_not_touch_the_token_budget(wired: Wired) -> None:
    """精排**不参与预算**：开关 rerank **不改变 token 计数的次数**（§13 的开关纯度）。

    ⚠ 为什么不直接断言"某个次数"：计数次数由**段数**决定（每段一次 + 最终串一次），
    而段数是数据的性质、不是常数。⇒ 正确的问法是**开关前后一不一样**——
    这也正是 §13 要的那个问题："关掉 rerank 是不是只动了顺序"。
    """
    ids = _ids(wired.store, 2)  # ★ 只落一次：两次 `_runs` 复用同一批对
    wired.qdrant.by_user["u1"] = ids

    def _runs(reranker: object) -> int:
        counting = _CountingCounter()
        wired.services.search = SearchPipeline(
            store=wired.store,
            qdrant=wired.qdrant,
            retriever=wired.retriever,
            checker=EvidenceChecker(),
            counter=counting,
            budget_tokens=wired.budget_tokens,
            reranker=reranker,  # type: ignore[arg-type]
        )
        wired.search(top_k=2)
        return counting.calls

    off = _runs(None)
    on = _runs(FakeReranker(want=[1, 0]))

    assert off > 0, "计数器一次都没被调用 —— 这条用例是空过的"
    assert on == off


class _CountingCounter:
    """数一下 `count()` 被调了几次（`FakeCounter` 只数字符，不记调用次数）。"""

    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def count(self, text: str) -> int:
        self.calls += 1
        return len(text)
