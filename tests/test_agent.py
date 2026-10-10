"""Chain-of-Query 多跳执行器（`agent/`）——**不发任何真请求**。

## 这个文件挡的五类错

| 类 | 挡什么 | 靠什么 |
| --- | --- | --- |
| **轮次** | 轮数不受控 / 提前停错 / 该继续却停了 | 脚本化的 LLM 回复 + 逐轮记录 |
| **索引映射** | 数字索引 → **稳定 ID** 错位（§5） | 断言文档序与索引**逐位对应** |
| **证据** | 历史已选证据被本轮名次挤掉（§4.4） | 直接读提示词文本，断言那条证据还在 |
| **降级（§10）** | LLM / 检索坏了却**变成 500** | 假客户端抛异常 + 断言 `fallback_reason` |
| **纯度（§13）** | 关掉开关却仍有 LLM 调用 / 仍有精排 | 假协作者的调用计数 |

⚠ 与 `test_router.py` 同一套做法：假 LLM 是**协议级替身**（`name` + `complete`），
不是 HTTP 桩；HTTP 层的形状由 [`../src/tianximem/llm/`](../src/tianximem/llm/) 自己的用例覆盖。

⚠ **`agent/` 里没有真 Qdrant、没有真 SQLite**：本文件用一个鸭子类型的假 store
（只实现 `read()` + `fetch_pairs_by_ids`），因为执行器**只用到这两样**——
用到第三样就说明它越界了（§8「不改写原文」，原文只有一条来源）。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest

from tianximem.agent import (
    COQ_PROMPT,
    COQ_PROMPT_VERSION,
    AgenticResult,
    CoqAgent,
    InvalidCoqOutput,
    normalize_query,
    parse_coq_decision,
    render_coq_prompt,
)
from tianximem.agent.parse import extract_json_object
from tianximem.llm import LlmUnavailable
from tianximem.rank import RerankUnavailable
from tianximem.retrieve import Candidate

# ── 假协作者 ────────────────────────────────────────────────────────────


class FakeLlm:
    """`LlmClient` 的替身：**按调用次序**回脚本化答案，并记下每一次的提示词。

    `replies` 可以是一段文本（每次都回它）、一个列表（用完最后一个就重复它），
    或一个异常（每次都抛）。要"第 2 次才坏"就给列表里塞一个异常。
    """

    def __init__(self, replies: str | BaseException | list[str | BaseException]) -> None:
        self._replies = replies if isinstance(replies, list) else [replies]
        self.calls: list[str] = []
        self.last_usage: dict[str, int] | None = None

    @property
    def name(self) -> str:
        return "fake-coq"

    def complete(self, prompt: str, *, max_tokens: int = 8, temperature: float = 0.0) -> str:
        index = min(len(self.calls), len(self._replies) - 1)
        self.calls.append(prompt)
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 20}
        reply = self._replies[index]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def prompts(self) -> str:
        return "\n\n====\n\n".join(self.calls)


class FakePair:
    """`QaPair` 的鸭子类型替身——`render_pair` 只要 `question` / `answer` / `event_time`。"""

    def __init__(self, memory_id: str, question: str, answer: str, event_time: int | None = None):
        self.id = memory_id
        self.question = question
        self.answer = answer
        self.event_time = event_time


class FakeStore:
    """只实现执行器真正用到的那两样。⚠ 缺任何一次读取都会**响亮**失败（不静默缺行）。"""

    def __init__(self, pairs: list[FakePair]) -> None:
        self._by_id = {pair.id: pair for pair in pairs}
        self.fetches: list[list[str]] = []

    @contextmanager
    def read(self):
        yield None

    def fetch_pairs_by_ids(self, conn: Any, ids: list[str]) -> list[FakePair]:
        self.fetches.append(list(ids))
        # ⚠ **不放回缺失的**：真 store 就是这么做的，而"悄悄补一条占位"会让
        #   `_rerank_round` 的"长度不等 ⇒ 不重排"那条判断永远不触发。
        return [self._by_id[i] for i in ids if i in self._by_id]


class FakeRetriever:
    """按 query 返回脚本化候选——**并记下每一次调用的形状**（§12.1 的"每轮融合召回"）。"""

    def __init__(
        self,
        by_query: dict[str, list[Candidate]] | None = None,
        *,
        default: list[Candidate] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._by_query = by_query or {}
        self._default = default or []
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    def search(self, *, user_id: str, query: str, top_k: int) -> list[Candidate]:
        self.calls.append({"user_id": user_id, "query": query, "top_k": top_k})
        if self._raises is not None:
            raise self._raises
        return list(self._by_query.get(query, self._default))


class FakeReranker:
    """`Reranker` 的替身：**按给定次序打分**（默认倒序 ⇒ 名次原序不变，便于断言）。"""

    def __init__(self, order: dict[str, float] | None = None, *, raises: bool = False) -> None:
        self._order = order or {}
        self._raises = raises
        self.calls: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "fake-rerank"

    def score(self, *, query: str, documents: list[str]) -> list[float]:
        self.calls.append({"query": query, "documents": list(documents)})
        if self._raises:
            raise RerankUnavailable("端点挂了")
        # 默认：按传入次序**递减**（0, -1, -2 …）⇒ 排序后名次与输入一致。
        return [
            self._order.get(document, float(-index)) for index, document in enumerate(documents)
        ]


# ── 夹具 ────────────────────────────────────────────────────────────────


def _pairs(*specs: tuple[str, str, str]) -> list[FakePair]:
    return [FakePair(mid, q, a) for mid, q, a in specs]


_FOUR = _pairs(
    ("m1", "Where did I join?", "You joined the Atlas lab."),
    ("m2", "Who directs the Atlas lab?", "Professor Wang directs the Atlas lab."),
    ("m3", "What did Professor Wang do before?", "Professor Wang worked at the North Lab."),
    ("m4", "Unrelated", "Something else entirely."),
    ("m5", "Another unrelated one", "Still nothing to do with the question."),
    ("m6", "A third filler", "Filler content, nothing more."),
)


def _candidates(*ids: str) -> list[Candidate]:
    return [Candidate(mid, rank) for rank, mid in enumerate(ids)]


def _reply(
    *,
    sufficient: bool = False,
    indices: list[int] | None = None,
    query: str = "next question",
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "is_sufficient": sufficient,
            "evidence_indices": indices or [],
            "new_query": query,
            "confidence_score": confidence,
        }
    )


def _agent(
    *,
    llm: FakeLlm,
    retriever: FakeRetriever,
    store: FakeStore | None = None,
    reranker: FakeReranker | None = None,
    counter=None,
    **kwargs,
) -> CoqAgent:
    from tests.conftest import FakeCounter

    return CoqAgent(
        store=store or FakeStore(_FOUR),
        retriever=retriever,
        llm=llm,
        counter=counter or FakeCounter(),
        reranker=reranker,
        **kwargs,
    )


# ── ① 原版提示词与填充 ──────────────────────────────────────────────────


def test_the_prompt_is_the_original_one_and_it_is_the_only_one() -> None:
    """⛔ **提示词逐字照用**（方案 §4.1）——它不许被"优化"。

    这里只钉住那些**改了就会改变语义**的锚点，不做整段比对（整段比对会在
    排版上过度敏感）；真正的防线是"文件里只有这一版"这条纪律。
    """
    assert COQ_PROMPT_VERSION == "coq-original"
    for anchor in (
        "You are a meticulous expert in retrieval-augmented question answering evaluation",
        '"is_sufficient" (boolean)',
        '"evidence_indices" (list of 0-based integers)',
        '"new_query" (string, single-line)',
        '"confidence_score" (number 0..1)',
        "If uncertain at any point, choose is_sufficient = false.",
        "NEXT BEST query objective",
        "Output the JSON object only.",
    ):
        assert anchor in COQ_PROMPT
    # ⛔ 提示词里**只有**那三个占位符——多一个就说明有人动过它。
    # ⚠ 一共 5 个花括号而不是 3 个：原版提示词里 `{used_query}` 与
    #   `{retrieved_episodes}` **各出现两次**（"Inputs (schemas)" 段一次、
    #   末尾那段作业说明一次）。填充会把两处都替上——那正是要的行为。
    assert COQ_PROMPT.count("{") == 5
    assert COQ_PROMPT.count("}") == 5
    assert COQ_PROMPT.count("{original_query}") == 1
    assert COQ_PROMPT.count("{used_query}") == 2
    assert COQ_PROMPT.count("{retrieved_episodes}") == 2


def test_filling_the_prompt_does_not_rewrite_it() -> None:
    """填三个变量，**不碰别的**：提示词正文一个字都不改。"""
    prompt = render_coq_prompt(
        original_query="Q1", used_queries=["Q1", "Q2"], documents=["[0]\ndoc"]
    )
    assert "Q1" in prompt and "Q2" in prompt and "[0]\ndoc" in prompt
    # 变量周围的正文照旧在（只替换占位符本身）
    assert "Now perform the task using:" in prompt
    assert "{original_query}" not in prompt
    assert "{used_query}" not in prompt
    assert "{retrieved_episodes}" not in prompt


def test_query_text_with_braces_survives_untouched() -> None:
    """⚠ Query 是**不可信输入**：里面有 `{` 不能把填充搞坏。

    `str.format` 在这里会抛 `KeyError`（好一点）或**替换出一个没人预期的输入**（更糟），
    所以填充走的是一遍扫描的 `re.sub`。这条用例钉住那个选择。
    """
    prompt = render_coq_prompt(
        original_query="What about {original_query} and {} braces?",
        used_queries=["a"],
        documents=["[0]\nb"],
    )
    assert "What about {original_query} and {} braces?" in prompt
    assert "\n**Rewritten Queries Tried**\na\n\n**Retrieved Documents**\n[0]\nb\n" in prompt


def test_a_query_containing_another_placeholder_is_not_substituted_twice() -> None:
    """⛔ 连续三次 `str.replace` 的**具体故障**：一句话里带字面 `{used_query}`。

    换成连续替换，这句 Query 会在**第二次替换时被查询列表顶掉**——
    提示词里凭空多出一段文本，**而且没有任何报错**。一遍扫描从根上排除它。
    """
    prompt = render_coq_prompt(
        original_query="why does it say {used_query} here?",
        used_queries=["first", "second"],
        documents=["[0]\nbody"],
    )
    assert "why does it say {used_query} here?" in prompt


# ── ② 解析：两阶段提取 + 严格校验（方案 §6）─────────────────────────────


def _payload(text: str, count: int = 3):
    return parse_coq_decision(text, document_count=count)


def test_think_block_before_the_json_is_still_parsed() -> None:
    """阶段①的**存在理由**：模型常先吐思考再吐对象。"""
    raw = '<think>\nlet me see\n</think>\n' + _reply(indices=[1], query="next")
    decision = _payload(raw)
    assert decision.evidence_indices == (1,)
    assert decision.new_query == "next"
    # ⚠ **要留下"是抠出来的"这个标记**：它开始退化时这是第一个信号。
    assert decision.extraction == "recovered"


def test_clean_json_is_marked_direct() -> None:
    assert _payload(_reply(indices=[0])).extraction == "direct"


def test_braces_inside_a_string_do_not_break_the_scanner() -> None:
    """⚠ 括号平衡必须**看懂字符串**——否则 `{"a": "}"}` 会被切在引号里。"""
    raw = 'noise {"is_sufficient": false, "evidence_indices": [], ' \
          '"new_query": "what about {braces} and \\"quotes\\"?", "confidence_score": 0.5} tail'
    found = extract_json_object(raw)
    assert found is not None
    payload, extraction = found
    assert extraction == "recovered"
    assert payload["new_query"] == 'what about {braces} and "quotes"?'


def test_no_json_object_at_all_is_rejected() -> None:
    with pytest.raises(InvalidCoqOutput, match="找不到 JSON 对象"):
        _payload("I could not decide.")


@pytest.mark.parametrize(
    "raw,reason",
    [
        ('{"is_sufficient": false, "evidence_indices": []}', "字段不匹配"),
        (
            '{"is_sufficient": false, "evidence_indices": [], "new_query": "q",'
            ' "confidence_score": 0.5, "extra": 1}',
            "多出",
        ),
        # ⛔ **不能把布尔读成整数索引**：`true` 是 `int` 的实例。
        (
            '{"is_sufficient": false, "evidence_indices": [true], "new_query": "q",'
            " \"confidence_score\": 0.5}",
            "必须是整数",
        ),
        # ⛔ **越界即异常**，不是"忽略这一条"。
        (
            '{"is_sufficient": false, "evidence_indices": [3], "new_query": "q",'
            " \"confidence_score\": 0.5}",
            "越界",
        ),
        (
            '{"is_sufficient": 1, "evidence_indices": [], "new_query": "q",'
            " \"confidence_score\": 0.5}",
            "必须是布尔",
        ),
        (
            '{"is_sufficient": false, "evidence_indices": [], "new_query": "a\\nb",'
            " \"confidence_score\": 0.5}",
            "单行",
        ),
        (
            '{"is_sufficient": false, "evidence_indices": [], "new_query": "q",'
            ' "confidence_score": 1.5}',
            "0..1",
        ),
        (
            '{"is_sufficient": false, "evidence_indices": [], "new_query": "q",'
            " \"confidence_score\": \"high\"}",
            "必须是数字",
        ),
    ],
)
def test_strict_validation_rejects_everything_off_spec(raw: str, reason: str) -> None:
    """§6.2 的表：**缺字段不补默认值、布尔不通过整数校验、越界即异常**。"""
    with pytest.raises(InvalidCoqOutput, match=reason):
        _payload(raw)


def test_the_exception_never_carries_the_raw_output() -> None:
    """⚠ 异常里只有原因、**没有原文**：原文是模型读着不可信输入写出来的东西。"""
    with pytest.raises(InvalidCoqOutput) as caught:
        _payload('{"is_sufficient": false}')
    message = str(caught.value)
    assert "{" not in message and "}" not in message
    assert "缺少" in message


def test_duplicate_indices_are_collapsed_not_rejected() -> None:
    """重复索引不是"越界"，只是啰嗦——去重后照常可用。"""
    assert _payload(_reply(indices=[2, 2, 0])).evidence_indices == (2, 0)


def test_normalize_query_is_case_and_whitespace_insensitive() -> None:
    assert normalize_query("  Who   IS\tShe? ") == normalize_query("who is she?")
    assert normalize_query("a b") != normalize_query("ab")


# ── ③ 轮次与停止条件（方案 §7）──────────────────────────────────────────


def test_first_round_sufficient_stops_immediately() -> None:
    """一轮就够 ⇒ **只跑一轮、只调一次 LLM、只精排一次**（§12.1 第一行）。"""
    llm = FakeLlm(_reply(sufficient=True, indices=[0], confidence=0.9))
    retriever = FakeRetriever(default=_candidates("m1", "m2"))
    reranker = FakeReranker()
    agent = _agent(llm=llm, retriever=retriever, reranker=reranker)

    result = agent.search(user_id="u", query="Who directs the lab?", top_k=100)

    assert len(llm.calls) == 1
    assert len(retriever.calls) == 1
    assert len(reranker.calls) == 1
    stats = result.stats
    assert stats.stop_reason == "sufficient"
    assert len(stats.rounds) == 1
    assert stats.rounds[0].is_sufficient is True
    assert stats.rounds[0].selected_ids == ("m1",)
    assert stats.selected_evidence_count == 1
    assert result.fallback_reason is None


def test_second_round_runs_when_evidence_is_insufficient() -> None:
    llm = FakeLlm(
        [
            _reply(indices=[0], query="Who directs the Atlas lab?"),
            _reply(sufficient=True, indices=[1], confidence=0.95),
        ]
    )
    retriever = FakeRetriever(
        by_query={
            "Where did I join?": _candidates("m1", "m4"),
            "Who directs the Atlas lab?": _candidates("m2", "m4"),
        }
    )
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="Where did I join?", top_k=100)

    assert [call["query"] for call in retriever.calls] == [
        "Where did I join?",
        "Who directs the Atlas lab?",
    ]
    assert result.stats.stop_reason == "sufficient"
    assert len(result.stats.rounds) == 2
    # 跨轮累积：第一轮选的 m1 与第二轮选的 m2 都在
    assert result.stats.selected_evidence_count == 2


def test_rounds_never_exceed_max_rounds() -> None:
    """三轮仍不足 ⇒ 停在 `max_rounds`，**绝不多跑第四轮**（§7.4）。"""
    llm = FakeLlm(
        [
            _reply(indices=[0], query="q2"),
            _reply(indices=[0], query="q3"),
            _reply(indices=[0], query="q4"),
        ]
    )
    agent = _agent(
        llm=llm,
        retriever=FakeRetriever(
            default=_candidates("m1", "m2", "m3"),
            by_query={
                "Where did I join?": _candidates("m1"),
                "q2": _candidates("m2"),
                "q3": _candidates("m3"),
                "q4": _candidates("m4"),
            },
        ),
    )
    result = agent.search(user_id="u", query="Where did I join?", top_k=100)

    assert len(llm.calls) == 3
    assert len(result.stats.rounds) == 3
    assert result.stats.stop_reason == "max_rounds"
    # ⚠ **q4 从没被执行过**（它在 max_rounds 那一步被丢掉了）⇒ 它不该出现在
    #   最终精排 Query 里（§9.1："不包含仅生成但没有执行过的 new_query"）。
    assert result.rerank_query is not None
    assert "q4" not in result.rerank_query


def test_a_rewritten_query_equal_to_a_previous_one_stops_the_loop() -> None:
    """§7.5：程序侧的去重保险。⚠ 大小写与空白差异也算重复。"""
    llm = FakeLlm(_reply(indices=[], query="  WHERE   did i join? "))
    retriever = FakeRetriever(by_query={"Where did I join?": _candidates("m1")})
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="Where did I join?", top_k=100)

    assert len(llm.calls) == 1
    assert result.stats.stop_reason == "duplicate_query"


def test_an_empty_next_query_stops_the_loop() -> None:
    llm = FakeLlm(_reply(indices=[], query="   "))
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="q", top_k=100)
    assert result.stats.stop_reason == "empty_query"


def test_sufficient_but_not_confident_is_recorded_as_such() -> None:
    """⚠ 这是个**真实状态**：模型判了"充分"却不敢担保（置信 < 阈值）。

    ⛔ **不要**把它读成"置信度不够就得继续"（§7.5 明令禁止那种读法）——
    提示词在判充分时会原样回吐原问题，所以那一步无论如何都停。
    这里要的是**它停下来的原因能被看见**：混进 `duplicate_query` 会让
    "阈值太高"与"模型提不出新词"变成同一件事。
    """
    llm = FakeLlm(_reply(sufficient=True, indices=[0], query="q", confidence=0.4))
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="q", top_k=100)
    assert result.stats.stop_reason == "low_confidence"
    # ⚠ 但**证据照样收下了**：模型指认的那条与"我们停不停"是两件事。
    assert result.stats.selected_evidence_count == 1


def test_the_threshold_only_gates_the_early_stop() -> None:
    """阈值 0.95 ⇒ 0.9 的"充分"不再提前停；阈值 0.5 ⇒ 同一份输出就停了。"""
    reply = _reply(sufficient=True, indices=[0], confidence=0.9)

    strict = _agent(
        llm=FakeLlm(reply), retriever=FakeRetriever(default=_candidates("m1")),
        sufficiency_threshold=0.95,
    )
    assert strict.search(user_id="u", query="q", top_k=100).stats.stop_reason == "low_confidence"

    loose = _agent(
        llm=FakeLlm(reply), retriever=FakeRetriever(default=_candidates("m1")),
        sufficiency_threshold=0.5,
    )
    assert loose.search(user_id="u", query="q", top_k=100).stats.stop_reason == "sufficient"


# ── ④ 每轮的检索与精排（方案 §3.2 / §12.4）──────────────────────────────


def test_every_round_reranks_with_that_round_query_before_taking_top_n() -> None:
    """⚠ 顺序必须是"融合池 → **精排** → 取 N"，而精排用的是**本轮**的 Query。

    反过来（先用 RRF 名次截 N 再精排）只会重排一堆已被提前排除的候选，
    **而结果看起来完全正常**（§3.2 专门点了这个坑）。
    """
    reranker = FakeReranker()
    agent = _agent(
        llm=FakeLlm(_reply(sufficient=True, indices=[0])),
        retriever=FakeRetriever(default=_candidates("m1", "m2", "m3")),
        reranker=reranker,
        per_round_top_k=2,
    )
    result = agent.search(user_id="u", query="the question", top_k=100)

    # 精排拿到的是**全部**召回候选（3 条），不是预先截出来的 2 条
    assert len(reranker.calls[0]["documents"]) == 3
    assert reranker.calls[0]["query"] == "the question"
    # 而交给 LLM 的只有精排后的 Top-2
    assert result.stats.rounds[0].candidate_ids == ("m1", "m2")
    assert len(result.stats.rounds[0].document_ids) <= 2


def test_round_candidate_depth_follows_the_request_top_k() -> None:
    """⚠ 每轮召回深度取**请求的 `top_k`**，不为多跳另造一套检索参数（§3.2）。"""
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=FakeLlm(_reply(sufficient=True, indices=[0])), retriever=retriever)
    agent.search(user_id="u", query="q", top_k=37)
    assert retriever.calls[0]["top_k"] == 37


def test_round_rerank_can_be_switched_off_for_the_ablation() -> None:
    """§12.4 的 A/B：关掉它 ⇒ **一次精排都不发生**（每轮与最终）。"""
    reranker = FakeReranker()
    retriever = FakeRetriever(default=_candidates("m3", "m1", "m2"))
    agent = _agent(
        llm=FakeLlm(_reply(sufficient=True, indices=[0])),
        retriever=retriever,
        reranker=reranker,
        round_rerank_enabled=False,
    )
    result = agent.search(user_id="u", query="q", top_k=100)

    assert reranker.calls == []
    assert result.stats.rerank_calls == 0
    assert result.stats.rounds[0].rerank_status == "disabled"
    # ⚠ **候选集合不变**（关的是名次，不是候选）——这是 §13 那条纯度要求的落点。
    assert result.stats.rounds[0].candidate_ids == ("m3", "m1", "m2")


def test_a_broken_reranker_degrades_without_killing_the_round() -> None:
    """D12：精排坏了 ⇒ **回原顺序**，本轮照跑。⚠ 但状态要记成 `degraded`。"""
    agent = _agent(
        llm=FakeLlm(_reply(sufficient=True, indices=[0])),
        retriever=FakeRetriever(default=_candidates("m1", "m2")),
        reranker=FakeReranker(raises=True),
    )
    result = agent.search(user_id="u", query="q", top_k=100)
    assert result.stats.rounds[0].rerank_status == "degraded"
    assert result.stats.rounds[0].candidate_ids == ("m1", "m2")
    assert result.stats.stop_reason == "sufficient"


def test_a_missing_source_row_disables_the_round_rerank_instead_of_misaligning() -> None:
    """真源缺行 ⇒ **不重排**（少了几条就没法一一对应），而不是按前缀对齐。"""
    store = FakeStore(_FOUR[:2])  # 只存 m1 / m2
    agent = _agent(
        llm=FakeLlm(_reply(sufficient=True, indices=[0])),
        retriever=FakeRetriever(default=_candidates("m1", "m2", "m3")),
        reranker=FakeReranker(),
        store=store,
    )
    result = agent.search(user_id="u", query="q", top_k=100)
    assert result.stats.rounds[0].rerank_status == "degraded"
    # m3 没有原文 ⇒ 它**进不了文档**，也就不可能被模型指认
    assert "m3" not in result.stats.rounds[0].document_ids


# ── ⑤ 证据索引与原文（方案 §5 / §8）────────────────────────────────────


def test_documents_are_rendered_with_the_shared_renderer_and_indexed_from_zero() -> None:
    """⚠ 文档正文必须走 `render_pair`（不变式 I1），且索引**从 0 起、与顺序逐一对应**。"""
    llm = FakeLlm(_reply(sufficient=True, indices=[0, 1]))
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1", "m2")))
    agent.search(user_id="u", query="q", top_k=100)

    prompt = llm.calls[0]
    assert "[0]\nQ: Where did I join?\nA: You joined the Atlas lab." in prompt
    assert "[1]\nQ: Who directs the Atlas lab?\nA: Professor Wang directs the Atlas lab." in prompt


def test_the_document_header_carries_only_real_metadata() -> None:
    """⚠ 头部只放**真源里真有的**东西（日期，**只到日**，D21）。

    ⛔ 不要为了凑成"Time / Role / Content"三行去编 `Role`：编出来的字段
    就是在给模型假信息，而它会照着它回答。
    """
    pairs = [FakePair("m1", "Q", "A", event_time=1_693_000_000_000)]  # 2023-08-24 UTC
    llm = FakeLlm(_reply(sufficient=True, indices=[0]))
    agent = _agent(
        llm=llm, retriever=FakeRetriever(default=_candidates("m1")), store=FakeStore(pairs)
    )
    agent.search(user_id="u", query="q", top_k=100)

    assert "[0] Time: 2023-08-25" in llm.calls[0]
    assert "Role:" not in llm.calls[0]


def test_the_index_mapping_is_rebuilt_every_round() -> None:
    """⭐ §5.2：**绝对不能复用上一轮的数字索引**。

    做法就是让同一条证据在两轮里**落在不同的索引上**：`m2` 第一轮是索引 1，
    第二轮（本轮只有两条新候选 m4 / m5，历史证据排在后面）变成索引 2。
    模型第二轮回的 `2` 必须解析成 `m2`——用上一轮的映射会解析成 `m5`。
    """
    llm = FakeLlm(
        [
            _reply(indices=[1], query="round two"),  # 第一轮：索引 1 = m2
            _reply(sufficient=True, indices=[2], confidence=0.95),  # 第二轮：索引 2 才是 m2
        ]
    )
    retriever = FakeRetriever(
        by_query={
            "q": _candidates("m1", "m2", "m3"),
            "round two": _candidates("m4", "m5"),
        }
    )
    agent = _agent(llm=llm, retriever=retriever, per_round_top_k=2)
    result = agent.search(user_id="u", query="q", top_k=100)

    first_ids = result.stats.rounds[0].document_ids
    second_ids = result.stats.rounds[1].document_ids
    # 第一轮：Top-2 是 m1 / m2 ⇒ m2 在索引 1
    assert first_ids == ("m1", "m2")
    # 第二轮：本轮 Top-2 是 m4 / m5，历史证据 m2 排在它们后面 ⇒ m2 在索引 2
    assert second_ids == ("m4", "m5", "m2")
    # ⇒ 同一个 `m2`，两轮里的索引**不一样**（1 → 2），而映射各轮重建、各自正确
    assert first_ids.index("m2") == 1
    assert second_ids.index("m2") == 2
    assert result.stats.rounds[1].selected_ids == ("m2",)


def test_the_document_order_never_changes_after_the_index_is_built() -> None:
    """§5.3 第 3 条：映射建立后**不得单独改动文档展示顺序**。

    ⚠ 这条只能靠"提示词里那一段就是 `document_ids` 的顺序"来钉——顺序一动，
    模型给的索引就指向别的证据，而**这件事不会报错**。
    """
    llm = FakeLlm(_reply(sufficient=True, indices=[0, 1, 2]))
    agent = _agent(
        llm=llm, retriever=FakeRetriever(default=_candidates("m1", "m2", "m3"))
    )
    result = agent.search(user_id="u", query="q", top_k=100)

    prompt = llm.calls[0]
    positions = [
        prompt.index(f"[{index}]\n{pair}")
        for index, pair in enumerate(
            (
                "Q: Where did I join?\nA: You joined the Atlas lab.",
                "Q: Who directs the Atlas lab?\nA: Professor Wang directs the Atlas lab.",
                "Q: What did Professor Wang do before?\nA: Professor Wang worked at the North Lab.",
            )
        )
    ]
    assert positions == sorted(positions)  # 索引序与文档序**同向**
    assert result.stats.rounds[0].document_ids == ("m1", "m2", "m3")
    assert result.stats.rounds[0].selected_ids == ("m1", "m2", "m3")


def test_a_duplicate_across_rounds_is_deduplicated_by_stable_id() -> None:
    """同一份证据在多轮命中，**最终候选里只留一份**（§8.2）。"""
    llm = FakeLlm(
        [
            _reply(indices=[], query="q2"),
            _reply(sufficient=True, indices=[], query="q2", confidence=0.95),
        ]
    )
    retriever = FakeRetriever(
        by_query={"q": _candidates("m1", "m2"), "q2": _candidates("m2", "m3")}
    )
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="q", top_k=100)

    ids = [candidate.memory_id for candidate in result.candidates]
    assert ids == ["m1", "m2", "m3"]
    assert len(ids) == len(set(ids))
    assert result.stats.unique_candidate_count == 3
    # ⚠ `candidate_count` 是**去重前**的出现次数（§8.1）：2 ＋ 2
    assert result.stats.candidate_count == 4


def test_history_evidence_is_never_shown_twice_in_one_round() -> None:
    """§5.3 第 6 条：同一轮里同一条只能出现一次。"""
    llm = FakeLlm(
        [
            _reply(indices=[0], query="q2"),
            _reply(sufficient=True, indices=[0], confidence=0.95),
        ]
    )
    retriever = FakeRetriever(
        by_query={"q": _candidates("m1", "m2"), "q2": _candidates("m1", "m3")}
    )
    agent = _agent(llm=llm, retriever=retriever)
    agent.search(user_id="u", query="q", top_k=100)

    # ⚠ 提示词里文档列表**出现两次**（原版提示词的 "Inputs (schemas)" 段与末尾各一次）
    #   ⇒ 只在末尾那份里数。
    tail = llm.calls[1].rsplit("**Retrieved Documents**", 1)[1]
    assert tail.count("Q: Where did I join?") == 1


def test_history_evidence_survives_a_round_where_it_ranks_last() -> None:
    """⭐ §4.4 的硬要求：**已选中的证据不能因为本轮名次低就被丢掉**。

    第二轮里 `m1` 掉出了 Top-N（`per_round_top_k=1` ⇒ 本轮只交一条），
    但它是第一轮选中的证据 ⇒ **它必须还在文档列表里**。
    """
    llm = FakeLlm(
        [
            _reply(indices=[0], query="q2"),
            _reply(sufficient=True, indices=[0], confidence=0.95),
        ]
    )
    retriever = FakeRetriever(by_query={"q": _candidates("m1"), "q2": _candidates("m2")})
    agent = _agent(llm=llm, retriever=retriever, per_round_top_k=1)
    result = agent.search(user_id="u", query="q", top_k=100)

    assert "Q: Where did I join?" in llm.calls[1]
    assert result.stats.rounds[1].document_ids == ("m2", "m1")


def test_the_document_budget_never_drops_history_evidence() -> None:
    """⭐ §4.4：历史证据**先占预算**，然后才轮到本轮候选。

    ⚠ 这里同时钉住一条**可证的不变式**：`history_tokens ≤ max_prompt_tokens` 恒成立
    （历史只可能由**发出去过的**文档组成，而那些已经过预算）⇒ 所以
    `_build_documents` 的"连历史都放不下"那条出口**今天不可达**。
    本用例断言的是**它不会因为预算紧就把历史挤掉**这个正面的性质。
    """
    llm = FakeLlm(
        [
            _reply(indices=[0, 1], query="q2"),
            _reply(sufficient=True, indices=[0], confidence=0.95),
        ]
    )
    retriever = FakeRetriever(by_query={"q": _candidates("m1", "m2"), "q2": _candidates("m3")})
    # 预算足够放下两条历史证据（53 + 78 token），但**放不下第三条**
    agent = _agent(llm=llm, retriever=retriever, max_prompt_tokens=140)
    result = agent.search(user_id="u", query="q", top_k=100)

    second = result.stats.rounds[1]
    # 第一轮选中的 m1 / m2 都在第二轮里；本轮新候选 m3 被预算挤掉了
    assert "Q: Where did I join?" in llm.calls[1]
    assert "Q: Who directs the Atlas lab?" in llm.calls[1]
    assert second.document_ids == ("m1", "m2")
    assert second.dropped_documents == 1


# ── ⑥ 最终候选与联合精排 Query（方案 §9）───────────────────────────────


def test_the_joint_query_lists_the_original_once_then_the_executed_rewrites() -> None:
    """⭐ §9.1：**原问题只出现一次**，后面跟着**实际执行过**的改写，按执行先后。"""
    llm = FakeLlm(
        [
            _reply(indices=[], query="Who directs it?"),
            _reply(indices=[], query="Where did that person work?"),
            _reply(sufficient=True, indices=[], confidence=0.9),
        ]
    )
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="Where did I join?", top_k=100)

    assert result.rerank_query == (
        "Where did I join?\nWho directs it?\nWhere did that person work?"
    )
    assert result.stats.joint_query_count == 3


def test_the_joint_query_never_contains_a_query_that_was_never_executed() -> None:
    """⚠ §9.1 最后一句：**不包含仅生成但没有执行过的 `new_query`**。"""
    llm = FakeLlm(_reply(sufficient=True, indices=[], query="never executed"))
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="the original", top_k=100)
    assert result.rerank_query == "the original"


def test_the_original_query_mode_leaves_the_rerank_query_empty() -> None:
    """§12.6 的对照项：只用原问题 ⇒ `None`（= 混合链那条路的行为）。"""
    agent = _agent(
        llm=FakeLlm(_reply(sufficient=True, indices=[])),
        retriever=FakeRetriever(default=_candidates("m1")),
        final_rerank_query_mode="original",
    )
    result = agent.search(user_id="u", query="q", top_k=100)
    assert result.rerank_query is None


def test_every_round_is_isolated_by_user_id() -> None:
    """用户隔离**每一轮**都要带着（§7.3）——漏掉某一轮就是跨用户检索。"""
    llm = FakeLlm([_reply(indices=[], query="q2"), _reply(sufficient=True, indices=[])])
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=llm, retriever=retriever)
    agent.search(user_id="alice", query="q", top_k=100)
    assert {call["user_id"] for call in retriever.calls} == {"alice"}


# ── ⑦ 降级（方案 §10）——**绝不让一次正常 Search 变成 500** ─────────────


def test_an_llm_failure_keeps_the_candidates_and_does_not_fall_back() -> None:
    """§10.1 第 4、5 条：**保留已经取得的合法候选**，能出合法结果就不回退。"""
    llm = FakeLlm([_reply(indices=[0], query="q2"), LlmUnavailable("端点挂了")])
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1", "m2")))
    result = agent.search(user_id="u", query="q", top_k=100)

    assert result.stats.stop_reason == "llm_unavailable"
    assert result.fallback_reason is None  # ⚠ **不回退**：手上已有候选
    assert [c.memory_id for c in result.candidates] == ["m1", "m2"]
    assert result.stats.rounds[-1].selected_ids == ()  # 无效输出**不更新证据集合**
    assert result.rerank_query == "q\nq2"


def test_an_unexpected_llm_error_is_recorded_differently_from_a_dead_endpoint() -> None:
    llm = FakeLlm(RuntimeError("bug"))
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    assert agent.search(user_id="u", query="q", top_k=100).stats.stop_reason == "llm_error"


def test_invalid_output_never_updates_the_evidence_set() -> None:
    """§10.1 第 2 条 + §6.2：**不补默认值、不用无效索引更新证据**。"""
    llm = FakeLlm('{"is_sufficient": true, "evidence_indices": [99], "new_query": "q",'
                  ' "confidence_score": 0.9}')
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="q", top_k=100)

    assert result.stats.stop_reason == "invalid_output"
    assert result.stats.selected_evidence_count == 0
    assert result.stats.rounds[0].selected_ids == ()
    # ⚠ 候选**照样交出去**（那是已经拿到的合法结果）
    assert result.candidates


def test_a_retrieval_failure_with_nothing_collected_falls_back() -> None:
    """§10.2：检索**调用失败** ≠ 没有相关记忆 ⇒ 交回混合检索（§10.1 第 6 条）。"""
    agent = _agent(
        llm=FakeLlm(_reply()),
        retriever=FakeRetriever(raises=RuntimeError("qdrant down")),
    )
    result = agent.search(user_id="u", query="q", top_k=100)

    assert result.fallback_reason == "retrieval_error"
    assert result.candidates == ()
    assert result.stats.stop_reason == "retrieval_error"


def test_a_retrieval_failure_after_evidence_was_collected_keeps_that_evidence() -> None:
    """已经攒到证据之后再坏 ⇒ **停下手上的证据交出去**，不回退。"""
    retriever = FakeRetriever(by_query={"q": _candidates("m1")})
    agent = _agent(llm=FakeLlm(_reply(indices=[0], query="q2")), retriever=retriever)

    original = retriever.search

    def flaky(*, user_id: str, query: str, top_k: int) -> list[Candidate]:
        if query == "q2":
            raise RuntimeError("qdrant down")
        return original(user_id=user_id, query=query, top_k=top_k)

    retriever.search = flaky  # type: ignore[method-assign]
    result = agent.search(user_id="u", query="q", top_k=100)

    assert result.fallback_reason is None
    assert result.stats.stop_reason == "retrieval_error"
    assert [c.memory_id for c in result.candidates] == ["m1"]


def test_an_empty_retrieval_is_not_a_failure() -> None:
    """⚠ §10.2 要区分的那两件事：**空结果**照常问模型（它会判"不足"），**异常**才降级。"""
    llm = FakeLlm(_reply(indices=[], query="a better question"))
    agent = _agent(llm=llm, retriever=FakeRetriever(default=[]))
    result = agent.search(user_id="u", query="q", top_k=100)

    assert len(llm.calls) >= 1  # 空候选**照样问了**（§10.2 要区分的那两件事）
    assert result.fallback_reason is None
    assert result.stats.rounds[0].recalled_candidates == 0
    # 文档列表为空 ⇒ 模型只能回 []，而它回不了越界的索引
    assert result.stats.rounds[0].document_ids == ()


# ── ⑧ 构造期断言 ────────────────────────────────────────────────────────


def test_construction_refuses_a_prompt_version_the_code_does_not_have() -> None:
    with pytest.raises(ValueError, match="只有一版提示词"):
        _agent(
            llm=FakeLlm(_reply()),
            retriever=FakeRetriever(),
            prompt_version="v2",
        )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_rounds": 0}, "max_rounds"),
        ({"per_round_top_k": 0}, "per_round_top_k"),
        ({"sufficiency_threshold": 1.5}, "sufficiency_threshold"),
        ({"max_prompt_tokens": 0}, "max_prompt_tokens"),
        ({"max_output_tokens": 0}, "max_output_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"final_rerank_query_mode": "whatever"}, "final_rerank_query_mode"),
    ],
)
def test_construction_refuses_impossible_limits(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _agent(llm=FakeLlm(_reply()), retriever=FakeRetriever(), **kwargs)


# ── ⑨ 它没有越界 ────────────────────────────────────────────────────────


def test_the_agent_never_imports_the_http_layer_or_the_router() -> None:
    """AST：执行器**不认识 HTTP、也不认识路由器**（两者接线在 `service/`）。

    ⚠ 这条挡的是"顺手从 agent 里调一下路由器"那类越界——它会让两个模块
    互相纠缠，而 §13 的消融需要它们**各自可关**。
    """
    import ast
    from pathlib import Path

    forbidden = ("tianximem.service", "tianximem.retrieve.router")
    for path in sorted(Path("src/tianximem/agent").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                for prefix in forbidden:
                    assert not node.module.startswith(prefix), f"{path.name} 引入了 {node.module}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    for prefix in forbidden:
                        assert not alias.name.startswith(prefix), f"{path.name} 引入了 {alias.name}"


def test_the_agent_result_carries_no_answer_text() -> None:
    """§2.1 的红线：产出物是**候选与名次**，里面没有答案。"""
    agent = _agent(
        llm=FakeLlm(_reply(sufficient=True, indices=[0])),
        retriever=FakeRetriever(default=_candidates("m1")),
    )
    result: AgenticResult = agent.search(user_id="u", query="q", top_k=100)
    assert result.candidates and all(
        isinstance(candidate, Candidate) for candidate in result.candidates
    )
    blob = json.dumps(
        {
            "stop_reason": result.stats.stop_reason,
            "queries": result.rerank_query,
        }
    )
    assert "Professor Wang" not in blob
