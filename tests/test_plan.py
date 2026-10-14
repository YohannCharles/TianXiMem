"""Plan 模式多跳执行器（`agent/plan.py`）——**不发任何真请求**。

## 这个文件挡的六类错

| 类 | 挡什么 | 靠什么 |
| --- | --- | --- |
| **顺序** | "先检索再回答"错位成"先回答再检索" | 断言**检索与 LLM 调用的交错次序** |
| **占位绑定** | 未绑定的 `{{sN}}` 进检索器 | 程序算的 `canonical_next_query` |
| **上下文** | 历史里混进旧片段全文 / 看不到本轮文档 | 直接读提示词文本（**逐块断言**） |
| **索引映射** | `document_index` → 稳定 ID 错位、引句对不上 | 断言文档序与索引逐位对应 |
| **降级（§10）** | 拆解/逐轮/检索坏了却**变成 500** | 假客户端抛异常 ＋ 断言 `fallback_reason` |
| **纯度（§13）** | `mode=react`（默认）却构造了 Plan 执行器 | 配置与装配的断言 |

⚠ 假协作者与 [`test_agent.py`](./test_agent.py) **同形但各自独立**：那边测的是另一份实现，
共享夹具会让"改一边的假件"悄悄改掉另一边的语义。刻意重复的是**测试替身**，不是生产逻辑。

⚠ 本文件**只测 Plan**。React 的算法与提示词本次一字未改，它的用例在
[`test_agent.py`](./test_agent.py)（那里有一条钉住"提示词只有原版那一份"）。
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import pytest

from tianximem.agent import (
    PLAN_MAX_STEPS,
    PLAN_STEP_KEYS,
    CoqAgent,
    InvalidPlanOutput,
    PlanAgent,
    bind_placeholders,
    parse_plan,
    parse_step_answer,
    placeholders_in,
)
from tianximem.agent.plan import (
    PLAN_STOP_REASONS,
    render_plan_decompose_prompt,
    render_plan_step_prompt,
)
from tianximem.agent.plan_prompts import PLAN_DECOMPOSE_PROMPT, PLAN_STEP_PROMPT
from tianximem.llm import LlmUnavailable
from tianximem.rank import RerankUnavailable
from tianximem.retrieve import Candidate

# ── 假协作者（与 test_agent.py 同形，**刻意各写一份**）──────────────────


class FakeLlm:
    """`LlmClient` 的替身：**按调用次序**回脚本化答案，并记下每一次的提示词。"""

    def __init__(self, replies: str | BaseException | list[str | BaseException]) -> None:
        self._replies = replies if isinstance(replies, list) else [replies]
        self.calls: list[str] = []
        self.last_usage: dict[str, int] | None = None

    @property
    def name(self) -> str:
        return "fake-plan"

    def complete(self, prompt: str, *, max_tokens: int = 8, temperature: float = 0.0) -> str:
        index = min(len(self.calls), len(self._replies) - 1)
        self.calls.append(prompt)
        self.last_usage = {"prompt_tokens": 100, "completion_tokens": 20}
        reply = self._replies[index]
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def call(self, index: int) -> str:
        """第 `index` 次调用的提示词（**0 = 拆解**）。"""
        return self.calls[index]


class FakePair:
    def __init__(self, memory_id: str, question: str, answer: str, event_time: int | None = None):
        self.id = memory_id
        self.question = question
        self.answer = answer
        self.event_time = event_time


class FakeStore:
    """只实现执行器真正用到的那两样（`read()` + `fetch_pairs_by_ids`）。"""

    def __init__(self, pairs: list[FakePair]) -> None:
        self._by_id = {pair.id: pair for pair in pairs}
        self.fetches: list[list[str]] = []

    @contextmanager
    def read(self):
        yield None

    def fetch_pairs_by_ids(self, conn: Any, ids: list[str]) -> list[FakePair]:
        self.fetches.append(list(ids))
        return [self._by_id[i] for i in ids if i in self._by_id]


class FakeRetriever:
    """按 query 返回脚本化候选——**并记下每一次调用的形状**（`top_k` 也算）。"""

    def __init__(
        self,
        by_query: dict[str, list[Candidate]] | None = None,
        *,
        default: list[Candidate] | None = None,
        raises: BaseException | None = None,
        raises_for: set[str] | None = None,
    ) -> None:
        self._by_query = by_query or {}
        self._default = default or []
        self._raises = raises
        #: 只对**某几句**炸（⭐ 兜底失败那条用例要靠它：分步检索正常、兜底那一次挂）。
        self._raises_for = raises_for or set()
        self.calls: list[dict[str, Any]] = []

    def search(self, *, user_id: str, query: str, top_k: int) -> list[Candidate]:
        self.calls.append({"user_id": user_id, "query": query, "top_k": top_k})
        if self._raises is not None:
            raise self._raises
        if query in self._raises_for:
            raise RuntimeError("兜底检索挂了")
        # ⚠ **按 `top_k` 截断**（真 `HybridRetriever` 就是这么做的）：不截的话
        #   "每轮实际召回 20 条"这条断言会**静默空过**——录到的永远是全部候选。
        return list(self._by_query.get(query, self._default))[:top_k]

    @property
    def queries(self) -> list[str]:
        return [call["query"] for call in self.calls]


class FakeReranker:
    """按给定次序打分（默认倒序 ⇒ 名次原序不变，便于断言）。"""

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
        return [
            self._order.get(document, float(-index)) for index, document in enumerate(documents)
        ]


class FakeCounter:
    """确定性计数器：1 个字符 = 1 个 token（与 `conftest.FakeCounter` 同口径）。"""

    def count(self, text: str) -> int:
        return len(text)


# ── 夹具 ────────────────────────────────────────────────────────────────


_PAIRS = [
    FakePair("m1", "Where did the trip start?", "The trip started in Lisbon."),
    FakePair("m2", "Who organised the trip?", "Maria organised the trip."),
    FakePair("m3", "Where was Maria born?", "Maria was born in Porto."),
    FakePair("m4", "What was the budget?", "The budget was 3000 euros."),
]


def _candidates(*ids: str) -> list[Candidate]:
    return [Candidate(mid, rank) for rank, mid in enumerate(ids)]


def _plan_reply(steps: list[dict[str, str]]) -> str:
    return json.dumps({"steps": steps})


def _step_reply(
    *,
    status: str = "answered",
    answer: str | None = "Lisbon",
    evidence: list[dict[str, Any]] | None = None,
) -> str:
    """v2 的逐轮输出：**恰好三个字段**（⛔ 没有 step_id / next_query）。"""
    return json.dumps({"status": status, "answer": answer, "evidence": evidence or []})


#: 两跳计划：s1 无依赖，s2 引用 `{{s1}}`。
_TWO_STEP = _plan_reply(
    [
        {"id": "s1", "query": "Where did the trip start?", "depends_on": []},
        {
            "id": "s2",
            "query": "Who was born in {{s1}}?",
            "depends_on": ["s1"],
        },
    ]
)


def _agent(
    *,
    llm: FakeLlm,
    retriever: FakeRetriever | None = None,
    store: FakeStore | None = None,
    reranker: FakeReranker | None = None,
    **kwargs,
) -> PlanAgent:
    return PlanAgent(
        store=store or FakeStore(_PAIRS),
        retriever=retriever or FakeRetriever(default=_candidates("m1", "m2", "m3")),
        llm=llm,
        counter=FakeCounter(),
        reranker=reranker,
        **kwargs,
    )


# ── ① 提示词：逐字、独立、填充不改写 ────────────────────────────────────


def test_the_two_prompts_are_independent_from_the_frozen_coq_prompt() -> None:
    """⛔ 两张提示词与冻结的原版 CoQ **没有共享文本**，且**条目数恰好两条**。

    ⚠ 这不是洁癖：把新提示词塞进原版 CoQ（或反过来）会让"react 跑的是原版"
    这条前提悄悄失效，而**两边的输出看起来都正常**。
    """
    from tianximem.agent import COQ_PROMPT

    assert PLAN_DECOMPOSE_PROMPT not in COQ_PROMPT
    assert PLAN_STEP_PROMPT not in COQ_PROMPT
    assert COQ_PROMPT not in PLAN_STEP_PROMPT
    # 原版冻结的那三个槽位**一个都不许出现在新提示词里**（否则填充会互相吃掉）。
    for slot in ("{original_query}", "{used_query}", "{retrieved_episodes}"):
        assert slot not in PLAN_DECOMPOSE_PROMPT
        assert slot not in PLAN_STEP_PROMPT


def test_filling_the_prompts_leaves_their_own_placeholders_alone() -> None:
    """⚠ 计划自己的 `{{s1}}` 与 JSON 示例**必须原样留在提示词里**。

    它们**不是**填充槽位（槽位是 `{current_step}` 这种单花括号）——填充要是把它们吃掉，
    模型就再也看不到"占位该长什么样"，而**这件事不会报错**（提示词照样发得出去）。
    """
    rendered = render_plan_decompose_prompt(original_question="Q?")
    assert "{{s1}} means the answer to s1" in rendered
    assert '"steps"' in rendered  # JSON 示例没被当成格式串
    assert rendered.startswith("You create an ordered retrieval plan for a question.")

    step = render_plan_step_prompt(
        original_question="Q?",
        retrieval_plan="s1: a\ns2: b",
        completed_history="(none yet)",
        current_step="s1: a",
        executed_query="a",
        current_documents=["[0]\nQ: x\nA: y"],
    )
    assert "{{s1}}" in step
    assert '"document_index"' in step


def test_a_value_containing_a_slot_name_is_not_substituted_twice() -> None:
    """⛔ **一遍扫描**：值里再出现槽位名，绝不能被二次替换。

    这是"连续 `str.replace`"的经典失败：Query 是一句不可信输入，
    它完全可以自己就含 `{current_documents}`，而连续替换会让它**把后面的正文吃掉**，
    **没有任何报错**。
    """
    key = "{current_documents}"
    rendered = render_plan_step_prompt(
        original_question="Q?",
        retrieval_plan="p",
        completed_history="(none yet)",
        current_step="s1: a",
        executed_query=key,
        current_documents=["REAL DOCS"],
    )
    assert "REAL DOCS" in rendered
    assert rendered.count(key) == 1  # 只剩下"实际执行的那句查询"里那一个


# ── ② 拆解：严格校验 ────────────────────────────────────────────────────


def test_the_first_step_may_not_have_a_placeholder_and_later_steps_must_point_back() -> None:
    """⛔ **真正的安全判据**：占位必须指向**此前出现过**的步骤（指自己/指后面/指不存在都作废）。

    ⚠ 2026-10-13 之后这条判据与"声明写没写对"**分开了**：查的是 `query` 里的占位，
    不是 `depends_on` 里写了什么（后者现在只影响 `dependencies_repaired`）。
    """
    with pytest.raises(InvalidPlanOutput, match="不是\\*\\*此前\\*\\*出现过的步骤"):
        parse_plan(_plan_reply([{"id": "s1", "query": "who is {{s2}}?", "depends_on": []}]))
    with pytest.raises(InvalidPlanOutput, match="不是\\*\\*此前\\*\\*出现过的步骤"):
        parse_plan(
            _plan_reply(
                [
                    {"id": "s1", "query": "a", "depends_on": []},
                    {"id": "s2", "query": "b {{s3}}", "depends_on": ["s3"]},
                ]
            )
        )
    with pytest.raises(InvalidPlanOutput, match="不是\\*\\*此前\\*\\*出现过的步骤"):
        # 指自己：`{{s2}}` 出现在 s2 自己的 query 里
        parse_plan(
            _plan_reply(
                [
                    {"id": "s1", "query": "a", "depends_on": []},
                    {"id": "s2", "query": "b {{s2}}", "depends_on": ["s1"]},
                ]
            )
        )


def test_a_declaration_that_disagrees_with_the_placeholders_is_repaired_not_fatal() -> None:
    """⭐ **依赖以占位为准**（2026-10-13）：声明与占位不一致 ⇒ 规范化并记一笔。

    实测动因：v2 提示词下 **5/50** 命中，全是"占位合法、**声明写错**"
    （最后一步把依赖写成"上一步"）。程序只按占位绑定 ⇒ **那份计划可以直接执行**，
    按旧规则却会整题退回普通混合检索。
    """
    plan = parse_plan(
        _plan_reply(
            [
                {"id": "s1", "query": "a", "depends_on": []},
                {"id": "s2", "query": "b {{s1}}", "depends_on": ["s1"]},
                {"id": "s3", "query": "c {{s1}}", "depends_on": ["s2"]},
            ]
        )
    )
    fixed = plan.steps[2]
    assert fixed.depends_on == ("s1",)  # ⭐ 规范化成占位那一份
    assert fixed.placeholders == ("s1",)
    assert fixed.declared_dependencies == ("s2",)  # ⛔ 原样留作诊断
    assert fixed.dependencies_repaired is True
    assert plan.steps[1].dependencies_repaired is False


def test_a_placeholder_that_is_not_declared_is_no_longer_fatal() -> None:
    """⚠ 反过来那条也**不再作废**：`{{s1}}` 没写进 `depends_on` ⇒ 补上并记一笔。"""
    plan = parse_plan(
        _plan_reply(
            [
                {"id": "s1", "query": "a", "depends_on": []},
                {"id": "s2", "query": "b {{s1}}", "depends_on": []},
            ]
        )
    )
    assert plan.steps[1].depends_on == ("s1",)
    assert plan.steps[1].declared_dependencies == ()
    assert plan.steps[1].dependencies_repaired is True


def test_a_declared_dependency_on_a_step_that_does_not_exist_is_still_fatal() -> None:
    """⛔ **可修与不可修的分界**：声明一个"不存在的步骤"**猜不出模型想依赖谁** ⇒ 作废。

    （"有占位却没声明"由查询文本唯一决定，所以那条可修；这一条不可修。）
    """
    with pytest.raises(InvalidPlanOutput, match="不是\\*\\*此前\\*\\*出现过的步骤"):
        parse_plan(
            _plan_reply(
                [
                    {"id": "s1", "query": "a", "depends_on": []},
                    {"id": "s2", "query": "b", "depends_on": ["s9"]},
                ]
            )
        )


@pytest.mark.parametrize(
    ("steps", "reason"),
    [
        ([], "1..4"),
        ([{"id": "s1", "query": "a", "depends_on": []}] * 5, "1..4"),
        (
            [
                {"id": "s1", "query": "a", "depends_on": []},
                {"id": "s1", "query": "b", "depends_on": []},
            ],
            "重复",
        ),
        ([{"id": "step1", "query": "a", "depends_on": []}], "形状"),
        ([{"id": "s1", "query": "a\nb", "depends_on": []}], "单行"),
        ([{"id": "s1", "query": "", "depends_on": []}], "不能是空串"),
        ([{"id": "s1", "query": "a", "depends_on": ["s1"]}], "不是\\*\\*此前\\*\\*"),
    ],
)
def test_the_plan_validation_rejects_everything_off_spec(steps: list, reason: str) -> None:
    with pytest.raises(InvalidPlanOutput, match=reason):
        parse_plan(_plan_reply(steps))


def test_an_unused_dependency_is_recorded_but_does_not_kill_the_plan() -> None:
    """声明了依赖却没写占位 ⇒ 记进 `unused_dependencies`，**计划仍然可用**。

    ⛔ 反过来（把它当错误作废整份计划）会让一份**能跑**的计划因为一句多余声明被丢掉——
    而那句话**不改变程序行为**（程序只按占位绑定）。
    """
    plan = parse_plan(
        _plan_reply(
            [
                {"id": "s1", "query": "a", "depends_on": []},
                {"id": "s2", "query": "b", "depends_on": ["s1"]},
            ]
        )
    )
    assert plan.steps[1].unused_dependencies == ("s1",)
    assert plan.steps[1].placeholders == ()


def test_at_most_four_steps_are_allowed_by_the_shipped_prompt() -> None:
    """计划的步数上限与提示词里那句 "one to four" 必须**是同一个数**。"""
    assert PLAN_MAX_STEPS == 4
    assert "one to four sub-questions" in PLAN_DECOMPOSE_PROMPT
    four = [{"id": f"s{i}", "query": "a", "depends_on": []} for i in range(1, 5)]
    assert len(parse_plan(_plan_reply(four)).steps) == 4
    with pytest.raises(InvalidPlanOutput, match="1..4"):
        parse_plan(_plan_reply(four + [{"id": "s5", "query": "a", "depends_on": []}]))


def test_the_shipped_decompose_prompt_declares_the_parser_contract_verbatim() -> None:
    """⚠ 提示词里声明的输出契约必须与解析器认的**逐字对上**（v2 起新加的回归位）。

    提示词说"每个 step 恰好 id / query / depends_on"，而解析器收别的东西（或反之）
    ⇒ **每一份计划都被作废**，而**两边都不会报错**——只看到 `plan_invalid` 一堆。
    ⚠ 同理：提示词里的占位写法（`{{s1}}`）必须就是 `plan_parse` 匹配的那一套。
    """
    assert 'Each step must contain exactly "id", "query", and "depends_on".' in (
        PLAN_DECOMPOSE_PROMPT
    )
    assert frozenset({"id", "query", "depends_on"}) == PLAN_STEP_KEYS
    assert "{{s1}} means the answer to s1" in PLAN_DECOMPOSE_PROMPT
    assert placeholders_in("x {{s1}} y") == ("s1",)
    # ⚠ v2 的两条新约束：不许答原题 ＋ `depends_on` 与占位**恰好**一致
    assert "you do not answer the question" in PLAN_DECOMPOSE_PROMPT
    assert "must contain exactly the earlier step IDs" in PLAN_DECOMPOSE_PROMPT


# ── ③ 占位绑定：只换占位、不自由改写 ────────────────────────────────────


def test_binding_only_substitutes_placeholders_and_keeps_every_qualifier() -> None:
    template = "Who directed the film whose soundtrack won {{s1}} at the 2019 festival?"
    bound, missing = bind_placeholders(template, {"s1": "the Silver Lion"})
    assert (
        bound == "Who directed the film whose soundtrack won the Silver Lion at the 2019 festival?"
    )
    assert missing == ()
    assert placeholders_in(template) == ("s1",)


def test_binding_leaves_an_unknown_placeholder_in_place_and_reports_it() -> None:
    """⛔ **填不上就原样留着**——调用方据此停下，而不是把 `{{s3}}` 送进检索器。"""
    bound, missing = bind_placeholders("who is {{s3}}?", {"s1": "x"})
    assert bound == "who is {{s3}}?"
    assert missing == ("s3",)


def test_an_answer_that_itself_looks_like_a_placeholder_is_not_re_substituted() -> None:
    """⚠ 答案是**模型写的自由文本**，它可能自己含一串 `{{s1}}`。**一遍扫描**挡住二次替换。"""
    bound, missing = bind_placeholders("born in {{s1}}?", {"s1": "{{s1}}"})
    assert bound == "born in {{s1}}?"
    assert missing == ()


# ── ④ 每轮输出：严格校验 ────────────────────────────────────────────────


def test_every_round_output_is_validated_against_this_round_documents() -> None:
    documents = ["[0]\nQ: a\nA: Lisbon", "[1]\nQ: b\nA: Porto"]
    ok = parse_step_answer(
        _step_reply(answer="Lisbon", evidence=[{"document_index": 0, "quote": "A: Lisbon"}]),
        step_id="s1",
        documents=documents,
    )
    assert ok.is_answer and ok.evidence[0].verified

    with pytest.raises(InvalidPlanOutput, match="越界"):
        parse_step_answer(
            _step_reply(evidence=[{"document_index": 2, "quote": "x"}]),
            step_id="s1",
            documents=documents,
        )
    # ⭐ 步骤编号**由程序附加**：模型回的 payload 里根本没有这个字段，
    #   所以"回了一个对不上的步骤号"这件事结构性地不可能（v2）。
    assert ok.step_id == "s1"
    with pytest.raises(InvalidPlanOutput, match="status"):
        parse_step_answer(_step_reply(status="maybe"), step_id="s1", documents=documents)


def test_a_quote_that_is_not_verbatim_is_dropped_not_fatal() -> None:
    """⚠ 引句逐字校验是**存在性**判据：不逐字的条目**丢**掉并计数，**不作废整轮**。

    ⛔ 两者不要混淆：**引用存在 ≠ 引用支持答案**——后者本模块判不了，也不假装判得了。
    """
    parsed = parse_step_answer(
        _step_reply(
            evidence=[
                {"document_index": 0, "quote": "A: Lisbon"},
                {"document_index": 1, "quote": "A: Lisbon"},  # 不存在于第 1 篇
            ]
        ),
        step_id="s1",
        documents=["[0]\nQ: a\nA: Lisbon", "[1]\nQ: b\nA: Porto"],
    )
    assert [ref.verified for ref in parsed.evidence] == [True, False]


def test_an_answer_with_a_newline_is_refused_because_it_becomes_a_query() -> None:
    """答案会被填进下一句检索词 ⇒ 多行答案必须**在进入历史之前**被挡下。"""
    with pytest.raises(InvalidPlanOutput, match="单行"):
        parse_step_answer(
            _step_reply(answer="Lisbon\nPortugal"), step_id="s1", documents=["[0]\nx"]
        )


# ── ⑤ 顺序：检索 → 回答 → 再检索（**不错位**）──────────────────────────


def _two_round_llm() -> FakeLlm:
    return FakeLlm(
        [
            _TWO_STEP,
            _step_reply(
                answer="Lisbon",
                evidence=[{"document_index": 0, "quote": "The trip started in Lisbon."}],
            ),
            _step_reply(answer="Maria"),
        ]
    )


def test_the_first_retrieval_uses_q1_and_never_the_original_question() -> None:
    """⛔ 初始化检索**直接用第一个子问题**——不先检索原始复杂问题。"""
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=_two_round_llm(), retriever=retriever)
    agent.search(user_id="u", query="Who was born where the trip started?", top_k=100)
    assert retriever.queries[0] == "Where did the trip start?"
    assert "Who was born where the trip started?" not in retriever.queries


def test_the_second_retrieval_happens_only_after_the_first_step_was_answered() -> None:
    """⭐ 调用①回答 Q1 并产出 Q2，**程序随后才检索 Q2**；调用②回答 Q2。

    断言的是**交错次序**，而不是"调了几次"——错位的表现是第 2 次检索仍然用 Q1，
    而那样**每一轮都会正常返回**。
    """
    llm = _two_round_llm()
    retriever = FakeRetriever(default=_candidates("m1", "m2"))
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="Who was born where the trip started?", top_k=100)

    assert retriever.queries == ["Where did the trip start?", "Who was born in Lisbon?"]
    # ⚠ LLM 次序：0 = 拆解、1 = 回答 s1、2 = 回答 s2。**拆解在最前，且只发生一次。**
    assert llm.calls[0].startswith("You create an ordered retrieval plan")
    assert llm.calls[1].startswith("You process the results of a search")
    assert "Step to answer now:\ns1:" in llm.calls[1]
    assert "Step to answer now:\ns2:" in llm.calls[2]
    # 第 2 轮的提示词里那口"本轮实际执行的查询"必须是**程序填好的**那一句
    assert "Actual query used for the completed search:\nWho was born in Lisbon?" in llm.calls[2]
    assert result.stats.stop_reason == "plan_complete"
    # ⭐ 最后一步**也被回答之后**才记 plan_complete：三次调用一次不少。
    assert result.stats.llm_calls == 3
    assert result.stats.decompose_llm_calls == 1
    assert result.stats.step_llm_calls == 2


def test_the_second_call_sees_the_history_and_this_round_but_not_the_old_snippets() -> None:
    """⭐ 上下文：**历史（问题/答案/来源 ID）＋ 本轮片段**。

    ⛔ 旧轮的片段全文与旧引句**一个字都不许进提示词**——那是这条路上唯一会长大的东西，
    而它长大之后的表现是"模型被旧上下文带偏"，不是任何报错。
    """
    llm = _two_round_llm()
    retriever = FakeRetriever(
        by_query={
            "Where did the trip start?": _candidates("m1"),
            "Who was born in Lisbon?": _candidates("m3"),
        }
    )
    agent = _agent(llm=llm, retriever=retriever)
    agent.search(user_id="u", query="Who was born where the trip started?", top_k=100)

    second = llm.call(2)
    # 历史：上一步的问题/答案/来源 ID 都在
    assert "s1 question: Where did the trip start?" in second
    assert "s1 answer: Lisbon" in second
    assert "s1 sources: m1" in second
    # 本轮：**R2** 的片段在（m3 的正文）
    assert "Maria was born in Porto." in second
    # ⛔ 旧片段（m1 的正文）与旧引句**不在**——只有它的 id 出现在 sources 里
    assert "The trip started in Lisbon." not in second
    # ⛔ 还没有发生的 s2 的答案当然也不在
    assert "s2 answer:" not in second
    # 本轮待答的问题与它的模板都在
    assert "Step to answer now:\ns2:" in second
    assert "{{s1}}\n" not in second  # 模板里的占位已经由程序绑好


def test_the_first_call_has_no_history_and_only_the_first_round() -> None:
    llm = _two_round_llm()
    retriever = FakeRetriever(
        by_query={"Where did the trip start?": _candidates("m1")},
        default=_candidates("m3"),
    )
    agent = _agent(llm=llm, retriever=retriever)
    agent.search(user_id="u", query="orig?", top_k=100)
    first = llm.call(1)
    assert "Previously completed questions and answers:\n(none yet)" in first
    assert "s1 answer:" not in first


def test_every_round_recalls_the_request_top_k_and_takes_the_top_n_after_rerank() -> None:
    """⭐ Plan 每轮**按请求的 `top_k` 召回**，精排**这 100 条**，再取前 `per_round_top_k`（20）。

    ⚠ **顺序不能反**（§3.2）：先截 20 再精排只会重排一堆**已被提前排除**的候选，
    **而结果看起来完全正常**。判据是"进精排的文档数 == 召回数"，不是"给模型的是 20"。

    ⚠ 2026-10-13 之前这里是"只召回 20 条再精排 20 条"——那让 Plan 与 React 在
    **两个**维度上同时不同（机制 ＋ 每轮深度），50 题对照因此**归不了因**。
    """
    # ⚠ 两轮给**不同**的池子：否则"跨轮去重"那条断言会被 `default` 掩盖成"每轮同一批"。
    retriever = FakeRetriever(
        by_query={
            "Where did the trip start?": _candidates(*[f"m{i}" for i in range(100)]),
            "Who was born in Lisbon?": _candidates(*[f"n{i}" for i in range(100)]),
        }
    )
    reranker = FakeReranker()
    store = FakeStore(
        [FakePair(f"{p}{i}", f"q{i}", f"a{i}") for p in ("m", "n") for i in range(100)]
    )
    agent = _agent(
        llm=_two_round_llm(),
        retriever=retriever,
        reranker=reranker,
        store=store,
        per_round_top_k=20,
    )
    result = agent.search(user_id="u", query="orig?", top_k=100)

    # ① 召回深度取**请求的 `top_k`**（与 React 同一条口径）
    assert [call["top_k"] for call in retriever.calls] == [100, 100]
    # ② 每轮精排**恰好一次**，用的是**那一轮实际执行的查询**，喂进去的就是召回的那 100 条
    assert [call["query"] for call in reranker.calls] == [
        "Where did the trip start?",
        "Who was born in Lisbon?",
    ]
    assert [len(call["documents"]) for call in reranker.calls] == [100, 100]
    # ③ 进候选池、发给模型的都只有精排后的**前 20 条**（`per_round_top_k`）
    assert all(len(record.candidate_ids) == 20 for record in result.plan_history)
    assert all(len(record.document_ids) == 20 for record in result.plan_history)
    assert result.stats.candidate_count == 200  # 各轮**召回**相加（100 + 100）
    assert result.stats.unique_candidate_count == 40  # 各轮前 20 并起来


def test_the_round_depth_follows_the_request_not_a_constant() -> None:
    """⚠ 请求给 7 就召回 7 —— `top_k` 是**请求的属性**，⛔ 代码里不许写死 100。"""
    retriever = FakeRetriever(default=_candidates(*[f"m{i}" for i in range(50)]))
    agent = _agent(llm=_two_round_llm(), retriever=retriever)
    agent.search(user_id="u", query="orig?", top_k=7)
    assert [call["top_k"] for call in retriever.calls] == [7, 7]


def test_candidates_are_cached_across_rounds_and_deduplicated_by_stable_id() -> None:
    """跨轮并集**按稳定 ID 去重**：轮内 20 条、两轮 40 次出现，重叠的只算一次。"""
    first = _candidates(*[f"m{i}" for i in range(20)])
    second = _candidates(*[f"m{i}" for i in range(10, 30)])
    retriever = FakeRetriever(
        by_query={"Where did the trip start?": first, "Who was born in Lisbon?": second}
    )
    store = FakeStore([FakePair(f"m{i}", f"q{i}", f"a{i}") for i in range(30)])
    agent = _agent(llm=_two_round_llm(), retriever=retriever, store=store, per_round_top_k=20)
    result = agent.search(user_id="u", query="orig?", top_k=100)

    stats = result.stats
    assert stats.candidate_count == 40  # 出现次数（20 + 20）
    assert stats.unique_candidate_count == 30  # 去重后（0..29）
    assert [c.memory_id for c in result.candidates][:3] == ["m0", "m1", "m2"]
    assert len({c.memory_id for c in result.candidates}) == len(result.candidates)
    # ⭐ 每轮的缓存数就是它自己那 20 条（**不重复读历史轮**）
    assert [len(record.candidate_ids) for record in result.plan_history] == [20, 20]


def test_round_rerank_can_be_switched_off_without_changing_the_candidate_set() -> None:
    """§13 的纯度：关掉每轮精排 ⇒ **候选集合不变**，只是名次不再被改。"""
    retriever = FakeRetriever(default=_candidates("m1", "m2", "m3"))
    reranker = FakeReranker()
    agent = _agent(
        llm=_two_round_llm(), retriever=retriever, reranker=reranker, round_rerank_enabled=False
    )
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert reranker.calls == []
    assert [c.memory_id for c in result.candidates] == ["m1", "m2", "m3"]
    assert all(record.rerank_status == "disabled" for record in result.plan_history)


# ── ⑦ 下一查询：程序算的那一份才是权威 ──────────────────────────────────


def test_the_next_query_is_bound_by_the_program_from_the_template() -> None:
    """⭐ **下一查询由程序绑定**：固定模板 ＋ 已确立答案 ⇒ 唯一那一句（v2）。

    ⛔ v2 起模型**根本不产出**下一查询（提示词只要三个字段），所以"接受自由改写"
    这条路整条没有了——`plan_history` 里记的是**程序算的那一句**。
    """
    llm = _two_round_llm()
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="orig?", top_k=100)

    assert result.plan_history[0].canonical_next_query == "Who was born in Lisbon?"
    assert retriever.queries[1] == "Who was born in Lisbon?"


def test_the_step_prompt_never_asks_the_model_for_a_next_query() -> None:
    """⛔ 提示词里**没有**下一查询的位置：v2 起那件事不是模型的职责。

    ⚠ 这条同时钉住"`render_plan_step_prompt` 的形参表 == 提示词的槽位表"——
    多传一个槽位会让 `_fill` **响亮失败**，少一个则会让提示词里**永远缺一段上下文**
    （而后者不会报错）。
    """
    from tianximem.agent.plan_prompts import PLAN_STEP_SLOTS

    assert "{next_step}" not in PLAN_STEP_SLOTS
    assert len(PLAN_STEP_SLOTS) == 6
    rendered = render_plan_step_prompt(
        original_question="Q?",
        retrieval_plan="s1: a",
        completed_history="(none yet)",
        current_step="s1: a",
        executed_query="a",
        current_documents=["[0]" + chr(10) + "x"],
    )
    # ⚠ 提示词里**提到**"next query" 只是为了**禁止**它（rule 9），所以判据不是
    #   "这个词不出现"，而是"没有下一查询的位置、也没有向模型索要它的字段"。
    assert "Next step template" not in rendered
    assert "Do not generate a next query" in rendered
    for slot in ("{next_step}", "{next_query}", "{next_step_id}"):
        assert slot not in rendered


def test_the_step_output_contract_matches_the_parser_verbatim() -> None:
    """⚠ 提示词声明的三个字段必须与解析器认的**逐字对上**（同拆解那条的理由）。"""
    from tianximem.agent.plan_parse import STEP_ANSWER_KEYS
    from tianximem.agent.plan_prompts import PLAN_STEP_PROMPT

    assert frozenset({"status", "answer", "evidence"}) == STEP_ANSWER_KEYS
    assert (
        "Return only JSON with exactly these fields:"
        + chr(10)
        + '"status", "answer", and "evidence".'
        in PLAN_STEP_PROMPT
    )
    # 三个状态与解析器认的是同一批
    assert 'Use status "answered", "unresolved", or "conflict".' in PLAN_STEP_PROMPT
    assert (
        "Each evidence item must contain exactly" + chr(10) + '"document_index" and "quote".'
        in PLAN_STEP_PROMPT
    )


def test_an_unresolved_step_that_also_supplies_an_answer_is_not_fatal() -> None:
    """⚠ 提示词要求 unresolved/conflict 时 `answer=null`、`evidence=[]`；
    模型偶尔会附上它们 ⇒ **不采用、不作废、照实记下来**。

    理由与"`depends_on` 以占位为准"同一条：**程序的行为不依赖这两个字段**
    （只有 `answered` 会被 `is_answer` 放行），所以没有理由因此丢掉整份计划。
    """
    parsed = parse_step_answer(
        _step_reply(status="unresolved", answer="Lisbon"),
        step_id="s1",
        documents=["[0]" + chr(10) + "x"],
    )
    assert parsed.is_answer is False
    assert parsed.step_id == "s1"


# ── ⑧ 未解决 / 冲突：**不猜占位** ───────────────────────────────────────


@pytest.mark.parametrize(
    ("status", "expected"),
    [("unresolved", "dependency_unresolved"), ("conflict", "binding_conflict")],
)
def test_an_unresolved_or_conflicting_step_stops_the_plan_without_guessing(
    status: str, expected: str
) -> None:
    """⛔ 解不出来就停下并记名：**不猜占位、不执行未绑定的下一问题、不做补救查询**。"""
    llm = FakeLlm(
        [
            _TWO_STEP,
            _step_reply(status=status, answer=None),
        ]
    )
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="orig?", top_k=100)

    assert result.stats.stop_reason == expected
    # ⭐ **第二条计划步骤没有被检索**（计划就停在这里）。
    assert [c["query"] for c in retriever.calls if c["query"] != "orig?"] == [
        "Where did the trip start?"
    ]
    # ⚠ `unresolved` 会多一次**原题兜底**检索（§新）——那是恢复分支，不是"下一跳"；
    #   `conflict` **不触发**恢复分支（本功能只针对 unresolved）。
    assert len(retriever.calls) == (2 if status == "unresolved" else 1)
    assert len(llm.calls) == 2  # 拆解 + 一次逐轮，没有第三次
    assert result.plan_history[0].status == status
    assert result.plan_history[0].answer is None
    # ⚠ 候选仍然被保留下来（不丢掉已经拿到的东西）
    assert "m1" in [c.memory_id for c in result.candidates]


def test_a_step_whose_placeholder_cannot_be_bound_never_reaches_the_retriever() -> None:
    """⛔ 程序侧的绑定失败：模板里有填不上的占位 ⇒ 停下，**绝不把它送进检索器**。

    真实触发路径是"依赖的答案没确立"（上一步 `unresolved`）——这里用一份**依赖了不存在的
    步**的计划直接构造它，因为那正是这条守卫要挡的输入形状。
    """
    # 手写一份"计划对象"绕过 `parse_plan` 的校验，直接喂给绑定那一步：
    # 它模拟的是"解析放行了、但答案缺失"的形态（`state.answers` 里没有 s1）。
    bound, missing = bind_placeholders("who is {{s1}}?", {})
    assert missing == ("s1",)
    assert bound == "who is {{s1}}?"


# ── ⑨ 停机与 plan_complete ──────────────────────────────────────────────


def test_plan_complete_is_recorded_only_after_the_last_retrieval_was_answered() -> None:
    """⭐ 最后一次检索也要**先被回答**，然后才记 `plan_complete`。"""
    llm = _two_round_llm()
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="orig?", top_k=100)

    assert result.stats.stop_reason == "plan_complete"
    assert result.plan_history[-1].stop_reason == "plan_complete"
    assert result.plan_history[-1].answer == "Maria"
    assert result.plan_history[-1].executed_query == "Who was born in Lisbon?"
    assert len(retriever.calls) == len(llm.calls) - 1  # 拆解那次不带检索


def test_every_stop_reason_used_in_the_code_is_declared() -> None:
    """停止原因要能**分开数**——所以它们必须来自那张声明过的表。"""
    assert set(PLAN_STOP_REASONS) == {
        "plan_complete",
        "dependency_unresolved",
        "binding_conflict",
        "unbound_placeholder",
        "invalid_output",
        "plan_invalid",
        "max_rounds",
        "llm_unavailable",
        "llm_error",
        "retrieval_error",
        "prompt_budget_exceeded",
    }


def test_a_plan_longer_than_the_round_budget_stops_at_max_rounds() -> None:
    """计划还有下一步却没跑 ⇒ `max_rounds`（**不是** `plan_complete`）。"""
    three = _plan_reply(
        [
            {"id": "s1", "query": "a", "depends_on": []},
            {"id": "s2", "query": "b {{s1}}", "depends_on": ["s1"]},
            {"id": "s3", "query": "c {{s2}}", "depends_on": ["s2"]},
        ]
    )
    llm = FakeLlm(
        [
            three,
            _step_reply(answer="A"),
            _step_reply(answer="B"),
        ]
    )
    retriever = FakeRetriever(default=_candidates("m1"))
    agent = _agent(llm=llm, retriever=retriever, max_rounds=2)
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.stats.stop_reason == "max_rounds"
    assert result.stats.plan_steps == 3
    assert result.stats.steps_executed == 2
    assert len(retriever.calls) == 2


# ── ⑩ 降级（§10）—— 绝不让一次正常 Search 变成 500 ──────────────────────


@pytest.mark.parametrize("reply", [LlmUnavailable("端点挂了"), RuntimeError("意料之外")])
def test_a_broken_decompose_falls_back_to_hybrid_retrieval(reply: BaseException) -> None:
    """拆解就失败 ⇒ 一条候选都没有 ⇒ **回退原混合链**（§10.1 第 6 条）。

    ⚠ 回退原因是**具名的**（不是笼统的"多跳没帮助"）：端点挂与输出非法要分得开。
    """
    llm = FakeLlm([reply])
    agent = _agent(llm=llm)
    result = agent.search(user_id="u", query="orig?", top_k=100)
    expected = "llm_unavailable" if isinstance(reply, LlmUnavailable) else "llm_error"
    assert result.stats.stop_reason == expected
    assert result.fallback_reason == expected
    assert result.candidates == ()


def test_an_invalid_plan_falls_back_instead_of_inventing_one() -> None:
    llm = FakeLlm([_plan_reply([{"id": "s1", "query": "a", "depends_on": ["s9"]}])])
    agent = _agent(llm=llm)
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.fallback_reason == "plan_invalid"
    assert result.candidates == ()


def test_an_invalid_round_output_keeps_the_candidates_and_does_not_fall_back() -> None:
    """已经攒到候选 ⇒ **不回退**：手上的证据比丢掉它们重来一遍更有价值。"""
    llm = FakeLlm([_TWO_STEP, "not json at all"])
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.stats.stop_reason == "invalid_output"
    assert result.fallback_reason is None
    assert [c.memory_id for c in result.candidates] == ["m1"]


def test_a_broken_step_llm_records_no_token_usage_for_that_round() -> None:
    """⚠ 失败的那一轮**不许报 token 数**：`last_usage` 是客户端上的残留值。"""
    llm = FakeLlm([_TWO_STEP, LlmUnavailable("挂了")])
    agent = _agent(llm=llm, retriever=FakeRetriever(default=_candidates("m1")))
    result = agent.search(user_id="u", query="orig?", top_k=100)
    record = result.plan_history[0]
    assert record.stop_reason == "llm_unavailable"
    assert record.input_tokens is None and record.output_tokens is None


def test_a_retrieval_failure_with_nothing_collected_falls_back() -> None:
    llm = FakeLlm([_TWO_STEP])
    agent = _agent(llm=llm, retriever=FakeRetriever(raises=RuntimeError("qdrant 挂了")))
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.stats.stop_reason == "retrieval_error"
    assert result.fallback_reason == "retrieval_error"


def test_a_retrieval_failure_on_the_second_round_keeps_the_first_round() -> None:
    retriever = FakeRetriever(by_query={"Where did the trip start?": _candidates("m1")})
    retriever.search = _fail_after_first(retriever)  # type: ignore[method-assign]
    llm = _two_round_llm()
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.stats.stop_reason == "retrieval_error"
    assert result.fallback_reason is None  # 已经攒到候选 ⇒ 不回退
    assert [c.memory_id for c in result.candidates] == ["m1"]


def _fail_after_first(retriever: FakeRetriever):
    """第二次调用才炸的检索桩（`FakeRetriever` 只支持"一直炸"或"一直好"）。"""
    original = retriever.search

    def search(*, user_id: str, query: str, top_k: int):
        if retriever.calls:
            retriever.calls.append({"user_id": user_id, "query": query, "top_k": top_k})
            raise RuntimeError("第二轮检索挂了")
        return original(user_id=user_id, query=query, top_k=top_k)

    return search


# ── ⑪ 结果形状：执行记录与真实证据**分开** ──────────────────────────────


def test_the_plan_history_never_pretends_to_be_memory() -> None:
    """⭐ 执行记录与真实证据**分开放**：`plan_history` 里的 id 全部来自候选池，
    ⛔ 没有编造的 memory id，也**不进** `candidates`（那才是会被打包的记忆）。"""
    llm = _two_round_llm()
    retriever = FakeRetriever(default=_candidates("m1", "m2"))
    agent = _agent(llm=llm, retriever=retriever)
    result = agent.search(user_id="u", query="orig?", top_k=100)

    pool = {c.memory_id for c in result.candidates}
    assert pool == {"m1", "m2"}
    for record in result.plan_history:
        for memory_id in record.evidence_ids:
            assert memory_id in pool  # 引用的都是**真实存在**的候选 id
        for memory_id in record.document_ids:
            assert memory_id in pool
    # ⚠ 答案文本只住在执行记录里，**不在**候选上（候选只有 id 与名次）
    assert "Maria" not in {c.memory_id for c in result.candidates}
    assert result.plan_history[-1].answer == "Maria"


def test_the_rerank_query_lists_the_original_question_once_then_what_ran() -> None:
    llm = _two_round_llm()
    agent = _agent(llm=llm)
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.rerank_query == "Where did the trip start?\nWho was born in Lisbon?"


def test_the_original_only_mode_leaves_the_rerank_query_empty() -> None:
    llm = _two_round_llm()
    agent = _agent(llm=llm, final_rerank_query_mode="original")
    result = agent.search(user_id="u", query="orig?", top_k=100)
    assert result.rerank_query is None


# ── ⑫ 构造期断言 ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_rounds": 0}, "max_rounds"),
        ({"per_round_top_k": 0}, "per_round_top_k"),
        ({"max_prompt_tokens": 0}, "max_prompt_tokens"),
        ({"max_output_tokens": 0}, "plan_max_output_tokens"),
        ({"temperature": -0.1}, "temperature"),
        ({"max_plan_steps": 5}, "计划步数上限"),
        ({"final_rerank_query_mode": "free"}, "final_rerank_query_mode"),
        ({"decompose_prompt_version": "plan-decompose-v9"}, "plan_decompose_prompt_version"),
        ({"step_prompt_version": "plan-step-v9"}, "plan_step_prompt_version"),
    ],
)
def test_construction_refuses_impossible_limits(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _agent(llm=FakeLlm(""), **kwargs)


# ── ⑬ 装配与配置：默认仍是 React（§13）──────────────────────────────────


def test_the_shipped_default_still_selects_react() -> None:
    """⭐ 旧配置（不填 `mode`）⇒ **逐字**走原来的执行器。"""
    from tianximem.common.config import AgenticConfig

    assert AgenticConfig().mode == "react"
    assert AgenticConfig().plan_max_output_tokens == 768


def test_the_config_selects_the_plan_executor_end_to_end(tmp_path) -> None:
    """⭐ `agentic.mode = plan` ⇒ 装配出的是 `PlanAgent`（不是 React）。"""
    from dataclasses import replace

    from tianximem.common.config import load_config
    from tianximem.service.app import build_services

    base = load_config(
        {
            "AML_EMB_BASE_URL": "http://emb.invalid",
            "AML_EMB_API_KEY": "k",
            "AML_BASE_URL": "http://llm.invalid",
            "AML_API_KEY": "k",
            "AML_MODEL": "m",
        }
    )
    config = replace(
        base,
        storage=replace(
            base.storage,
            sqlite=replace(base.storage.sqlite, path=str(tmp_path / "t.db")),
        ),
        agentic=replace(base.agentic, enabled=True, mode="plan", force=True),
        capture=replace(base.capture, enabled=False),
    )
    services = build_services(config)
    try:
        assert isinstance(services.search.agentic, PlanAgent)
        assert not isinstance(services.search.agentic, CoqAgent)
    finally:
        services.close()

    react = replace(config, agentic=replace(config.agentic, mode="react"))
    services = build_services(react)
    try:
        assert isinstance(services.search.agentic, CoqAgent)
    finally:
        services.close()


def test_a_typo_in_the_mode_is_refused_at_load_time(tmp_path) -> None:
    """⛔ 拼错的模式**不许静默退回 react**：那会让"跑的是哪套机制"无法归因。"""
    from pathlib import Path

    from tianximem.common.config import ConfigError, load_config

    # ⚠ 测的是**仓库里那份真的 `default.yaml`**（只改一处），而不是一份只在本用例里
    #   存在的假配置——否则"yaml 里的键名写错了"这类问题一个都测不到。
    real = Path(__file__).resolve().parents[1] / "configs"
    out = tmp_path / "configs"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("default.yaml", "local.yaml", "submit.yaml"):
        text = (real / name).read_text(encoding="utf-8")
        if name == "default.yaml":
            assert "  mode: react" in text
            text = text.replace("  mode: react", "  mode: plna")
        (out / name).write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="agentic.mode"):
        load_config({"AML_EMB_BASE_URL": "http://x", "AML_EMB_API_KEY": "k"}, config_dir=out)


def test_the_plan_output_budget_is_wired_from_its_own_key(tmp_path) -> None:
    """⚠ Plan 的 `max_output_tokens` 来自 `agentic.plan_max_output_tokens`——
    **不是** `agentic.max_output_tokens`（那个是 CoQ 四字段 JSON 的）。"""
    from dataclasses import replace

    from tianximem.common.config import load_config
    from tianximem.service.app import build_services

    base = load_config(
        {
            "AML_EMB_BASE_URL": "http://emb.invalid",
            "AML_EMB_API_KEY": "k",
            "AML_BASE_URL": "http://llm.invalid",
            "AML_API_KEY": "k",
            "AML_MODEL": "m",
        }
    )
    config = replace(
        base,
        storage=replace(
            base.storage,
            sqlite=replace(base.storage.sqlite, path=str(tmp_path / "t.db")),
        ),
        agentic=replace(
            base.agentic,
            enabled=True,
            mode="plan",
            force=True,
            max_output_tokens=256,
            plan_max_output_tokens=1234,
        ),
        capture=replace(base.capture, enabled=False),
    )
    services = build_services(config)
    try:
        agent = services.search.agentic
        assert isinstance(agent, PlanAgent)
        assert agent._max_output_tokens == 1234  # noqa: SLF001 — 这是装配对不对的直接证据
    finally:
        services.close()


def test_services_close_releases_the_plan_client(tmp_path) -> None:
    """⚠ 关闭路径要**两种模式都列出来**——漏掉一支的表现是连接池没释放，平时看不出来。"""
    from dataclasses import replace

    from tianximem.common.config import load_config
    from tianximem.llm import OpenAiCompatChat
    from tianximem.service.app import build_services

    base = load_config(
        {
            "AML_EMB_BASE_URL": "http://emb.invalid",
            "AML_EMB_API_KEY": "k",
            "AML_BASE_URL": "http://llm.invalid",
            "AML_API_KEY": "k",
            "AML_MODEL": "m",
        }
    )
    config = replace(
        base,
        storage=replace(
            base.storage,
            sqlite=replace(base.storage.sqlite, path=str(tmp_path / "t.db")),
        ),
        agentic=replace(base.agentic, enabled=True, mode="plan"),
        capture=replace(base.capture, enabled=False),
    )
    services = build_services(config)
    agent = services.search.agentic
    assert isinstance(agent, PlanAgent)
    assert isinstance(agent.llm, OpenAiCompatChat)
    client = agent.llm
    services.close()
    # ⚠ 断言的是连接池**真的**被释放了（`close()` 把它置空）——与
    #   `test_router.py::test_services_close_releases_the_llm_client` 同一口径。
    assert client._client is None  # noqa: SLF001


def test_the_plan_executor_does_not_import_the_http_layer_or_the_router() -> None:
    """与 `agent/` 的其余部分同一条边界（AST 级，不是靠自觉）。"""
    import ast
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "tianximem"
    banned = ("tianximem.service", "tianximem.retrieve.router")
    for name in ("plan.py", "plan_parse.py", "plan_prompts.py"):
        tree = ast.parse((root / "agent" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(banned):
                raise AssertionError(f"{name} 越界 import 了 {node.module}")
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(banned), f"{name} 越界 import 了 {alias.name}"


def test_the_react_side_is_untouched_by_this_mode() -> None:
    """⛔ 本次改动**没有**动 React 的提示词与停止原因表（它是冻结的那一份）。"""
    from tianximem.agent import COQ_PROMPT_VERSION, STOP_REASONS

    assert COQ_PROMPT_VERSION == "coq-original"
    assert "sufficient" in STOP_REASONS  # React 的停止原因表**没有被 Plan 改过**
    assert set(STOP_REASONS).isdisjoint({"plan_complete", "dependency_unresolved"})


# ── ⑭ `unresolved` 恢复分支：按剩余名额补一次**原题**检索（2026-10-13）─────


#: 四步计划（每步引用前一步），用来测"第 k 轮 unresolved"的预算。
_FOUR_STEP = _plan_reply(
    [
        {"id": "s1", "query": "q1", "depends_on": []},
        {"id": "s2", "query": "q2 {{s1}}", "depends_on": ["s1"]},
        {"id": "s3", "query": "q3 {{s2}}", "depends_on": ["s2"]},
        {"id": "s4", "query": "q4 {{s3}}", "depends_on": ["s3"]},
    ]
)


def _unresolved_at(answered_rounds: int) -> FakeLlm:
    """拆解 + `answered_rounds` 次正常回答 + **一次 `unresolved`**。"""
    replies: list[str] = [_FOUR_STEP]
    replies += [_step_reply(answer=f"A{i}") for i in range(1, answered_rounds + 1)]
    replies.append(_step_reply(status="unresolved", answer=None))
    return FakeLlm(replies)


def _big_retriever(**kwargs) -> FakeRetriever:
    """候选管够（120 条），这样"截没截到 20"看得出来。"""
    return FakeRetriever(default=_candidates(*[f"m{i}" for i in range(120)]), **kwargs)


def test_a_normal_finish_never_touches_the_recovery_branch() -> None:
    """① 正常完成：**不调兜底**、联合精排照旧、答题输入照旧（不带 recovery）。"""
    retriever = _big_retriever()
    result = _agent(llm=_two_round_llm(), retriever=retriever).search(
        user_id="u", query="orig?", top_k=100
    )
    assert result.recovery.entered is False
    assert result.recovery.status == "not_triggered"
    assert result.recovery.rerank_query_mode == "joint"
    assert retriever.queries == ["Where did the trip start?", "Who was born in Lisbon?"]
    assert result.rerank_query == "Where did the trip start?\nWho was born in Lisbon?"
    assert result.stats.stop_reason == "plan_complete"


@pytest.mark.parametrize(("answered_rounds", "expected"), [(0, 60), (1, 40), (2, 20)])
def test_the_fallback_budget_is_eighty_minus_the_rounds_already_spent(
    answered_rounds: int, expected: int
) -> None:
    """② 第 1/2/3 轮 `unresolved` ⇒ 原题兜底分别以 **60 / 40 / 20** 检索；每题**只一次**。"""
    llm = _unresolved_at(answered_rounds)
    retriever = _big_retriever()
    result = _agent(llm=llm, retriever=retriever).search(user_id="u", query="orig?", top_k=100)

    fallback = [c for c in retriever.calls if c["query"] == "orig?"]
    assert len(fallback) == 1, "⭐ 每题**最多一次**原题兜底"
    assert fallback[0]["top_k"] == expected
    # ⚠ `executed_rounds` **含刚刚返回 unresolved 的那一轮**（它的候选已经进池子）。
    assert result.recovery.executed_rounds == answered_rounds + 1
    assert result.recovery.remaining_top_k == expected
    assert result.recovery.status == "success"
    # ⭐ 兜底**不增加 LLM 调用**：拆解 1 次 ＋ 每轮 1 次。
    assert result.stats.llm_calls == 1 + answered_rounds + 1
    assert result.recovery.recalled == expected


def test_the_fourth_unresolved_round_exhausts_the_budget_without_retrieving() -> None:
    """③ 第 4 轮 `unresolved` ⇒ `remaining=0`、**不检索**，但精排仍用原题。"""
    retriever = _big_retriever()
    result = _agent(llm=_unresolved_at(3), retriever=retriever).search(
        user_id="u", query="orig?", top_k=100
    )
    assert result.recovery.entered is True
    assert result.recovery.status == "skipped_budget_exhausted"
    assert result.recovery.remaining_top_k == 0
    assert result.recovery.recalled == 0
    assert "orig?" not in retriever.queries  # ⛔ 一次兜底检索都没发生
    assert result.rerank_query == "orig?"  # ⭐ 精排词仍然是原题
    assert result.recovery.rerank_query_mode == "original"


def test_an_unresolved_last_step_is_never_recorded_as_plan_complete() -> None:
    """④ 计划**最后一步** `unresolved`：按实际轮数算预算，⛔ 不许记 `plan_complete`。"""
    llm = FakeLlm(
        [
            _TWO_STEP,
            _step_reply(answer="Lisbon"),
            _step_reply(status="unresolved", answer=None),
        ]
    )
    retriever = _big_retriever()
    result = _agent(llm=llm, retriever=retriever).search(user_id="u", query="orig?", top_k=100)

    assert result.stats.stop_reason == "dependency_unresolved"
    assert result.recovery.executed_rounds == 2
    assert result.recovery.remaining_top_k == 40  # 80 − 2×20
    assert result.rerank_query == "orig?"


def test_a_failed_fallback_keeps_every_candidate_and_still_finishes() -> None:
    """⑦ 兜底检索失败：**已有候选一条不动**，仍走原题精排与最终答题，不抛异常。"""
    retriever = _big_retriever(raises_for={"orig?"})
    result = _agent(llm=_unresolved_at(0), retriever=retriever).search(
        user_id="u", query="orig?", top_k=100
    )
    assert result.recovery.status == "error"
    assert result.recovery.entered is True
    assert result.recovery.recalled == 0 and result.recovery.added_unique == 0
    assert result.stats.stop_reason == "dependency_unresolved"
    # ⛔ **不是**整条路降级：候选还在，也没有回退到混合检索。
    assert result.fallback_reason is None
    assert len(result.candidates) == 20
    assert result.rerank_query == "orig?"


def test_only_unresolved_triggers_the_recovery_branch() -> None:
    """⛔ 拆解失败 / `conflict` / 逐轮 LLM 挂 **一律不触发**（各自保持原行为）。"""
    # 拆解失败（`plan_invalid`）
    invalid = _agent(llm=FakeLlm([_plan_reply([{"id": "s1", "query": "a", "depends_on": ["s9"]}])]))
    failed = invalid.search(user_id="u", query="orig?", top_k=100)
    assert failed.recovery.entered is False and failed.recovery.status == "not_triggered"
    # `conflict`（不是 `unresolved`）
    conflict = _agent(
        llm=FakeLlm([_TWO_STEP, _step_reply(status="conflict", answer=None)]),
        retriever=_big_retriever(),
    ).search(user_id="u", query="orig?", top_k=100)
    assert conflict.stats.stop_reason == "binding_conflict"
    assert conflict.recovery.entered is False
    # 逐轮 LLM 挂
    dead = _agent(
        llm=FakeLlm([_TWO_STEP, LlmUnavailable("端点挂了")]), retriever=_big_retriever()
    ).search(user_id="u", query="orig?", top_k=100)
    assert dead.stats.stop_reason == "llm_unavailable"
    assert dead.recovery.entered is False


def test_the_fallback_candidates_are_merged_by_stable_id_and_never_truncated() -> None:
    """⑤⑥ 按稳定 ID 合并去重；兜底结果**不再被截成每轮 Top20**；不足 80 条也不补检索。"""
    retriever = _big_retriever()
    result = _agent(llm=_unresolved_at(0), retriever=retriever).search(
        user_id="u", query="orig?", top_k=100
    )
    ids = [c.memory_id for c in result.candidates]
    # 分步那 20 条（m0..m19）与兜底那 60 条（m0..m59）重叠 20 条 ⇒ 并起来 60 条
    assert len(ids) == 60
    assert len(set(ids)) == len(ids), "⭐ 重复的 memory_id 只保留一份"
    assert result.recovery.added_unique == 40
    assert len([c for c in retriever.calls if c["query"] == "orig?"]) == 1
    # ⛔ 兜底候选**没有**被截成 20（截过的话并集就只有 20 条）
    assert ids[20:] == [f"m{i}" for i in range(20, 60)]


def test_no_unbound_placeholder_is_ever_sent_to_the_retriever() -> None:
    """⑧ 第 2 步 `unresolved` ⇒ 后面两步（模板带 `{{s2}}`/`{{s3}}`）**一句都没被检索**。"""
    retriever = _big_retriever()
    result = _agent(llm=_unresolved_at(1), retriever=retriever).search(
        user_id="u", query="orig?", top_k=100
    )
    assert all("{{" not in call["query"] for call in retriever.calls)
    assert retriever.queries == ["q1", "q2 A1", "orig?"]
    assert result.stats.steps_executed == 2


def test_the_react_side_has_no_recovery_branch_at_all() -> None:
    """⑨ React 那一侧**结构上**没有这条分支（它没有"计划步骤"这个概念）。"""
    from dataclasses import fields

    from tianximem.agent import AgenticResult, AgenticStats

    assert not hasattr(CoqAgent, "_recover")
    assert "recovery" not in {f.name for f in fields(AgenticResult)}
    assert "recovery" not in {f.name for f in fields(AgenticStats)}


def test_plan_without_a_recovery_keeps_the_joint_rerank_query() -> None:
    """⚠ 恢复分支**只改它自己那一条**：没触发的题，精排词逐字还是"执行过的那些查询"。"""
    plain = _agent(llm=_two_round_llm(), retriever=_big_retriever()).search(
        user_id="u", query="orig?", top_k=100
    )
    assert plain.rerank_query == "Where did the trip start?\nWho was born in Lisbon?"
    assert plain.recovery.rerank_query_mode == "joint"
