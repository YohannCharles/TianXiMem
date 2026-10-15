"""Chain-of-Query 多跳检索执行器（`MULTI_HOP` 那条路）。

**它不是一个"通用 Agent"**：没有工具调用、没有自主规划、不生成答案。
它只做一件事——**把同一条查询问好几轮，每轮换一个说法**，并把每轮的证据攒起来。

```text
原始 Query
   │
   ├─→ ① 本轮 Query → 现有混合召回（BM25 + Dense + RRF）→ 候选
   │                   → 现有 Reranker → Top-N（`per_round_top_k`）
   ├─→ ② 攒证据：本轮 Top-N ＋ 此前已选中的证据（按稳定 ID 去重、重新编号）
   ├─→ ③ 一次 LLM：充分性判断 ＋ 选证据 ＋ 下一跳 Query（原版 CoQ 提示词）
   └─→ ④ 停？（充分且置信 ≥ 阈值 / 轮数用尽 / Query 重复或为空 / 发生故障）
         否 ⇒ 换 Query 回 ①
         是 ⇒ 把所有轮次的 Top-N 去重合并，交回调用方
```

## 三条边界（写在这里，因为越界不会报错）

1. **最终精排不在这里做。** 调用方（[`../service/pipeline.py`](../service/pipeline.py)）
   对"合并后的那一份列表"精排**恰好一次**——本执行器只把**该用哪个 Query 去精排**
   交上去（`rerank_query` = 原问题 ＋ 实际执行过的改写，§9.1）。
   ⛔ 这样安排**不是**省事：它让"精排恰好一次"这条纪律**结构性地**成立，
   而不是靠"我记得不要在最后一轮再排一次"（§9.4 明确点出了这个陷阱）。
   ⚠ 每轮内部那次精排是**另一件事**：它的作用是把交给 LLM 的证据筛干净，
   产出物**不进响应**。两者共用同一个 `Reranker` 对象，不是两套精排。
2. **不生成答案**（§2.1 的红线）。产出物是候选 + 名次。
3. **它不改任何一条记忆的原文**：文档文本一律走
   [`../common/render.py`](../common/render.py) 的 `render_pair`（不变式 I1），
   与索引侧、精排输入是同一份渲染。模型给出的 `new_query` **只当检索词**用，
   ⛔ **绝不把它当成一条记忆内容**（§8.3）。

## 与 PRD §9 的关系（**这里的参数与 §9 不同，是刻意的**）

PRD §9 给的是 ReFind 的形状（最多 4 轮、每轮 `top_k=5`、三个工具）。本阶段**不用那套**：
方案要求 `max_rounds`(默认 4) 轮、每轮 Top20、**没有工具**（工具集那一块属于通用 Agent，本阶段明确不做）。
⚠ 因此**不要**拿 §9 的每轮 `top_k=5` 来"校正" `per_round_top_k`——两者描述的不是同一件事。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from time import monotonic
from typing import Final

from tianximem.agent.coq_prompt import COQ_PROMPT, COQ_PROMPT_SLOTS, COQ_PROMPT_VERSION
from tianximem.agent.parse import CoqDecision, InvalidCoqOutput, parse_coq_decision
from tianximem.common.render import day_granularity, render_pair
from tianximem.common.tokens import TokenCounter
from tianximem.llm import LlmClient, LlmUnavailable
from tianximem.rank import Reranker, RerankUnavailable
from tianximem.retrieve.fusion import Candidate, HybridRetriever
from tianximem.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "FINAL_QUERY_MODES",
    "STOP_REASONS",
    "AgenticResult",
    "AgenticStats",
    "CoqAgent",
    "RoundRecord",
    "normalize_query",
    "render_coq_prompt",
]

#: 停止原因。**它们要能在 §14 里分开数**——"轮数用尽"与"LLM 挂了"是两回事，
#: 混成一个"没找到完整证据"会让失败归因整个失效。
STOP_REASONS: Final[tuple[str, ...]] = (
    "sufficient",
    "low_confidence",
    "max_rounds",
    "empty_query",
    "duplicate_query",
    "invalid_output",
    "llm_unavailable",
    "llm_error",
    "retrieval_error",
    "prompt_budget_exceeded",
)

#: 最终精排 Query 的两种构造方式（§9.1 与 §12.6 的对照项）。
FINAL_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")

#: 三个占位符的**一遍扫描**匹配器。见 `render_coq_prompt` 的第一条说明——
#: 换成连续三次 `str.replace` 会让 Query 里的字面占位符被**二次替换**，且不报错。
_PLACEHOLDER_RE: Final[re.Pattern[str]] = re.compile(
    "|".join(re.escape(slot) for slot in COQ_PROMPT_SLOTS)
)


def normalize_query(query: str) -> str:
    """Query 的**程序侧**去重口径（§7.5）：小写 + 折叠空白。

    ⚠ 它是**确定性保险**，不是"语义去重"：原版提示词已经要求模型自己避免重复，
    这里只挡住"字面一模一样"的那种。**不为它再加一次 LLM 调用**（§7.5 明说第一版不做）。
    """
    return " ".join(query.casefold().split())


def render_coq_prompt(
    *, original_query: str, used_queries: Sequence[str], documents: Sequence[str]
) -> str:
    """把三样东西填进原版提示词。**不改提示词本身**。

    ⚠ **一遍扫描、绝不重扫替换结果**（`re.sub` 而不是连续三次 `str.replace`）：
    Query 是**不可信输入**。连续替换的话，一句含字面 `{used_query}` 的 Query 会在
    **第二次替换时被替掉**——提示词里凭空多出那段查询列表，而**没有任何报错**。
    一遍扫描从根上排除这件事：替换进去的文本不再参与匹配。

    ⚠ 它也**不用 `str.format`**：Query 里的 `{` 会让它抛异常（好一点），
    或者**替换出一个没人预期的输入**（更糟）。

    ⚠ 替换前先断言三个占位符都在——**提示词被人动过**这件事要在构造期响亮失败，
    而不是把一个还带着 `{...}` 的提示词发给模型（它会照着抄）。
    """
    values = {
        "{original_query}": original_query,
        "{used_query}": "\n".join(used_queries),
        "{retrieved_episodes}": "\n\n".join(documents),
    }
    missing = [slot for slot in COQ_PROMPT_SLOTS if slot not in COQ_PROMPT]
    if missing:
        raise ValueError(f"提示词缺少占位符：{missing}")
    return _PLACEHOLDER_RE.sub(lambda match: values[match.group(0)], COQ_PROMPT)


@dataclass(frozen=True, slots=True)
class RoundRecord:
    """一轮的**完整过程**（§11）。实验模式下要用它重建"模型当时看到了什么"。"""

    index: int
    #: **这一轮实际执行的 Query**（不是下一轮要跑的）。
    query: str
    #: 本轮混合召回的候选数。⚠ `HybridRetriever` 已经按请求的 `top_k` 截过 ⇒
    #: 它是**融合池的下界**，不是池子本身的大小。
    recalled_candidates: int
    #: 本轮发给模型的文档 ID，**顺序即索引序**（§5.1 的临时索引映射）。
    document_ids: tuple[str, ...]
    #: 本轮精排 Top-N 的稳定 ID（§8.1 的 `all_candidate_ids` 那一轮的贡献）。
    candidate_ids: tuple[str, ...]
    #: 模型指认的证据 ID——**已由数字索引换成稳定 ID**（§5.3 第 5 条）。
    selected_ids: tuple[str, ...]
    #: 因为提示词 token 预算没进文档列表的候选数。**不为 0 就要能看见**（§9.3）。
    dropped_documents: int
    #: `disabled` / `called` / `degraded`（D12 的三分口径）。
    rerank_status: str
    is_sufficient: bool | None = None
    confidence_score: float | None = None
    new_query: str | None = None
    #: `direct` / `recovered`（§6.1 的日志标记）。
    extraction: str | None = None
    stop_reason: str | None = None
    retrieval_latency_ms: float = 0.0
    rerank_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class AgenticStats:
    """一次多跳 Search 的汇总（§11 的"最终记录"那一段）。**不进响应。**"""

    rounds: tuple[RoundRecord, ...]
    stop_reason: str
    fallback_reason: str | None
    #: 去重**前**的候选出现次数（三轮最多 60）与去重后的条数（§8.1 / §9.2）。
    candidate_count: int
    unique_candidate_count: int
    selected_evidence_count: int
    joint_query_count: int
    llm_calls: int
    rerank_calls: int
    latency_ms: float


@dataclass(frozen=True, slots=True)
class AgenticResult:
    """执行器交给编排层的全部东西。

    `fallback_reason is not None` ⇒ **调用方改走混合检索**：执行器在拿到任何证据之前
    就失败了（§10.1 第 6 条）。那是"回退"这个动作**唯一**的触发条件——
    一旦已经攒到候选就**不再回退**：手上已有的证据比丢掉它们重来一遍更有价值。
    """

    candidates: tuple[Candidate, ...]
    rerank_query: str | None
    stats: AgenticStats
    fallback_reason: str | None = None


@dataclass(slots=True)
class _State:
    """一次 Search 的临时状态（§7.1）。**不持久化，也不跨请求共享。**"""

    #: 实际执行过的 Query，**按顺序**（第一轮就是原始问题）。既喂给提示词，
    #: 也用来拼最终的精排 Query（§9.1）。
    executed: list[str] = field(default_factory=list)
    normalized: set[str] = field(default_factory=set)
    #: 各轮 Top-N 的稳定 ID，**保序去重**（用 `dict` 的插入序）——第一轮的名次最先。
    ordered_ids: dict[str, None] = field(default_factory=dict)
    #: 模型选中的证据 ID，保序去重（§8.1 的 `selected_evidence_ids`）。
    selected: list[str] = field(default_factory=list)
    #: 稳定 ID → 文档正文（`render_pair` 的产物）。**原文只从这里或真源来**（§8.3）。
    bodies: dict[str, str] = field(default_factory=dict)
    #: 稳定 ID → 日期串（**只到日**，D21）。
    days: dict[str, str] = field(default_factory=dict)


class CoqAgent:
    """`MULTI_HOP` 的执行器。**同步、无状态：一次 `search()` 一份请求局部状态。**

    它**不抛异常**（构造参数非法除外）：所有失败都变成 `AgenticResult` 上的
    一个具名原因。理由与路由器同一条——一条可选路径的失败不该让正常 Search 变 500。
    """

    def __init__(
        self,
        *,
        store: SqliteStore,
        retriever: HybridRetriever,
        llm: LlmClient,
        counter: TokenCounter,
        reranker: Reranker | None = None,
        max_rounds: int = 3,
        per_round_top_k: int = 20,
        sufficiency_threshold: float = 0.8,
        max_prompt_tokens: int = 6000,
        max_output_tokens: int = 256,
        temperature: float = 0.0,
        round_rerank_enabled: bool = True,
        final_rerank_query_mode: str = "combined",
        inject_abs_time: bool = False,
        prompt_version: str = COQ_PROMPT_VERSION,
    ) -> None:
        if prompt_version != COQ_PROMPT_VERSION:
            # ⚠ 与 `QueryRouter` / `rerank.envelope` 同一套写法：**这是一条断言**。
            #   "配了哪一版提示词"与"跑的是哪一版"必须永远是同一件事。
            raise ValueError(
                f"agentic.prompt_version = {prompt_version!r}，"
                f"但代码里只有一版提示词 {COQ_PROMPT_VERSION!r}"
            )
        if max_rounds < 1:
            raise ValueError("agentic.max_rounds 必须 ≥ 1")
        if per_round_top_k < 1:
            raise ValueError("agentic.per_round_top_k 必须 ≥ 1")
        if not 0.0 <= sufficiency_threshold <= 1.0:
            raise ValueError("agentic.sufficiency_threshold 必须在 0..1 之间")
        if max_prompt_tokens <= 0:
            raise ValueError("agentic.max_prompt_tokens 必须为正")
        if max_output_tokens <= 0:
            raise ValueError("agentic.max_output_tokens 必须为正")
        if temperature < 0:
            raise ValueError("agentic.temperature 不得为负")
        if final_rerank_query_mode not in FINAL_QUERY_MODES:
            raise ValueError(
                f"agentic.final_rerank_query_mode 必须是 {FINAL_QUERY_MODES} 之一，"
                f"收到 {final_rerank_query_mode!r}"
            )

        self._store = store
        self._retriever = retriever
        self._llm = llm
        self._counter = counter
        self._reranker = reranker
        self._max_rounds = max_rounds
        self._per_round_top_k = per_round_top_k
        self._sufficiency_threshold = sufficiency_threshold
        self._max_prompt_tokens = max_prompt_tokens
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._round_rerank = round_rerank_enabled
        self._final_query_mode = final_rerank_query_mode
        self._inject_abs_time = inject_abs_time

    @property
    def llm(self) -> LlmClient:
        """当前接的对话客户端。

        只读暴露给**装配侧**（[`../service/app.py`](../service/app.py) 的
        `Services.close()` 要关掉它的连接池）与**测试**（断言真的接上了）。
        ⚠ 与 `QueryRouter.client` 同一个理由：**不要让关闭路径自己再存一份引用**——
        两份状态的结果是"装配时接上了、退出时漏关"，而那平时完全看不出来。
        """
        return self._llm

    # ── 对外唯一入口 ────────────────────────────────────────────────────
    def search(self, *, user_id: str, query: str, top_k: int) -> AgenticResult:
        started = monotonic()
        state = _State()
        records: list[RoundRecord] = []
        llm_calls = 0
        rerank_calls = 0
        # `None` = 还没决定停 ⇒ 循环继续。收口时才落到 `max_rounds` 上，
        # ⛔ 反过来（初值就写 `max_rounds`）会让循环**永远只跑一轮**，而且不报错。
        stop_reason: str | None = None
        fallback_reason: str | None = None
        current = query

        for index in range(1, self._max_rounds + 1):
            round_query = current
            round_started = monotonic()

            # ① 本轮融合召回。⚠ 深度用请求的 `top_k`：那是**当前默认 Search 的既有口径**
            #    （混合链也是先取 top_k 条再精排）⇒ 不为多跳另造一套检索参数（§3.2）。
            #    于是"融合池 → 精排 → Top-N"这条顺序是结构性成立的，而不是
            #    "先 RRF 截 20 再精排 20"——那两件事的结果不一样，而且不会报错。
            try:
                recalled = self._retriever.search(user_id=user_id, query=round_query, top_k=top_k)
            except Exception:  # noqa: BLE001 —— 见下
                # §10.2：**检索调用失败 ≠ 没有相关记忆**。这里刻意捕得宽：
                # `HybridRetriever.search` 的下游是远端 embedding 与 Qdrant，
                # 它们的异常类型不是本模块该知道的。**吞掉它不等于掩盖它**——
                # ① 一条候选都没攒到时，`fallback_reason` 让调用方去跑原混合链，
                #    同一个错误会在那里**以本来的样子**冒出来；
                # ② 已经攒到证据时停下，并把原因记进 `stop_reason`。
                stop_reason = "retrieval_error"
                if not state.ordered_ids:
                    fallback_reason = "retrieval_error"
                records.append(
                    RoundRecord(
                        index=index,
                        query=round_query,
                        recalled_candidates=0,
                        document_ids=(),
                        candidate_ids=(),
                        selected_ids=(),
                        dropped_documents=0,
                        rerank_status="disabled",
                        stop_reason=stop_reason,
                        retrieval_latency_ms=_ms(round_started),
                    )
                )
                break
            retrieval_ms = _ms(round_started)

            # ② 每轮内部精排（只为了提高交给 LLM 的证据质量，§2.3）
            rerank_started = monotonic()
            ranked, rerank_status = self._rerank_round(query=round_query, candidates=recalled)
            if rerank_status == "called":
                rerank_calls += 1
            rerank_ms = _ms(rerank_started)

            # ③ 本轮 Top-N + 缓存原文
            top = ranked[: self._per_round_top_k]
            top_ids = tuple(candidate.memory_id for candidate in top)
            self._cache_documents(state, top_ids)
            for memory_id in top_ids:
                # 合并多轮必然产生重复 ⇒ 保序去重：第一轮的名次最先，后面各轮只补新的。
                state.ordered_ids.setdefault(memory_id, None)

            # ④ 本轮真正执行过的 Query 记进历史。**必须在调用 LLM 之前**：
            #    §4.3 要求第一轮的 `used_query` 里就有原始问题。
            state.executed.append(round_query)
            state.normalized.add(normalize_query(round_query))

            # ⑤ 组织文档列表（本轮 Top-N ＋ 历史已选证据）
            documents, document_ids, dropped, budget_exceeded = self._build_documents(
                state, top_ids
            )
            if budget_exceeded:
                stop_reason = "prompt_budget_exceeded"
                records.append(
                    RoundRecord(
                        index=index,
                        query=round_query,
                        recalled_candidates=len(recalled),
                        document_ids=(),
                        candidate_ids=top_ids,
                        selected_ids=(),
                        dropped_documents=dropped,
                        rerank_status=rerank_status,
                        stop_reason=stop_reason,
                        retrieval_latency_ms=retrieval_ms,
                        rerank_latency_ms=rerank_ms,
                    )
                )
                break

            # ⑥ 一次 LLM：充分性 + 选证据 + 下一跳
            llm_started = monotonic()
            llm_calls += 1
            decision, failure = self._decide(
                original=query, used=state.executed, documents=documents
            )
            llm_ms = _ms(llm_started)
            input_tokens, output_tokens = _usage(self._llm)

            picked: tuple[str, ...] = ()
            if decision is None:
                # §10.1 第 4、5 条：**保留已经取得的合法候选**，不回退、不重试。
                stop_reason = failure or "llm_error"
                # ⚠ **失败的那一轮不许报 token 数**：`last_usage` 是**客户端上的残留值**
                #   （成功过一次之后它一直留着），照抄会把**上一轮**的用量记到这一轮头上
                #   ——那是 §14 里一个看不出来的假数。
                input_tokens = output_tokens = None
            else:
                # ⑦ 索引 → 稳定 ID（每轮重建，§5.2）
                picked = tuple(document_ids[i] for i in decision.evidence_indices)
                for memory_id in picked:
                    if memory_id not in state.selected:
                        state.selected.append(memory_id)

                # ⑧ 停？——顺序有讲究（§7.4 / §7.5）
                sufficient = decision.is_sufficient
                confident = decision.confidence_score >= self._sufficiency_threshold
                if sufficient and confident:
                    stop_reason = "sufficient"
                else:
                    next_query = decision.new_query.strip()
                    if not next_query or normalize_query(next_query) in state.normalized:
                        # **提不出可用的下一跳。** 三种来源要分得开：
                        #   * 判过"充分"却不够确定 ⇒ `low_confidence`（门槛与提示词标尺的
                        #     交界处，是个真实状态，混进 `duplicate_query` 会让它看不见）；
                        #   * 下一跳是空 ⇒ `empty_query`；
                        #   * **原样回吐原问题 ⇒ `duplicate_query`**。
                        #     ⚠ 最后这条**不是异常**：原版提示词在证据不足却提不出更好
                        #     Query 时就是这么做的。⛔ **不要**把这里实现成
                        #     "置信度不够就必须继续"（§7.5 明确禁止那种读法）。
                        if sufficient:
                            stop_reason = "low_confidence"
                        elif not next_query:
                            stop_reason = "empty_query"
                        else:
                            stop_reason = "duplicate_query"
                    elif index >= self._max_rounds:
                        stop_reason = "max_rounds"
                    else:
                        current = next_query

            records.append(
                RoundRecord(
                    index=index,
                    query=round_query,
                    recalled_candidates=len(recalled),
                    document_ids=document_ids,
                    candidate_ids=top_ids,
                    selected_ids=picked,
                    dropped_documents=dropped,
                    rerank_status=rerank_status,
                    is_sufficient=None if decision is None else decision.is_sufficient,
                    confidence_score=None if decision is None else decision.confidence_score,
                    new_query=None if decision is None else decision.new_query,
                    extraction=None if decision is None else decision.extraction,
                    stop_reason=stop_reason,
                    retrieval_latency_ms=retrieval_ms,
                    rerank_latency_ms=rerank_ms,
                    llm_latency_ms=llm_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                )
            )
            if stop_reason is not None:
                break

        return self._finish(
            state=state,
            records=records,
            stop_reason=stop_reason or "max_rounds",
            fallback_reason=fallback_reason,
            llm_calls=llm_calls,
            rerank_calls=rerank_calls,
            started=started,
        )

    # ── 每一步 ──────────────────────────────────────────────────────────
    def _rerank_round(
        self, *, query: str, candidates: list[Candidate]
    ) -> tuple[list[Candidate], str]:
        """对**本轮候选**精排一次；返回 `(候选, 状态)`。

        ⚠ 状态三分（D12 的口径）：`disabled` = 没打算调（没配 reranker / 关掉了 /
        没有候选）；`called` = 真调了；`degraded` = 打算调但没成。

        ⛔ **降级回原顺序，不中断本轮**：本轮的产出物是"交给 LLM 的证据"，
        名次不完美远好过整条路失败。**但它必须被记下来**——否则一个坏掉的端点
        会表现成"多跳本来就没帮助"，而归因正好错。

        ⚠ 精排**不改变候选集合**，只改名次（与 `pipeline._maybe_rerank` 同一条性质），
        所以本轮的"进/出候选数"是同一个数，记录里只有一个字段。
        """
        if not self._round_rerank or self._reranker is None or not candidates:
            return candidates, "disabled"
        pairs = self._fetch([candidate.memory_id for candidate in candidates])
        documents = [
            render_pair(pair, inject_abs_time=self._inject_abs_time)
            for pair in pairs
            if pair is not None
        ]
        if len(documents) != len(candidates):
            # 真源缺行 ⇒ 拼不出一一对应 ⇒ 不重排（与 `_maybe_rerank` 同一条判断）
            return candidates, "degraded"
        try:
            scores = self._reranker.score(query=query, documents=documents)
        except RerankUnavailable:
            return candidates, "degraded"
        if len(scores) != len(candidates):
            # 形状不符与端点挂了一样不可用——按前缀对齐会让后半段名次错位。
            return candidates, "degraded"
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return (
            [Candidate(memory_id=candidates[i].memory_id, rank=new) for new, i in enumerate(order)],
            "called",
        )

    def _build_documents(
        self, state: _State, top_ids: Sequence[str]
    ) -> tuple[list[str], tuple[str, ...], int, bool]:
        """本轮发给模型的文档列表 + 索引映射 + 丢弃数 + 是否**因预算整体失败**。

        两个约束同时成立：

        * **历史已选证据不能因为本轮名次低就丢掉**（§4.4）⇒ 先按预算给历史**留够**位置；
        * 本轮 Top-N 是这一轮的目的 ⇒ 剩下的预算按名次补。

        ⚠ 顺序是 **本轮优先、历史在后**（§4.4 的图）。索引映射随后**一次性**建立，
        之后**不再改动文档顺序**（§5.3 第 3 条）——顺序一动，模型给的索引就指向别的证据。

        ## 那个 `budget_exceeded` 出口（**今天是不可达的，而且这是可证的**）

        论文式的证明：每一轮都有 `history ⊆ 上一轮的文档`，而上一轮的文档总量
        `≤ max_prompt_tokens`（`fresh` 被 `budget = max_prompt_tokens - history` 卡住）
        ⇒ **归纳可得 `history_tokens ≤ max_prompt_tokens` 恒成立**，
        前提是"只从**发出去的**那份列表里选证据"——那正是第 ⑤ 步做的事。

        ⇒ 它**今天**不可能触发。留着它不是为了兜底某种输入（那不存在），
        而是**为了一个将来才会出现的改动**：一旦有人让模型选中一条**没出现在本轮
        文档里**的记忆，这个归纳就不成立，那时正确的行为是**响亮地停下**，
        而不是安静地交出一份被截断的输入。⛔ 别为了让这个分支"有覆盖"去构造假场景。
        """
        history_ids = [memory_id for memory_id in state.selected if memory_id in state.bodies]
        history = [(memory_id, self._document_body(state, memory_id)) for memory_id in history_ids]
        history_tokens = sum(self._document_cost(state, memory_id) for memory_id in history_ids)
        if history_tokens > self._max_prompt_tokens:
            # 连历史证据都放不下 ⇒ **不组织一个残缺的输入**，安全停下（§4.4 最后一段）
            return [], (), len(top_ids), True

        budget = self._max_prompt_tokens - history_tokens
        selected_set = set(history_ids)
        fresh: list[tuple[str, str]] = []
        used_tokens = 0
        dropped = 0
        for memory_id in top_ids:
            if memory_id in selected_set:
                continue  # 同一轮里同一条只能出现一次（§5.3 第 6 条）
            if memory_id not in state.bodies:
                dropped += 1  # 真源缺行 ⇒ 它进不了文档，也就不能被选中
                continue
            cost = self._document_cost(state, memory_id)
            if used_tokens + cost > budget:
                dropped += 1
                continue
            used_tokens += cost
            fresh.append((memory_id, self._document_body(state, memory_id)))

        # ⚠ **渲染只在这里做一次**：`body` 是 `render_pair` 的裸产物，索引头由下面这一行加上。
        #   把已经带头的文本再嵌一次会得到 `[1]\n[0]\nQ: …`——**而模型读得懂它**，
        #   于是这件事不会报错，只会让"索引 ↔ 证据"的对应悄悄错位。
        ordered = fresh + history
        documents = [
            _render_document(index, body, state.days.get(memory_id, ""))
            for index, (memory_id, body) in enumerate(ordered)
        ]
        return documents, tuple(memory_id for memory_id, _ in ordered), dropped, False

    def _decide(
        self, *, original: str, used: Sequence[str], documents: Sequence[str]
    ) -> tuple[CoqDecision | None, str | None]:
        """一次 LLM 调用 + 严格解析。**失败返回 `(None, 原因)`，从不抛异常。**"""
        prompt = render_coq_prompt(original_query=original, used_queries=used, documents=documents)
        try:
            raw = self._llm.complete(
                prompt, max_tokens=self._max_output_tokens, temperature=self._temperature
            )
        except LlmUnavailable:
            return None, "llm_unavailable"
        except Exception:  # noqa: BLE001 —— 换一个客户端实现就可能抛别的类型
            # 与 `QueryRouter.route` 同一处理：**"预期内的不可用"与"意料之外的 bug"
            # 要分得开**（前者 `llm_unavailable`，后者 `llm_error`），但两者都不中断服务。
            return None, "llm_error"
        try:
            # ⚠ `document_count` 是**本轮真正发出去的那份列表的长度**——索引的合法上界
            #   由它决定，而不是由 Top-N 或任何常量决定。
            return parse_coq_decision(raw, document_count=len(documents)), None
        except InvalidCoqOutput:
            # ⛔ 绝不"补默认值再当作合法决策"（§6.2）⇒ 本轮无效，带着已有候选停下。
            return None, "invalid_output"

    def _finish(
        self,
        *,
        state: _State,
        records: list[RoundRecord],
        stop_reason: str,
        fallback_reason: str | None,
        llm_calls: int,
        rerank_calls: int,
        started: float,
    ) -> AgenticResult:
        candidates = tuple(
            Candidate(memory_id=memory_id, rank=rank)
            for rank, memory_id in enumerate(state.ordered_ids)
        )
        executed = list(state.executed)
        rerank_query = (
            "\n".join(executed) if self._final_query_mode == "combined" and executed else None
        )
        return AgenticResult(
            candidates=candidates,
            rerank_query=rerank_query,
            stats=AgenticStats(
                rounds=tuple(records),
                stop_reason=stop_reason,
                fallback_reason=fallback_reason,
                candidate_count=sum(record.recalled_candidates for record in records),
                unique_candidate_count=len(candidates),
                selected_evidence_count=len(state.selected),
                joint_query_count=len(executed),
                llm_calls=llm_calls,
                rerank_calls=rerank_calls,
                latency_ms=_ms(started),
            ),
            fallback_reason=fallback_reason,
        )

    # ── 原文 ────────────────────────────────────────────────────────────
    def _cache_documents(self, state: _State, memory_ids: Sequence[str]) -> None:
        """把还没有正文的候选从真源取回来（**一次批量读**）。

        ⚠ 缓存是**请求局部**的（挂在 `state` 上）：跨请求留缓存会让"原文"的时效性
        不再可控，而它没有任何收益——同一份真源、同一套渲染。
        """
        missing = [memory_id for memory_id in memory_ids if memory_id not in state.bodies]
        if not missing:
            return
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, missing)
        by_id = {pair.id: pair for pair in pairs}
        for memory_id in missing:
            pair = by_id.get(memory_id)
            if pair is None:
                continue  # 真源缺行：不缓存 ⇒ `_build_documents` 会把它记成 dropped
            state.bodies[memory_id] = render_pair(pair, inject_abs_time=self._inject_abs_time)
            state.days[memory_id] = day_granularity(getattr(pair, "event_time", None))

    def _document_body(self, state: _State, memory_id: str) -> str:
        """一段文档的**正文**（`render_pair` 的产物，**不带索引头**）。"""
        return state.bodies[memory_id]

    def _document_cost(self, state: _State, memory_id: str) -> int:
        """一条文档的 token 成本——**按它最终呈现的样子算**（含索引头与日期行）。

        ⚠ 只数正文会让预算永远偏松，而"超了"这件事**不会报错**：它表现为
        模型收到的提示词比配置说的更大（更慢、更贵、更不准），仅此而已。
        """
        day = state.days.get(memory_id, "")
        return self._counter.count(_render_document(0, self._document_body(state, memory_id), day))

    def _fetch(self, memory_ids: Sequence[str]) -> list[QaPair | None]:
        with self._store.read() as conn:
            pairs = self._store.fetch_pairs_by_ids(conn, list(memory_ids))
        by_id = {pair.id: pair for pair in pairs}
        return [by_id.get(memory_id) for memory_id in memory_ids]


def _render_document(index: int, body: str, day: str = "") -> str:
    """一篇文档。**索引是模型看到的那一个**（§5.1）。

    ⚠ 头部只放**真源里真有的**元数据（日期，且**只到日**，D21）。⛔ 不要为了
    凑成"Time / Role / Content"三行去编 `Role`：本项目的一条记忆是一个 QA 块，
    它的正文自己就带着 `Q:` / `A:` 两个前缀，**编出来的字段就是在给模型假信息**。
    """
    head = f"[{index}] Time: {day}" if day else f"[{index}]"
    return f"{head}\n{body}"


def _ms(started: float) -> float:
    return round((monotonic() - started) * 1000.0, 3)


def _usage(client: LlmClient) -> tuple[int | None, int | None]:
    """从客户端上尽力读一次 token 用量。

    ⚠ `last_usage` **不在 `LlmClient` 协议里**（它是某个具体端点的实现细节，见
    [`../llm/base.py`](../llm/base.py)）⇒ 这里只能"有就读、没有就 `None`"。
    **不要**为了这两个数去扩协议：那会让每个实现都被迫产出一个它未必知道的东西。
    """
    usage = getattr(client, "last_usage", None)
    if not isinstance(usage, dict):
        return None, None
    return _as_tokens(usage.get("prompt_tokens")), _as_tokens(usage.get("completion_tokens"))


def _as_tokens(value: object) -> int | None:
    """`usage` 里的一个计数 → `int`；形状不对就 `None`。

    ⚠ **排除 `bool`**：`True` 是 `int` 的实例，不排除的话一次
    `"prompt_tokens": true` 会被读成 1 个 token，而这在 §14 的汇总里**看不出来**。
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value
