# ruff: noqa: E501
# ⛔ **本文件豁免行宽，而且只因为一件事**：`ROUTER_PROMPT` 是**逐字定稿的提示词正文**。
# 它的规则句子本来就超过 100 列，而抑制指令**不能**写在字符串里面——那会变成提示词的
# 一部分，直接改掉发给模型的内容。★ 这是本仓唯一一处这种豁免：**新加的行仍该守 100 列**，
# 豁免不是把 E501 关掉的理由，是那段字符串动不得的结果。

"""二分类查询路由器 —— **Search 共同取证失败后的策略选择**（基于 `llm/`，不绑模型）。

```text
Search(query, top_k)
   → 基础参数 / 空白 query 检查（入口，不外包给 LLM）
   → 共同事实取证（确定性）—— **先于本模块**；完整 ⇒ 直接返回
   → 【QueryRouter】─ MULTI_HOP → 多跳执行器（不在 / 拿不到证据 ⇒ 回退 DEFAULT）
                   └ DEFAULT   → 原混合检索链
```

⚠ **路由不是 Search 的第一站**：共同事实取证无条件先试。计数和名单类问题由共同取证
处理；无法处理时走 `DEFAULT`。路由器只输出 `MULTI_HOP` 或 `DEFAULT`。

## 三条不变量（本模块的存在理由）

1. **它只选策略**：不查库、不改写问题、不生成答案、**不产出任何事实绑定字段**。
   让模型吐绑定键只有两种结局：静默匹配不上，或**带着逐字引句静默选错事实**。
2. **它的失败不改变失败面的形状**：端点挂 / 超时 / 输出非法 / 执行器未接 ⇒ **一律降级**，
   绝不让一次正常的 Search 变成 500。
3. **关闭它时不发生路由 LLM 调用**，Search 保持默认混合检索行为。

## 为什么"只要二选一"是安全的

Query 是**不可信输入**，注入在原理上防不住。所以本模块**不靠提示词设防**，
靠**输出域极小 + 两条路都安全**：输出只可能落在两个枚举值里（严格匹配，见 `parse_route`），
两条路本身都是既有的、测过的路径。⇒ 注入最多让它**选错分支**，不能引入新路径。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from tianximem.llm.base import DEFAULT_MAX_OUTPUT_TOKENS, LlmClient, LlmUnavailable

__all__ = [
    "NONE_OUTPUT",
    "ROUTER_PROMPT",
    "ROUTER_PROMPT_VERSION",
    "ROUTE_STATUSES",
    "Route",
    "RouteDecision",
    "QueryRouter",
    "parse_route",
]


class Route(StrEnum):
    """两个互斥的检索策略类别。**值就是模型该原样吐出来的那个词。**

    继承 `str` 使路由值可直接写入 JSON 指标快照。

    **只有两个取值**。`parse_route` 对枚举值做整串匹配；其他输出按非法输出降级到
    `DEFAULT`。计数和名单类问题由先运行的共同取证处理。
    """

    MULTI_HOP = "MULTI_HOP"
    DEFAULT = "DEFAULT"


#: 当前路由提示词版本，也是 `router.prompt_version` 的合法值。
#:
#: 这是配置身份断言：`QueryRouter` 在构造时要求配置值与此常量相等，否则启动失败。
#: 修改提示词文字时必须同时更新本常量和 `configs/*.yaml`。
ROUTER_PROMPT_VERSION: Final[str] = "v5"

#: 当前生效的路由提示词，逐字定稿，不要润色。
#:
#: 这是本仓库唯一的 Router 提示词，版本号用于校验提示词身份。
#:
#: 提示词包含三条关键约束：
#:
#: 1. **没有「不确定就选 DEFAULT」的兜底**——它会让绝大多数题落进 `DEFAULT`；
#: 2. **`MULTI_HOP` 的判据压成一句可执行的检验 + 一串显式依赖信号词**
#:    （`"then"` / `"after"` / `"using that"` …），
#:    并显式排除「多个具名实体 / 多条事实 / 普通比较」；
#: 3. **不包含 `COUNT`**（计数 / 名单类由共同取证处理）。
#:
#: ## 没有 `NONE` 输出
#:
#: * AML 的响应只有 `data` 数组，空结果也是空数组，因此不存在第四个对外取值；
#: * 空或无效输入属于入口校验，由 HTTP 边界和 `SearchPipeline` 的 `blank_query`
#:   短路处理，不交给 LLM。
#:
#: 解析层可以区分游离的 `NONE`（`model_none`）和其他非法输出，但两者都降级到
#: `DEFAULT`。格式中的前导空白由解析层去除，提示词文本保持不变。
ROUTER_PROMPT: Final[
    str
] = """You are a  router. Your task is to select exactly ONE tool name from the provided list that best fits the user query. Do not call any tools. Use only the text in {query}; do not assume external context or missing metadata. You may interpret dependency only from explicit linguistic structure (e.g., "use X to find Y", possessive/relationship chains).

GOAL
- Choose exactly one of: MULTI_HOP, DEFAULT.

MECHANISM (what to do, then how to do it)

1) Classify the query type from {query} using ONLY the operational criteria below (pick exactly one):

A) MULTI-HOP (dependency chain; later step depends on earlier result)
Choose MULTI-HOP if the query requires two or more dependent steps where you must first determine X, then use X to determine Y (or more).
- Explicit dependency signals: "then", "after", "using that", "based on that result", "from there", "which of those", "once you find", "given the answer to", "trace/derive".
- Relationship-chain dependency: queries that require resolving an intermediate entity/attribute to proceed (e.g., "X's spouse's company", "author of the book that inspired the film").
- Comparison/timeline becomes MULTI-HOP ONLY if it requires derived attributes that must be found first (e.g., "Which is older, the CEO of A or the founder of B?" requires finding CEO/founder first).
- Tie-breaker: If any explicit dependency chain exists, classify as MULTI-HOP even if multiple entities/keywords appear.

B) DEFAULT (direct or independent retrieval)

Choose DEFAULT when the query is a straightforward lookup or contains multiple independent lookups that do not require dependent resolution.

- Multiple known entities, subjects, or questions without dependency belong here.
- Direct comparisons of independently identifiable facts belong here.
- Use DEFAULT when A does not apply.

2) Deterministic routing (hard mapping)

- If MULTI-HOP dependency exists -> MULTI_HOP.
- Else -> DEFAULT.

AVAILABLE ROUTES

MULTI_HOP: Iterative retrieval that searches step by step, uses retrieved evidence to identify missing relationships, and generates subsequent queries.

DEFAULT: Standard hybrid memory retrieval without iterative query rewriting.

INPUT
Query:
{query}

OUTPUT FORMAT (strict)
Output a single line containing only:
MULTI_HOP
  DEFAULT

Do not output explanations, confidence scores, JSON, or any additional text.

CLASSIFICATION EXAMPLES (for calibration; do not output these)

1) "Who is the author of Dune?" -> DEFAULT
2) "Give the capitals of Spain and Portugal." -> DEFAULT
3) "Find the spouse of Marie Curie, then name his primary field." -> MULTI_HOP
4) "Compare GDP per capita of Japan and South Korea." -> DEFAULT
5) "Which is older: the founder of Company A or the CEO of Company B?" -> MULTI_HOP
6) "What type of car did Evan get after his old Prius broke down?" -> MULTI_HOP
7) "How many conferences did the person who hired Melanie attend?" -> MULTI_HOP"""

#: 版本 → 提示词。**这是"哪些版本存在"的唯一声明处。**
#:
#: ⚠ **当前只启用这一版提示词**。若要评估另一版，必须同时声明新的提示词与版本号；
#: `cfg` 和 `configs/*.yaml` 两处不一致时会在构造期报错。


#: `NONE` —— **v3 引入的"模型明确拒答"**。⛔ **它不是第三条执行路**：
#: 管线只有两条（`MULTI_HOP` / `DEFAULT`），所以它走**降级**路径，
#: 落到 `DEFAULT`，并记 `reason = "model_none"`。
#:
#: ⚠ **必须与"输出非法"分开记**：一个是"模型说它判不了"，一个是"模型说了一句我们看不懂的话"
#: （`NOT MULTI_HOP`、一段 JSON、一句解释）。混成一个数，就再也看不出
#: "新提示词是不是让模型更容易拒答"——而那是 v3 最该被盯住的一格。
NONE_OUTPUT: Final[str] = "NONE"


def is_none_output(text: str) -> bool:
    """模型是不是明确拒答了。**归一化规则与 `parse_route` 完全一致**（只去空白 + 大小写）。"""
    return text.strip().upper() == NONE_OUTPUT


def parse_route(text: str) -> Route | None:
    """严格解析模型输出；**不合法 ⇒ `None`**（调用方回退 `DEFAULT`）。

    ## 为什么是"整串等于枚举值"，不是"包含某个词"

    ⛔ **禁止子串匹配**：模型回 `NOT MULTI_HOP`（"这不是多跳"）时，
    子串匹配会读成 `MULTI_HOP`——**把否定读成肯定**，而且**方向正好是反的**。
    `DEFAULT` 那一档也会被 `"No, this is DEFAULT behavior"` 这类句子随机命中。

    ## 允许的两步归一化，以及为什么它们不会放宽判据

    1. **首尾空白**（含换行）去掉——模型常带一个结尾换行，这不是"输出非法"。
    2. **大小写归一**（`upper()`）——`multi_hop` 与 `MULTI_HOP` 是同一个词的不同排版。

    ⚠ **这两步都不可能把一个非法串变成合法串**：归一化只改空白与大小写，
    **不做任何子串、前缀、相似度或模糊匹配**。`"not multi_hop".upper()`
    仍然不等于 `"MULTI_HOP"`。

    ⚠ **多行输出一律非法**（`"DEFAULT\\nbecause ..."` ⇒ `None`）：
    提示词要求恰好一行，而收下"第一行"就等于**默许模型附加解释**——
    那正是 `OUTPUT FORMAT` 那一段要挡的东西。
    """
    parsed = text.strip().upper()
    for route in Route:
        if parsed == route.value:
            return route
    return None


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """一次路由的结果和执行元数据；元数据只进入指标与测试，不进入响应。

    | 字段 | 含义 |
    | --- | --- |
    | `route` | 实际要走的策略。**降级时恒为 `DEFAULT`** |
    | `status` | `success`（模型正常分类）/ `degraded`（降级到 DEFAULT） |
    | `reason` | 降级的**具体原因**（成功时为 `None`）——`llm_unavailable` / `invalid_output` / … |
    | `raw` | 模型原样回的那段文本（成功时=枚举值；用于诊断"它到底说了什么"） |
    | `latency_ms` | 这一次路由的墙钟耗时 |
    """

    route: Route
    status: str
    reason: str | None
    raw: str
    latency_ms: float


#: `RouteDecision.status` 的取值域（**唯一声明处**）。与 `rerank` 的
#: `calls` / `disabled` / `degraded` 三分法同形：**"没打算用它"与"想用但没成"必须分得开**
#: ——混为一谈会把"端点一直挂"看成"我们本来就没开路由"。
ROUTE_STATUSES: Final[tuple[str, ...]] = ("success", "degraded")


class QueryRouter:
    """把 Query 分成 `MULTI_HOP` / `DEFAULT` 中的一个。**永不抛异常。**

    ⚠ **它不知道任何检索细节**：不 import `store/`、不查 Qdrant、不读 SQLite，
    也**不 import `../retrieve/evidence.py`**——它只把一个字符串映射成两个枚举值之一。
    "选出来的路怎么走"是 [`../service/pipeline.py`](../service/pipeline.py) 的事。
    """

    def __init__(
        self,
        client: LlmClient,
        *,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
        prompt_version: str = ROUTER_PROMPT_VERSION,
    ) -> None:
        # ⚠ **版本不符 ⇒ 构造期响亮失败**（而不是跑起来之后在 run record 里留下一个假指纹）。
        #   它与 `rerank.envelope` 是同一条纪律：值只有一个正确答案，写错**不报错**才是灾难。
        if prompt_version != ROUTER_PROMPT_VERSION:
            raise ValueError(
                f"`router.prompt_version` 必须是 {ROUTER_PROMPT_VERSION!r}，"
                f"收到 {prompt_version!r}。\n"
                "  它是一条【断言】不是说明：本实现带的是哪一版提示词，配置就得声称哪一版。\n"
                "  ⚠ 换了提示词，就必须同时改 `ROUTER_PROMPT_VERSION` 与 `configs/*.yaml`——"
                "否则两轮实验的 run record 会声称同一份提示词，而那种漂移查不出来。"
            )
        if max_output_tokens <= 0:
            raise ValueError(f"`router.max_output_tokens` 必须为正：{max_output_tokens}")
        if temperature < 0:
            raise ValueError(f"`router.temperature` 不得为负：{temperature}")
        self._client = client
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._prompt_version = prompt_version
        #: 本实例用的那一份提示词（= 全仓唯一的那一份）。
        self._prompt = ROUTER_PROMPT

    # ── 标识 ───────────────────────────────────────────────────────────

    @property
    def client(self) -> LlmClient:
        """底下的 LLM 客户端（**只读暴露给测试与诊断**：装配漏接是静默的）。"""
        return self._client

    @property
    def model(self) -> str:
        """模型标识（进 run record 的配置指纹）。**没有硬编码的模型名**——
        它由 `LlmClient` 的实现给出，端点与名字都在 `.env`。"""
        return self._client.name

    @property
    def prompt_version(self) -> str:
        """**这一实例实际用的那一版**——不是"当前是哪一版"（两臂并存时它们不同）。"""
        return self._prompt_version

    # ── 路由 ───────────────────────────────────────────────────────────

    def route(self, query: str) -> RouteDecision:
        """分类一次。**任何失败都退成 `DEFAULT` + `degraded`，绝不抛。**

        ⚠ **Query 用 `replace` 注入，不用 `format`**：`format` 会把 Query 里的 `{}` 当成格式串
        （那是用户可控的输入），轻则 `KeyError`，重则**把模板的其余部分吃掉**。
        """
        started = time.perf_counter()
        prompt = self._prompt.replace("{query}", query)
        try:
            raw = self._client.complete(
                prompt,
                max_tokens=self._max_output_tokens,
                temperature=self._temperature,
            )
        except LlmUnavailable as exc:
            return self._degrade("llm_unavailable", str(exc), started)
        except Exception as exc:  # noqa: BLE001 — 见下
            # ★ **兜底捕获是刻意的**（与 `RemoteReranker.score` 同一条理由）：
            #   "Router 的失败不能直接导致一次正常 Search 返回 500"。协议级假实现
            #   在测试里会抛任意异常，而生产上"端点行为超出预期"也是常态。
            # reason 保留异常类型，使未预期缺陷与端点不可用在指标中可区分。
            return self._degrade("llm_error", f"{type(exc).__name__}: {exc}", started)

        if is_none_output(raw):
            # ⚠ **`NONE` 走降级，不是走第四条路**（见 `NONE_OUTPUT` 的说明）。
            #   它意味着"模型拒绝分类"——对 Search 而言与"没分成类"同一条出口。
            return self._degrade("model_none", None, started, raw=raw)
        route = parse_route(raw)
        if route is None:
            # 非法输出 ⇒ DEFAULT。这里不重试，避免与 AML 的重试叠加，也避免无依据地
            # 重复询问没有置信度信号的模型。
            return self._degrade("invalid_output", None, started, raw=raw)
        return RouteDecision(
            route=route,
            status="success",
            reason=None,
            raw=raw,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )

    # ── 降级 ───────────────────────────────────────────────────────────

    def _degrade(
        self, reason: str, detail: str | None, started: float, *, raw: str = ""
    ) -> RouteDecision:
        """**唯一的降级出口**——所有失败形态都从这里回到 `DEFAULT`。

        ⚠ `raw` 只在"输出非法"时有值，且它**进的是诊断**：模型到底回了什么，
        是排查"为什么这一类题老是 `DEFAULT`"的第一手材料。
        ⚠ `detail`（异常消息）**不拼进 `raw`**：模型原样输出与端点错误是两回事，
        混在一个字段里就没法按"谁的问题"筛了。
        """
        return RouteDecision(
            route=Route.DEFAULT,
            status="degraded",
            reason=f"{reason}: {detail}" if detail else reason,
            raw=raw,
            latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
