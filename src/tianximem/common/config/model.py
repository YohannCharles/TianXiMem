"""配置树的形状：`ConfigError` + 全部 `@dataclass(frozen=True, slots=True)` 配置类。

⚠ 每个键的归类（A / B / C）与完整清单在 `docs/config-reference.md`——本文件不另列一份。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from tianximem.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

from .constants import (
    DEFAULT_AGENTIC_MODE,
    DEFAULT_CAPTURE_MAX_BYTES,
    DEFAULT_COQ_MAX_OUTPUT_TOKENS,
    DEFAULT_COQ_MAX_PROMPT_TOKENS,
    DEFAULT_COQ_MAX_ROUNDS,
    DEFAULT_COQ_PER_ROUND_TOP_K,
    DEFAULT_COQ_PROMPT_VERSION,
    DEFAULT_COQ_SUFFICIENCY_THRESHOLD,
    DEFAULT_COQ_TIMEOUT_S,
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_GROUNDED_PLANNER_MAX_OUTPUT_TOKENS,
    DEFAULT_GROUNDED_PLANNER_PROMPT_VERSION,
    DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION,
    DEFAULT_PLAN_MAX_OUTPUT_TOKENS,
    DEFAULT_PLAN_STEP_PROMPT_VERSION,
    DEFAULT_PREFETCH_LIMIT,
    DEFAULT_PROFILE,
    DEFAULT_RADIUS,
    DEFAULT_RERANK_ENVELOPE,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_ROUTER_MAX_OUTPUT_TOKENS,
    DEFAULT_ROUTER_PROMPT_VERSION,
    DEFAULT_ROUTER_TIMEOUT_S,
    DEFAULT_WEIGHTS,
    RRF_K,
)


class ConfigError(ValueError):
    """配置不合法。**必须在启动阶段响亮失败**，不要等到第一个请求才炸。"""


# ── 配置树 ─────────────────────────────────────────────────────────────
# 字段名与 `docs/config-reference.md` 的点分键**逐段对应**（`storage.sqlite.path`
# ↔ `AppConfig.storage.sqlite.path`），免得两份文档各说各的。


@dataclass(frozen=True, slots=True)
class SqliteConfig:
    """真源文件（§6.1）。"""

    #: `TIANXIMEM_SQLITE_PATH`。【路径归 env】
    path: str = "var/tianxi.db"
    #: 抢不到写锁时**等待**而不是立刻抛 `SQLITE_BUSY` 的毫秒数（D17）。【阈值归 yaml】
    busy_timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    """派生索引（§6.3）。**必须 server 模式**（local 模式会静默丢弃 payload 索引）。"""

    #: `TIANXIMEM_QDRANT_URL`。【端点归 env】
    url: str = "http://localhost:6333"
    #: 集合名。单集合 + `user_id` 作 tenant 过滤（D3）。【名字归 yaml】
    collection: str = "memories"


@dataclass(frozen=True, slots=True)
class StorageConfig:
    sqlite: SqliteConfig = field(default_factory=SqliteConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


@dataclass(frozen=True, slots=True)
class EmbedCacheConfig:
    """落盘的向量缓存（§7.2）。"""

    #: `TIANXIMEM_EMBED_CACHE_DIR`。**必须落盘**，否则每次重启重付一遍 embedding。【路径归 env】
    dir: str = "var/embed_cache"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    embed: EmbedCacheConfig = field(default_factory=EmbedCacheConfig)


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """模型名（§2.3）。"""

    #: ⚠ **这里是模型名的家**，不是 `.env`：它是 profile 之间**唯一真正该变**的东西，
    #: 而 `local.yaml` / `submit.yaml` 存在的理由就是让"哪些量随模型变"能被看见
    #: （config-reference §9）。
    #: ⚠ 开发期使用 **4B**（`memory2.021130.xyz` 提供 embedding）——与 `default.yaml`
    #: 及 `embed/qwen3_embedding.py::DEFAULT_MODEL` **三处同值**，由
    #: `tests/test_config.py::test_model_default_is_the_same_in_both_places` 钉住。
    embedder: str = "Qwen/Qwen3-Embedding-4B"

    #: **期望的向量维度**——它是**断言，不是赋值**：维度仍然来自真实响应（§7.4 的
    #: `dim` 只读属性），这一项只回答"这个部署接受多大的向量"，不符就响亮失败。
    #: ⚠ 它**进缓存坐标系**（`embed/base.py` 的 `EmbeddingCoordinate`）⇒ 换维度自动换一个
    #: 缓存文件，旧缓存不可能被静默命中（2026-10-09 那次"改了维度却命中旧向量"就是这么发生的）。
    #: ⚠ §2.3 规定的两个模型——`text-embedding-v4`（提交期）与开发期那个
    #: （2026-10-13 起是 Qwen3-Embedding-4B，此前的 8B 与它**也**同维）
    #: ——**同为 1024 维**，所以这一个值同时覆盖两个 profile。
    #: ⛔ 别把这个巧合当成保障：**换模型不改维度时，没有任何一层会拒绝**，
    #:   只表现为"检索变差"⇒ 换模型必须重建集合、作废缓存（`models.embedder` 进缓存坐标）。
    embed_dim: int = 1024


@dataclass(frozen=True, slots=True)
class RrfConfig:
    """Weighted RRF 的参数（§7.3）。"""

    #: **正确性常量，不是旋钮**（D5）。见模块顶部的 `RRF_K`。
    k: int = RRF_K
    #: 顺序与 §7.3 的 `prefetch` 顺序**一一对应**：固定 `(bm25, dense)`。
    #: **任何调整都必须有 ablation 数据支撑**（C 类）。
    weights: tuple[float, float] = DEFAULT_WEIGHTS


@dataclass(frozen=True, slots=True)
class GroundedPlannerConfig:
    """Optional LLM interpretation after deterministic query compilation."""

    enabled: bool = False
    timeout_seconds: float = DEFAULT_ROUTER_TIMEOUT_S
    max_output_tokens: int = DEFAULT_GROUNDED_PLANNER_MAX_OUTPUT_TOKENS
    temperature: float = 0.0
    prompt_version: str = DEFAULT_GROUNDED_PLANNER_PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """混合检索（§7.1–§7.3）。"""

    #: 每路进入 RRF 的候选池大小（§7.3 的 `N`）。
    #: ⚠ **不是**种子数（§10 的 20）、**不是** Top-K（§2.2 的 100）——三个不同的量。
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT
    #: 查询侧的 instruction 前缀。`""` ⇒ **与"原样送"逐字节相同**（v1 现状）。
    #: 要不要开是**待定的规格问题**，见 `embed/query_instruction.py`。
    query_instruction: str = ""
    #: 所有来源支持的事实共用一个同步索引/查询执行器。
    grounded_evidence: bool = False
    evidence_limit: int = 12
    evidence_hop_limit: int = 4
    fact_backfill_limit: int = 1024
    grounded_planner: GroundedPlannerConfig = field(default_factory=GroundedPlannerConfig)
    rrf: RrfConfig = field(default_factory=RrfConfig)


@dataclass(frozen=True, slots=True)
class NeighborConfig:
    """Neighbor Expansion（§10）。"""

    #: 主动扩窗的种子数——**只对 rerank 后的前 N 条扩窗**，其余候选仍保留、只是不扩展。
    #: ⚠ 为什么值的家在这里、为什么它不是调参项：见模块顶部的 `DEFAULT_EXPANSION_SEED_LIMIT`。
    expansion_seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT
    #: 扩窗半径，**单位是 QA 对**：±1 拿回前后各**一整对**（最多 4 条消息）。
    radius: int = DEFAULT_RADIUS
    #: 段内**种子放在哪**（§11.2 的"组内顺序"，明文列为可消融项）。
    #:
    #: * `keep`（默认）：纯位置（`local_index`）时间序，
    #:   种子在它本来的时间位置上
    #: * `front`：种子移到**段首**，其余照时间序 —— ⚠ **段内时间连续性会断**，
    #:   而"窗口是一段连续对话、按时间序读才成立"正是 §11.2 选时间序的理由
    #: * `echo`：种子在段首**重复一遍**，下面**完整的时间序块原样保留** ——
    #:   时间是连续的，代价是多花一对的 token（实测 ≈ +3.5%）
    #:
    #: 实验动机：multi-hop 错的 29 题里 **28 题的证据就在上下文里**、模型没抓住
    #: （答得太笼统 / 抓错事实）⇒ 试"把命中的那对放显眼"。**三个变体都跑过再定**。
    seed_placement: str = "keep"


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """token 预算（§6.4）。**双预算里的 token 那一半**；槽位那一半是请求里的 `top_k`。"""

    #: 上限（**A 类外部契约常量**：AML 定的，无权改，写进来只为追溯）。
    max_tokens: int = MAX_INPUT_TOKENS
    #: 分词器名。**必须是答案模型自己的那个**（§6.4）——用近似会在 100 个对上放大到几千 token。
    tokenizer: str = DEFAULT_TOKENIZER


@dataclass(frozen=True, slots=True)
class PackagingConfig:
    """上下文打包的渲染变体（§11.3）；键语义以 config-reference 为准。"""

    #: `false`（**v1 定稿**）⇒ `content` 里没有任何绝对时间戳，时间只走 `event_time` 筛选。
    #: `true`（**D21 起为默认**）⇒ 每对正文前加日粒度日期前缀。
    #:
    #: ⚠ 它是**"贵"消融项**：正文一改，embedding 输入就改 ⇒ **向量索引要重建**，
    #: 而且必须跑在**另一个集合**上（`storage.qdrant.collection`）——两臂混在一个集合里，
    #: 检索到的是哪一臂的向量**根本看不出来**。
    inject_abs_time: bool = True

    #: `false` ⇒ `content` 逐字等于被索引的文本（**消融臂**）。
    #: `true`（**2026-09-26 起默认**）⇒ 把每对正文里的**相对时间就地注解**成绝对日期
    #: （`last Tues (July 18, 2023)`），原文一字不动——实现与口径见 [`annotate.py`](./annotate.py)。
    #:
    #: ⚠ **它与 `inject_abs_time` 有一处根本区别**：那个改的是 embedding 输入，
    #: **本键只改 `content`** ⇒ 随时开关，**不用重建索引、不用换集合**。
    #: 代价是 `content` 不再逐字等于被索引的文本（不变式 I1 的**一个声明式例外**）：
    #: `content` = 被索引的文本 + 一层确定性注解，去掉注解后逐字相同（`tests/test_annotate.py`）。
    #:
    #: ⚠ 它是 **C 类**（自设阈值那一类），**任何调整都要有 ablation 数据**（§7.3）——
    #: 开关本身与它买到的东西见
    #: [`../../../eval/reports/ledger.md`](../../../eval/reports/ledger.md)。
    annotate_relatives: bool = True
    #: 仅在 Search 打包交易事实时补单笔金额；不改索引或聚合跨记录答案。
    annotate_unit_totals: bool = False


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """远程 reranker（§11.2 / D12）。**只放阈值与开关；端点与密钥在 env。**

    ⚠ **本段刻意只有三个键**：一个键要有消费者才收（§6.1 对 DDL 的同一条纪律）——
    而 rerank 的真实旋钮只有"开不开"、"等多久"与"发哪个字段名"。想调模型就换 `.env` 的
    `TIANXIMEM_RERANKER_MODEL`（那是端点身份，不是阈值）。

    ⚠ `envelope` 是三个里**唯一跟端点走**的：它描述的是"对面那个网关说哪种线格式"，
    与 `TIANXIMEM_RERANKER_BASE_URL` 是同一件事的两半 ⇒ **换网关必须一起换**。
    """

    #: §15 的消融开关之一。`false` ⇒ **不构造 reranker**，Search 直接用融合名次。
    #:
    #: ⚠ 它的"关"分支**只该改变排名**，不得改变候选数量（§13 的开关纯度）——
    #: `tests/test_rerank.py` 有这条断言。
    enabled: bool = True
    #: 单次 rerank 请求的超时（秒）。见模块顶部 `DEFAULT_RERANK_TIMEOUT_S` 的两条约束。
    timeout_seconds: float = DEFAULT_RERANK_TIMEOUT_S
    #: 请求里那一个字段叫什么。见模块顶部 `DEFAULT_RERANK_ENVELOPE`——
    #: **填错不报错，只静默降级**（两端各有自己的拒绝码：400 / 422）。
    envelope: str = DEFAULT_RERANK_ENVELOPE


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """`Add` / `Search` 的**原文采集**（诊断旁路，**默认关**）。

    ⚠ **它不属于 A / B / C 任何一类**：那三类说的都是"影响结果的量"，而本项
    只把官方发来的请求抄一份落盘（[`../service/capture.py`](../service/capture.py)）。
    默认关是因为它按设计**只在"要看清官方发了什么"时打开**（S6），其余时间是净开销。

    实现与四条纪律（在解析之前抄 / 不改下游 body / 不吞异常 / 写失败不影响响应）见那个模块。
    """

    #: 开关。`false` ⇒ **不装中间件**（零开销、零行为差异）。
    enabled: bool = False
    #: **每份文件**的字节上限，写满就换下一份（`<name>.part2.jsonl`…）。
    #: 见模块顶部的常量说明——**它不是总量上限**。
    max_bytes: int = DEFAULT_CAPTURE_MAX_BYTES
    #: 落点（`TIANXIMEM_CAPTURE_PATH`）。【路径归 env】——与 SQLite / 向量缓存同一条规矩。
    path: str = "var/capture/requests.jsonl"


@dataclass(frozen=True, slots=True)
class RouterConfig:
    """二分类查询路由器（`retrieve/router.py`）。**共同取证失败之后的策略选择。**

    ⚠ **它不是 Search 的第一站**（2026-10-14 起）：共同事实取证（确定性、不调 LLM）
    无条件先试，只有它没给出完整证据时才轮到本模块。路由开着时，
    `DEFAULT` 类问题**不再先试共同取证**是这条接缝原先的代价——⚠ **该代价已随
    取证前移而消失**：今天两条分支都在取证失败之后。

    ⚠ **默认 `false`，而且这是刻意的**：打开它仍然**改变 Search 的路径选择**，
    而 `configs/default.yaml` 是**提交口径**——一个未经代理评测标定的行为改动
    **不该默认生效**。要跑对照就用 profile 覆盖（`TIANXIMEM_PROFILE`）或臂快照。

    ⚠ **模型名不在这里**：`router.model` 这一项由**全局的 `models.llm`** 满足
    （`.env` 的 `AML_MODEL`）——**不另开一个键**。理由与 §4.4 的"不得引入第二套独立的
    模型配置或环境变量加载机制"是同一条：两处都能设的模型名，最终会变成
    "run record 里写的模型不是实际打的那个"。

    ⚠ **没有置信度阈值，这是有意的**（§4：不设置路由置信度阈值）。分类是**类别选择**，
    不是打分；引入一个未经标定的数字，等于把一个"要么对要么错"的问题换成一个
    "阈值定在哪"的问题，而后者需要一份**我们没有**的校准集。
    """

    #: §15 的消融开关之一。`false` ⇒ **不构造路由器**，这条链上**一次 LLM 调用都不发生**，
    #: 且共同取证分支照旧无条件先试（= 本开关存在之前的行为）。
    #:
    #: ⚠ **本内置缺省是 `false`，而 `default.yaml` 写的是 `true`**——与
    #: `grounded_evidence` 同一条口径：缺这个键的快照就按"不构造路由器"跑，
    #: 基线（`configs/default.yaml`）则把它打开（2026-10-14 起）。
    enabled: bool = False
    #: 单次路由调用的超时（秒）。见模块顶部 `DEFAULT_ROUTER_TIMEOUT_S` 的说明
    #: （它与 reranker 的超时**方向不同**，别照抄那一条）。
    timeout_seconds: float = DEFAULT_ROUTER_TIMEOUT_S
    #: 单次生成的输出上限（token）。见 `DEFAULT_ROUTER_MAX_OUTPUT_TOKENS`。
    max_output_tokens: int = DEFAULT_ROUTER_MAX_OUTPUT_TOKENS
    #: 生成温度。**默认 0 是"这一步要可复现"的声明**（§6 的可复现性要求）。
    #: ⚠ 网关**可能忽略它**（实测有些自研封装不接这个字段）——那就没有确定性保证，
    #: 而"重放同一批问题得到不同路由"会表现为分数抖动。所以 §6 那条"在一小组边界问题上
    #: 重复调用、观察路由是否稳定"**必须真跑**，不能靠这个键的存在当保证。
    temperature: float = 0.0
    #: 提示词版本（`retrieve/router.py` 的 `ROUTER_PROMPT_VERSION`）。
    #:
    #: ⚠ **它是一条断言，不是说明**：与代码里的常量不符 ⇒ **构造期响亮失败**。
    #: 理由是 §6 的可复现性——"改了两轮提示词却忘了改版本号"的表现是
    #: **两份 run record 声称同一份提示词**，而那种漂移查不出来。
    prompt_version: str = DEFAULT_ROUTER_PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class AgenticConfig:
    """多跳执行器（`MULTI_HOP` 分支）——**执行器已经实现**，默认仍然关着。

    ⚠ 与 `RouterConfig` 的关系是**两条独立的开关**，别把它们读成一个：

    | | 作用 |
    | --- | --- |
    | `router.enabled` | **选路**（两个枚举值之一）。开它 ⇒ 按模型判的走 `MULTI_HOP` 或 `DEFAULT` |
    | `agentic.enabled` | **执行器在不在**。开它 ⇒ `MULTI_HOP` 真的跑多轮检索，而不是降级 |

    今天**只有两个都开着**，多跳那条路才会真的跑起来；只开 `agentic` 而不开 `router`
    则一条请求都不会走到这里（除非用了 `force`，那是实验开关，见下）。

    ⚠ **默认全关**：一条会多花 3 次 LLM ＋ 最多 3 次精排的路，**不该默认生效在提交口径上**。
    """

    #: 是否构造多跳执行器。`false`（默认）⇒ `build_agentic` 返回 `None`
    #: ⇒ 每条 `MULTI_HOP` 降级回 `DEFAULT` 并记 `route_fallback = agentic_unavailable`。
    enabled: bool = False

    #: **执行器的模式**（[`AGENTIC_MODES`][AGENTIC_MODES] 之一）。
    #: **内置缺省 `"react"`**（[`DEFAULT_AGENTIC_MODE`][DEFAULT_AGENTIC_MODE] 有完整理由）；
    #: ⭐ **`configs/default.yaml` 是提交口径，那里写的是 `plan`**（2026-10-14 起）。
    #: ⇒ 从 yaml 加载出来的配置是 `plan`，直接构造 `AgenticConfig()` 的是 `react`。
    #:
    #: ⚠ 它选的是**执行器**，⛔ **不是"走不走这条路"**——那个仍然归 `router.enabled`
    #: 与 `force`。三条键的关系：
    #:
    #: ```text
    #: router.enabled / force  ⇒  走不走 MULTI_HOP
    #: agentic.enabled         ⇒  装不装执行器
    #: agentic.mode            ⇒  装的是哪一种执行器
    #: ```
    mode: str = DEFAULT_AGENTIC_MODE

    #: **实验开关：绕过路由器，把所有查询按 `MULTI_HOP` 处理**（§12.3 的 E1）。
    #:
    #: ⚠ 它存在的**唯一**理由是 E1 那一臂要"独立验证 CoQ 的检索能力"——不绕过路由器，
    #: 测出来的就是"路由器 × 多跳"的合成效果，两者分不开（§12.3 明说了这件事）。
    #: ⛔ **它不是给提交口径用的**：打开它等于关掉 `MULTI_HOP` / `DEFAULT` 两条路的选择能力。
    #: 关闭它（默认）时这条链与本开关存在之前**逐字相同**。
    force: bool = False

    #: 检索轮数上限，**含第一轮**（§7.4）。见 `DEFAULT_COQ_MAX_ROUNDS` 里与 PRD §9 的差异说明。
    #:
    #: ⚠ **它同时是 Plan 模式的检索次数上限**（两个模式共用这一道预算键，
    #: 因为"轮数 × 每轮条数"本来就是同一道题）。Plan 的计划本身最多 4 步
    #: （提示词写死"one to four"），所以默认 3 ⇒ **一个 4 步计划只会跑 3 步**
    #: 并记 `stop_reason = "max_rounds"`。那不是错误，但它必须能被看见。
    max_rounds: int = DEFAULT_COQ_MAX_ROUNDS

    #: 每轮交给 CoQ 的候选数（精排**之后**取前 N 条）。
    per_round_top_k: int = DEFAULT_COQ_PER_ROUND_TOP_K

    #: 提前停止的置信度门槛。⚠ 只在"已判定充分"之后起作用，见该常量的说明。
    sufficiency_threshold: float = DEFAULT_COQ_SUFFICIENCY_THRESHOLD

    #: 每轮 prompt 的 token 上限。**与 `budget.max_tokens` 不是同一个量**。
    max_prompt_tokens: int = DEFAULT_COQ_MAX_PROMPT_TOKENS

    #: 单次生成的输出上限（token）。⚠ **React 模式用这个值**（回一个四字段的 JSON）。
    #: Plan 模式用 `plan_max_output_tokens`——两者的输出形状不同，见那个常量的说明。
    max_output_tokens: int = DEFAULT_COQ_MAX_OUTPUT_TOKENS

    #: **Plan 模式**单次生成的输出上限（token）。见 `DEFAULT_PLAN_MAX_OUTPUT_TOKENS`。
    #: ⚠ React 模式**不看它**——两条路的输出结构不一样，合成一个数会让其中一边被卡住。
    plan_max_output_tokens: int = DEFAULT_PLAN_MAX_OUTPUT_TOKENS

    #: 单次调用的超时（秒）。
    timeout_seconds: float = DEFAULT_COQ_TIMEOUT_S

    #: 生成温度。**默认 0 是"这一步要可复现"的声明**（§6）。
    temperature: float = 0.0

    #: **每轮内部精排开关**（§12.4 的那个消融：A = 直接用 RRF 顺序取 Top-N；
    #: B = 先精排再取 Top-N）。默认 `true` = 方案的主方案。
    #:
    #: ⚠ 它**只影响交给 LLM 的那 20 条**：最终精排由调用方做，与这个开关无关。
    #: 关掉它**不改候选集合**，只改名次顺序 —— 所以它是一个合格的对照
    #: （§13："开关必须只影响它命名的那一件事"）。
    round_rerank_enabled: bool = True

    #: 最终精排 Query 的构造方式（§9.1）。取值域见 `FINAL_RERANK_QUERY_MODES`。
    final_rerank_query_mode: str = "combined"

    #: 提示词版本（`agent/coq_prompt.py` 的 `COQ_PROMPT_VERSION`）。
    #:
    #: ⚠ **它是一条断言，不是说明**：与代码里的常量不符 ⇒ **构造期响亮失败**。
    #: 同 `router.prompt_version`：删掉了历史版本之后，"改提示词忘了改版本号"的表现是
    #: **两份 run record 声称同一份提示词**，而那种漂移查不出来。
    #: ⚠ **它只属于 `react` 模式**（原版 CoQ 是冻结的那一份）。
    prompt_version: str = DEFAULT_COQ_PROMPT_VERSION

    #: Plan 模式的**拆解**提示词版本（`agent/plan_prompts.py` 的常量）。同一条断言纪律。
    plan_decompose_prompt_version: str = DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION

    #: Plan 模式的**逐轮**提示词版本。**与拆解那条分开**——两条各自演进。
    plan_step_prompt_version: str = DEFAULT_PLAN_STEP_PROMPT_VERSION


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """服务进程形态（§15）。"""

    #: `TIANXIMEM_WORKERS`。**v1 只允许 1**——理由见 `validate()` 与 `assert_single_process()`。
    workers: int = 1


@dataclass(frozen=True, slots=True)
class AppConfig:
    """全量运行配置。**`build_services()` 的输入就是它。**"""

    profile: str = DEFAULT_PROFILE
    storage: StorageConfig = field(default_factory=StorageConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    neighbor: NeighborConfig = field(default_factory=NeighborConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    router: RouterConfig = field(default_factory=RouterConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    #: ⚠ **密钥与端点不在 yaml 里**——它们随机器与密钥而异、不进 git（见模块 docstring）。
    #: 名字与 `.env` 的 `AML_EMB_*` 一一对应，**没有默认值**：缺了在启动时响亮失败。
    embed_base_url: str = ""
    embed_api_key: str = ""

    #: reranker 的端点与密钥（`TIANXIMEM_RERANKER_*`）。
    #:
    #: ⚠ **与上面两个正相反：缺了不报错。** 空值 ⇒ 不构造 reranker、走 `rerank_disabled`
    #: ——那是 D12 要求的降级形态，**不是配置错误**。所以 `validate()` 不 `_require` 它们。
    reranker_base_url: str = ""
    reranker_api_key: str = ""
    #: 请求里声明的模型名（进 run record 的配置指纹；D12 要求提交时不得更换）。
    #:
    #: ⚠ 实测：**网关会忽略这个字段**，响应里 `model` 回的是服务端路径
    #: （新 host 则**校验**它：不认识就 404）。保留它是因为①请求该带上自己声明的模型、
    #: ②它是 run record 里"这次用的哪个 reranker"的唯一来源。**不要拿它做路由或校验。**
    reranker_model: str = ""

    #: §14 指标快照的落点（`TIANXIMEM_METRICS_PATH`）。【路径归 env】
    #:
    #: 空 ⇒ `NullMetricsSink`（不写任何东西）——**缺省不是错误**，与 reranker 同一套口径。
    metrics_path: str = ""

    #: LLM 对话端点的三件套（`AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL`）。
    #:
    #: ⚠ **它们是"端点身份"那一档，所以跟 `base_url` / `key` 一起住 `.env`**
    #: （与 `reranker_*` 完全同型；理由见 `docs/config-reference.md` §9 那条
    #: "embedding 走 yaml、另两个走 env，这不是笔误"）。
    #:
    #: ⚠ **允许为空**（默认就是空的）：绝大多数部署**没有** `router.enabled`
    #: ⇒ 它们不需要这三个值，而"缺了就拒绝启动"会让服务平白多一个启动依赖。
    #: 真正需要它们的是 `validate()` 里那一条——**只在 `router.enabled = true` 时**才要求。
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
