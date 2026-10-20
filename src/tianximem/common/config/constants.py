"""配置常量：环境变量名 / 正确性常量 / 自设阈值 / 取值域——**值的家**。

⚠ 这一层只有**值**，没有逻辑；分工与加载顺序的声明在包 docstring
（`common/config/__init__.py`），本文件不复制。
"""

from __future__ import annotations

from typing import Final

# ── 环境变量名（**唯一声明处**）──────────────────────────────────────────
# 新增一个环境变量 = 在本表加一行 + 在 `.env.example` 加一行说明。**别在别处硬写字面量。**

ENV_PROFILE: Final[str] = "TIANXIMEM_PROFILE"
ENV_CONFIG_DIR: Final[str] = "TIANXIMEM_CONFIG_DIR"
ENV_SQLITE_PATH: Final[str] = "TIANXIMEM_SQLITE_PATH"
ENV_QDRANT_URL: Final[str] = "TIANXIMEM_QDRANT_URL"
ENV_EMBED_BASE_URL: Final[str] = "AML_EMB_BASE_URL"
ENV_EMBED_API_KEY: Final[str] = "AML_EMB_API_KEY"
ENV_EMBED_CACHE_DIR: Final[str] = "TIANXIMEM_EMBED_CACHE_DIR"
ENV_WORKERS: Final[str] = "TIANXIMEM_WORKERS"

#: reranker 的三个变量。**名字沿用 `.env.example` 里那三个**（`.env.example` 是它们的家，
#: 本表只是代码侧的引用点，两处由 `tests/test_config.py` 的静态断言钉住）。
#:
#: ⚠ **主网关**（`memory.021130.xyz`），不是 memory2——两个网关 host 与 key 都不同。
#: ⚠ 它对服务**不是必需**的：缺了照常启动、走 `rerank_disabled` 路径
#: ——**这与 embedding 的三个变量正相反**。
ENV_RERANKER_BASE_URL: Final[str] = "TIANXIMEM_RERANKER_BASE_URL"
ENV_RERANKER_API_KEY: Final[str] = "TIANXIMEM_RERANKER_API_KEY"
ENV_RERANKER_MODEL: Final[str] = "TIANXIMEM_RERANKER_MODEL"

#: LLM 对话端点的三个变量。
#:
#: 这组变量与服务配置共用，并由本包统一读取。不要另起一套前缀，否则同一个端点会出现
#: 两个配置来源，填错其中一个时可能静默失效。
#:
#: ⚠ **端点是按"对"配的**：`AML_BASE_URL_2` / `AML_API_KEY_2` 是**次端点**，
#: 它们服务的是 harness 的**判分并行**（"这一次判分用第几个端点"由 `AML_ENDPOINT_INDEX`
#: 指派）。那是**调度概念，不是这台机器的配置** ⇒ **服务侧只接主端点这一对**，
#: 不把 `_2` 那一套拉进来（拉进来会得到一个"服务用哪个端点取决于环境变量"的静默分叉）。
ENV_LLM_BASE_URL: Final[str] = "AML_BASE_URL"
ENV_LLM_API_KEY: Final[str] = "AML_API_KEY"
ENV_LLM_MODEL: Final[str] = "AML_MODEL"
#: `.env` 文件的位置（默认 cwd 下的 `.env`）。⚠ 它**不是**一个配置项，是"去哪读环境"。
ENV_FILE: Final[str] = "TIANXIMEM_ENV_FILE"

#: 指标快照的落点。**空 ⇒ 不写**（`NullMetricsSink`）——缺省不是错误。
#:
#: 指标供 HTTP-only harness 读取；响应形状固定且 harness 不导入服务内部模块，
#: 因此磁盘快照是两个进程之间的公共接口。
#: 两臂对照时**每个服务给一份**，否则两个进程的计数会互相盖掉。
ENV_METRICS_PATH: Final[str] = "TIANXIMEM_METRICS_PATH"

#: `Add` / `Search` **原文采集**的落点（`TIANXIMEM_CAPTURE_PATH`）。【路径归 env】
#:
#: ⚠ 与 `TIANXIMEM_METRICS_PATH` 的"空 ⇒ 不写"**不同**：本项**有**一个默认值
#: （`var/capture/requests.jsonl`），开不开由 yaml 的 `capture.enabled` 决定
#: ——一个键只有一个家（开关归 yaml、路径归 env，见 `configs/CLAUDE.md`）。
#: 它存在的理由是：平台实际发送的 `request_id` 和其他请求字段需要从真实流量确认。
ENV_CAPTURE_PATH: Final[str] = "TIANXIMEM_CAPTURE_PATH"

DEFAULT_PROFILE: Final[str] = "default"
DEFAULT_CONFIG_DIR: Final[str] = "configs"
DEFAULT_ENV_FILE: Final[str] = ".env"

#: **正确性常量，不是可调项**：Qdrant 默认 `k=2`、文献是 60，而 Qdrant 的秩 0-based，
#: 只有 61 才等价于文献的 60。**填错不报错**——两种写法的名次都"看起来正常"，
#: 只有分数差一个数量级。⇒ 这里**拒绝**而不是警告。
#:
#: ⚠ 同一个值在 [`../retrieve/fusion.py`](../retrieve/fusion.py) 里也有一份（那边是**策略层**
#: 对直接调用方的防御，不能只靠"配置层已经查过"）。两处相等由
#: `tests/test_config.py::test_rrf_k_is_the_same_constant_in_both_places` 钉住。
RRF_K: Final[int] = 61

#: Neighbor Expansion 的默认值。值的唯一来源在这里，消费方直接引用，避免默认值漂移。
#: 它们与 `default.yaml` 中的取值必须一致。
#:
#: `DEFAULT_RADIUS = 0` 关闭扩窗。评分池全部是 user 消息，无法形成邻接链；现有评测也未
#: 显示扩窗带来超过噪声的收益，而上下文成本明显增加。代码仍保留扩窗能力，启用时需要
#: 同时修改这里和 `default.yaml`，并重新运行消融。
#:
#: `DEFAULT_EXPANSION_SEED_LIMIT` 在半径为 0 时不参与执行，也不应作为调优依据。
#: 种子数、窗口大小、Top-K 和 token 预算需要一起评估；修改前必须提供消融数据。
DEFAULT_EXPANSION_SEED_LIMIT: Final[int] = 1000
DEFAULT_RADIUS: Final[int] = 0

#: `neighbor.seed_placement` 的取值域。
SEED_PLACEMENTS: Final[tuple[str, ...]] = ("keep", "front", "echo")

#: 本层没有 `ingest.*` 键：`request_id` 是不透明字符串，写入位置由
#: `(request_id, local_index)` 决定。平台实际发送的是 `r_3115…` 一类不透明 ID，
#: 因此不存在解析 `request_id` 的配置。

#: 混合检索默认值：每路进入 RRF 的候选池大小（`N`）与 `[w_bm25, w_dense]` 权重。
#:
#: 值集中在这里，供 `retrieve/` 与 `store/` 的 `HybridParams` 引用。
#: 这两项是可调参数，任何修改都需要评测数据支撑；没有评测集时手动调权重通常不会稳定
#: 优于默认值。
DEFAULT_PREFETCH_LIMIT: Final[int] = 200
DEFAULT_WEIGHTS: Final[tuple[float, float]] = (0.5, 0.5)

# Grounded query planner. The planner interprets query wording; evidence
# execution remains deterministic and source-audited.
DEFAULT_GROUNDED_PLANNER_MAX_OUTPUT_TOKENS: Final[int] = 256
DEFAULT_GROUNDED_PLANNER_PROMPT_VERSION: Final[str] = "v3"

#: rerank 调用的超时（秒），对应 `rerank.timeout_seconds`。
#:
#: 过紧会把网关排队误报为 reranker 故障；过松会让关键路径上的 Search 等待过久。
#:
#: （开发机、主网关）：**100 篇 ≈ 2.2s、200 篇 ≈ 5.3s** ⇒ 默认 30s 约 10 倍余量。
#: **这是观测值，不是规格**；换模型或换网关后要重新量。
DEFAULT_RERANK_TIMEOUT_S: Final[float] = 30.0

#: rerank 请求里那**一个字段叫什么**（`rerank.envelope`）。
#:
#: 自托管网关的线格式**互斥**，而且**填错哪边都不报错**：
#:
#: * `"query"` —— 单个查询字符串，对应自研封装的 rerank 接口。
#: * `"queries"` —— 查询字符串数组，对应 vLLM 原生 score 接口。
#:
#: 字段名错误不会改变响应形状，只会使每次检索静默降级，并体现在
#: `rerank.degraded` 指标中。因此它是端点身份的一部分，需要随
#: `TIANXIMEM_RERANKER_BASE_URL` 一起更换，不是可调旋钮。
DEFAULT_RERANK_ENVELOPE: Final[str] = "query"

#: `rerank.envelope` 的取值域。见 `DEFAULT_RERANK_ENVELOPE` 的两条格式与各自的拒绝码。
RERANK_ENVELOPES: Final[frozenset[str]] = frozenset({"queries", "query"})

#: `capture.max_bytes` 的默认值：**每份文件**写这么多就换下一份（50 MiB）。
#:
#: 它是文件切分粒度，不是总量上限。分卷避免生成无法直接打开的单个大 JSONL；
#: `capture.enabled` 长时间开启仍可能占满磁盘，因此运行后应关闭。
DEFAULT_CAPTURE_MAX_BYTES: Final[int] = 50 * 1024 * 1024

#: 查询路由器的超时（秒）。**C 类自设阈值**（`router.timeout_seconds`）。
#:
#: 路由器位于检索链起点，只输出一个分类结果。超时后回退到 `DEFAULT` 混合检索，
#: 因此该值比 reranker 超时更紧。这里的默认值来自观测，换模型或网关后应重新测量。
DEFAULT_ROUTER_TIMEOUT_S: Final[float] = 10.0

#: 路由器单次生成的输出上限（token），对应 `router.max_output_tokens`。
#:
#: 该上限是成本护栏：模型仍可能附加解释。不要设为 1，否则 `MULTI_HOP` 会被截断，
#: 所有输出都会非法并静默降级为 `DEFAULT`。
DEFAULT_ROUTER_MAX_OUTPUT_TOKENS: Final[int] = 8

#: 路由器提示词的默认版本（`router.prompt_version`）。
#:
#: 该值必须与 `retrieve/router.py` 的 `ROUTER_PROMPT_VERSION` 一致。`common/` 不能导入
#: `retrieve/`，因此两处各维护字面量，并由 `QueryRouter` 在构造期校验；
#: 不一致时服务直接启动失败。
DEFAULT_ROUTER_PROMPT_VERSION: Final[str] = "v5"

#: 多跳执行器的**检索轮数上限**（`agentic.max_rounds`），包含第一轮。
#: 内置默认值供直接构造配置对象使用；产品 profile 在 `configs/default.yaml` 中设置
#: 实际值。轮数、每轮候选数和上下文预算必须一起调整。
DEFAULT_COQ_MAX_ROUNDS: Final[int] = 3

#: 每轮交给 CoQ 的候选数（`agentic.per_round_top_k`），在轮内精排之后取前 N 条。
#:
#: 它与进入精排的候选数是两个量：后者取请求的 `top_k`，顺序必须是
#: "融合池 → 精排 → 取 N"。先截 N 再精排会提前排除候选而不会报错。
DEFAULT_COQ_PER_ROUND_TOP_K: Final[int] = 20

#: 提前停止所需的置信度（`agentic.sufficiency_threshold`）。
#:
#: 它只在模型已经判定证据充分后使用：先看 `is_sufficient = true`，再看置信度。
#: 不能反过来把低置信度解释为必须继续检索；证据不足且没有更好查询时停止是正常路径。
#: 数值来自提示词自带标尺，尚未由本仓库评测校准。
DEFAULT_COQ_SUFFICIENCY_THRESHOLD: Final[float] = 0.8

#: 多跳执行器单次 LLM 调用的超时（秒，`agentic.timeout_seconds`）。
#:
#: 路由器只输出一个分类词，而这里要输出包含改写 Query 的 JSON 对象。
#: 总耗时约为轮数乘以此超时值。
DEFAULT_COQ_TIMEOUT_S: Final[float] = 30.0

#: 多跳执行器单次生成的输出上限（token，`agentic.max_output_tokens`）。
#:
#: ⚠ 它**不是**路由器的 8：那里只回一个枚举值，这里要回四个字段的 JSON
#: （其中 `new_query` 是一句自然语言）。给 256 是**成本护栏**——
#: 够写一句查询加上 JSON 骨架，而超出的部分一定是模型在附加解释。
DEFAULT_COQ_MAX_OUTPUT_TOKENS: Final[int] = 256

#: 每轮喂给模型的文档总量上限（token，`agentic.max_prompt_tokens`）。
#:
#: 它与 `budget.max_tokens` 不是同一个量：后者是返回给 AML 的答案窗口预算，
#: 这里是多跳执行器单次提示词的预算。预算先为已选证据保留空间，再按名次加入本轮候选，
#: 避免历史证据因当前轮名次较低而被静默挤掉。
DEFAULT_COQ_MAX_PROMPT_TOKENS: Final[int] = 6000

#: 多跳提示词的默认版本（`agentic.prompt_version`）。
#:
#: 该值必须与 [`../agent/coq_prompt.py`](../agent/coq_prompt.py) 中的版本常量一致。
#: `common/` 不能导入 `agent/`，因此由 `CoqAgent` 在构造期校验，不一致时启动失败。
DEFAULT_COQ_PROMPT_VERSION: Final[str] = "coq-original"

#: 多跳执行器的**两种模式**（`agentic.mode`）。
#:
#: | 值 | 执行器 | 怎么推进 |
#: | --- | --- | --- |
#: | `"plan"`（**`default.yaml` 的默认**） | `PlanAgent`（先拆一份固定计划） | 每轮**只**回答"刚检索完的那一步"，**不重规划** |
#: | `"react"`（**内置缺省**） | `CoqAgent` | 每轮重新决定“充分了吗” |
#:
#: 内置缺省是保守的 `react`；`configs/default.yaml` 是产品默认，使用 `plan`。
#: 直接构造 `AgenticConfig()` 的测试和工具因此不会隐式启用产品模式。
#: 两种模式各自维护提示词、解析和停止判据，见 `agent/` 的模块说明。
DEFAULT_AGENTIC_MODE: Final[str] = "react"

#: `agentic.mode` 的取值域。
AGENTIC_MODES: Final[tuple[str, ...]] = ("react", "plan")

#: Plan 模式单次生成的输出上限（token，`agentic.plan_max_output_tokens`）。
#:
#: ⚠ **它不是 `DEFAULT_COQ_MAX_OUTPUT_TOKENS`（256）**，两者不能合并：
#: CoQ 回的是**定长四字段**，而 Plan 的两次调用都回**带自由文本的 JSON**——
#: 拆解回 1～4 步（每步一句子问题），逐轮回一段答案 ＋ 若干条逐字引句 ＋ 一句下一查询。
#: 256 会把四步计划**从中间截断**，而截断的表现是"JSON 不完整 ⇒ 计划作废 ⇒
#: 这一题退回混合检索"，**看起来像模型不行**。给 768 是成本护栏：
#: 够写满这两份结构，而超出的部分一定是模型在附加解释。
DEFAULT_PLAN_MAX_OUTPUT_TOKENS: Final[int] = 768

#: Plan 拆解提示词的默认版本（`agentic.plan_decompose_prompt_version`）。
#:
#: ⚠ 与 `DEFAULT_COQ_PROMPT_VERSION` **同一条纪律**（`common/` 不能 import `agent/`，
#: 两处各写一个字面量、由构造期兜住）。
DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION: Final[str] = "plan-decompose-v2"

#: Plan 逐轮提示词的默认版本（`agentic.plan_step_prompt_version`）。
#: **与拆解那条分开**：两条提示词各自演进，一个版本号管不住两条。
DEFAULT_PLAN_STEP_PROMPT_VERSION: Final[str] = "plan-step-v2"

#: `agentic.final_rerank_query_mode` 的取值域。
#:
#: * `"combined"` —— 使用实际执行过的查询进行联合精排
#: * `"original"` —— 只使用原问题进行精排
#:
#: ⚠ 两种都**不重复原问题**：`combined` 拼的是"执行过的那些"，而第一轮执行的就是原问题，
#: 所以原问题**天然只出现一次**。
FINAL_RERANK_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")
