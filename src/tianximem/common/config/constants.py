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
#: ⚠ **主网关**（`memory.021130.xyz`），不是 memory2——两个网关 host 与 key 都不同（D18）。
#: ⚠ 它对服务**不是必需**的（D12）：缺了照常启动、走 `rerank_disabled` 路径
#: ——**这与 embedding 的三个变量正相反**。
ENV_RERANKER_BASE_URL: Final[str] = "TIANXIMEM_RERANKER_BASE_URL"
ENV_RERANKER_API_KEY: Final[str] = "TIANXIMEM_RERANKER_API_KEY"
ENV_RERANKER_MODEL: Final[str] = "TIANXIMEM_RERANKER_MODEL"

#: LLM 对话端点的三个变量（§2.3 的 `models.llm`）。
#:
#: ⚠ **名字不是新起的**——`AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL` 是**仓库里已有的**
#: 那一组（`.env.example` 与 `docs/config-reference.md` §9 早就声明了它们，口子是给
#: 归档 pipeline 与 eval harness 用的，经 `eval/harness/api_config.py` 读）。
#: 本表只是**把同一组变量搬进唯一的读环境变量入口**——服务侧要读它们，
#: 而"全包只有 `common/config/` 这个包能读环境变量"是一条有静态测试钉住的结构性质
#: （`tests/test_config.py::test_only_config_reads_the_environment`）。
#: ⛔ **不许另起一套名字**（例如 `TIANXIMEM_LLM_*`）：那会造出"同一个端点有两个家"，
#: 而照着 `.env.example` 填的人只会填对其中一个（另一处永远是空，且**不报错**）。
#:
#: ⚠ **端点是按"对"配的**（D39）：`AML_BASE_URL_2` / `AML_API_KEY_2` 是**次端点**，
#: 它们服务的是 harness 的**判分并行**（"这一次判分用第几个端点"由 `AML_ENDPOINT_INDEX`
#: 指派）。那是**调度概念，不是这台机器的配置** ⇒ **服务侧只接主端点这一对**，
#: 不把 `_2` 那一套拉进来（拉进来会得到一个"服务用哪个端点取决于环境变量"的静默分叉）。
ENV_LLM_BASE_URL: Final[str] = "AML_BASE_URL"
ENV_LLM_API_KEY: Final[str] = "AML_API_KEY"
ENV_LLM_MODEL: Final[str] = "AML_MODEL"
#: `.env` 文件的位置（默认 cwd 下的 `.env`）。⚠ 它**不是**一个配置项，是"去哪读环境"。
ENV_FILE: Final[str] = "TIANXIMEM_ENV_FILE"

#: §14 指标快照的落点。**空 ⇒ 不写**（`NullMetricsSink`）——缺省不是错误。
#:
#: ⚠ 它**不是**"服务自己要看的东西"，是**给 harness 看的**：响应形状是契约、
#: harness 又禁 import `src/` ⇒ 磁盘是两边唯一的公共面
#: （[`../observability/`](../observability/) 的模块 docstring）。
#: 两臂对照时**每个服务给一份**，否则两个进程的计数会互相盖掉。
ENV_METRICS_PATH: Final[str] = "TIANXIMEM_METRICS_PATH"

#: `Add` / `Search` **原文采集**的落点（`TIANXIMEM_CAPTURE_PATH`）。【路径归 env】
#:
#: ⚠ 与 `TIANXIMEM_METRICS_PATH` 的"空 ⇒ 不写"**不同**：本项**有**一个默认值
#: （`var/capture/requests.jsonl`），开不开由 yaml 的 `capture.enabled` 决定
#: ——一个键只有一个家（开关归 yaml、路径归 env，见 `configs/CLAUDE.md`）。
#: 它存在的理由是 **S6**：平台的 `request_id` 到底长什么样，我们只听过转述。
ENV_CAPTURE_PATH: Final[str] = "TIANXIMEM_CAPTURE_PATH"

DEFAULT_PROFILE: Final[str] = "default"
DEFAULT_CONFIG_DIR: Final[str] = "configs"
DEFAULT_ENV_FILE: Final[str] = ".env"

#: **正确性常量，不是可调项**（D5）：Qdrant 默认 `k=2`、文献是 60，而 Qdrant 的秩 0-based，
#: 只有 61 才等价于文献的 60。**填错不报错**——两种写法的名次都"看起来正常"，
#: 只有分数差一个数量级。⇒ 这里**拒绝**而不是警告。
#:
#: ⚠ 同一个值在 [`../retrieve/fusion.py`](../retrieve/fusion.py) 里也有一份（那边是**策略层**
#: 对直接调用方的防御，不能只靠"配置层已经查过"）。两处相等由
#: `tests/test_config.py::test_rrf_k_is_the_same_constant_in_both_places` 钉住。
RRF_K: Final[int] = 61

#: Neighbor Expansion 的初值（§10）。**值的家在这里**（不是 `rank/neighbor.py`）：
#: `common/` 是**最底层**、被所有层依赖，反过来（`common/` import `rank/`）是分层错误；
#: 由消费方**引用**，也就不存在"代码默认值 vs 配置默认值"两处漂移。
#: 它们与 `default.yaml` 里的取值**必须一致**——两处不一致时，"代码默认值 vs
#: 配置默认值"就又分叉了。
#:
#: ⛔ **`DEFAULT_RADIUS = 0`（2026-10-01，D31）：扩窗 + 段合并未挣到自己的位置。**
#: 两条依据：① 判分池 100% `role: user` ⇒ D29 之后链是 0 ⇒ **扩窗在那里根本不执行**
#: （结构性事实）；② 唯一能测的地方（locomo `official` 346 题）读出 −0.87pt，
#: 落在 ~1pt 噪声底之内。"条件不对所以看不出好"这条路已堵死：那次对照里链几乎不断
#: （跨 Add 只 4.4%）、上下文**翻了一倍**、而预算**只用到 12%**（没被截断）。
#: ⚠ **代码保留**（`rank/neighbor.py` 一行未删）⇒ 开回来只需把这两个常量与
#: `default.yaml` 一起改回去。数字见 `eval/reports/ledger.md` 的「扩窗为什么是 0」。
#:
#: ⚠ **`DEFAULT_EXPANSION_SEED_LIMIT = 1000` 在 `DEFAULT_RADIUS = 0` 下空转**
#: （没有扩窗，种子数就没有含义）。它当年那个依据（N1，+3.5pt）**早于 D28**，
#: **没有被重跑过，不要引用它**。
#:
#: ⚠ 这两个量**不是调参项**：§10 明确"种子数、窗口大小、Top-K、token 预算**是同一道题**，
#: 任何一项调整都要重算其余三项"⇒ 改它们之前要有消融数据（§12.1 R1 对冲 3；config-reference §6）。
DEFAULT_EXPANSION_SEED_LIMIT: Final[int] = 1000
DEFAULT_RADIUS: Final[int] = 0

#: `neighbor.seed_placement` 的取值域（§11.2 的组内顺序消融，2026-09-25）。
SEED_PLACEMENTS: Final[tuple[str, ...]] = ("keep", "front", "echo")

#: ⛔ **本层没有任何 `ingest.*` 键**：`request_id` 是 **opaque string**，
#: 位置 = `(request_id, local_index)`（D28）。**不要再加回任何"解析 `request_id`"的配置项**
#: ——平台实发的是 `r_3115…` 这种不透明 id，那条路在真实流量上 100% 失败。

#: §7.3 的两个检索参数初值：每路进入 RRF 的候选池大小（`N`）与 `[w_bm25, w_dense]` 权重。
#:
#: ⚠ **值的家在这里**（而不是 `retrieve/fusion.py`）：理由与上面两个**同一条**——
#: `retrieve/` 与 `store/` 的 `HybridParams` 都要用到它们。
#:
#: ⚠ 这两个是 **C 类可调项**（不同于 `RRF_K` 那种正确性常量）：**任何调整都必须有
#: ablation 数据支撑**——Qdrant 官方明确警告"无评测集时手调权重不太可能稳定优于默认值"（§7.3）。
DEFAULT_PREFETCH_LIMIT: Final[int] = 200
DEFAULT_WEIGHTS: Final[tuple[float, float]] = (0.5, 0.5)

# Grounded query planner. The planner interprets query wording; evidence
# execution remains deterministic and source-audited.
DEFAULT_GROUNDED_PLANNER_MAX_OUTPUT_TOKENS: Final[int] = 256
DEFAULT_GROUNDED_PLANNER_PROMPT_VERSION: Final[str] = "v3"

#: rerank 调用的超时（秒）。**C 类自设阈值**（`rerank.timeout_seconds`，config-reference §7）。
#:
#: ⚠ 定这个值的两条约束**方向相反**，别只看着一边调：
#:
#: * **太紧 ⇒ 伪降级**。网关是开发环路的单点（V8：answer / judge / embed / rerank 四条都打它），
#:   排队时延会抬高——那时超时会把"排队"报成"reranker 坏了"，而两者在计数器上长得一样。
#: * **太松 ⇒ Search 被拖住**。单请求上限 30 分钟（§2.2），而 rerank 在关键路径上。
#:
#: 实测（开发机、主网关）：**100 篇 ≈ 2.2s、200 篇 ≈ 5.3s** ⇒ 默认 30s 约 10 倍余量。
#: **这是观测值，不是规格**；换模型或换网关后要重新量。
DEFAULT_RERANK_TIMEOUT_S: Final[float] = 30.0

#: rerank 请求里那**一个字段叫什么**（`rerank.envelope`）。
#:
#: 自托管网关的线格式**互斥**，而且**填错哪边都不报错**：
#:
#: * `"query"` —— **自研封装**的形状（单数字符串）。现役主网关
#:   `memory.021130.xyz` 就是这一档；本机自托管网关（容器里
#:   `host.docker.internal:9002` → 宿主机 `127.0.0.1:8082`）也是。给它发 `queries` ⇒ **422**。
#: * `"queries"` —— **vLLM 原生 score 形状**（`queries: [...]`，数组）。
#:   2026-09-28 起 `memory3.021130.xyz` 是这一档（当天网关只改了 nginx.conf，
#:   `/v1/rerank` rewrite 到 vLLM 原生 `/v1/score`）。给它发 `query` ⇒ **400**。
#:   ⚠ `memory3` 自 2026-10-09 起改提供对话模型，**这条已无部署在用**，留着只为认出它。
#:
#: ⚠ **代价是静默的**：D12 把远端问题一律吞成 `RerankUnavailable` ⇒ 发错字段名的表现是
#:   **每次检索都降级、服务不报错、响应照旧合法**，只在 §14 的 `rerank.degraded` 计数上看得见。
#:   ⇒ 它是**端点身份的一部分**，跟着 `TIANXIMEM_RERANKER_BASE_URL` 一起换，**不是可调旋钮**。
DEFAULT_RERANK_ENVELOPE: Final[str] = "query"

#: `rerank.envelope` 的取值域。见 `DEFAULT_RERANK_ENVELOPE` 的两条格式与各自的拒绝码。
RERANK_ENVELOPES: Final[frozenset[str]] = frozenset({"queries", "query"})

#: `capture.max_bytes` 的默认值：**每份文件**写这么多就换下一份（50 MiB）。
#:
#: ⚠ 它是**切分粒度，不是总量上限**：一轮 Full 的请求原文按 ~1.5 GB 估（§2.2），
#: 而单个 1.5 GB 的 JSONL 打不开 ⇒ 分成 ~30 份、每份都能直接看。
#: `capture.enabled` 忘了关的代价就是"记满一块盘"，由人盯着（跑完关掉）。
DEFAULT_CAPTURE_MAX_BYTES: Final[int] = 50 * 1024 * 1024

#: 查询路由器的超时（秒）。**C 类自设阈值**（`router.timeout_seconds`）。
#:
#: ⚠ 与 `DEFAULT_RERANK_TIMEOUT_S` 的**方向不同**，别照抄那一条：
#: * reranker 的超时是"等多久"——它排在检索链**中间**，太紧会把**排队**报成"端点坏了"；
#: * 路由器的超时是"**在检索开始之前**允许拖多久"。它在链的**最前面**，而
#:   一次 Search 的上限是 30 分钟（§2.2）⇒ 它拖住的不是别的组件，是**整条链的起点**。
#:
#: 所以本值**刻意比 reranker 紧**（10s vs 30s）：路由器只回一个词，
#: 而它的降级路径**有定义**（回 `DEFAULT` = 原混合检索）。超时的代价是"少省一次检索"，
#: 不是"检索不到"——**没有理由为它等 30 秒**。
#: ⚠ 这是**观测值不是规格**：换模型或换网关后要重新量（开发机主网关上的 P95 见
#: `eval/reports/` 的路由报告）。
DEFAULT_ROUTER_TIMEOUT_S: Final[float] = 10.0

#: 路由器单次生成的输出上限（token）。**C 类自设阈值**（`router.max_output_tokens`）。
#:
#: ⚠ 它是**成本护栏，不是"提示词已经要求它只回一个词"的冗余**：生成是逐 token 的，
#: 而模型没有义务遵守 `OUTPUT FORMAT`。给 8 已经比最长的那个枚举值（`MULTI_HOP`，2 个
#: token 上下）宽出几倍 ⇒ **超出的部分一定是模型在附加解释**，而那正是我们要截断的东西。
#: ⚠ 别调到 1：`MULTI_HOP` 会被截成 `MULTI`（输出非法 ⇒ 全部降级成 `DEFAULT` 且**不报错**）。
DEFAULT_ROUTER_MAX_OUTPUT_TOKENS: Final[int] = 8

#: 路由器提示词的默认版本（`router.prompt_version`）。
#:
#: ⚠ **它必须与 `retrieve/router.py` 的 `ROUTER_PROMPT_VERSION` 一致**，而这里**不能 import 那边**
#: （`common/` 是最底层，反过来 import `retrieve/` 是分层错误）。
#: ⇒ 两处各写一个字面量、由**构造期**兜住：`QueryRouter` 在版本不认识时**响亮失败**，
#: 所以不一致的表现是"服务起不来"，**不是**"悄悄用了另一版"。
#: ⚠ 本常量存在的理由只有一个：**别让同一个值在 `RouterConfig` 的默认值与
#: `_router()` 的缺省里各写一遍**（2026-10-10 就真的漂过一次：dataclass 是 v3、缺省还是 v1）。
DEFAULT_ROUTER_PROMPT_VERSION: Final[str] = "v5"

#: 多跳执行器的**检索轮数上限**（`agentic.max_rounds`）。**含第一轮**。
#:
#: ⚠ PRD §9 写的是 4 轮，**这里是 3**，两处不一致是**刻意的**：
#: §9 描述的是 ReFind 形状的通用 Agent（4 轮、每轮 top_k=5、三个工具），
#: 而本阶段只实现方案要求的那部分（3 轮、每轮 Top20、**没有工具**）。
#: ⛔ **不要拿 §9 的 4 去"修正"这个值**——那会同时改掉轮数与每轮候选数两个量，
#: 而它们是一道题（§10："种子数、窗口大小、Top-K、token 预算，是同一道题"）。
#: ⚠ §9 还有一句必须记住：**"4 是检索轮数的上限，不是 LLM 调用次数的上限"**
#: ⇒ §14 要看的是**平均轮数与平均调用数**，不是拿这个数去卡。
DEFAULT_COQ_MAX_ROUNDS: Final[int] = 3

#: 每轮交给 CoQ 的候选数（`agentic.per_round_top_k`）——**每轮精排之后取前 N 条**。
#:
#: ⚠ 它与"进精排的候选数"是**两个量**：后者取请求的 `top_k`（AML 固定 100），
#: 顺序必须是"融合池 → 精排 → 取 N"。反过来（先截 N 再精排）只会重排一堆已经被
#: 提前排除掉的候选，**而结果看起来完全正常**（§3.2 专门点了这个坑）。
DEFAULT_COQ_PER_ROUND_TOP_K: Final[int] = 20

#: 提前停止所需的置信度（`agentic.sufficiency_threshold`）。
#:
#: ⚠ 它**只用在"已经判定充分"之后**：先 `is_sufficient = true`，再看置信度够不够。
#: ⛔ **不要反过来**读成"置信度低于它就要继续检索"——原版提示词在证据不足却提不出
#: 更好 Query 时会原样回吐原问题，那也是一种正常的停止（§7.5 明确禁止那种实现）。
#: ⚠ 这个数来自提示词**自带的**标尺（0.90–1.00 很确定 / 0.60–0.89 中等），
#: 不是本项目的标定结果 ⇒ 它没被本仓的评测标过，别当成已校准的阈值。
DEFAULT_COQ_SUFFICIENCY_THRESHOLD: Final[float] = 0.8

#: 多跳执行器单次 LLM 调用的超时（秒，`agentic.timeout_seconds`）。
#:
#: ⚠ **与路由器的 10s 不是一回事**：路由器只回一个词（`max_output_tokens = 8`），
#: 而这里要回一个 JSON 对象（含一句改写 Query）。三轮各等一次 ⇒ 它按
#: **3 × 单次超时**计算，仍远在 30 分钟的单请求上限之内（§2.2）。
DEFAULT_COQ_TIMEOUT_S: Final[float] = 30.0

#: 多跳执行器单次生成的输出上限（token，`agentic.max_output_tokens`）。
#:
#: ⚠ 它**不是**路由器的 8：那里只回一个枚举值，这里要回四个字段的 JSON
#: （其中 `new_query` 是一句自然语言）。给 256 是**成本护栏**——
#: 够写一句查询加上 JSON 骨架，而超出的部分一定是模型在附加解释。
DEFAULT_COQ_MAX_OUTPUT_TOKENS: Final[int] = 256

#: 每轮喂给模型的文档总量上限（token，`agentic.max_prompt_tokens`）。
#:
#: ⚠ **它和 `budget.max_tokens`（117,760）不是同一个东西**，⛔ 不要复用那一个：
#: 那个是 **AML 答案窗口**的预算，而这里是**发给 9B 模型的一次 prompt**。
#: 拿 117k 当上限等于没有上限——每轮 20 篇文档会被原样全塞进去，
#: 而"超了"这件事**不会报错**，只会让检索变慢、变贵、变不准。
#: ⚠ 预算的分配顺序是有约束的：**先给此前已选中的证据留够**，剩下的才按名次补本轮
#: 候选（§4.4）——历史证据不能因为本轮名次低就被悄悄挤掉。
DEFAULT_COQ_MAX_PROMPT_TOKENS: Final[int] = 6000

#: 多跳提示词的默认版本（`agentic.prompt_version`）。
#:
#: ⚠ 与 `DEFAULT_ROUTER_PROMPT_VERSION` **同一条纪律**：这里**不能 import**
#: [`../agent/coq_prompt.py`](../agent/coq_prompt.py)（`common/` 是最底层）。
#: 两处各写一个字面量、由**构造期**兜住：`CoqAgent` 在版本不认识时**响亮失败**，
#: 所以不一致的表现是"服务起不来"，**不是**"悄悄用了另一版"。
#: 名字沿用方案 §10.4 的 `coq-original`（含义：原版，没有本项目自己的改动）。
DEFAULT_COQ_PROMPT_VERSION: Final[str] = "coq-original"

#: 多跳执行器的**两种模式**（`agentic.mode`）。
#:
#: | 值 | 执行器 | 怎么推进 |
#: | --- | --- | --- |
#: | `"plan"`（**`default.yaml` 的默认**） | `PlanAgent`（先拆一份固定计划） | 每轮**只**回答"刚检索完的那一步"，**不重规划** |
#: | `"react"`（**内置缺省**） | `CoqAgent`（原版 CoQ） | 每轮重新决定"充分了吗" |
#:
#: ⚠ **内置缺省必须是 `react`**：`AgenticConfig()` 直接构造（不经过任何 yaml）的场景——
#: 测试、`tools/` 里的臂快照——要按"不填这个键"处理，那**不是**提交口径。
#: ⇒ 两者分工与 `retrieval.grounded_evidence` 相同：**内置缺省保守、`default.yaml` 是提交口径**。
#: ⛔ **不要为了让"默认是 plan"看起来一致就把这个常量也翻掉**：那会一次性改掉所有
#: 不带 `mode` 的调用方，而它们是**刻意**要保住旧行为的。
#: 两者的提示词、解析、停止判据**各自独立**（⛔ 不共享，见 `agent/` 的包说明）。
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
#: * `"combined"` —— 原问题 ＋ **实际执行过的**改写 Query（§9.1 的联合精排，**主方案**）
#: * `"original"` —— 只用原问题（§12.6 列出的后续消融项，本阶段**不做那个实验**，
#:   但把开关留下：它单独改变的量只有一个，是个合格的对照）
#:
#: ⚠ 两种都**不重复原问题**：`combined` 拼的是"执行过的那些"，而第一轮执行的就是原问题，
#: 所以原问题**天然只出现一次**。
FINAL_RERANK_QUERY_MODES: Final[tuple[str, ...]] = ("combined", "original")
