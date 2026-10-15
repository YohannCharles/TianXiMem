"""配置的唯一入口（§15 / §12.1 R1 对冲 3）。

## 本模块是全包**唯一**读 `os.environ` 的地方

`service/` / `retrieve/` / `rank/` / `store/` / `embed/` 都**不得**自行读环境变量——
它们只接收这里构造出来的配置对象。理由不是洁癖：环境变量的读取点一多，
"这次跑的是哪套值"就无法从一个地方回答，而 §13 的每一条对照实验都依赖那个回答。

## 两层来源，**每个键只有一个家**

env（`.env`）拥有密钥 / 端点 / **路径** / 进程形态；`configs/<profile>.yaml` 拥有
阈值 / 权重 / 模型名 / 集合名。**不允许一个键两边都能设**——两处都能设的值，最终会变成
"跑出来的结果和 yaml 里写的不一样，而没人知道为什么"；而**阈值尤其不能藏进环境变量**：
那会让 Step 5 的模型切换变成"改 shell 变量"、**改了什么无法 diff**，而归因恰恰是那一步
唯一的目的（§12.1 R1 对冲 4）。分工的一处声明在
[`../../../configs/CLAUDE.md`](../../../configs/CLAUDE.md)。

## 加载顺序

```text
内置默认值（本文件的 dataclass 默认值）
  ← configs/default.yaml                  （缺失 ⇒ 报错，不静默退回内置默认值）
  ← configs/<profile>.yaml                （profile 来自 TIANXIMEM_PROFILE，默认 default）
  ← .env                                  （**最低优先级的"环境"**，见下）
  ← 真实的环境变量                         （**压过 .env**：export 过的值说了算）
  → 校验                                  （不合法的值在这里响亮失败）
```

**`.env` 由本模块读取**：它是"这台机器的环境"的本地副本，不是一条独立来源 ⇒ 排在真实
环境变量**之下**（`uv run` 与 `Makefile` 都不加载 `.env`，不读它的话 `make serve` 启动时缺密钥）。

> ⚠ **只有 `env=None`（生产路径）才读 `.env`。** 测试传一个 `env` dict 时**绝不碰磁盘**——
> 否则"测试不依赖真实环境变量"这条纪律就破了。换 `.env` 的位置用 `TIANXIMEM_ENV_FILE`。

**选了一个不存在的 profile 同样报错**（不静默忽略）——静默回退会让"我明明选了 submit"
变成一个查不出来的问题。

配置项与类别（A 外部契约常量 / B 正确性常量 / C 自设阈值）的完整清单在
[`../../../docs/config-reference.md`](../../../docs/config-reference.md)——**本模块不另列一份**。
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

import yaml

from tianximem.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

__all__ = [
    "ENV_CONFIG_DIR",
    "ENV_EMBED_API_KEY",
    "ENV_EMBED_BASE_URL",
    "ENV_EMBED_CACHE_DIR",
    "ENV_FILE",
    "ENV_LLM_API_KEY",
    "ENV_LLM_BASE_URL",
    "ENV_LLM_MODEL",
    "ENV_METRICS_PATH",
    "ENV_PROFILE",
    "ENV_QDRANT_URL",
    "ENV_RERANKER_API_KEY",
    "ENV_RERANKER_BASE_URL",
    "ENV_RERANKER_MODEL",
    "ENV_SQLITE_PATH",
    "ENV_WORKERS",
    "DEFAULT_EXPANSION_SEED_LIMIT",
    "DEFAULT_PREFETCH_LIMIT",
    "DEFAULT_RADIUS",
    "DEFAULT_RERANK_TIMEOUT_S",
    "DEFAULT_ROUTER_MAX_OUTPUT_TOKENS",
    "DEFAULT_ROUTER_PROMPT_VERSION",
    "DEFAULT_ROUTER_TIMEOUT_S",
    "DEFAULT_WEIGHTS",
    "RRF_K",
    "AgenticConfig",
    "AppConfig",
    "ConfigError",
    "RouterConfig",
    "assert_single_process",
    "load_config",
]


class ConfigError(ValueError):
    """配置不合法。**必须在启动阶段响亮失败**，不要等到第一个请求才炸。"""


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
#: 而"全包只有 `common/config.py` 能读环境变量"是一条有静态测试钉住的结构性质
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
    #: ⚠ 开发期 2026-10-13 起是 **4B**（`memory2.021130.xyz` 上只有它）——与 `default.yaml`
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


# ── yaml 读取：**不认识的键一律报错** ────────────────────────────────────

#: yaml 顶层允许的段。**env 拥有的键不出现在这里**（见模块 docstring 的分工表）：
#: 路径 / 端点 / 密钥 / worker 数都只从环境变量来。
_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {
        "storage",
        "models",
        "retrieval",
        "neighbor",
        "budget",
        "packaging",
        "rerank",
        "router",
        "agentic",
        "capture",
    }
)


def _group(raw: object, /, *, where: str, allowed: set[str]) -> Mapping[str, Any]:
    """取出一段 yaml 并校验键名。

    ⚠ **拼错的键必须响亮失败**：`prefetch_limt` 被静默忽略，跑出来的就是"默认值"，
    而没有任何人会发现——这正是本项目反复要避免的那一类失败。
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"配置段 `{where}` 必须是一个映射，收到 {type(raw).__name__}")
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ConfigError(
            f"配置段 `{where}` 里有不认识的键：{unknown}；允许的是 {sorted(allowed)}（拼错了吗？）"
        )
    return raw


def _int(value: object, /, *, where: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"`{where}` 必须是整数，收到 {value!r}")
    return value


def _str(value: object, /, *, where: str, default: str) -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise ConfigError(f"`{where}` 必须是字符串，收到 {value!r}")
    return value


def _float(value: object, /, *, where: str, default: float) -> float:
    """浮点配置项。

    ⚠ **`bool` 要先挡掉**：Python 里 `True` 是 `int` 的实例，`isinstance(True, float)` 为假、
    但 `isinstance(True, (int, float))` 为真——`timeout_seconds: true` 若不挡会变成 `1.0`，
    于是超时被悄悄改成 1 秒，而**配置看起来完全正常**。
    """
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"`{where}` 必须是数字，收到 {value!r}")
    return float(value)


def _weights(value: object, /, *, where: str) -> tuple[float, float]:
    if value is None:
        return DEFAULT_WEIGHTS
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ConfigError(
            f"`{where}` 必须是两个数（`[bm25, dense]`，顺序见 §7.3 的 prefetch）：{value!r}"
        )
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"`{where}` 必须是两个数：{value!r}") from exc


def _read_yaml(path: Path, /, *, required: bool) -> Mapping[str, Any]:
    if not path.exists():
        if required:
            raise ConfigError(
                f"配置文件不存在：{path}。"
                "阈值与模型名的家在这里（见 configs/CLAUDE.md）；"
                f"换目录用 `{ENV_CONFIG_DIR}`。"
            )
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"解析 {path} 失败：{exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{path} 的顶层必须是一个映射，收到 {type(raw).__name__}")
    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigError(
            f"{path} 顶层有不认识的段：{unknown}；允许的是 {sorted(_TOP_LEVEL_KEYS)}。"
            f"（路径 / 端点 / 密钥 / worker 数只在 .env 里，见 configs/CLAUDE.md）"
        )
    return raw


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """把 overlay 深合并进 base（只对映射递归，其余整体替换）。

    ⚠ 用**递归合并**而不是整体替换：profile 只该写它真正改变的那几个键，
    否则 `local.yaml` 会退化成"另一份完整配置"，而"哪些量随模型变"这个信息就丢了
    （`configs/CLAUDE.md` 的原文理由）。
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


# ── `.env` 的读取 ──────────────────────────────────────────────────────


def _parse_env_line(raw: str) -> tuple[str, str] | None:
    """把一行 `.env` 解析成 `(key, value)`；空行、注释行与非 `KEY=value` 行返回 `None`。

    支持我们**自己的** `.env.example` 用到的全部写法，不多支持：
    `KEY=value` · `export KEY=value` · 行内 `# 注释` · 引号包起来的值。

    ⚠ **`#` 前必须有空白才算注释**——否则 `KEY=a#b` 会被截成 `a`（这是 dotenv 的通例）。
    ⚠ **坏的行使值缺失 ⇒ 由 `validate()` 响亮失败**（"AML_EMB_API_KEY 没填"），
    所以这里**跳过**而不是报错：`.env` 是给人写的，一个多出来的空行不该让服务起不来。
    """
    line = raw.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export ") :].lstrip()
    key, sep, value = line.partition("=")
    if not sep or not key.strip():
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return key.strip(), value[1:-1]  # 引号包起来 ⇒ 原样，不剥注释
    for index, char in enumerate(value):
        if char == "#" and (index == 0 or value[index - 1].isspace()):
            value = value[:index].strip()
            break
    return key.strip(), value


def _read_env_file(path: Path) -> dict[str, str]:
    """读一个 `.env`。**文件不存在 ⇒ 空 dict**（生产上可能直接用真实环境变量）。"""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        parsed = _parse_env_line(raw)
        if parsed is not None:
            values[parsed[0]] = parsed[1]
    return values


# ── 各段的构造（显式写出来，错误信息才能指名道姓）──────────────────────


def _storage(raw: object) -> StorageConfig:
    g = _group(raw, where="storage", allowed={"sqlite", "qdrant"})
    s = _group(g.get("sqlite"), where="storage.sqlite", allowed={"busy_timeout_ms"})
    q = _group(g.get("qdrant"), where="storage.qdrant", allowed={"collection"})
    return StorageConfig(
        sqlite=SqliteConfig(
            busy_timeout_ms=_int(
                s.get("busy_timeout_ms"), where="storage.sqlite.busy_timeout_ms", default=5000
            )
        ),
        qdrant=QdrantConfig(
            collection=_str(
                q.get("collection"), where="storage.qdrant.collection", default="memories"
            )
        ),
    )


def _models(raw: object) -> ModelsConfig:
    g = _group(raw, where="models", allowed={"embedder", "embed_dim"})
    return ModelsConfig(
        # ⚠ 这里必须**再写一遍字面值**：`ModelsConfig` 是 `slots=True` 的 dataclass，
        #   `ModelsConfig.embedder` 拿到的是 slot 描述符、**不是默认值**（踩过）。
        #   两处的同值由 `tests/test_config.py::test_model_default_is_the_same_in_both_places`
        #   钉住。
        embedder=_str(
            g.get("embedder"), where="models.embedder", default="Qwen/Qwen3-Embedding-4B"
        ),
        embed_dim=_int(g.get("embed_dim"), where="models.embed_dim", default=1024),
    )


def _retrieval(raw: object) -> RetrievalConfig:
    g = _group(
        raw,
        where="retrieval",
        allowed={
            "prefetch_limit",
            "query_instruction",
            "grounded_evidence",
            "evidence_limit",
            "evidence_hop_limit",
            "fact_backfill_limit",
            "rrf",
        },
    )
    r = _group(g.get("rrf"), where="retrieval.rrf", allowed={"k", "weights"})
    evidence = g.get("grounded_evidence", False)
    if not isinstance(evidence, bool):
        raise ConfigError("`retrieval.grounded_evidence` 必须是布尔值")
    return RetrievalConfig(
        prefetch_limit=_int(
            g.get("prefetch_limit"),
            where="retrieval.prefetch_limit",
            default=DEFAULT_PREFETCH_LIMIT,
        ),
        query_instruction=_str(
            g.get("query_instruction"), where="retrieval.query_instruction", default=""
        ),
        grounded_evidence=evidence,
        evidence_limit=_int(g.get("evidence_limit"), where="retrieval.evidence_limit", default=12),
        evidence_hop_limit=_int(
            g.get("evidence_hop_limit"), where="retrieval.evidence_hop_limit", default=4
        ),
        fact_backfill_limit=_int(
            g.get("fact_backfill_limit"), where="retrieval.fact_backfill_limit", default=1024
        ),
        rrf=RrfConfig(
            k=_int(r.get("k"), where="retrieval.rrf.k", default=RRF_K),
            weights=_weights(r.get("weights"), where="retrieval.rrf.weights"),
        ),
    )


def _neighbor(raw: object) -> NeighborConfig:
    g = _group(raw, where="neighbor", allowed={"expansion_seed_limit", "radius", "seed_placement"})
    placement = _str(g.get("seed_placement"), where="neighbor.seed_placement", default="keep")
    if placement not in SEED_PLACEMENTS:
        raise ConfigError(
            f"`neighbor.seed_placement` 只能是 {SEED_PLACEMENTS} 之一，收到 {placement!r}"
        )
    return NeighborConfig(
        expansion_seed_limit=_int(
            g.get("expansion_seed_limit"),
            where="neighbor.expansion_seed_limit",
            default=DEFAULT_EXPANSION_SEED_LIMIT,
        ),
        radius=_int(g.get("radius"), where="neighbor.radius", default=DEFAULT_RADIUS),
        seed_placement=placement,
    )


def _budget(raw: object) -> BudgetConfig:
    g = _group(raw, where="budget", allowed={"max_tokens", "tokenizer"})
    return BudgetConfig(
        max_tokens=_int(g.get("max_tokens"), where="budget.max_tokens", default=MAX_INPUT_TOKENS),
        tokenizer=_str(g.get("tokenizer"), where="budget.tokenizer", default=DEFAULT_TOKENIZER),
    )


def _packaging(raw: object) -> PackagingConfig:
    g = _group(
        raw,
        where="packaging",
        allowed={"inject_abs_time", "annotate_relatives", "annotate_unit_totals"},
    )
    inject = g.get("inject_abs_time")
    if inject is not None and not isinstance(inject, bool):
        raise ConfigError(
            f"`packaging.inject_abs_time` 必须是布尔值，收到 {inject!r}——"
            "它是「带 / 不带时间戳前缀」那个对照臂的开关，不是日期格式"
        )
    annotate = g.get("annotate_relatives")
    if annotate is not None and not isinstance(annotate, bool):
        raise ConfigError(
            f"`packaging.annotate_relatives` 必须是布尔值，收到 {annotate!r}——"
            "它是「正文里的相对时间要不要就地注解成绝对日期」那个开关"
        )
    unit_totals = g.get("annotate_unit_totals", False)
    if not isinstance(unit_totals, bool):
        raise ConfigError("`packaging.annotate_unit_totals` 必须是布尔值")
    return PackagingConfig(
        inject_abs_time=True if inject is None else inject,
        annotate_relatives=False if annotate is None else annotate,
        annotate_unit_totals=unit_totals,
    )


def _rerank(raw: object) -> RerankConfig:
    g = _group(raw, where="rerank", allowed={"enabled", "timeout_seconds", "envelope"})
    enabled = g.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ConfigError(f"`rerank.enabled` 必须是布尔值，收到 {enabled!r}")
    envelope = _str(g.get("envelope"), where="rerank.envelope", default=DEFAULT_RERANK_ENVELOPE)
    # ⚠ **这一条必须是硬错误**（哪怕两个值都能让服务起来）：它是"对面那个网关说什么格式"
    #   的声明，而写错的表现是**每次检索都静默降级**——那种失败只在 §14 的计数里看得见，
    #   排查成本远高于在这里响亮地拒绝。同 `rrf.k` / `workers` 的口径（配置化 ≠ 可调）。
    if envelope not in RERANK_ENVELOPES:
        raise ConfigError(
            f"`rerank.envelope` 只能是 {sorted(RERANK_ENVELOPES)}，收到 {envelope!r}\n"
            "  `query`   = 自研封装（现役主网关 `memory.021130.xyz`；本机网关同上）；\n"
            "  `queries` = vLLM 原生 score 形状（`memory3.021130.xyz` 在 2026-09-28 之后\n"
            "              那一阵用过，2026-10-09 起改提供对话模型）；\n"
            "  ⚠ 它与 `TIANXIMEM_RERANKER_BASE_URL` 是同一件事的两半：**换网关要一起换**。"
        )
    return RerankConfig(
        enabled=True if enabled is None else enabled,
        timeout_seconds=_float(
            g.get("timeout_seconds"),
            where="rerank.timeout_seconds",
            default=DEFAULT_RERANK_TIMEOUT_S,
        ),
        envelope=envelope,
    )


def _router(raw: object) -> RouterConfig:
    """查询路由器。**两个布尔项都要显式挡掉 `bool` 之外的取值**（同 `_rerank`）。"""
    g = _group(
        raw,
        where="router",
        allowed={
            "enabled",
            "timeout_seconds",
            "max_output_tokens",
            "temperature",
            "prompt_version",
        },
    )
    enabled = g.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ConfigError(f"`router.enabled` 必须是布尔值，收到 {enabled!r}")
    return RouterConfig(
        enabled=False if enabled is None else enabled,
        timeout_seconds=_float(
            g.get("timeout_seconds"),
            where="router.timeout_seconds",
            default=DEFAULT_ROUTER_TIMEOUT_S,
        ),
        max_output_tokens=_int(
            g.get("max_output_tokens"),
            where="router.max_output_tokens",
            default=DEFAULT_ROUTER_MAX_OUTPUT_TOKENS,
        ),
        temperature=_float(g.get("temperature"), where="router.temperature", default=0.0),
        prompt_version=_str(
            g.get("prompt_version"),
            where="router.prompt_version",
            default=DEFAULT_ROUTER_PROMPT_VERSION,
        ),
    )


def _agentic(raw: object) -> AgenticConfig:
    """多跳执行器（`MULTI_HOP`）。**每一个键都只影响它命名的那一件事**（§13）。

    ⚠ `enabled` 是唯一的开关；`force` 是**实验**开关（E1 要绕过路由器），
    其余都是上限/阈值。⛔ 这里刻意**没有**"轮数 × 每轮条数"之外的检索参数：
    每轮召回深度取**请求的 `top_k`**，不为多跳另造一套（§3.2）。
    """
    g = _group(
        raw,
        where="agentic",
        allowed={
            "enabled",
            "mode",
            "force",
            "max_rounds",
            "per_round_top_k",
            "sufficiency_threshold",
            "max_prompt_tokens",
            "max_output_tokens",
            "plan_max_output_tokens",
            "timeout_seconds",
            "temperature",
            "round_rerank_enabled",
            "final_rerank_query_mode",
            "prompt_version",
            "plan_decompose_prompt_version",
            "plan_step_prompt_version",
        },
    )
    for name in ("enabled", "force", "round_rerank_enabled"):
        value = g.get(name)
        if value is not None and not isinstance(value, bool):
            raise ConfigError(f"`agentic.{name}` 必须是布尔值，收到 {value!r}")
    mode = _str(g.get("mode"), where="agentic.mode", default=DEFAULT_AGENTIC_MODE)
    if mode not in AGENTIC_MODES:
        # 取值域在**这里**就挡住：一个拼错的模式会让执行器**静默退回 react**，
        # 而"跑的是哪套机制"正是所有多跳结论的自变量——那种静默会直接污染结论。
        raise ConfigError(f"`agentic.mode` 必须是 {AGENTIC_MODES} 之一，收到 {mode!r}")
    rerank_query_mode = _str(
        g.get("final_rerank_query_mode"),
        where="agentic.final_rerank_query_mode",
        default="combined",
    )
    if rerank_query_mode not in FINAL_RERANK_QUERY_MODES:
        # 取值域在**这里**就挡住：一个拼错的模式会让"联合精排"静默退化成"只用原问题"，
        # 而两者的分数差正好是本阶段要测的东西（§9.1）——**那种静默会直接污染结论**。
        raise ConfigError(
            f"`agentic.final_rerank_query_mode` 必须是 {FINAL_RERANK_QUERY_MODES} 之一，"
            f"收到 {rerank_query_mode!r}"
        )
    return AgenticConfig(
        enabled=bool(g.get("enabled") or False),
        mode=mode,
        force=bool(g.get("force") or False),
        max_rounds=_int(
            g.get("max_rounds"), where="agentic.max_rounds", default=DEFAULT_COQ_MAX_ROUNDS
        ),
        per_round_top_k=_int(
            g.get("per_round_top_k"),
            where="agentic.per_round_top_k",
            default=DEFAULT_COQ_PER_ROUND_TOP_K,
        ),
        sufficiency_threshold=_float(
            g.get("sufficiency_threshold"),
            where="agentic.sufficiency_threshold",
            default=DEFAULT_COQ_SUFFICIENCY_THRESHOLD,
        ),
        max_prompt_tokens=_int(
            g.get("max_prompt_tokens"),
            where="agentic.max_prompt_tokens",
            default=DEFAULT_COQ_MAX_PROMPT_TOKENS,
        ),
        max_output_tokens=_int(
            g.get("max_output_tokens"),
            where="agentic.max_output_tokens",
            default=DEFAULT_COQ_MAX_OUTPUT_TOKENS,
        ),
        plan_max_output_tokens=_int(
            g.get("plan_max_output_tokens"),
            where="agentic.plan_max_output_tokens",
            default=DEFAULT_PLAN_MAX_OUTPUT_TOKENS,
        ),
        timeout_seconds=_float(
            g.get("timeout_seconds"),
            where="agentic.timeout_seconds",
            default=DEFAULT_COQ_TIMEOUT_S,
        ),
        temperature=_float(g.get("temperature"), where="agentic.temperature", default=0.0),
        round_rerank_enabled=bool(g.get("round_rerank_enabled", True)),
        final_rerank_query_mode=rerank_query_mode,
        prompt_version=_str(
            g.get("prompt_version"),
            where="agentic.prompt_version",
            default=DEFAULT_COQ_PROMPT_VERSION,
        ),
        plan_decompose_prompt_version=_str(
            g.get("plan_decompose_prompt_version"),
            where="agentic.plan_decompose_prompt_version",
            default=DEFAULT_PLAN_DECOMPOSE_PROMPT_VERSION,
        ),
        plan_step_prompt_version=_str(
            g.get("plan_step_prompt_version"),
            where="agentic.plan_step_prompt_version",
            default=DEFAULT_PLAN_STEP_PROMPT_VERSION,
        ),
    )


def _capture(raw: object) -> CaptureConfig:
    """诊断开关——**路径不在这里**（它归 env，由 `load_config` 覆盖）。"""
    g = _group(raw, where="capture", allowed={"enabled", "max_bytes"})
    enabled = g.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ConfigError(f"`capture.enabled` 必须是布尔值，收到 {enabled!r}")
    return CaptureConfig(
        enabled=False if enabled is None else enabled,
        max_bytes=_int(
            g.get("max_bytes"), where="capture.max_bytes", default=DEFAULT_CAPTURE_MAX_BYTES
        ),
    )


# ── 校验 ───────────────────────────────────────────────────────────────

_MISSING_HINT: Final[str] = (
    "服务需要它才能启动——**不要在缺配置的情况下带着空值跑起来**，"
    "那样要等到第一次请求才炸。见 `.env.example` 与 `configs/default.yaml`。"
)


def _require(value: str, /, *, where: str, source: str) -> None:
    if not value or not value.strip():
        raise ConfigError(f"缺少配置 `{where}`（来源：{source}）。{_MISSING_HINT}")


def validate(cfg: AppConfig) -> AppConfig:
    """校验并**归一化**（yaml 里的 list → 不可变 tuple）。不合法 ⇒ `ConfigError`。"""
    _require(cfg.storage.sqlite.path, where="storage.sqlite.path", source=f"`{ENV_SQLITE_PATH}`")
    _require(cfg.storage.qdrant.url, where="storage.qdrant.url", source=f"`{ENV_QDRANT_URL}`")
    _require(
        cfg.storage.qdrant.collection, where="storage.qdrant.collection", source="configs/*.yaml"
    )
    _require(cfg.cache.embed.dir, where="cache.embed.dir", source=f"`{ENV_EMBED_CACHE_DIR}`")
    _require(cfg.models.embedder, where="models.embedder", source="configs/*.yaml")
    _require(cfg.embed_base_url, where="embed.base_url", source=f"`{ENV_EMBED_BASE_URL}`")
    _require(cfg.embed_api_key, where="embed.api_key", source=f"`{ENV_EMBED_API_KEY}`")

    if cfg.storage.sqlite.busy_timeout_ms <= 0:
        raise ConfigError(
            f"`storage.sqlite.busy_timeout_ms` 必须为正：{cfg.storage.sqlite.busy_timeout_ms}"
        )
    if cfg.retrieval.prefetch_limit <= 0:
        raise ConfigError(f"`retrieval.prefetch_limit` 必须为正：{cfg.retrieval.prefetch_limit}")
    for name in ("evidence_limit", "evidence_hop_limit", "fact_backfill_limit"):
        if getattr(cfg.retrieval, name) <= 0:
            raise ConfigError(f"`retrieval.{name}` 必须为正")
    if cfg.retrieval.rrf.k != RRF_K:
        raise ConfigError(
            f"`retrieval.rrf.k` 必须是 {RRF_K}，收到 {cfg.retrieval.rrf.k}。\n"
            "  它是【正确性常量】不是可调项：Qdrant 的 k 默认是 2、文献是 60，"
            f"而 Qdrant 的秩 0-based，只有 {RRF_K} 才等价于文献的 60（D5）。\n"
            "  填错【不会报错】——两种写法的名次都看起来正常，只有分数差一个数量级。"
        )
    w_bm25, w_dense = cfg.retrieval.rrf.weights
    if w_bm25 < 0 or w_dense < 0:
        raise ConfigError(f"`retrieval.rrf.weights` 不得为负：{[w_bm25, w_dense]!r}")
    if w_bm25 == 0 and w_dense == 0:
        raise ConfigError("`retrieval.rrf.weights` 两项不能同时为 0——那等于不检索")
    if cfg.rerank.timeout_seconds <= 0:
        raise ConfigError(
            f"`rerank.timeout_seconds` 必须为正：{cfg.rerank.timeout_seconds}\n"
            "  它挡的是【reranker 挂住不返回】——非正值会让每次 rerank 都立刻超时，"
            "于是**每次都降级**，而响应看起来完全合法（只是名次没被精排）。"
        )
    if cfg.capture.max_bytes <= 0:
        raise ConfigError(
            f"`capture.max_bytes` 必须为正：{cfg.capture.max_bytes}\n"
            "  它是**护栏**不是阈值：`capture.enabled` 忘了关时，代价必须是有界的"
            "（写满就停），而不是把磁盘写满。"
        )
    if cfg.server.workers != 1:
        raise ConfigError(
            f"`server.workers`（`{ENV_WORKERS}`）必须是 1，收到 {cfg.server.workers}。\n"
            "  §15 要求 `--workers 1`。\n"
            "  ⚠ **放开多 worker 需要的验证没有做过**（并发写入压力、`busy_timeout` 争用、"
            "多进程各自的 Qdrant 客户端）⇒ 在那之前保持这条约束，"
            "**不要用「位置已经是请求的纯函数了」当理由把它去掉**。"
        )
    _validate_router(cfg)
    _validate_agentic(cfg)
    return replace(
        cfg,
        retrieval=replace(cfg.retrieval, rrf=replace(cfg.retrieval.rrf, weights=(w_bm25, w_dense))),
    )


def _validate_router(cfg: AppConfig) -> None:
    """路由器的三件套——**只在它真的开着时才要求**。

    ⚠ **为什么这里配错了要拒绝启动，而不是像 reranker 那样"缺了就降级 + WARNING"**
    （那条口径见 `build_reranker`）——两者的**失败后果不对称**：

    * 精排缺端点 ⇒ 名次退回 RRF。**检索仍然完整**，降级是**有定义的**，
      而且 run record 里 `rerank_disabled` 会如实记着。D12 接受这个代价，因为
      精排是"唯一不被规则保证可用"的组件。
    * 路由开着却没端点 ⇒ **每一次请求都退 `DEFAULT`**。结果不是"少一点排序质量"，
      而是**"Router On" 这个实验臂根本没有路由器**——跑出来的分数会被读成
      "路由没有帮助"，而归因完全错。**这正是 §13 要挡的那一类静默失败。**

    所以：**开着就必须配齐**（`validate()` 在启动阶段响亮失败），而"不想配"
    的出口是**把 `enabled` 设为 `false`**（默认值）。这与 `embed_base_url` 那种
    "缺了就拒绝启动"同型，只是加了 `enabled` 这个前提条件。
    """
    if not cfg.router.enabled:
        return
    for value, name, env in (
        (cfg.llm_base_url, "llm.base_url", ENV_LLM_BASE_URL),
        (cfg.llm_api_key, "llm.api_key", ENV_LLM_API_KEY),
        (cfg.llm_model, "llm.model", ENV_LLM_MODEL),
    ):
        if not value or not value.strip():
            raise ConfigError(
                f"`router.enabled = true`，但缺少配置 `{name}`（来源：`{env}`）。\n"
                "  路由器要调用对话端点才能分类，而缺端点的表现是**每次请求都降级**——"
                "跑出来的就是「Router On」这一臂其实没开路由器（见 `_validate_router` 的说明）。\n"
                "  ⇒ 填好 `.env` 的 `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL`，"
                "或把 `router.enabled` 改回 `false`。"
            )
    if cfg.router.timeout_seconds <= 0:
        raise ConfigError(
            f"`router.timeout_seconds` 必须为正：{cfg.router.timeout_seconds}\n"
            "  非正值会让每次路由都立刻超时 ⇒ **每次都降级**，而响应看起来完全合法。"
        )
    if cfg.router.max_output_tokens <= 0:
        raise ConfigError(
            f"`router.max_output_tokens` 必须为正：{cfg.router.max_output_tokens}\n"
            "  ⚠ 尤其不要设成 1：`MULTI_HOP` 会被截断成 `MULTI` ⇒ 输出非法 ⇒ "
            "**全部降级成 `DEFAULT`，而且不报错**。"
        )
    if cfg.router.temperature < 0:
        raise ConfigError(f"`router.temperature` 不得为负：{cfg.router.temperature}")
    _require(
        cfg.router.prompt_version,
        where="router.prompt_version",
        source="configs/*.yaml",
    )


def _validate_agentic(cfg: AppConfig) -> None:
    """多跳执行器的键——**只在它真的开着时才要求端点，但取值范围**任何时候都要对**。

    ⚠ **端点那一半与 `_validate_router` 的理由完全相同**（都是"开着就必须配齐"）：
    `agentic.enabled` 开着却没有对话端点 ⇒ 每一轮 LLM 都失败 ⇒ 每条 `MULTI_HOP`
    都带着候选回退 ⇒ 跑出来的分数被读成"多跳没有帮助"。**归因错了，不是质量差。**

    ⚠ 取值范围那一半**不看 `enabled`**：一个 `max_rounds: 0` 在关着的时候无害，
    但**打开它的那一刻**会造出"一轮都不跑、却声称走过多跳"的状态——
    而这正是本模块存在的意义（把启动期的错误挡在启动期）。

    它**不检查 `force` 本身**：那是实验开关，本身合法，只是不该出现在提交口径里
    （口径由 run record 的配置指纹负责留痕，不是由校验器负责禁止）。
    ⚠ 但它**必须与 `enabled` 一起用**——见下面那条检查。
    """
    if cfg.agentic.force and not cfg.agentic.enabled:
        raise ConfigError(
            "`agentic.force = true` 但 `agentic.enabled = false`。\n"
            "  `force` 的含义是「绕过路由器、一律走 MULTI_HOP」，而**执行器没装**时"
            "这条指令无处可去 ⇒ 每一个查询都会安静地按原路走，**而你以为在跑 E1**。\n"
            "  ⇒ 两个一起打开，或两个都关掉。"
        )
    if cfg.agentic.enabled:
        for value, name, env in (
            (cfg.llm_base_url, "llm.base_url", ENV_LLM_BASE_URL),
            (cfg.llm_api_key, "llm.api_key", ENV_LLM_API_KEY),
            (cfg.llm_model, "llm.model", ENV_LLM_MODEL),
        ):
            if not value or not value.strip():
                raise ConfigError(
                    f"`agentic.enabled = true`，但缺少配置 `{name}`（来源：`{env}`）。\n"
                    "  多跳执行器每一轮都要调一次对话端点；缺端点 ⇒ **每一轮都失败**，"
                    "于是「Agent On」这一臂其实没有执行器（见 `_validate_agentic` 的说明）。\n"
                    "  ⇒ 填好 `.env` 的 `AML_BASE_URL` / `AML_API_KEY` / `AML_MODEL`，"
                    "或把 `agentic.enabled` 改回 `false`。"
                )
    if cfg.agentic.max_rounds < 1:
        raise ConfigError(
            f"`agentic.max_rounds` 必须 ≥ 1（**含第一轮**）：{cfg.agentic.max_rounds}\n"
            "  0 会让多跳一轮都不跑却仍然返回**空候选**——那不是降级，是「假装跑过了」。"
        )
    if cfg.agentic.mode not in AGENTIC_MODES:
        # `_agentic()` 已经挡过（它更早、消息更具体）；这里再挡一次是**纵深**：
        # 手写一个 `AgenticConfig(...)` 绕过加载器时，本函数是唯一的守卫。
        raise ConfigError(f"`agentic.mode` 必须是 {AGENTIC_MODES} 之一：{cfg.agentic.mode!r}")
    if cfg.agentic.per_round_top_k < 1:
        raise ConfigError(f"`agentic.per_round_top_k` 必须 ≥ 1：{cfg.agentic.per_round_top_k}")
    if not 0.0 <= cfg.agentic.sufficiency_threshold <= 1.0:
        raise ConfigError(
            "`agentic.sufficiency_threshold` 必须在 0..1 之间："
            f"{cfg.agentic.sufficiency_threshold}\n"
            "  ⚠ 它只在**已经判定充分**之后起作用（见 `DEFAULT_COQ_SUFFICIENCY_THRESHOLD`）。"
        )
    for name in ("max_prompt_tokens", "max_output_tokens", "plan_max_output_tokens"):
        if getattr(cfg.agentic, name) <= 0:
            raise ConfigError(f"`agentic.{name}` 必须为正：{getattr(cfg.agentic, name)}")
    if cfg.agentic.timeout_seconds <= 0:
        raise ConfigError(f"`agentic.timeout_seconds` 必须为正：{cfg.agentic.timeout_seconds}")
    if cfg.agentic.temperature < 0:
        raise ConfigError(f"`agentic.temperature` 不得为负：{cfg.agentic.temperature}")
    _require(cfg.agentic.prompt_version, where="agentic.prompt_version", source="configs/*.yaml")
    _require(
        cfg.agentic.plan_decompose_prompt_version,
        where="agentic.plan_decompose_prompt_version",
        source="configs/*.yaml",
    )
    _require(
        cfg.agentic.plan_step_prompt_version,
        where="agentic.plan_step_prompt_version",
        source="configs/*.yaml",
    )


def assert_single_process() -> None:
    """确认当前进程**不是**被多进程 supervisor 派生出来的 worker。

    `validate()` 只能管住"配置里写的 worker 数"，而 `--workers 4` 是**命令行**给的 ⇒
    这里补一道**运行期**的守卫，否则那条约束仍然只是文档。

    **判据**：uvicorn 的 `--workers N`（N>1）与 `--reload` 都会用 `multiprocessing`
    派生一个子进程来跑 app，于是子进程里 `multiprocessing.parent_process()` **不是 None**；
    而单进程启动时它是 `None`。

    ⚠ **无法与 `--reload` 区分**（两者在子进程里形状完全相同），所以 `--reload` 也会被拦下
    ——**这是可接受的**：开发期请用 `make serve`（无 `--reload`、`--workers 1`）。

    ⚠ 前提假设：**v1 的服务入口只有 uvicorn**。若将来有别的进程宿主（例如 harness 用
    `multiprocessing` 跑服务），这条会误伤——**届时必须重审**，而不是加个开关绕过去。
    """
    if (os.environ.get(ENV_WORKERS) or "").strip() not in ("", "1"):
        return  # 已经（或即将）在 validate() 里拦过同一个问题，不重复报
    parent = multiprocessing.parent_process()
    if parent is None:
        return
    raise ConfigError(
        f"检测到本进程是多进程派生的 worker（pid={os.getpid()}，父进程 pid={parent.pid}）。\n"
        "  §15 要求 `--workers 1`（理由见 validate() 里那条错误消息的说明）。\n"
        "  ⇒ 去掉 `--workers N`（N>1）。若你在用 `--reload`：本守卫无法把它与多 worker 区分"
        "（两者在子进程里形状相同），请改用 `make serve`。\n"
        "  ⚠ uvicorn 的 `--workers N` 监督进程会**反复重启**这些 worker，于是这条错误会反复刷——"
        "**每个 worker 都被拦下，服务不会变得可用**，但你要按 Ctrl-C 退出，别等它自己停。"
    )


# ── 加载 ───────────────────────────────────────────────────────────────


def load_config(
    env: Mapping[str, str] | None = None,
    /,
    *,
    config_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """按"默认值 ← default.yaml ← <profile>.yaml ← .env ← 真实环境变量"合成并校验。

    `env` 是为了**可测试**：传一个 dict 就完全不碰真实环境、**也不读磁盘上的 `.env`**
    （`tests/test_config.py` 用它）。生产入口 `create_app_from_env()` 不传，于是会读 `.env`。

    `env_file` 指定 `.env` 的位置；不传则由 `TIANXIMEM_ENV_FILE` 决定，再退回 cwd 下的 `.env`。
    """
    if env is None:
        # .env 是"这台机器的环境"的本地副本 ⇒ 排在真实环境变量**之下**，自动让位
        file_path = Path(
            env_file if env_file is not None else (os.environ.get(ENV_FILE) or DEFAULT_ENV_FILE)
        )
        src: Mapping[str, str] = {**_read_env_file(file_path), **os.environ}
    else:
        src = env

    profile = (src.get(ENV_PROFILE) or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    where = Path(
        config_dir if config_dir is not None else (src.get(ENV_CONFIG_DIR) or DEFAULT_CONFIG_DIR)
    )

    data = _read_yaml(where / "default.yaml", required=True)
    if profile != DEFAULT_PROFILE:
        data = _deep_merge(data, _read_yaml(where / f"{profile}.yaml", required=True))

    storage = _storage(data.get("storage"))
    storage = replace(
        storage,
        sqlite=replace(storage.sqlite, path=src.get(ENV_SQLITE_PATH) or storage.sqlite.path),
        qdrant=replace(storage.qdrant, url=src.get(ENV_QDRANT_URL) or storage.qdrant.url),
    )
    cache = CacheConfig(
        embed=EmbedCacheConfig(dir=src.get(ENV_EMBED_CACHE_DIR) or "var/embed_cache")
    )

    # 采集的**路径归 env**（与 SQLite / 向量缓存同一条规矩）；开不开在 yaml 的
    # `capture.enabled`——一个键只有一个家，两处都能设的值最后没人知道为什么。
    capture = _capture(data.get("capture"))
    capture = replace(capture, path=src.get(ENV_CAPTURE_PATH) or capture.path)

    try:
        workers = int((src.get(ENV_WORKERS) or "1").strip() or "1")
    except ValueError as exc:
        raise ConfigError(f"`{ENV_WORKERS}` 必须是整数，收到 {src.get(ENV_WORKERS)!r}") from exc

    return validate(
        AppConfig(
            profile=profile,
            storage=storage,
            cache=cache,
            models=_models(data.get("models")),
            retrieval=_retrieval(data.get("retrieval")),
            neighbor=_neighbor(data.get("neighbor")),
            budget=_budget(data.get("budget")),
            packaging=_packaging(data.get("packaging")),
            rerank=_rerank(data.get("rerank")),
            router=_router(data.get("router")),
            agentic=_agentic(data.get("agentic")),
            capture=capture,
            server=ServerConfig(workers=workers),
            # 密钥与端点不在 yaml 里（见模块 docstring 的两层分工）
            embed_base_url=(src.get(ENV_EMBED_BASE_URL) or "").strip(),
            embed_api_key=(src.get(ENV_EMBED_API_KEY) or "").strip(),
            # ⚠ 这三个**允许为空**（默认就是空的）：不需要路由器的部署不必配它们。
            #   真正要求它们的是 `_validate_router()`——**只在 `router.enabled` 时**。
            #   ⚠ 端点与 key 是**一对**，`_2` 那一套是 harness 的调度概念，服务不接（见常量注释）。
            llm_base_url=(src.get(ENV_LLM_BASE_URL) or "").strip(),
            llm_api_key=(src.get(ENV_LLM_API_KEY) or "").strip(),
            llm_model=(src.get(ENV_LLM_MODEL) or "").strip(),
            # ⚠ 这三个**允许为空**：空 ⇒ 不构造 reranker（D12 的降级形态，不是配置错误）
            reranker_base_url=(src.get(ENV_RERANKER_BASE_URL) or "").strip(),
            reranker_api_key=(src.get(ENV_RERANKER_API_KEY) or "").strip(),
            reranker_model=(src.get(ENV_RERANKER_MODEL) or "").strip(),
            # ⚠ 同样允许为空：空 ⇒ 不写指标快照（缺省不是错误，见字段的注释）
            metrics_path=(src.get(ENV_METRICS_PATH) or "").strip(),
        )
    )
