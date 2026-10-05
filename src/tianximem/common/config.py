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
    "DEFAULT_WEIGHTS",
    "RRF_K",
    "AppConfig",
    "ConfigError",
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
#: ⚠ **主网关**（`memory3.021130.xyz`），不是 memory2——两个网关 host 与 key 都不同（D18）。
#: ⚠ 它对服务**不是必需**的（D12）：缺了照常启动、走 `rerank_disabled` 路径
#: ——**这与 embedding 的三个变量正相反**。
ENV_RERANKER_BASE_URL: Final[str] = "TIANXIMEM_RERANKER_BASE_URL"
ENV_RERANKER_API_KEY: Final[str] = "TIANXIMEM_RERANKER_API_KEY"
ENV_RERANKER_MODEL: Final[str] = "TIANXIMEM_RERANKER_MODEL"
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
#: 两个自托管网关的线格式**互斥**，而且**填错哪边都不报错**：
#:
#: * `"queries"` —— **vLLM 原生 score 形状**（`queries: [...]`，数组）。提交期的主网关
#:   `memory3.021130.xyz` 就是这一档。给它发 `query` ⇒ **400**。
#: * `"query"` —— **自研封装**的形状（单数字符串）。本机自托管网关
#:   （容器里 `host.docker.internal:9002` → 宿主机 `127.0.0.1:8082`），
#:   给它发 `queries` ⇒ **422**。
#:
#: ⚠ **代价是静默的**：D12 把远端问题一律吞成 `RerankUnavailable` ⇒ 发错字段名的表现是
#:   **每次检索都降级、服务不报错、响应照旧合法**，只在 §14 的 `rerank.degraded` 计数上看得见。
#:   ⇒ 它是**端点身份的一部分**，跟着 `TIANXIMEM_RERANKER_BASE_URL` 一起换，**不是可调旋钮**。
DEFAULT_RERANK_ENVELOPE: Final[str] = "queries"

#: `rerank.envelope` 的取值域。见 `DEFAULT_RERANK_ENVELOPE` 的两条格式与各自的拒绝码。
RERANK_ENVELOPES: Final[frozenset[str]] = frozenset({"queries", "query"})

#: `capture.max_bytes` 的默认值：**每份文件**写这么多就换下一份（50 MiB）。
#:
#: ⚠ 它是**切分粒度，不是总量上限**：一轮 Full 的请求原文按 ~1.5 GB 估（§2.2），
#: 而单个 1.5 GB 的 JSONL 打不开 ⇒ 分成 ~30 份、每份都能直接看。
#: `capture.enabled` 忘了关的代价就是"记满一块盘"，由人盯着（跑完关掉）。
DEFAULT_CAPTURE_MAX_BYTES: Final[int] = 50 * 1024 * 1024


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
    embedder: str = "qwen3-embedding-8b"


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
    """上下文打包的渲染变体（§11.3）。**两个键，默认值都是 v1 定稿口径。**"""

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
    g = _group(raw, where="models", allowed={"embedder"})
    return ModelsConfig(
        embedder=_str(g.get("embedder"), where="models.embedder", default="qwen3-embedding-8b")
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
    g = _group(raw, where="packaging", allowed={"inject_abs_time", "annotate_relatives"})
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
    return PackagingConfig(
        inject_abs_time=True if inject is None else inject,
        annotate_relatives=False if annotate is None else annotate,
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
            "  `queries` = vLLM 原生 score 形状（`memory3.021130.xyz`）；\n"
            "  `query`   = 自研封装（本机网关，容器里走 `host.docker.internal:9002`）。\n"
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

    return replace(
        cfg,
        retrieval=replace(cfg.retrieval, rrf=replace(cfg.retrieval.rrf, weights=(w_bm25, w_dense))),
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
            capture=capture,
            server=ServerConfig(workers=workers),
            # 密钥与端点不在 yaml 里（见模块 docstring 的两层分工）
            embed_base_url=(src.get(ENV_EMBED_BASE_URL) or "").strip(),
            embed_api_key=(src.get(ENV_EMBED_API_KEY) or "").strip(),
            # ⚠ 这三个**允许为空**：空 ⇒ 不构造 reranker（D12 的降级形态，不是配置错误）
            reranker_base_url=(src.get(ENV_RERANKER_BASE_URL) or "").strip(),
            reranker_api_key=(src.get(ENV_RERANKER_API_KEY) or "").strip(),
            reranker_model=(src.get(ENV_RERANKER_MODEL) or "").strip(),
            # ⚠ 同样允许为空：空 ⇒ 不写指标快照（缺省不是错误，见字段的注释）
            metrics_path=(src.get(ENV_METRICS_PATH) or "").strip(),
        )
    )
