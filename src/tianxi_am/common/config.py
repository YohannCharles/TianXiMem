"""配置的唯一入口（§15 / §12.1 R1 对冲 3）。

## 本模块是全包**唯一**读 `os.environ` 的地方

`service/` / `retrieve/` / `rank/` / `store/` / `embed/` 都**不得**自行读环境变量——
它们只接收这里构造出来的配置对象。理由不是洁癖：环境变量的读取点一多，
"这次跑的是哪套值"就无法从一个地方回答，而 §13 的每一条对照实验都依赖那个回答。

## 两层来源，**每个键只有一个家**

| 层 | 拥有哪些键 | 为什么 |
| --- | --- | --- |
| **环境变量**（`.env`） | 密钥、端点、**路径**、进程形态（worker 数） | **不进 git**，随机器而异 |
| **`configs/<profile>.yaml`** | 阈值、权重、模型名、集合名 | 要能 **diff、评审、归档** |

**没有重叠**：一个键要么在 env、要么在 yaml，**不允许两边都能设**。这条是刻意的——
两处都能设的值，最终会变成"跑出来的结果和 yaml 里写的不一样，而没人知道为什么"。

> ⚠ **为什么阈值不能藏进环境变量**：那会让 Step 5 的模型切换变成"改 shell 变量"——
> **改了什么无法 diff**，而归因恰恰是那一步唯一的目的（§12.1 R1 对冲 4）。
> 一处声明在 [`../../../configs/CLAUDE.md`](../../../configs/CLAUDE.md)。

## 加载顺序

```text
内置默认值（本文件的 dataclass 默认值）
  ← configs/default.yaml                  （缺失 ⇒ 报错，不静默退回内置默认值）
  ← configs/<profile>.yaml                （profile 来自 TIANXI_PROFILE，默认 default）
  ← .env                                  （**最低优先级的"环境"**，见下）
  ← 真实的环境变量                         （**压过 .env**：export 过的值说了算）
  → 校验                                  （不合法的值在这里响亮失败）
```

**`.env` 由本模块读取**（2026-09-24 补）。它**不是**一条独立来源，而是"这台机器的环境"的
本地副本——所以它排在真实环境变量**之下**，且在 CI / `export` 过的场景下自动让位。
**这样 `.env` 才真的生效**：在此之前它只是被文档声明成"密钥的家"，而没有任何东西读它
（`uv run` 不加载 `.env`，Makefile 也不 include 它）⇒ `make serve` 会在启动时缺密钥。

> ⚠ **只有 `env=None`（生产路径）才读 `.env`。** 测试传一个 `env` dict 时**绝不碰磁盘**——
> 否则"测试不依赖真实环境变量"这条纪律就破了。换 `.env` 的位置用 `TIANXI_ENV_FILE`。

**选了一个不存在的 profile 同样报错**（不静默忽略）——静默回退会让"我明明选了 submit"
变成一个查不出来的问题。

配置项与类别的完整清单（A 外部契约常量 / B 正确性常量 / C 自设阈值）在
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

from tianxi_am.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

__all__ = [
    "ENV_CONFIG_DIR",
    "ENV_EMBED_API_KEY",
    "ENV_EMBED_BASE_URL",
    "ENV_EMBED_CACHE_DIR",
    "ENV_FILE",
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

ENV_PROFILE: Final[str] = "TIANXI_PROFILE"
ENV_CONFIG_DIR: Final[str] = "TIANXI_CONFIG_DIR"
ENV_SQLITE_PATH: Final[str] = "TIANXI_SQLITE_PATH"
ENV_QDRANT_URL: Final[str] = "TIANXI_QDRANT_URL"
ENV_EMBED_BASE_URL: Final[str] = "AML_EMB_BASE_URL"
ENV_EMBED_API_KEY: Final[str] = "AML_EMB_API_KEY"
ENV_EMBED_CACHE_DIR: Final[str] = "TIANXI_EMBED_CACHE_DIR"
ENV_WORKERS: Final[str] = "TIANXI_WORKERS"

#: reranker 的三个变量（2026-09-24 接线）。**名字沿用 `.env.example` 里早就声明的那三个**，
#: 没有另起一套——`.env.example` 是它们的家，本表只是代码侧的引用点，两处由
#: `tests/test_config.py` 的静态断言钉住。
#:
#: ⚠ **主网关**（`memory.021130.xyz`），不是 memory2——两个网关 host 与 key 都不同（D18）。
#: ⚠ 它对服务**不是必需**的：它是唯一不被规则保证可用的组件（D12），
#: 所以缺了它服务照常启动、走 `rerank_disabled` 路径——**这与 embedding 的三个变量正相反**。
ENV_RERANKER_BASE_URL: Final[str] = "TIANXI_RERANKER_BASE_URL"
ENV_RERANKER_API_KEY: Final[str] = "TIANXI_RERANKER_API_KEY"
ENV_RERANKER_MODEL: Final[str] = "TIANXI_RERANKER_MODEL"
#: `.env` 文件的位置（默认 cwd 下的 `.env`）。⚠ 它**不是**一个配置项，是"去哪读环境"。
ENV_FILE: Final[str] = "TIANXI_ENV_FILE"

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

#: Neighbor Expansion 的 v1 初值（§10）。
#:
#: ⚠ **值的家在这里**（而不是 `rank/neighbor.py`）：`common/` 是**最底层**、被所有层依赖，
#: 而反过来（`common/` import `rank/`）是分层错误。所以默认值只能住在这里，
#: 由 `rank/neighbor.py` **引用**——这样也就不存在"代码默认值 vs 配置默认值"两处漂移。
#:
#: ⚠ 这两个量**不是调参项**：§10 明确"种子数、窗口大小、Top-K、token 预算**是同一道题**，
#: 任何一项调整都要重算其余三项"。改它们之前要有消融数据（§12.1 R1 对冲 3）。
DEFAULT_EXPANSION_SEED_LIMIT: Final[int] = 30
DEFAULT_RADIUS: Final[int] = 1

#: §7.3 的两个检索参数初值：每路进入 RRF 的候选池大小（`N`）与 `[w_bm25, w_dense]` 权重。
#:
#: ⚠ **值的家在这里**（而不是 `retrieve/fusion.py`）：理由与上面两个**同一条**——
#: `common/` 是最底层，而 `retrieve/` 与 `store/` 的 `HybridParams` 都要用到它们。
#: （2026-09-25 之前，`200` 与 `(0.5, 0.5)` 在全仓**各有 4 份裸字面量且没有一致性测试**；
#: 现在只剩这一处，`retrieve/` 与 `store/` 都引用它。）
#:
#: ⚠ 这两个是 **C 类可调项**（不同于 `RRF_K` 那种正确性常量）：**任何调整都必须有
#: ablation 数据支撑**——Qdrant 官方明确警告"无评测集时手调权重不太可能稳定优于默认值"（§7.3）。
DEFAULT_PREFETCH_LIMIT: Final[int] = 200
DEFAULT_WEIGHTS: Final[tuple[float, float]] = (0.5, 0.5)

#: rerank 调用的超时（秒）。**C 类自设阈值**（`rerank.timeout_seconds`）。
#:
#: ⚠ 定这个值的两条约束**方向相反**，别只看着一边调：
#:
#: * **太紧 ⇒ 伪降级**。网关是开发环路的单点（V8：answer / judge / embed / rerank 四条都打它），
#:   排队时延会抬高——那时超时会把"排队"报成"reranker 坏了"，而两者在计数器上长得一样。
#: * **太松 ⇒ Search 被拖住**。单请求上限 30 分钟（§2.2），而 rerank 在关键路径上。
#:
#: 实测（2026-09-24，开发机，主网关）：**100 篇 ≈ 2.2s、200 篇 ≈ 5.3s**。
#: 默认取 30s ⇒ 约 10 倍余量。**这是观测值，不是规格**；换模型或换网关后要重新量。
DEFAULT_RERANK_TIMEOUT_S: Final[float] = 30.0


# ── 配置树 ─────────────────────────────────────────────────────────────
# 字段名与 `docs/config-reference.md` 的点分键**逐段对应**（`storage.sqlite.path`
# ↔ `AppConfig.storage.sqlite.path`），免得两份文档各说各的。


@dataclass(frozen=True, slots=True)
class SqliteConfig:
    """真源文件（§6.1）。"""

    #: `TIANXI_SQLITE_PATH`。【路径归 env】
    path: str = "var/tianxi.db"
    #: 抢不到写锁时**等待**而不是立刻抛 `SQLITE_BUSY` 的毫秒数（D17）。【阈值归 yaml】
    busy_timeout_ms: int = 5000


@dataclass(frozen=True, slots=True)
class QdrantConfig:
    """派生索引（§6.3）。**必须 server 模式**（local 模式会静默丢弃 payload 索引）。"""

    #: `TIANXI_QDRANT_URL`。【端点归 env】
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

    #: `TIANXI_EMBED_CACHE_DIR`。**必须落盘**，否则每次重启重付一遍 embedding。【路径归 env】
    dir: str = "var/embed_cache"


@dataclass(frozen=True, slots=True)
class CacheConfig:
    embed: EmbedCacheConfig = field(default_factory=EmbedCacheConfig)


@dataclass(frozen=True, slots=True)
class ModelsConfig:
    """模型名（§2.3）。"""

    #: ⚠ **这里是模型名的家**，不是 `.env`：它是 profile 之间**唯一真正该变**的东西，
    #: 而 `local.yaml` / `submit.yaml` 存在的理由就是让"哪些量随模型变"能被看见。
    embedder: str = "Qwen/Qwen3-Embedding-8B"


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
    rrf: RrfConfig = field(default_factory=RrfConfig)


@dataclass(frozen=True, slots=True)
class PairingConfig:
    """批次切分上限（§6.5）。"""

    batch_max_messages: int = 20
    batch_max_words: int = 2000


@dataclass(frozen=True, slots=True)
class NeighborConfig:
    """Neighbor Expansion（§10）。"""

    #: 主动扩窗的种子数——**只对 rerank 后的前 N 条扩窗**，其余候选仍保留、只是不扩展。
    #:
    #: ⚠ v1 初值。§10 明确"种子数、窗口大小、Top-K、token 预算**是同一道题**，
    #: 任何一项调整都要重算其余三项" ⇒ **改它之前要有消融数据**（§12.1 R1 对冲 3）。
    expansion_seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT
    #: 扩窗半径，**单位是 QA 对**：±1 拿回前后各**一整对**（最多 4 条消息）。
    radius: int = DEFAULT_RADIUS


@dataclass(frozen=True, slots=True)
class BudgetConfig:
    """token 预算（§6.4）。**双预算里的 token 那一半**；槽位那一半是请求里的 `top_k`。"""

    #: 上限（**A 类外部契约常量**：AML 定的，无权改，写进来只为追溯）。
    max_tokens: int = MAX_INPUT_TOKENS
    #: 分词器名。**必须是答案模型自己的那个**（§6.4）——用近似会在 100 个对上放大到几千 token。
    tokenizer: str = DEFAULT_TOKENIZER


@dataclass(frozen=True, slots=True)
class PackagingConfig:
    """上下文打包的渲染变体（§11.3）。**只有一个键，而且默认值就是 v1 定稿口径。**

    ⚠ 这个键**存在的唯一理由是 T1**（§13 的"时间戳前缀 带 / 不带"）——
    `config-reference.md` §2 明文要求那个对照项"同样要可配"。它**不是**一个调参项：
    §11.3 的结论是"不注入绝对时间"（两条互相独立的裁判规则都指向它），
    打开它只会让分数变差；打开它的目的是**把那条结论在代理评测上验一遍**。
    """

    #: `false`（**v1 定稿**）⇒ `content` 里没有任何绝对时间戳，时间只走 `event_time` 筛选。
    #: `true`（**只为 T1 的"带"臂**）⇒ 每对正文前加日粒度日期前缀。
    #:
    #: ⚠ 它是**"贵"消融项**：正文一改，embedding 输入就改 ⇒ **向量索引要重建**，
    #: 而且必须跑在**另一个集合**上（`storage.qdrant.collection`）——两臂混在一个集合里，
    #: 检索到的是哪一臂的向量**根本看不出来**。
    inject_abs_time: bool = False


@dataclass(frozen=True, slots=True)
class RerankConfig:
    """远程 reranker（§11.2 / D12）。**只放阈值与开关；端点与密钥在 env。**

    ⚠ **本段刻意只有两个键**（2026-09-24 接线时定的）。理由：一个键要有消费者才收
    （§6.1 对 DDL 的同一条纪律）——而 rerank 的真实旋钮只有"开不开"与"等多久"。
    想调模型就换 `.env` 的 `TIANXI_RERANKER_MODEL`（那是端点身份，不是阈值）。
    """

    #: §15 的消融开关之一。`false` ⇒ **不构造 reranker**，Search 直接用融合名次。
    #:
    #: ⚠ 它的"关"分支**只该改变排名**，不得改变候选数量（§13 的开关纯度）——
    #: `tests/test_rerank.py` 有这条断言。
    enabled: bool = True
    #: 单次 rerank 请求的超时（秒）。见模块顶部 `DEFAULT_RERANK_TIMEOUT_S` 的两条约束。
    timeout_seconds: float = DEFAULT_RERANK_TIMEOUT_S


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """服务进程形态（§15）。"""

    #: `TIANXI_WORKERS`。**v1 只允许 1**——理由见 `validate()` 与 `assert_single_process()`。
    workers: int = 1


@dataclass(frozen=True, slots=True)
class AppConfig:
    """全量运行配置。**`build_services()` 的输入就是它。**"""

    profile: str = DEFAULT_PROFILE
    storage: StorageConfig = field(default_factory=StorageConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    neighbor: NeighborConfig = field(default_factory=NeighborConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    packaging: PackagingConfig = field(default_factory=PackagingConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    server: ServerConfig = field(default_factory=ServerConfig)

    #: ⚠ **密钥与端点不在 yaml 里**——它们随机器与密钥而异、不进 git（见模块 docstring）。
    #: 名字与 `.env` 的 `AML_EMB_*` 一一对应，**没有默认值**：缺了在启动时响亮失败。
    embed_base_url: str = ""
    embed_api_key: str = ""

    #: reranker 的端点与密钥（`TIANXI_RERANKER_*`）。
    #:
    #: ⚠ **与上面两个正相反：缺了不报错。** 空值 ⇒ 不构造 reranker、走 `rerank_disabled`
    #: ——那是 D12 要求的降级形态，**不是配置错误**。所以 `validate()` 不 `_require` 它们。
    reranker_base_url: str = ""
    reranker_api_key: str = ""
    #: 请求里声明的模型名（进 run record 的配置指纹；D12 要求提交时不得更换）。
    #:
    #: ⚠ 实测（2026-09-24）：**网关会忽略这个字段**，响应里 `model` 回的是服务端路径
    #: （`/data/…/Qwen3-Reranker-4B`）。保留它是因为①请求该带上自己声明的模型、
    #: ②它是 run record 里"这次用的哪个 reranker"的唯一来源。**不要拿它做路由或校验。**
    reranker_model: str = ""


# ── yaml 读取：**不认识的键一律报错** ────────────────────────────────────

#: yaml 顶层允许的段。**env 拥有的键不出现在这里**（见模块 docstring 的分工表）：
#: 路径 / 端点 / 密钥 / worker 数都只从环境变量来。
_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset(
    {"storage", "models", "retrieval", "pairing", "neighbor", "budget", "packaging", "rerank"}
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
        embedder=_str(g.get("embedder"), where="models.embedder", default="Qwen/Qwen3-Embedding-8B")
    )


def _retrieval(raw: object) -> RetrievalConfig:
    g = _group(raw, where="retrieval", allowed={"prefetch_limit", "query_instruction", "rrf"})
    r = _group(g.get("rrf"), where="retrieval.rrf", allowed={"k", "weights"})
    return RetrievalConfig(
        prefetch_limit=_int(
            g.get("prefetch_limit"),
            where="retrieval.prefetch_limit",
            default=DEFAULT_PREFETCH_LIMIT,
        ),
        query_instruction=_str(
            g.get("query_instruction"), where="retrieval.query_instruction", default=""
        ),
        rrf=RrfConfig(
            k=_int(r.get("k"), where="retrieval.rrf.k", default=RRF_K),
            weights=_weights(r.get("weights"), where="retrieval.rrf.weights"),
        ),
    )


def _neighbor(raw: object) -> NeighborConfig:
    g = _group(raw, where="neighbor", allowed={"expansion_seed_limit", "radius"})
    return NeighborConfig(
        expansion_seed_limit=_int(
            g.get("expansion_seed_limit"),
            where="neighbor.expansion_seed_limit",
            default=DEFAULT_EXPANSION_SEED_LIMIT,
        ),
        radius=_int(g.get("radius"), where="neighbor.radius", default=DEFAULT_RADIUS),
    )


def _budget(raw: object) -> BudgetConfig:
    g = _group(raw, where="budget", allowed={"max_tokens", "tokenizer"})
    return BudgetConfig(
        max_tokens=_int(g.get("max_tokens"), where="budget.max_tokens", default=MAX_INPUT_TOKENS),
        tokenizer=_str(g.get("tokenizer"), where="budget.tokenizer", default=DEFAULT_TOKENIZER),
    )


def _packaging(raw: object) -> PackagingConfig:
    g = _group(raw, where="packaging", allowed={"inject_abs_time"})
    inject = g.get("inject_abs_time")
    if inject is not None and not isinstance(inject, bool):
        raise ConfigError(
            f"`packaging.inject_abs_time` 必须是布尔值，收到 {inject!r}——"
            "它是「带 / 不带时间戳前缀」那个对照臂的开关，不是日期格式"
        )
    return PackagingConfig(inject_abs_time=False if inject is None else inject)


def _rerank(raw: object) -> RerankConfig:
    g = _group(raw, where="rerank", allowed={"enabled", "timeout_seconds"})
    enabled = g.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        raise ConfigError(f"`rerank.enabled` 必须是布尔值，收到 {enabled!r}")
    return RerankConfig(
        enabled=True if enabled is None else enabled,
        timeout_seconds=_float(
            g.get("timeout_seconds"),
            where="rerank.timeout_seconds",
            default=DEFAULT_RERANK_TIMEOUT_S,
        ),
    )


def _pairing(raw: object) -> PairingConfig:
    g = _group(raw, where="pairing", allowed={"batch_max_messages", "batch_max_words"})
    return PairingConfig(
        batch_max_messages=_int(
            g.get("batch_max_messages"), where="pairing.batch_max_messages", default=20
        ),
        batch_max_words=_int(
            g.get("batch_max_words"), where="pairing.batch_max_words", default=2000
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
    if cfg.pairing.batch_max_messages <= 0:
        raise ConfigError(
            f"`pairing.batch_max_messages` 必须为正：{cfg.pairing.batch_max_messages}"
        )
    if cfg.pairing.batch_max_words <= 0:
        raise ConfigError(f"`pairing.batch_max_words` 必须为正：{cfg.pairing.batch_max_words}")
    if cfg.rerank.timeout_seconds <= 0:
        raise ConfigError(
            f"`rerank.timeout_seconds` 必须为正：{cfg.rerank.timeout_seconds}\n"
            "  它挡的是【reranker 挂住不返回】——非正值会让每次 rerank 都立刻超时，"
            "于是**每次都降级**，而响应看起来完全合法（只是名次没被精排）。"
        )
    if cfg.server.workers != 1:
        raise ConfigError(
            f"`server.workers`（`{ENV_WORKERS}`）必须是 1，收到 {cfg.server.workers}。\n"
            "  §15：Add 的按 session 串行化用的是【进程内】锁（service/locks.py 的 SessionLocks），"
            "多 worker 会**静默失效**——每个 worker 各有各的锁，同 session 的两个批次照旧并发，"
            "于是 pair_idx 撞车。\n"
            "  ⇒ v1 必须 `--workers 1`。这不是可调项，是约束。"
        )

    return replace(
        cfg,
        retrieval=replace(cfg.retrieval, rrf=replace(cfg.retrieval.rrf, weights=(w_bm25, w_dense))),
    )


def assert_single_process() -> None:
    """确认当前进程**不是**被多进程 supervisor 派生出来的 worker。

    `validate()` 只能管住"配置里写的 worker 数"。而 `--workers 4` 是**命令行**给的，
    配置层看不见它——所以这里补一道**运行期**的守卫，否则那条约束仍然只是文档。

    **判据**：uvicorn 的 `--workers N`（N>1）与 `--reload` 都会用 `multiprocessing`
    派生一个子进程来跑 app，于是子进程里 `multiprocessing.parent_process()` **不是 None**；
    而单进程启动时它是 `None`。

    ⚠ **无法与 `--reload` 区分**（两者在子进程里形状完全相同——父进程的 pid 与名字是唯一
    可见的信息）。因此 `--reload` 也会被拦下。**这是可接受的**：开发期请用 `make serve`
    （无 `--reload`、`--workers 1`），或手工去掉 `--reload`。

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
        "  §15 要求 `--workers 1`：SessionLocks 是【进程内】锁，多 worker 会**静默失效**。\n"
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

    `env_file` 指定 `.env` 的位置；不传则由 `TIANXI_ENV_FILE` 决定，再退回 cwd 下的 `.env`。
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
            pairing=_pairing(data.get("pairing")),
            neighbor=_neighbor(data.get("neighbor")),
            budget=_budget(data.get("budget")),
            packaging=_packaging(data.get("packaging")),
            rerank=_rerank(data.get("rerank")),
            server=ServerConfig(workers=workers),
            # 密钥与端点不在 yaml 里（见模块 docstring 的两层分工）
            embed_base_url=(src.get(ENV_EMBED_BASE_URL) or "").strip(),
            embed_api_key=(src.get(ENV_EMBED_API_KEY) or "").strip(),
            # ⚠ 这三个**允许为空**：空 ⇒ 不构造 reranker（D12 的降级形态，不是配置错误）
            reranker_base_url=(src.get(ENV_RERANKER_BASE_URL) or "").strip(),
            reranker_api_key=(src.get(ENV_RERANKER_API_KEY) or "").strip(),
            reranker_model=(src.get(ENV_RERANKER_MODEL) or "").strip(),
        )
    )
