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
  ← 环境变量                              （**只覆盖它拥有的那几项**）
  → 校验                                  （不合法的值在这里响亮失败）
```

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

__all__ = [
    "ENV_CONFIG_DIR",
    "ENV_EMBED_API_KEY",
    "ENV_EMBED_BASE_URL",
    "ENV_EMBED_CACHE_DIR",
    "ENV_PROFILE",
    "ENV_QDRANT_URL",
    "ENV_SQLITE_PATH",
    "ENV_WORKERS",
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

DEFAULT_PROFILE: Final[str] = "default"
DEFAULT_CONFIG_DIR: Final[str] = "configs"

#: **正确性常量，不是可调项**（D5）：Qdrant 默认 `k=2`、文献是 60，而 Qdrant 的秩 0-based，
#: 只有 61 才等价于文献的 60。**填错不报错**——两种写法的名次都"看起来正常"，
#: 只有分数差一个数量级。⇒ 这里**拒绝**而不是警告。
#:
#: ⚠ 同一个值在 [`../retrieve/fusion.py`](../retrieve/fusion.py) 里也有一份（那边是**策略层**
#: 对直接调用方的防御，不能只靠"配置层已经查过"）。两处相等由
#: `tests/test_config.py::test_rrf_k_is_the_same_constant_in_both_places` 钉住。
RRF_K: Final[int] = 61


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
    weights: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """混合检索（§7.1–§7.3）。"""

    #: 每路进入 RRF 的候选池大小（§7.3 的 `N`）。
    #: ⚠ **不是**种子数（§10 的 20）、**不是** Top-K（§2.2 的 100）——三个不同的量。
    prefetch_limit: int = 200
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
    server: ServerConfig = field(default_factory=ServerConfig)

    #: ⚠ **密钥与端点不在 yaml 里**——它们随机器与密钥而异、不进 git（见模块 docstring）。
    #: 名字与 `.env` 的 `AML_EMB_*` 一一对应，**没有默认值**：缺了在启动时响亮失败。
    embed_base_url: str = ""
    embed_api_key: str = ""


# ── yaml 读取：**不认识的键一律报错** ────────────────────────────────────

#: yaml 顶层允许的段。**env 拥有的键不出现在这里**（见模块 docstring 的分工表）：
#: 路径 / 端点 / 密钥 / worker 数都只从环境变量来。
_TOP_LEVEL_KEYS: Final[frozenset[str]] = frozenset({"storage", "models", "retrieval", "pairing"})


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


def _weights(value: object, /, *, where: str) -> tuple[float, float]:
    if value is None:
        return (0.5, 0.5)
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
        prefetch_limit=_int(g.get("prefetch_limit"), where="retrieval.prefetch_limit", default=200),
        query_instruction=_str(
            g.get("query_instruction"), where="retrieval.query_instruction", default=""
        ),
        rrf=RrfConfig(
            k=_int(r.get("k"), where="retrieval.rrf.k", default=RRF_K),
            weights=_weights(r.get("weights"), where="retrieval.rrf.weights"),
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
) -> AppConfig:
    """按"默认值 ← default.yaml ← <profile>.yaml ← 环境变量"合成并校验。

    `env` 是为了**可测试**：传一个 dict 就完全不碰真实环境（`tests/test_config.py` 用它）。
    生产入口 `create_app_from_env()` 不传，走 `os.environ`。
    """
    src: Mapping[str, str] = os.environ if env is None else env

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
            server=ServerConfig(workers=workers),
            # 密钥与端点不在 yaml 里（见模块 docstring 的两层分工）
            embed_base_url=(src.get(ENV_EMBED_BASE_URL) or "").strip(),
            embed_api_key=(src.get(ENV_EMBED_API_KEY) or "").strip(),
        )
    )
