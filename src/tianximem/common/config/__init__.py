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

# 本文件是**再导出面**：导入即导出——`__all__`（逐字沿自原 `config.py`）之外的
# 名字照样必须可导入，全仓的 import 点与 `tests/test_config.py` 的静态断言依赖它们。
# ruff: noqa: F401


from __future__ import annotations

from tianximem.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

from .constants import (
    AGENTIC_MODES,
    DEFAULT_AGENTIC_MODE,
    DEFAULT_CAPTURE_MAX_BYTES,
    DEFAULT_CONFIG_DIR,
    DEFAULT_COQ_MAX_OUTPUT_TOKENS,
    DEFAULT_COQ_MAX_PROMPT_TOKENS,
    DEFAULT_COQ_MAX_ROUNDS,
    DEFAULT_COQ_PER_ROUND_TOP_K,
    DEFAULT_COQ_PROMPT_VERSION,
    DEFAULT_COQ_SUFFICIENCY_THRESHOLD,
    DEFAULT_COQ_TIMEOUT_S,
    DEFAULT_ENV_FILE,
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
    ENV_CAPTURE_PATH,
    ENV_CONFIG_DIR,
    ENV_EMBED_API_KEY,
    ENV_EMBED_BASE_URL,
    ENV_EMBED_CACHE_DIR,
    ENV_FILE,
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_METRICS_PATH,
    ENV_PROFILE,
    ENV_QDRANT_URL,
    ENV_RERANKER_API_KEY,
    ENV_RERANKER_BASE_URL,
    ENV_RERANKER_MODEL,
    ENV_SQLITE_PATH,
    ENV_WORKERS,
    FINAL_RERANK_QUERY_MODES,
    RERANK_ENVELOPES,
    RRF_K,
    SEED_PLACEMENTS,
)
from .loader import load_config
from .model import (
    AgenticConfig,
    AppConfig,
    BudgetConfig,
    CacheConfig,
    CaptureConfig,
    ConfigError,
    EmbedCacheConfig,
    GroundedPlannerConfig,
    ModelsConfig,
    NeighborConfig,
    PackagingConfig,
    QdrantConfig,
    RerankConfig,
    RetrievalConfig,
    RouterConfig,
    RrfConfig,
    ServerConfig,
    SqliteConfig,
    StorageConfig,
)
from .validate import assert_single_process, validate

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
    "DEFAULT_GROUNDED_PLANNER_MAX_OUTPUT_TOKENS",
    "DEFAULT_GROUNDED_PLANNER_PROMPT_VERSION",
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
    "GroundedPlannerConfig",
    "RouterConfig",
    "assert_single_process",
    "load_config",
]
