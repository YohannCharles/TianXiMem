"""启动期校验（不合法 ⇒ `ConfigError`）与进程形态守卫——把错误挡在启动期。"""

from __future__ import annotations

import multiprocessing
import os
from dataclasses import replace
from typing import Final

from .constants import (
    AGENTIC_MODES,
    ENV_EMBED_API_KEY,
    ENV_EMBED_BASE_URL,
    ENV_EMBED_CACHE_DIR,
    ENV_LLM_API_KEY,
    ENV_LLM_BASE_URL,
    ENV_LLM_MODEL,
    ENV_QDRANT_URL,
    ENV_SQLITE_PATH,
    ENV_WORKERS,
    RRF_K,
)
from .model import AppConfig, ConfigError

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
            f"Qdrant 的秩从 0 开始，只有 {RRF_K} 才等价于文献公式中的 60。\n"
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
    planner = cfg.retrieval.grounded_planner
    if planner.timeout_seconds <= 0:
        raise ConfigError("`retrieval.grounded_planner.timeout_seconds` 必须为正")
    if planner.max_output_tokens <= 0:
        raise ConfigError("`retrieval.grounded_planner.max_output_tokens` 必须为正")
    if planner.temperature < 0:
        raise ConfigError("`retrieval.grounded_planner.temperature` 不得为负")
    if not planner.prompt_version.strip():
        raise ConfigError("`retrieval.grounded_planner.prompt_version` 不能为空")
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
      而且 run record 里 `rerank_disabled` 会如实记着。 接受这个代价，因为
      精排是"唯一不被规则保证可用"的组件。
    * 路由开着却没端点 ⇒ **每一次请求都退 `DEFAULT`**。结果不是"少一点排序质量"，
      而是**"Router On" 这个实验臂根本没有路由器**——跑出来的分数会被读成
      "路由没有帮助"，导致归因完全错误。这是启动校验必须阻止的静默失败。

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
    for name in ("max_output_tokens", "plan_max_output_tokens"):
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

    当前服务入口是 uvicorn。其他进程模型未经验证，不能通过新增旁路开关绕过此检查。
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
