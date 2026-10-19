"""`configs/*.yaml` 的读取、深合并与各段构造（**不认识的键一律报错**）。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import yaml

from tianximem.common.tokens import DEFAULT_TOKENIZER, MAX_INPUT_TOKENS

from .constants import (
    AGENTIC_MODES,
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
    DEFAULT_RADIUS,
    DEFAULT_RERANK_ENVELOPE,
    DEFAULT_RERANK_TIMEOUT_S,
    DEFAULT_ROUTER_MAX_OUTPUT_TOKENS,
    DEFAULT_ROUTER_PROMPT_VERSION,
    DEFAULT_ROUTER_TIMEOUT_S,
    DEFAULT_WEIGHTS,
    ENV_CONFIG_DIR,
    FINAL_RERANK_QUERY_MODES,
    RERANK_ENVELOPES,
    RRF_K,
    SEED_PLACEMENTS,
)
from .model import (
    AgenticConfig,
    BudgetConfig,
    CaptureConfig,
    ConfigError,
    GroundedPlannerConfig,
    ModelsConfig,
    NeighborConfig,
    PackagingConfig,
    QdrantConfig,
    RerankConfig,
    RetrievalConfig,
    RouterConfig,
    RrfConfig,
    SqliteConfig,
    StorageConfig,
)

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


def _bool(
    value: object,
    /,
    *,
    where: str,
    default: bool,
    hint: str = "",
    show_value: bool = True,
    present: bool = False,
) -> bool:
    """布尔项校验：缺键 ⇒ `default`；非布尔 ⇒ 响亮失败。

    ⚠ 不用 `isinstance(value, int)` 之类的宽松判断：`0/1` 会被静默当成 `False/True`。
    `show_value=False` 供那些历史消息里不带 `收到 …` 的键保留原文案；
    `present=False` 表示"键可以缺"，显式 `null` 仍然算非法（与旧行为一致）。
    """
    if value is None:
        if present:
            detail = f"，收到 {value!r}" if show_value else ""
            raise ConfigError(f"`{where}` 必须是布尔值{detail}")
        return default
    if not isinstance(value, bool):
        detail = f"，收到 {value!r}" if show_value else ""
        suffix = f"——{hint}" if hint else ""
        raise ConfigError(f"`{where}` 必须是布尔值{detail}{suffix}")
    return value


def _read_yaml(path: Path, /) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigError(
            f"配置文件不存在：{path}。"
            "阈值与模型名的家在这里（见 configs/CLAUDE.md）；"
            f"换目录用 `{ENV_CONFIG_DIR}`。"
        )
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
            "grounded_planner",
            "rrf",
        },
    )
    r = _group(g.get("rrf"), where="retrieval.rrf", allowed={"k", "weights"})
    p = _group(
        g.get("grounded_planner"),
        where="retrieval.grounded_planner",
        allowed={"enabled", "timeout_seconds", "max_output_tokens", "temperature", "prompt_version"},
    )
    evidence = _bool(
        g.get("grounded_evidence"),
        where="retrieval.grounded_evidence",
        default=False,
        show_value=False,
        present="grounded_evidence" in g,
    )
    planner_enabled = _bool(
        p.get("enabled"),
        where="retrieval.grounded_planner.enabled",
        default=False,
        show_value=False,
        present="enabled" in p and p["enabled"] is not None,
    )
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
        grounded_planner=GroundedPlannerConfig(
            enabled=planner_enabled,
            timeout_seconds=_float(
                p.get("timeout_seconds"),
                where="retrieval.grounded_planner.timeout_seconds",
                default=DEFAULT_ROUTER_TIMEOUT_S,
            ),
            max_output_tokens=_int(
                p.get("max_output_tokens"),
                where="retrieval.grounded_planner.max_output_tokens",
                default=DEFAULT_GROUNDED_PLANNER_MAX_OUTPUT_TOKENS,
            ),
            temperature=_float(
                p.get("temperature"),
                where="retrieval.grounded_planner.temperature",
                default=0.0,
            ),
            prompt_version=_str(
                p.get("prompt_version"),
                where="retrieval.grounded_planner.prompt_version",
                default=DEFAULT_GROUNDED_PLANNER_PROMPT_VERSION,
            ),
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
    inject = _bool(
        g.get("inject_abs_time"),
        where="packaging.inject_abs_time",
        default=True,
        hint="它是「带 / 不带时间戳前缀」那个对照臂的开关，不是日期格式",
    )
    annotate = _bool(
        g.get("annotate_relatives"),
        where="packaging.annotate_relatives",
        default=False,
        hint="它是「正文里的相对时间要不要就地注解成绝对日期」那个开关",
    )
    # ⚠ **缺键取 `False`、dataclass 取 `True` 是有意的**（不是漏改）：D22 之前冻结的
    #   快照按"无注解"跑。口径见 docs/config-reference.md §2；别把它"清理"成同值。
    unit_totals = _bool(
        g.get("annotate_unit_totals"),
        where="packaging.annotate_unit_totals",
        default=False,
        show_value=False,
        present="annotate_unit_totals" in g,
    )
    return PackagingConfig(
        inject_abs_time=inject,
        annotate_relatives=annotate,
        annotate_unit_totals=unit_totals,
    )


def _rerank(raw: object) -> RerankConfig:
    g = _group(raw, where="rerank", allowed={"enabled", "timeout_seconds", "envelope"})
    enabled = _bool(g.get("enabled"), where="rerank.enabled", default=True)
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
        enabled=enabled,
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
    enabled = _bool(g.get("enabled"), where="router.enabled", default=False)
    return RouterConfig(
        enabled=enabled,
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
    for name in ("enabled", "force"):
        _bool(g.get(name), where=f"agentic.{name}", default=False)
    # 这一项**取返回值**（默认 True，与 model.py / config-reference 一致）；显式 null
    # 也响亮失败——`bool(None)` 会把它静默当成 False。
    round_rerank_enabled = _bool(
        g.get("round_rerank_enabled"),
        where="agentic.round_rerank_enabled",
        default=True,
        present="round_rerank_enabled" in g,
    )
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
        round_rerank_enabled=round_rerank_enabled,
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
    enabled = _bool(g.get("enabled"), where="capture.enabled", default=False)
    return CaptureConfig(
        enabled=enabled,
        max_bytes=_int(
            g.get("max_bytes"), where="capture.max_bytes", default=DEFAULT_CAPTURE_MAX_BYTES
        ),
    )
