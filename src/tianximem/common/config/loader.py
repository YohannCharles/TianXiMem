"""`load_config` —— 加载顺序（默认值 ← `default.yaml` ← `<profile>.yaml` ← `.env` ← 真实环境变量）的唯一实现。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .constants import (
    DEFAULT_CONFIG_DIR,
    DEFAULT_PROFILE,
    ENV_CAPTURE_PATH,
    ENV_CONFIG_DIR,
    ENV_EMBED_API_KEY,
    ENV_EMBED_BASE_URL,
    ENV_EMBED_CACHE_DIR,
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
)
from .env import _env_source
from .model import AppConfig, CacheConfig, ConfigError, EmbedCacheConfig, ServerConfig
from .validate import validate
from .yaml_source import (
    _agentic,
    _budget,
    _capture,
    _deep_merge,
    _models,
    _neighbor,
    _packaging,
    _read_yaml,
    _rerank,
    _retrieval,
    _router,
    _storage,
)


def load_config(
    env: Mapping[str, str] | None = None,
    /,
    *,
    config_dir: str | Path | None = None,
    env_file: str | Path | None = None,
) -> AppConfig:
    """按"默认值 ← default.yaml ← <profile>.yaml ← .env ← 真实环境变量"合成并校验。

    `env` is an optional caller-provided mapping. Passing it bypasses both the
    real environment and disk dotenv loading. `create_app_from_env()` passes None.

    `env_file` 指定 `.env` 的位置；不传则由 `TIANXIMEM_ENV_FILE` 决定，再退回 cwd 下的 `.env`。
    """
    src = _env_source(env, env_file)

    profile = (src.get(ENV_PROFILE) or DEFAULT_PROFILE).strip() or DEFAULT_PROFILE
    where = Path(
        config_dir if config_dir is not None else (src.get(ENV_CONFIG_DIR) or DEFAULT_CONFIG_DIR)
    )

    data = _read_yaml(where / "default.yaml")
    if profile != DEFAULT_PROFILE:
        data = _deep_merge(data, _read_yaml(where / f"{profile}.yaml"))

    storage = _storage(data.get("storage"))
    storage = replace(
        storage,
        sqlite=replace(storage.sqlite, path=src.get(ENV_SQLITE_PATH) or storage.sqlite.path),
        qdrant=replace(storage.qdrant, url=src.get(ENV_QDRANT_URL) or storage.qdrant.url),
    )
    cache = CacheConfig(
        embed=EmbedCacheConfig(dir=src.get(ENV_EMBED_CACHE_DIR) or EmbedCacheConfig().dir)
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
            # ⚠ 这三个**允许为空**：空 ⇒ 不构造 reranker
            reranker_base_url=(src.get(ENV_RERANKER_BASE_URL) or "").strip(),
            reranker_api_key=(src.get(ENV_RERANKER_API_KEY) or "").strip(),
            reranker_model=(src.get(ENV_RERANKER_MODEL) or "").strip(),
            # ⚠ 同样允许为空：空 ⇒ 不写指标快照（缺省不是错误，见字段的注释）
            metrics_path=(src.get(ENV_METRICS_PATH) or "").strip(),
        )
    )
