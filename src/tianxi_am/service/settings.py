"""`service/` 的配置面。

> ⚠ **过渡位置**：本文件按 `src/tianxi_am/CLAUDE.md` 的"**不在本包内读环境变量**——
> 统一走 `common/` 的配置加载"本不该存在。`common/config.py` 是 Step 1 的 **③-d** 切片，
> 尚未实现；而 `uvicorn tianxi_am.service.app:...` 需要一个可运行的入口。
>
> ⇒ **环境绑定的唯一落点在这里**（`ServiceSettings.from_env`），③-d 时整体搬走。
> **不要在别处再读一次 `os.environ`。**

## 这里只放**密钥与路径**，不放阈值

`.env.example` 的原则：阈值 / 权重 / 消融开关属于 `configs/*.yaml`，不属于环境变量——
否则 §12.1 R1 的模型切换会变成改 shell 脚本。

所以本文件**不含** `prefetch_limit` / `weights` / `rrf_k` / `top_k` / 各类阈值：
那些用各自模块里已声明的**规格初值**（`make_hybrid_params()` 的默认参数就是 §7.3 的初值，
其中 `rrf_k` 是正确性常量而非旋钮）。③-d 再把它们接到 `configs/`。
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = ["ENV_KEYS", "ServiceSettings"]

#: 环境变量名（③-d 会连同这份清单一起搬进 `common/`）。
ENV_KEYS: tuple[str, ...] = (
    "TIANXI_SQLITE_PATH",
    "TIANXI_QDRANT_URL",
    "TIANXI_EMBED_CACHE_DIR",
    "AML_EMB_BASE_URL",
    "AML_EMB_API_KEY",
    "AML_EMB_MODEL",
)


@dataclass(frozen=True, slots=True)
class ServiceSettings:
    """服务启动所需的**路径与密钥**。

    `emb_*` 三项**没有默认值**：缺了就在启动时**响亮地失败**，
    而不是带着一个空 base_url 跑起来、等到第一次 Add 才炸。
    """

    emb_base_url: str
    emb_api_key: str
    sqlite_path: str = "var/tianxi.db"
    qdrant_url: str = "http://localhost:6333"
    embed_cache_dir: str = "var/embed_cache"
    emb_model: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServiceSettings:
        src = os.environ if env is None else env
        missing = [k for k in ("AML_EMB_BASE_URL", "AML_EMB_API_KEY") if not src.get(k)]
        if missing:
            raise ValueError(
                f"缺少环境变量：{', '.join(missing)}。"
                "服务需要 embedding 端点才能启动（见 .env.example）"
            )
        return cls(
            emb_base_url=src["AML_EMB_BASE_URL"],
            emb_api_key=src["AML_EMB_API_KEY"],
            emb_model=src.get("AML_EMB_MODEL", ""),
            sqlite_path=src.get("TIANXI_SQLITE_PATH") or "var/tianxi.db",
            qdrant_url=src.get("TIANXI_QDRANT_URL") or "http://localhost:6333",
            embed_cache_dir=src.get("TIANXI_EMBED_CACHE_DIR") or "var/embed_cache",
        )
