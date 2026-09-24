"""HTTP 层：`POST /add`、`POST /search`、`GET /health` 探活（§2.1、§2.2、§15）。

**本层只做三件事**：

1. **把契约形状钉死**——[`schemas.py`](./schemas.py) 是 §2.1 的字面翻译；
2. **串行化 Add**——按 `(user_id, session_id)` 加锁，保护 `pairing/` 的"读位置 → 写位置"；
3. **保持可重试**——内部异常**不吞、不返回部分成功**（[`errors.py`](./errors.py)）。

**它不做检索、不做配对、不碰存储**——那些逻辑全在下面各层；
把"按什么顺序调"放进了 [`pipeline.py`](./pipeline.py)。
"""

from tianxi_am.common.config import AppConfig, ConfigError, assert_single_process, load_config
from tianxi_am.service.app import Services, build_services, create_app, create_app_from_env
from tianxi_am.service.locks import SessionLocks, SessionLockTimeout
from tianxi_am.service.pipeline import AddOutcome, AddPipeline, SearchPipeline
from tianxi_am.service.schemas import (
    AddMessage,
    AddRequest,
    AddResponse,
    SearchRequest,
    SearchResponse,
    SearchResultItem,
)

__all__ = [
    "AddMessage",
    "AddOutcome",
    "AddPipeline",
    "AddRequest",
    "AddResponse",
    "AppConfig",
    "ConfigError",
    "SearchPipeline",
    "SearchRequest",
    "SearchResponse",
    "SearchResultItem",
    "Services",
    "SessionLockTimeout",
    "SessionLocks",
    "assert_single_process",
    "build_services",
    "create_app",
    "create_app_from_env",
    "load_config",
]
