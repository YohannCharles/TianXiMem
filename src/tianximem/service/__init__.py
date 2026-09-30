"""HTTP 层：`POST /add`、`POST /search`、`GET /health` 探活（§2.1、§2.2、§15）。

**本层只做两件事**：钉死契约形状（[`schemas.py`](./schemas.py) 是 §2.1 的字面翻译）、
保持可重试（内部异常**不吞、不返回部分成功**，见 [`errors.py`](./errors.py)）。

> ⚠ **它不串行化 Add**：位置是**请求的纯函数**（`(request_id, local_index)`，**D28**；
> `request_id` 是 opaque string、**不解析**）⇒ 同 session 并发与乱序到达都安全。
> 并发唯一的排队点是 `store/` 层的 `BEGIN IMMEDIATE`。

**它不做检索、不做配对、不碰存储**——那些逻辑全在下面各层；**"按什么顺序调"在
[`pipeline.py`](./pipeline.py)**。
"""

from tianximem.common.config import AppConfig, ConfigError, assert_single_process, load_config
from tianximem.service.app import Services, build_services, create_app, create_app_from_env
from tianximem.service.pipeline import AddOutcome, AddPipeline, SearchPipeline
from tianximem.service.schemas import (
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
    "assert_single_process",
    "build_services",
    "create_app",
    "create_app_from_env",
    "load_config",
]
