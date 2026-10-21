"""`text-embedding-v4`：提交链路的 embedding 模型。

调用的是 DashScope（阿里云百炼）的 **OpenAI 兼容** `/embeddings`。形状与
[`qwen3_embedding.py`](./qwen3_embedding.py) 相同——差的是**这个模型自己的 API 事实**：

| 事实 | 值 | 不照做的后果 |
| --- | --- | --- |
| 模型名 | `text-embedding-v4` | —— |
| 单请求输入条数上界 | **10** | 基类默认 `batch_size = 64` ⇒ **第一批 `Add` 就全线 400** |

`base_url` 和 `api_key` 由装配层传入；本模块不读取环境变量。

不要让开发网关运行提交 profile：部分网关忽略 `model` 字段，而开发模型与提交模型
同为 1024 维，错误向量可能落到同一缓存坐标并静默复用。缓存坐标包含模型和维度身份。

本实现只产出 dense 向量；词法检索由 Qdrant 原生 `qdrant/bm25` 提供。
单条文本超过 8,192 token 会被服务端静默截断，修改渲染模板时必须遵守该上限。

维度来自真实返回，**不在此声明**——见 [`base.py`](./base.py)。
"""

from __future__ import annotations

import httpx

from tianximem.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_MODEL", "TextEmbeddingV4Embedder"]

#: 提交期模型名，必须与 `configs/submit.yaml` 的 `models.embedder` 一致；
#: 装配层按该值选择实现。
DEFAULT_MODEL: str = "text-embedding-v4"

#: 单次 `/embeddings` 请求至多 10 条输入——**提供方的硬限，不是调优参数**。
#: 超了报 `batch size is invalid, it should not be larger than 10`（HTTP 400）；
#: 调小只是多几次往返。⚠ 它**不能与模型名分开配**：那会造出"配得出、跑不通"的状态。
DEFAULT_BATCH_SIZE: int = 10


class TextEmbeddingV4Embedder(OpenAICompatEmbedder):
    """提交期 embedder（OpenAI 兼容端点）。

    ⚠ `batch_size` 的默认值是**这个模型的 API 事实**（≤10 条/请求），不是调优参数——
    调大不是变慢，是**报错**。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            batch_size=batch_size,
            # 基类的测试接缝（桩客户端）**透传**——否则"请求真的被切成 10 条一段"
            # 这件事只能打真端点才验得了。
            client=client,
        )
