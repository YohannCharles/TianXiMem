"""`text-embedding-v4` —— **提交链路的唯一 embedding 模型**（§2.3 规定，不可协商）。

调用的是 DashScope（阿里云百炼）的 **OpenAI 兼容** `/embeddings`。形状与
[`qwen3_embedding.py`](./qwen3_embedding.py) 相同——差的是**这个模型自己的 API 事实**：

| 事实 | 值 | 不照做的后果 |
| --- | --- | --- |
| 模型名 | `text-embedding-v4` | —— |
| 单请求输入条数上界 | **10** | 基类默认 `batch_size = 64` ⇒ **第一批 `Add` 就全线 400** |

⚠ **`base_url` / `api_key` 由装配点传进来**（本模块不读环境变量，③-d）：它们来自 `.env` 的
`AML_EMB_*`——**那两个变量是"embedding 端点"这个位置的名字，值随部署而变**
（开发期 = 自建主网关，提交期 = DashScope）。

> ⛔ **别拿开发网关跑提交 profile**：它与 reranker 那台一样**忽略 `model` 字段**，而开发期的
> Qwen3-Embedding-8B 与本模型**同为 1024 维** ⇒ 向量会挂到 `text-embedding-v4` 这个**缓存坐标**下，
> **不报错**，之后真的换端点时会**全部命中缓存**。机制与处置见 `docs/decisions.md` **D27**。

⚠ **本实现只产出 Dense 向量**（v4 也不提供 sparse / ColBERT）：词法那一路走 Qdrant 原生
`qdrant/bm25`（§7.1），**任何依赖多向量能力的代码都是死重**（§2.3）。

⚠ 单条文本超过 **8,192 token** 会被服务端**静默截断**。v1 的正文是一对一块、远小于它，
但**改渲染模板时要记得这个上界**。

维度来自真实返回，**不在此声明**——见 [`base.py`](./base.py)。
"""

from __future__ import annotations

import httpx

from tianxi_am.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_MODEL", "TextEmbeddingV4Embedder"]

#: 提交期模型名。**§2.3 规定只能用这一个**，且与 `configs/submit.yaml` 的
#: `models.embedder` 是同一个值——装配按这个值挑实现（`service/app.py` 的 `build_embedder`）。
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
