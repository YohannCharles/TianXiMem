"""Qwen3-Embedding-4B 开发期 embedding 适配器。

该适配器不进入提交链路；提交链路使用 `text-embedding-v4`。开发网关部署 4B 模型，
与提交模型同为 1024 维，因此模型不匹配不会由维度校验发现，必须通过配置和缓存坐标控制。

本实现只产出 dense 向量。词法检索由 Qdrant 原生 `qdrant/bm25` 提供。
查询默认原样送入模型；可选的 instruction 前缀位于
[`query_instruction.py`](./query_instruction.py)，并且必须在缓存之外应用。

维度来自真实响应，不在此声明。本模块不读取环境变量；端点和凭据由装配层传入。
"""

from __future__ import annotations

from tianximem.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_MODEL", "Qwen3EmbeddingEmbedder"]

#: Development model ID. It must match `ModelsConfig.embedder` and
#: `configs/default.yaml`; the embedding gateway validates it case-sensitively.
#: Copy the exact ID advertised by `/v1/models`.
DEFAULT_MODEL: str = "Qwen/Qwen3-Embedding-4B"


class Qwen3EmbeddingEmbedder(OpenAICompatEmbedder):
    """远程 Qwen3-Embedding-4B（自建网关 `memory2.021130.xyz`，OpenAI 兼容 `/embeddings`）。

    ⚠ 它与 LLM 那一路**不同 host、不同 key**（`.env` 里 `AML_EMB_*` 与 `AML_*`
    是两套值，别混用）；**reranker 在主网关**（`memory.021130.xyz`），用**第三个**变量名
    （`TIANXIMEM_RERANKER_*`）——embedding 与 reranker **使用不同 host**。
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            batch_size=batch_size,
        )
