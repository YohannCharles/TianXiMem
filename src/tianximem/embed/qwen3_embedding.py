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

#: 开发期模型名。⚠ 与 `common/config/model.py` 的 `ModelsConfig.embedder` 默认值是**同一个值**
#: （那边是配置的默认，这里是直接构造这个类时的默认）。两处相等由
#: `tests/test_config.py::test_model_default_is_the_same_in_both_places` 钉住。
#:
#: ⚠ 这个 id 是**服务端的事**，随网关而变。当前开发网关
#: `memory2.021130.xyz` 使用 **`Qwen/Qwen3-Embedding-4B`**。
#: ⚠ **两个网关对 `model` 的态度不同**：主网关（LLM/reranker）忽略它，填错不报错；
#: `memory2` **会校验**，而且**大小写敏感**（`...-4b` → 404 `The model ... does not exist.`）
#: ⇒ 写端点 `/v1/models` 声称的那个 id，逐字照抄。
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
