"""Qwen3-Embedding-8B —— **开发期替代模型**（§12.1 R1，团队主动接受的偏离）。

**不进提交链路**（§2.3 规定 embedding 只能用 `text-embedding-v4`）。它存在只是为控 API 成本，
Step 5 之后停止使用。

⚠ **本实现只产出 Dense 向量。** 不产出 sparse、不产出 ColBERT——词法那一路走 Qdrant 原生
`qdrant/bm25`（§7.1），理由见 [`CLAUDE.md`](./CLAUDE.md)：学出来的稀疏权重既不是 BM25，
而 `text-embedding-v4` 也不提供该能力 ⇒ 任何依赖它的代码都是**死重**。

⚠ **查询侧照原样送**（§7.2）：v1 不做任何查询改写。**模型卡另有推荐用法**——查询侧加
`Instruct: …\nQuery:…` 前缀、文档侧不加，不加会让检索掉约 1%–5%。那一层在
[`query_instruction.py`](./query_instruction.py)，**默认关着**，要不要开是待定的规格问题。
**唯一不可违反的顺序：前缀必须加在缓存之上**（该模块 docstring 讲了为什么）。

维度来自真实返回，**不在此声明**——见 `base.py` 的说明。
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from tianxi_am.embed.base import OpenAICompatEmbedder

__all__ = ["Qwen3EmbeddingEmbedder"]

DEFAULT_MODEL: str = "Qwen/Qwen3-Embedding-8B"

ENV_BASE_URL: str = "AML_EMB_BASE_URL"
ENV_API_KEY: str = "AML_EMB_API_KEY"
ENV_MODEL: str = "AML_EMB_MODEL"


class Qwen3EmbeddingEmbedder(OpenAICompatEmbedder):
    """远程 Qwen3-Embedding-8B（自建网关，OpenAI 兼容 `/embeddings`）。

    ⚠ 网关与 LLM 的那一路**同 host 但 key 不同**（`.env` 里 `AML_EMB_API_KEY`
    与 `AML_API_KEY` 是两个值，别混用）。
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

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Qwen3EmbeddingEmbedder:
        """从环境变量构造。

        ⚠ **这是过渡措施**：正经的配置加载属 `common/config.py`（§15：不得硬编码），
        本切片里 `common/config.py` 还没实现，所以给一个显式的入口而不是散落的
        `os.environ[...]`。缺变量时**响亮地失败**，不静默退回默认值。
        """
        src = os.environ if env is None else env
        missing = [k for k in (ENV_BASE_URL, ENV_API_KEY) if not src.get(k)]
        if missing:
            raise ValueError(f"缺少环境变量：{', '.join(missing)}（见 .env.example）")
        return cls(
            base_url=src[ENV_BASE_URL],
            api_key=src[ENV_API_KEY],
            model=src.get(ENV_MODEL) or DEFAULT_MODEL,
        )
