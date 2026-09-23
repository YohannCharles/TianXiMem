"""BGE-M3 —— **开发期替代模型**（§12.1 R1，团队主动接受的偏离）。

**不进提交链路**（§2.3 规定 embedding 只能用 `text-embedding-v4`）。它存在只是为控 API 成本，
Step 5 之后应停止使用。

⚠ **本实现只产出 Dense 向量。** 不产出 sparse、不产出 ColBERT——理由见
[`CLAUDE.md`](./CLAUDE.md)：词法那一路走 Qdrant 原生 `qdrant/bm25`（§7.1），
**不使用 BGE-M3 的 sparse 输出**（它的 sparse 是学出来的词法权重而非 BM25，
而且提交时的 `text-embedding-v4` 不提供该能力 ⇒ 任何依赖它的代码在提交时都是**死重**）。

维度来自真实返回（1024），不在此声明——见 `base.py` 的说明。
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from tianxi_am.embed.base import OpenAICompatEmbedder

__all__ = ["BGEM3Embedder"]

DEFAULT_MODEL: str = "BAAI/bge-m3"

ENV_BASE_URL: str = "AML_EMB_BASE_URL"
ENV_API_KEY: str = "AML_EMB_API_KEY"
ENV_MODEL: str = "AML_EMB_MODEL"


class BGEM3Embedder(OpenAICompatEmbedder):
    """远程 BGE-M3（自建网关，OpenAI 兼容 `/embeddings`）。

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
    def from_env(cls, env: Mapping[str, str] | None = None) -> BGEM3Embedder:
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
