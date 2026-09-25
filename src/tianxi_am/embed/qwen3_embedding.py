"""Qwen3-Embedding-8B —— **开发期替代模型**（§12.1 R1，团队主动接受的偏离）。

**不进提交链路**（§2.3 规定 embedding 只能用 `text-embedding-v4`）。它存在只是为控 API 成本，
Step 5 之后停止使用。

⚠ **本实现只产出 Dense 向量。** 不产出 sparse、不产出 ColBERT——词法那一路走 Qdrant 原生
`qdrant/bm25`（§7.1），而 `text-embedding-v4` 也不提供该能力 ⇒ 任何依赖它的代码都是**死重**
（理由见 [`CLAUDE.md`](./CLAUDE.md)）。

⚠ **查询侧照原样送**（§7.2）：v1 不做任何查询改写。模型卡推荐的那层 instruction 前缀
（查询侧加、文档侧不加）在 [`query_instruction.py`](./query_instruction.py)，**默认关着**，
要不要开是待定的规格问题；而**唯一不可违反的顺序：前缀必须加在缓存之上**。

维度来自真实返回，**不在此声明**——见 `base.py` 的说明。

⚠ **本模块不读环境变量**：`base_url` / `api_key` / `model` 都由装配点传进来
（`common/config.py` 是全包唯一读环境变量的地方，③-d）。
"""

from __future__ import annotations

from tianxi_am.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_MODEL", "Qwen3EmbeddingEmbedder"]

#: 开发期模型名。⚠ 与 `common/config.py` 的 `ModelsConfig.embedder` 默认值是**同一个值**
#: （那边是配置的默认，这里是直接构造这个类时的默认）。两处相等由
#: `tests/test_config.py::test_model_default_is_the_same_in_both_places` 钉住。
DEFAULT_MODEL: str = "Qwen/Qwen3-Embedding-8B"


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
