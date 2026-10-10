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

from tianximem.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_MODEL", "Qwen3EmbeddingEmbedder"]

#: 开发期模型名。⚠ 与 `common/config.py` 的 `ModelsConfig.embedder` 默认值是**同一个值**
#: （那边是配置的默认，这里是直接构造这个类时的默认）。两处相等由
#: `tests/test_config.py::test_model_default_is_the_same_in_both_places` 钉住。
#:
#: ⚠ 这个 id 是**服务端的事**，随网关而变：2026-09-28 迁到 vLLM 直服时是
#: `qwen3-embedding-8b`，2026-10-09 回到自研封装（`memory.021130.xyz`）后是
#: **`Qwen/Qwen3-Embedding-8B`**（该网关实测**忽略** `model` 字段，两个名字都通——
#: 写端点声称的那个，声明与 `/v1/models` 才对得上）。
DEFAULT_MODEL: str = "Qwen/Qwen3-Embedding-8B"


class Qwen3EmbeddingEmbedder(OpenAICompatEmbedder):
    """远程 Qwen3-Embedding-8B（自建主网关 `memory.021130.xyz`，OpenAI 兼容 `/embeddings`）。

    ⚠ 它与 LLM 那一路**不同 host、不同 key**（`.env` 里 `AML_EMB_*` 与 `AML_*`
    是两套值，别混用）；同网关的 reranker 用**第三个**变量名（`TIANXIMEM_RERANKER_*`）。
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
