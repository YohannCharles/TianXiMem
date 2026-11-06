from __future__ import annotations

from tianximem.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_MODEL", "Qwen3EmbeddingEmbedder"]

DEFAULT_MODEL: str = "Qwen/Qwen3-Embedding-4B"


class Qwen3EmbeddingEmbedder(OpenAICompatEmbedder):
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
