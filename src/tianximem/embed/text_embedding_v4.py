from __future__ import annotations

import httpx

from tianximem.embed.base import OpenAICompatEmbedder

__all__ = ["DEFAULT_BATCH_SIZE", "DEFAULT_MODEL", "TextEmbeddingV4Embedder"]

DEFAULT_MODEL: str = "text-embedding-v4"

DEFAULT_BATCH_SIZE: int = 10


class TextEmbeddingV4Embedder(OpenAICompatEmbedder):
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
            client=client,
        )
