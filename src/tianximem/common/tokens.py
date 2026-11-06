from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Final, Protocol

__all__ = [
    "DEFAULT_TOKENIZER",
    "EMBED_MAX_TOKENS",
    "LLM_MAX_INPUT_TOKENS",
    "MAX_INPUT_TOKENS",
    "O200kCounter",
    "TokenCounter",
    "TokenCounterError",
    "load_counter",
]

DEFAULT_TOKENIZER: Final[str] = "o200k_base"

MAX_INPUT_TOKENS: Final[int] = 117_760

LLM_MAX_INPUT_TOKENS: Final[int] = 110_000

EMBED_MAX_TOKENS: Final[int] = 8_192


class TokenCounterError(RuntimeError):
    pass


class TokenCounter(Protocol):
    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...

    def encode_ids(self, text: str) -> list[int]: ...

    def decode_ids(self, ids: Sequence[int]) -> str: ...


class O200kCounter:
    def __init__(self, encoding_name: str = DEFAULT_TOKENIZER) -> None:
        try:
            import tiktoken
        except ImportError as exc:  # pragma: no cover
            raise TokenCounterError(
                "tiktoken 没装上，无法数 token。它是 `[project] dependencies` 里的一项。"
            ) from exc
        try:
            self._encoding = tiktoken.get_encoding(encoding_name)
        except Exception as exc:  # noqa: BLE001
            raise TokenCounterError(
                f"加载分词器 {encoding_name!r} 失败：{type(exc).__name__}: {exc}\n"
                "  ⚠ `tiktoken` **第一次使用会联网下载** BPE 文件并缓存到本地；"
                "之后离线可用。离线机器上请先在有网时预热一次。"
            ) from exc
        self._name = encoding_name

    @property
    def name(self) -> str:
        return self._name

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode_ids(text))

    def encode_ids(self, text: str) -> list[int]:
        if not text:
            return []
        return self._encoding.encode(text, disallowed_special=())

    def decode_ids(self, ids: Sequence[int]) -> str:
        return self._encoding.decode(list(ids))


@lru_cache(maxsize=4)
def _cached(encoding_name: str) -> O200kCounter:
    return O200kCounter(encoding_name)


def load_counter(encoding_name: str = DEFAULT_TOKENIZER) -> TokenCounter:
    return _cached(encoding_name)
