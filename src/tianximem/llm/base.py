from __future__ import annotations

from typing import Protocol, runtime_checkable

__all__ = ["DEFAULT_MAX_OUTPUT_TOKENS", "LlmClient", "LlmUnavailable"]

DEFAULT_MAX_OUTPUT_TOKENS: int = 8


class LlmUnavailable(RuntimeError):
    pass


@runtime_checkable
class LlmClient(Protocol):
    @property
    def name(self) -> str: ...

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        temperature: float = 0.0,
    ) -> str: ...
