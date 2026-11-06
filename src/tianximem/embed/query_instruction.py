from __future__ import annotations

from collections.abc import Sequence

from tianximem.embed.base import Embedder

__all__ = [
    "DEFAULT_TASK",
    "DEFAULT_TEMPLATE",
    "QueryInstructionEmbedder",
    "apply_query_instruction",
]

DEFAULT_TEMPLATE: str = "Instruct: {task}\nQuery:{query}"

DEFAULT_TASK: str = "Given a web search query, retrieve relevant passages that answer the query"


def apply_query_instruction(query: str, instruction: str, template: str = DEFAULT_TEMPLATE) -> str:
    if not instruction:
        return query
    return template.format(task=instruction, query=query)


class QueryInstructionEmbedder:
    def __init__(
        self,
        inner: Embedder,
        *,
        instruction: str = "",
        template: str = DEFAULT_TEMPLATE,
    ) -> None:
        self._inner = inner
        self._instruction = instruction
        self._template = template

    @property
    def inner(self) -> Embedder:
        return self._inner

    @property
    def instruction(self) -> str:
        return self._instruction

    @property
    def dim(self) -> int:
        return self._inner.dim

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        return self._inner.encode(texts)

    def encode_query(self, texts: Sequence[str]) -> list[list[float]]:
        prefixed = [
            apply_query_instruction(text, self._instruction, self._template) for text in texts
        ]
        encode_query = getattr(self._inner, "encode_query", None)
        if callable(encode_query):
            return encode_query(prefixed)
        return self._inner.encode(prefixed)
