from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MemoryBlock",
    "Message",
    "RoleBlock",
    "compose_memory_blocks",
    "encode_answer",
    "is_user",
    "join_question",
    "link_blocks",
]


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    timestamp: int | None = None

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("role 不得为空")
        object.__setattr__(self, "content", self.content.strip())


def is_user(message: Message) -> bool:
    return message.role == "user"


def encode_answer(messages: Sequence[Message]) -> str:
    return "\n".join(f"[{m.role}] {m.content}" if m.content else f"[{m.role}]" for m in messages)


def join_question(messages: Sequence[Message]) -> str:
    return "\n".join(m.content for m in messages)


@dataclass(frozen=True, slots=True)
class RoleBlock:
    role: str
    messages: tuple[Message, ...]
    source_idxs: tuple[int, ...]

    @property
    def is_user(self) -> bool:
        return self.role == "user"


@dataclass(frozen=True, slots=True)
class MemoryBlock:
    question: str | None

    answer: str | None

    event_time: int | None

    messages: tuple[Message, ...]

    source_idxs: tuple[int, ...]

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(m.role for m in self.messages)

    @property
    def timestamps(self) -> tuple[int | None, ...]:
        return tuple(m.timestamp for m in self.messages)

    @property
    def is_paired(self) -> bool:
        return self.question is not None and self.answer is not None


def compose_memory_blocks(messages: Sequence[Message]) -> tuple[MemoryBlock, ...]:
    role_blocks = _merge_consecutive_roles(messages)
    blocks: list[MemoryBlock] = []
    i = 0
    while i < len(role_blocks):
        if not role_blocks[i].is_user:
            blocks.append(_block_from(role_blocks[i]))
            i += 1
            continue
        end = i + 1
        while end < len(role_blocks) and not role_blocks[end].is_user:
            end += 1
        if end > i + 1:
            blocks.append(_block_from(*role_blocks[i:end]))
        else:
            blocks.append(_block_from(role_blocks[i]))
        i = end
    return tuple(blocks)


def link_blocks(
    blocks: Sequence[MemoryBlock],
) -> tuple[tuple[int | None, int | None], ...]:
    links: list[tuple[int | None, int | None]] = [(None, None)] * len(blocks)
    complete = [index for index, block in enumerate(blocks) if block.is_paired]
    for position, index in enumerate(complete):
        prev_index = complete[position - 1] if position else None
        next_index = complete[position + 1] if position + 1 < len(complete) else None
        links[index] = (prev_index, next_index)
    return tuple(links)


def _merge_consecutive_roles(messages: Sequence[Message]) -> tuple[RoleBlock, ...]:
    blocks: list[RoleBlock] = []
    for idx, message in enumerate(messages):
        if blocks and blocks[-1].role == message.role:
            last = blocks.pop()
            blocks.append(
                RoleBlock(
                    role=last.role,
                    messages=(*last.messages, message),
                    source_idxs=(*last.source_idxs, idx),
                )
            )
        else:
            blocks.append(RoleBlock(role=message.role, messages=(message,), source_idxs=(idx,)))
    return tuple(blocks)


def _block_from(*role_blocks: RoleBlock) -> MemoryBlock:
    messages = tuple(m for rb in role_blocks for m in rb.messages)
    source_idxs = tuple(i for rb in role_blocks for i in rb.source_idxs)
    user_messages = [m for rb in role_blocks if rb.is_user for m in rb.messages]
    other_messages = [m for rb in role_blocks if not rb.is_user for m in rb.messages]
    return MemoryBlock(
        question=join_question(user_messages) or None,
        answer=encode_answer(other_messages) or None,
        event_time=messages[0].timestamp if messages else None,
        messages=messages,
        source_idxs=source_idxs,
    )
