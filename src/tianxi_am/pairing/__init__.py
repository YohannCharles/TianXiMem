"""记忆块组合（D24）。

**一次 `Add` = 组合的唯一边界**：`Add` 内连续同 role 合并、相邻 UserBlock + 非UserBlock
配对、配不上的独立成块；**不同 `Add` 永不拼接**。

组合规则见 [`pairing.py`](./pairing.py)（纯函数、无跨 Add 状态），落库见
[`apply.py`](./apply.py)。
"""

from tianxi_am.pairing.apply import AddBatch, ApplyBatchResult, apply_batch
from tianxi_am.pairing.pairing import (
    MemoryBlock,
    Message,
    RoleBlock,
    compose_memory_blocks,
    encode_answer,
    is_user,
    join_question,
    parse_chunk_ordinal,
)

__all__ = [
    "AddBatch",
    "ApplyBatchResult",
    "MemoryBlock",
    "Message",
    "RoleBlock",
    "apply_batch",
    "compose_memory_blocks",
    "encode_answer",
    "is_user",
    "join_question",
    "parse_chunk_ordinal",
]
