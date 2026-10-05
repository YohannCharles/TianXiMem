"""记忆块组合与落库（D24 / **D28**）。

**一次 `Add` = 组合与邻接的唯一边界**：`Add` 内连续同 role 合并、相邻 UserBlock +
非UserBlock 配对、配不上的独立成块；邻接（`prev` / `next`）也**只在这一批内**相连。
**不同 `Add` 永不拼接、也不建立邻接**——`request_id` 是 opaque string，
跨 Add 不存在可信的全局顺序（D28）。

组合规则见 [`pairing.py`](./pairing.py)（纯函数、无跨 Add 状态），落库见
[`apply.py`](./apply.py)。
"""

from tianximem.pairing.apply import (
    AddBatch,
    ApplyBatchResult,
    PayloadMismatchError,
    apply_batch,
    payload_fingerprint,
)
from tianximem.pairing.pairing import (
    MemoryBlock,
    Message,
    RoleBlock,
    compose_memory_blocks,
    encode_answer,
    is_user,
    join_question,
    link_blocks,
)

__all__ = [
    "AddBatch",
    "ApplyBatchResult",
    "MemoryBlock",
    "Message",
    "PayloadMismatchError",
    "RoleBlock",
    "apply_batch",
    "compose_memory_blocks",
    "encode_answer",
    "is_user",
    "join_question",
    "link_blocks",
    "payload_fingerprint",
]
