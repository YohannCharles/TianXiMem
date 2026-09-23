"""配对与批次续接（§6.2 / §6.5）。

**全项目逻辑最绕的一层，也是最容易静默出错的一层**——出错的表现是
"某些记忆永远检索不到"，而**不会有任何报错**。
"""

from tianxi_am.pairing.continuation import AddBatch, ApplyBatchResult, apply_batch
from tianxi_am.pairing.instrument import (
    InMemoryPendingInstrument,
    NullPendingInstrument,
    PendingCounters,
    PendingInstrument,
)
from tianxi_am.pairing.pairing import (
    BatchLimits,
    BatchPlan,
    Message,
    NewPairDraft,
    ResumeActions,
    encode_answer,
    is_user,
    plan_batch,
)

__all__ = [
    "AddBatch",
    "ApplyBatchResult",
    "BatchLimits",
    "BatchPlan",
    "InMemoryPendingInstrument",
    "Message",
    "NewPairDraft",
    "NullPendingInstrument",
    "PendingCounters",
    "PendingInstrument",
    "ResumeActions",
    "apply_batch",
    "encode_answer",
    "is_user",
    "plan_batch",
]
