"""§6.5 的三个 `pending` 计数器。

这三个计数器是**判断 AML 切分是否频繁打断 QA 对的唯一手段**（pairing/CLAUDE.md），
持续偏高说明配对规则需要调整。它们属于 `observability/` 的聚合，但在本层发射。

### ⚠ 一个必须说清的局限：`pending_orphaned` 有两种来源，本层**无法区分**

| 来源 | 说明 |
| --- | --- |
| **(i) 真·残缺** | session 就此结束，对里确实少了一半。**非零是正常的** |
| **(ii) 误判残留** | 本批恰好命中上限、最后一对**其实已完整**。**非零说明续接漏了 3b——是 bug** |

pairing/CLAUDE.md 要求把两者分开，否则会把 bug 当成 AML 的切分行为。**但单批处理拿不到
区分它们所需的信息**：服务端不知道 AML 为什么在这里切断（S2，"Adapter 计数的词"官方
从未定义）；而 (ii) 的判据是"本应被 3b 关掉却没关"——3b 一旦正确执行，那一对就走
`pending_completed` 而不是 `pending_orphaned`，**根本不进这个计数器**。

**因此本层给 `pending_orphaned` 的定义是保守的、可实现的**：

> **session 结束判定时**（本批两限都未命中）那个**仍然处于 `pending` 的既有对**
> ——它在被 3d 标 `complete` 之前计入。

这个判据可从单批信息算出，**非零确实说明"有一个 pending 活到了 session 末"**，
但它**不区分 (i) 与 (ii)**；要区分需要在 `docs/open-questions.md` 里单列一条
（见 `docs/decisions.md` D16 之后的记录）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "PendingCounters",
    "PendingInstrument",
    "InMemoryPendingInstrument",
    "NullPendingInstrument",
]


@dataclass(slots=True)
class PendingCounters:
    """三个计数器的值。字段名与 pairing/CLAUDE.md 一致。"""

    pending_created: int = 0
    pending_completed: int = 0
    pending_orphaned: int = 0


@runtime_checkable
class PendingInstrument(Protocol):
    """发射口。真正的聚合在 `observability/`，本层只负责发。"""

    def record_created(self, count: int = 1) -> None: ...
    def record_completed(self, count: int = 1) -> None: ...
    def record_orphaned(self, count: int = 1) -> None: ...


@dataclass(slots=True)
class InMemoryPendingInstrument:
    """进程内累加。用于单元测试与单进程服务（§15 要求 `--workers 1`）。"""

    counters: PendingCounters = field(default_factory=PendingCounters)

    def record_created(self, count: int = 1) -> None:
        self.counters.pending_created += count

    def record_completed(self, count: int = 1) -> None:
        self.counters.pending_completed += count

    def record_orphaned(self, count: int = 1) -> None:
        self.counters.pending_orphaned += count


class NullPendingInstrument:
    """什么都不做——用于不关心计数的调用点，避免到处写 `if instrument is not None`。"""

    __slots__ = ()

    def record_created(self, count: int = 1) -> None:
        pass

    def record_completed(self, count: int = 1) -> None:
        pass

    def record_orphaned(self, count: int = 1) -> None:
        pass
