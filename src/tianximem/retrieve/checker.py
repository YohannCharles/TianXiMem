from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "CheckerDecision",
    "CheckerInstrument",
    "CheckerThresholds",
    "EvidenceChecker",
    "InMemoryCheckerInstrument",
    "NullCheckerInstrument",
    "rank_consistency",
]


@dataclass(frozen=True, slots=True)
class CheckerThresholds:
    top1_agree: bool = True
    topk_overlap: int = 5
    topk_overlap_min: int = 3
    bm25_top1_within: int = 3


def rank_consistency(
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    thresholds: CheckerThresholds | None = None,
) -> bool:
    t = CheckerThresholds() if thresholds is None else thresholds
    if not bm25_ids or not dense_ids:
        return False

    if t.top1_agree and bm25_ids[0] == dense_ids[0]:
        return True

    overlap = set(bm25_ids[: t.topk_overlap]) & set(dense_ids[: t.topk_overlap])
    if len(overlap) >= t.topk_overlap_min:
        return True

    head = dense_ids[: t.bm25_top1_within]
    return bm25_ids[0] in head


@dataclass(frozen=True, slots=True)
class CheckerDecision:
    enough: bool

    criterion_would_say: bool | None

    reason: str


@runtime_checkable
class CheckerInstrument(Protocol):
    def record(self, decision: CheckerDecision) -> None: ...


class NullCheckerInstrument:
    __slots__ = ()

    def record(self, decision: CheckerDecision) -> None:
        pass


@dataclass(slots=True)
class InMemoryCheckerInstrument:
    decisions: list[CheckerDecision] = field(default_factory=list)

    def record(self, decision: CheckerDecision) -> None:
        self.decisions.append(decision)

    @property
    def would_trigger_rate(self) -> float | None:
        known = [d for d in self.decisions if d.criterion_would_say is not None]
        if not known:
            return None
        return sum(1 for d in known if not d.criterion_would_say) / len(known)


@dataclass(slots=True)
class EvidenceChecker:
    instrument: CheckerInstrument = field(default_factory=NullCheckerInstrument)
    thresholds: CheckerThresholds = field(default_factory=CheckerThresholds)

    def decide(
        self,
        *,
        query: str,
        arm_rankings: tuple[Sequence[str], Sequence[str]] | None = None,
    ) -> CheckerDecision:
        would_say: bool | None = None
        if arm_rankings is not None:
            would_say = rank_consistency(arm_rankings[0], arm_rankings[1], self.thresholds)
            reason = "名次一致性判据本来判定为充足" if would_say else "名次一致性判据本来判定为不足"
        else:
            reason = "未跑两路分离查询"
        decision = CheckerDecision(enough=True, criterion_would_say=would_say, reason=reason)
        self.instrument.record(decision)
        return decision
