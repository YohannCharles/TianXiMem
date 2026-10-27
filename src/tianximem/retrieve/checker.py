"""Rank-consistency evidence checker.

The checker evaluates agreement between BM25-only and dense-only rankings. Fused RRF
scores are not calibrated relevance probabilities and are not used as thresholds.

The current Search path keeps checker decisions observational: the query router owns
route selection. When arm rankings are not supplied, `criterion_would_say` is `None`
rather than guessing a decision.
"""

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
    """名次一致性判据的四个默认值；调整前需要评测数据支撑。"""

    #: 两路 top-1 相同 ⇒ 足够
    top1_agree: bool = True
    #: 比的是**两路各取前 `topk_overlap` 条**的重叠数（窗口大小，不是阈值）
    topk_overlap: int = 5
    #: 上一条真正被判据比较的那个数：重叠 ≥ 它 ⇒ 足够
    topk_overlap_min: int = 3
    #: 或：bm25 的 top-1 在 dense 结果里位于前几名 ⇒ 足够
    bm25_top1_within: int = 3


def rank_consistency(
    bm25_ids: Sequence[str],
    dense_ids: Sequence[str],
    thresholds: CheckerThresholds | None = None,
) -> bool:
    """名次一致性判据。纯函数，不访问 Qdrant。

    ```text
    足够（Direct Return）：
      两路 top1 相同
      或 两路 top-5 重叠 ≥ 3
      或 bm25 top1 的名次在 dense 结果中位于前 3
    否则：
      → Agentic Search
    ```

    两路都为空时返回 False（没有候选谈不上"证据充足"）。
    """
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
    """一轮判定。**它就是"必须记录"的那个东西。**"""

    enough: bool
    """Whether evidence is considered sufficient. The current pipeline keeps this observational."""

    criterion_would_say: bool | None
    """The rank-consistency result, or `None` when arm rankings were not supplied."""

    reason: str


@runtime_checkable
class CheckerInstrument(Protocol):
    """判定记录的出口。

    各业务层发射判定，聚合由 [`../observability/`](../observability/) 负责。
    """

    def record(self, decision: CheckerDecision) -> None: ...


class NullCheckerInstrument:
    """不记录——只有在明确不需要数据时才用。"""

    __slots__ = ()

    def record(self, decision: CheckerDecision) -> None:
        pass


@dataclass(slots=True)
class InMemoryCheckerInstrument:
    """Keep checker decisions in memory for diagnostics and experiment records."""

    decisions: list[CheckerDecision] = field(default_factory=list)

    def record(self, decision: CheckerDecision) -> None:
        self.decisions.append(decision)

    @property
    def would_trigger_rate(self) -> float | None:
        """Fraction of decisions that would trigger the multi-hop path.

        Return `None` when no decision supplied both ranking arms. `None` means
        "not measured", while `0.0` means "measured and never triggered".
        """
        known = [d for d in self.decisions if d.criterion_would_say is not None]
        if not known:
            return None
        return sum(1 for d in known if not d.criterion_would_say) / len(known)


@dataclass(slots=True)
class EvidenceChecker:
    """Record the rank-consistency result while leaving route selection observational."""

    instrument: CheckerInstrument = field(default_factory=NullCheckerInstrument)
    thresholds: CheckerThresholds = field(default_factory=CheckerThresholds)

    def decide(
        self,
        *,
        query: str,
        arm_rankings: tuple[Sequence[str], Sequence[str]] | None = None,
    ) -> CheckerDecision:
        """判定一次。

        `arm_rankings` is `(BM25-only IDs, dense-only IDs)`.
        Callers may omit it when they do not need the counterfactual diagnostic.
        """
        would_say: bool | None = None
        if arm_rankings is not None:
            would_say = rank_consistency(arm_rankings[0], arm_rankings[1], self.thresholds)
            reason = "名次一致性判据本来判定为充足" if would_say else "名次一致性判据本来判定为不足"
        else:
            reason = "未跑两路分离查询"
        decision = CheckerDecision(enough=True, criterion_would_say=would_say, reason=reason)
        self.instrument.record(decision)
        return decision
