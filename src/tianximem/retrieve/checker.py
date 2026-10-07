"""§8 Evidence Checker —— **v1 恒返回"充足"的带日志空实现**（D13）。

## v1 的契约

它**不做门控**（下游 `agent/` 归 v2），但它**必须存在**，理由有两条：

1. **接缝先存在**——Step 4 到来时是**插入**，不是重写拓扑；
2. **要攒下反事实分布**——每轮记下"本来会不会触发多轮搜索"。**不记，Step 4 之前
   就永远无法用数据回答 A4「agent 到底值不值」**。

## 判据为什么用名次、不用分数

> **RRF 融合后的分数不是校准量。** Qdrant 官方明确警告不要把单路阈值用到根级
> `score_threshold`——"**照搬 dense-only 的阈值会静默截断结果**"。

所以判据只能建立在**两路各自的原始排名**上（`bm25-only` 与 `dense-only`）——
这也是 [`../retrieve/CLAUDE.md`](./CLAUDE.md) 说"唯一需要交接的信息是那两次分离查询的
原始排名"的原因。

## ⚠ 一条**刻意的未完成**（不是遗漏）

`decide()` 接受**可选**的 `arm_rankings`，而 **v1 的调用方不传**（省掉两次额外的 Qdrant
查询）⇒ `criterion_would_say` 记成 `None`。**判据本身在这里写全并测到了**（"判据与接入点
要在 v1 就写对"），缺的只是喂给它数据的那一步：**要真的攒到 A4 需要的那份分布，必须有人
传这两路排名**，而那需要 `store/qdrant_store.py` 暴露一个**单路查询**（现有的
`hybrid_search` 只做融合查询）。链条见 [`docs/roadmap.md`](../../../docs/roadmap.md) Step 1 末尾。
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
    """§8 判据里的四个初值：**自设阈值，可调，但调整要有 ablation 数据支撑**（§15）。

    它们与 `rrf_k` 不同类——`rrf_k` 是换算结果，错了就**静默出错**（值见 config-reference §4）。
    """

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
    """§8 的**名次一致性判据**。纯函数——判据本身不碰 Qdrant。

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
    """v1 恒为 True（不做门控）。"""

    criterion_would_say: bool | None
    """判据**本来**会说什么。`None` = 没跑两路分离查询（v1 的调用方不传排名）。"""

    reason: str


@runtime_checkable
class CheckerInstrument(Protocol):
    """判定记录的出口。

    聚合在 [`../observability/`](../observability/)（§14），本层只负责发——
    与 `rank/reranker.py` 的 `rerank_calls` / `rerank_degraded` 同一套"发射在各自层、
    聚合在 observability"的分工。
    """

    def record(self, decision: CheckerDecision) -> None: ...


class NullCheckerInstrument:
    """不记录——只有在明确不需要数据时才用。"""

    __slots__ = ()

    def record(self, decision: CheckerDecision) -> None:
        pass


@dataclass(slots=True)
class InMemoryCheckerInstrument:
    """进程内保留每一轮判定。**A4 的反事实分布就是从这串记录里数出来的。**"""

    decisions: list[CheckerDecision] = field(default_factory=list)

    def record(self, decision: CheckerDecision) -> None:
        self.decisions.append(decision)

    @property
    def would_trigger_rate(self) -> float | None:
        """ "本来会触发多轮搜索"的比例——**A4 需要的那个量**。

        没有任何一轮带上判据输入时返回 `None`（**不是 0**：那会把"没测"说成"不会触发"，
        而两者对 A4 的结论完全相反）。
        """
        known = [d for d in self.decisions if d.criterion_would_say is not None]
        if not known:
            return None
        return sum(1 for d in known if not d.criterion_would_say) / len(known)


@dataclass(slots=True)
class EvidenceChecker:
    """v1：恒返回"充足"，但**每轮判定都发出去**。"""

    instrument: CheckerInstrument = field(default_factory=NullCheckerInstrument)
    thresholds: CheckerThresholds = field(default_factory=CheckerThresholds)

    def decide(
        self,
        *,
        query: str,
        arm_rankings: tuple[Sequence[str], Sequence[str]] | None = None,
    ) -> CheckerDecision:
        """判定一次。

        `arm_rankings` = `(bm25-only 的 id 名次, dense-only 的 id 名次)`。
        **v1 的调用方不传**——见模块 docstring 的"刻意的未完成"。
        """
        would_say: bool | None = None
        if arm_rankings is not None:
            would_say = rank_consistency(arm_rankings[0], arm_rankings[1], self.thresholds)
            reason = f"v1 不做门控（恒返回充足）；判据本来判定为{'充足' if would_say else '不足'}"
        else:
            reason = "v1 不做门控（恒返回充足）；未跑两路分离查询，判据无输入"
        decision = CheckerDecision(enough=True, criterion_would_say=would_say, reason=reason)
        self.instrument.record(decision)
        return decision
