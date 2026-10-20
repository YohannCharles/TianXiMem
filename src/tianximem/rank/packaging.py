"""Context Packaging：段级打包与 token 预算。

```text
ContextSegment[]（已按 best_rank 升序）
  → 按 best_rank 依次加入，直到撞上 top_k 或 token 预算
  → data[]（id = 锚点，content = 整段，created_at = 锚点的日粒度，score = 1/(final_rank+1)）
```

> **分组顺序（组内 `local_index` / 组间 `best_rank`）在 [`neighbor.py`](./neighbor.py) 定**，
> 这里只按给定的顺序消费。**不要把这个文件当成整条 Search 链**（[`CLAUDE.md`](./CLAUDE.md)）。

## 三条本文件独有、且**必须**守住的东西

**1. 本函数是最终数量的守门人。** `len(data) <= top_k` 精确成立。
⚠ 这里的 `top_k` 约束的是**段数**，不是中间 raw memory 数——扩窗会把 raw 数抬到
远大于 `top_k`，而合并又会把它降回来（见 `neighbor.py` 的模块 docstring）。

**2. 段是**原子单位**，装不下就停。** 不截半个段、不拆回单条、不跳过当前段再塞后面的。
理由：段是"一段连续对话"，**截断点必须落在段边界上**——否则模型读到的上下文缺了一环，
而 `content` 本身看不出这个缺口。

**3. `content` 不通过参数注入。** [`common/render`](../common/render.py) 是**唯一实现**
（不变式 I1）：同一份渲染既是 embedding 的输入、也是返回给 AML 的 `content`。
⇒ 所以这里**没有 `renderer` 参数**——不是省事，是让漂移**不可能发生**。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tianximem.common.render import SEGMENT_SEP, day_granularity
from tianximem.common.tokens import TokenCounter
from tianximem.rank.neighbor import ContextSegment

__all__ = [
    "PackagedResponse",
    "ResponseItem",
    "day_granularity",
    "package",
    "placeholder_score",
]

#: `created_at` 固定使用 UTC 日粒度，不提供时区配置，保证同一数据在不同机器上一致。
#:
#: `event_time` 由 harness 按 UTC 合成；没有时区的时间按 floating calendar time 处理。
#: 日期生成统一使用 [`common/render.py`](../common/render.py) 的 `day_granularity`；
#: 本模块只重新导出该名称，保持 `packaging.day_granularity` 路径可用。
UTC_ONLY: Final[str] = "UTC"


def placeholder_score(rank: int) -> float:
    """`1/(rank+1)`：按 0-based 输出位置单调递减的占位分。

    不返回原始 RRF 或 reranker 分数；这些分数不是校准量。占位分即使被 AML 再次排序，
    也保持当前输出顺序。

    ⚠ 这里的 `rank` 是**最终输出位置**（段级），不是 rerank 名次。扩窗与合并后，
    这两个位置含义不同。
    """
    if rank < 0:
        raise ValueError(f"rank 不得为负：{rank}")
    return 1.0 / (rank + 1.0)


@dataclass(frozen=True, slots=True)
class ResponseItem:
    """`data[]` 的一项，只包含契约规定的四个字段。

    ⚠ 这里**没有** `session_id` / `request_id` / `local_index` /
    `source_memory_ids` 之类的内部字段。
    多余的键通常被忽略，但**没有理由冒这个险**。
    """

    id: str
    content: str
    created_at: str
    score: float


@dataclass(frozen=True, slots=True)
class PackagedResponse:
    """打包结果。**除 `items` 外都是诊断量**——它们不进响应，但要能被看见。

    | 字段 | 含义 |
    | --- | --- |
    | `dropped_missing` | **真源里查不到正文**的候选条数。恒应为 0；非 0 = 索引与真源脱钩 |
    | `considered_segments` | 合并后的段总数（= 排序后、截断前的候选段数） |
    | `truncated_by_top_k` | 因 `top_k` 而截断 |
    | `truncated_by_budget` | 因 token 预算而提前停止（**不是**截断，是"装不下就停"） |
    | `output_tokens` | 最终拼接字符串的真实 token 数 |
    """

    items: tuple[ResponseItem, ...]
    dropped_missing: int = 0
    considered_segments: int = 0
    truncated_by_top_k: bool = False
    truncated_by_budget: bool = False
    output_tokens: int = 0

    @property
    def count(self) -> int:
        return len(self.items)


def _fit_by_budget(
    segments: Sequence[ContextSegment],
    *,
    counter: TokenCounter,
    max_tokens: int,
) -> tuple[list[ContextSegment], bool]:
    """按 `best_rank` 顺序依次加入，直到**下一个段装不下** ⇒ 停。

    ### 决策用的是"上界"，报告的是"真值"

    决策必须便宜（不能每加一个段就把整段文本重新数一遍，那是 O(S²) 的字符量），
    所以这里用 `已用 = Σ 各段的 token + (段数 - 1) × 连接符的 token`。

    ⚠ **它是一个上界，不是恒等式**：BPE 的合并可以跨越拼接边界，所以
    `count(a) + count(sep) + count(b) ≥ count(a + sep + b)`。用上界做预算**只会少装、
    不会超装**——这是预算该有的偏向（超装的后果是 AML 按前缀截断，**排在后面的证据整段作废**）。

    而**报告**出去的那个 `output_tokens` 是最终字符串的**真值**（见 `package`）。
    """
    if max_tokens <= 0:
        return [], bool(segments)

    separator_tokens = counter.count(SEGMENT_SEP)
    chosen: list[ContextSegment] = []
    used = 0
    for segment in segments:
        extra = segment.token_count + (separator_tokens if chosen else 0)
        if used + extra > max_tokens:
            return chosen, True
        used += extra
        chosen.append(segment)
    return chosen, False


def package(
    segments: Sequence[ContextSegment],
    *,
    top_k: int,
    counter: TokenCounter,
    max_tokens: int,
    dropped_missing: int = 0,
) -> PackagedResponse:
    """把**已排序**的段打包成 `data[]`。**本函数是最终数量与预算的守门人。**

    * 段的顺序由调用方保证（`neighbor.merge_segments` 按 `best_rank` 升序）
    * **`top_k` 先于预算**：先截到 `top_k` 个段，再按预算决定装几个——
      顺序反过来的话，被预算砍掉的段会**占掉名额**，实际返回数就少于该有的
    * **`top_k <= 0` ⇒ 空响应**，且**不数 token**（那时结果必然是空的）
    * `score` 按**输出位置**重算，不是照抄段的名次——段可能因预算被跳过，
      照抄会让 `score` 出现空洞（而它必须单调递减、且 `rank=0` 就是 `1.0`）
    """
    total_segments = len(segments)
    if top_k <= 0 or total_segments == 0:
        return PackagedResponse(items=(), dropped_missing=dropped_missing)

    head = list(segments)[:top_k]
    truncated_by_top_k = total_segments > len(head)

    chosen, truncated_by_budget = _fit_by_budget(head, counter=counter, max_tokens=max_tokens)

    items = tuple(
        ResponseItem(
            id=segment.anchor_memory_id,  # ★ id 是**锚点**，不是段里第一条
            content=segment.content,
            created_at=day_granularity(segment.anchor_event_time),
            score=placeholder_score(output_rank),  # 名次按【输出位置】——见 docstring
        )
        for output_rank, segment in enumerate(chosen)
    )

    # 对最终拼接字符串计数，报告真实值。
    output_text = SEGMENT_SEP.join(segment.content for segment in chosen)
    return PackagedResponse(
        items=items,
        dropped_missing=dropped_missing,
        considered_segments=total_segments,
        truncated_by_top_k=truncated_by_top_k,
        truncated_by_budget=truncated_by_budget,
        output_tokens=counter.count(output_text),
    )
