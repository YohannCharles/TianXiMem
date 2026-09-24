"""Context Packaging —— **Step 1 的最小切片**（§11.3）。

> ⚠⚠ **这是阶段性实现，不是最终的 Search Pipeline。**
> 当前：`Hybrid → Candidates → Checker(v1 passthrough) → **Minimal Packaging** → ≤ top_k`
> 最终：`... → Checker → **Remote Rerank** → **Neighbor Expansion** → Context Packaging → ≤ top_k`
> 两者都写在 [`CLAUDE.md`](./CLAUDE.md)，**不要把这个切片当成最终形态**。

## 本切片只做五件事

1. 按 `Candidate` 的**当前名次**生成响应
2. 从 SQLite 真源取正文（`fetch_pairs_by_ids`；**正文不进 Qdrant**，§6.3）
3. `content` 用 [`../common/render.py`](../common/render.py) 的**唯一实现**
4. `created_at` 只给到**日粒度**，`event_time` 为 NULL 时发 `""`
5. `score = 1/(rank+1)`，且**最终数量严格 ≤ `top_k`**

## 本切片**不**做（免得被当成遗漏）

Rerank（Step 3）· Neighbor Expansion（Step 2）· 双预算截断（Step 2）·
组内/组间顺序（扩窗出现后才有意义）。

## 为什么 `content` 不通过参数注入

`common/render` 是**唯一实现**（不变式 I1）：同一份渲染既是 embedding 的输入、也是
返回给 AML 的 `content`。若这里允许传入别的渲染函数，"检索命中的是什么"与"模型读到的
是什么"就会**漂移，且不报错**。

⇒ 所以**没有 `renderer` 参数**——不是省事，是让漂移**不可能发生**。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from typing import Final

from tianxi_am.common.render import render
from tianxi_am.retrieve.fusion import Candidate
from tianxi_am.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "DEFAULT_TZ",
    "PackagedResponse",
    "ResponseItem",
    "day_granularity",
    "package",
    "placeholder_score",
]

#: `created_at` 日粒度的**时区口径**。
#:
#: ⚠ **契约没规定时区**（`docs/contract.md` §3 只说"只给到日粒度"）。取 **UTC** 的理由：
#: 它是**与机器无关**的——用本地时区会让同一份数据在不同机器上差一天，而本项目最怕
#: 的就是不可复现。
#:
#: ⚠ **它与加载层必须一致**：`event_time` 是由 harness 合成的，
#: **合成时用哪个时区、这里就得用哪个时区**，否则日期会整体偏一天（且不报错）。
#: ⇒ 这个量要进 `configs/`，与加载层共享（`config-reference` 的 `packaging.created_at_tz`）。
DEFAULT_TZ: Final[tzinfo] = UTC

#: 日期格式：`YYYY-MM-DD`（§11.3 的示例是 `2026-07-26`）。
#: ⚠ **不要用裸 Unix 毫秒**——渲染出来是 `- [1753512557000] ...`，对模型无意义。
_DATE_FORMAT: Final[str] = "%Y-%m-%d"


def day_granularity(event_time: int | None, *, tz: tzinfo = DEFAULT_TZ) -> str:
    """Unix 毫秒 → `YYYY-MM-DD`；`None` → **空串 `""`**。

    * `event_time` 是**该对首条消息**的 timestamp（§6.1），可空
    * **NULL 时发 `""`**（§11.3）：渲染代码是 `str(item.get("created_at") or "")`，
      假值会退化成 `- {text}`——那是一条**有定义的降级路径**
    * ⚠ **绝不拿 Add 的到达时间兜底**：那是"何时写入"而不是"何时发生"，
      会给模型**错误信息**
    * **只给到日粒度**：粒度会被模型看见，秒级会诱发它按秒级回答，从而踩中
      "粒度变细"那条判负规则（§11.3）
    """
    if event_time is None:
        return ""
    return datetime.fromtimestamp(event_time / 1000, tz=tz).strftime(_DATE_FORMAT)


def placeholder_score(rank: int) -> float:
    """`1/(rank+1)`——**单调递减的占位值**，`rank` 是 **0-based**。

    ⚠ **不要返回原始 RRF 分数**：融合分数**不是校准量**（§8），对我们没有意义；
    但**万一 AML 按 `score` 重排，省略 `score` 就有风险**。取最安全的一侧：
    这个值无论 AML 是否重排，**顺序都不变**（§11.3）。

    ⚠ Rerank 接入后**名次会变**，届时这个值要基于**最终名次**重算——
    所以它只能在这里现算，不能固化在上游类型里。
    """
    if rank < 0:
        raise ValueError(f"rank 不得为负：{rank}")
    return 1.0 / (rank + 1.0)


@dataclass(frozen=True, slots=True)
class ResponseItem:
    """`data[]` 的一项——**§2.1 的四个字段，一个不多**。

    ⚠ 这里**没有** `session_id` / `pair_idx` / `score` 之外的任何内部字段。
    多余的键通常被忽略，但**没有理由冒这个险**。
    """

    id: str
    content: str
    created_at: str
    score: float


@dataclass(frozen=True, slots=True)
class PackagedResponse:
    """打包结果。

    `dropped_missing` 是**真源里查不到正文的条数**——正常情况下应为 0
    （Qdrant 是派生索引，`id` 与 SQLite 一一对应）。**非 0 说明索引与真源脱钩了**
    （例如 SQLite 被回滚到旧状态），**不该静默**：调用方要能看到这个数。
    """

    items: tuple[ResponseItem, ...]
    dropped_missing: int

    @property
    def count(self) -> int:
        return len(self.items)


def package(
    candidates: Sequence[Candidate],
    *,
    store: SqliteStore,
    top_k: int,
    tz: tzinfo = DEFAULT_TZ,
) -> PackagedResponse:
    """把候选打包成 `data[]`。**本函数是最终数量的守门人。**

    * **`top_k` 的边界**：`<= 0` ⇒ 空响应（**不查库**）；上游给得更多 ⇒ 先截断
    * **绝不为凑满 `top_k` 复制或补造结果**——有多少真源行就有多少项
    * `score` 按**输出位置**重算，不是照抄输入名次：真源缺行时中间会被跳过，
      照抄会让 `score` 出现空洞（而它必须单调递减、且 `rank=0` 就是 `1.0`）
    """
    if top_k <= 0:
        return PackagedResponse(items=(), dropped_missing=0)

    head = list(candidates)[:top_k]
    ids = [c.memory_id for c in head if c.memory_id]
    if not ids:
        return PackagedResponse(items=(), dropped_missing=0)

    # 只读路径：用 `read()` 开一个**短生命周期只读连接**，用完即关。
    # **不要**用 transaction()——那会拿写锁（BEGIN IMMEDIATE），让一次只读去和写事务抢。
    with store.read() as conn:
        pairs = store.fetch_pairs_by_ids(conn, ids)
    by_id: dict[str, QaPair] = {p.id: p for p in pairs}

    items: list[ResponseItem] = []
    for candidate in head:
        pair = by_id.get(candidate.memory_id)
        if pair is None:
            continue  # 真源里没有 ⇒ 数进 dropped_missing，不补造
        items.append(
            ResponseItem(
                id=pair.id,
                content=render(pair.question, pair.answer),
                created_at=day_granularity(pair.event_time, tz=tz),
                # 名次按【输出位置】——见 docstring
                score=placeholder_score(len(items)),
            )
        )

    return PackagedResponse(
        items=tuple(items),
        dropped_missing=len(head) - len(items),
    )
