"""Neighbor Expansion + Context Segment Merge（§10 / §11.2）。

```text
ranked（已去重、已 rerank 的名次）
  → ① 全部保留进 selected（rerank_rank = 真实名次）
  → ② 只对前 expansion_seed_limit 条扩窗 ± radius（**SQL 在 store/**）
  → ③ 新扩出来的进 selected（rerank_rank = None —— **不造排名**）
  → ④ 按 (user_id, session_id) 分组、组内按 pair_idx 升序，**一次线性扫描**合并成段
  → ⑤ 段按 best_rank（段内最小的真实名次）升序
```

## 五条不能越过的线

1. **`rank31+` 只保留、不扩展**（§10 的种子数）
   —— 扩窗代价随种子数线性增长；名次靠后的种子本就可能在预算里被砍掉
2. **不给新扩出来的邻居造排名**
   —— 编个假名次会让"段优先级只看真实名次"失去意义，锚点也会指向一条从没被选中过的记忆
3. **禁止跨 session 扩窗**
   —— 那是跨对话拼接，语义上不成立；而这两个字段是**隔离契约**（§2.2）
4. **只有 `pair_idx` 相邻才合并**
   —— 段的定义就是"连续对话"；硬拼不连续的对会让模型读到**断裂的上下文**，而 `content` 里看不出来
5. **段内顺序只能是 `pair_idx` 升序**
   —— 把种子提到最前会把一段话**拦腰截断**（§11.2），而答案阶段按前缀截断

## 为什么扩窗要读 SQLite 而不是 Qdrant

§6.3 的分工：**正文不进 Qdrant**。邻域查询正好命中
`UNIQUE(user_id, session_id, pair_idx)` 的索引（不变式 4）——一次 `BETWEEN`、不扫全表。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from tianxi_am.common.config import DEFAULT_EXPANSION_SEED_LIMIT, DEFAULT_RADIUS
from tianxi_am.common.render import render, render_date, render_segment
from tianxi_am.common.tokens import TokenCounter
from tianxi_am.retrieve.fusion import Candidate
from tianxi_am.store.sqlite_store import QaPair, SqliteStore

__all__ = [
    "DEFAULT_EXPANSION_SEED_LIMIT",
    "DEFAULT_RADIUS",
    "ContextSegment",
    "ExpansionResult",
    "SelectedMemory",
    "expand_neighbors",
    "merge_segments",
]

#: 主动扩窗的种子数与半径（§10）——**从 `common/config.py` 再导出**。
#:
#: ⚠ **值的家不在这里**：`common/` 不得 import 本层（分层要求），所以默认值住在配置层。
#: 各写一份会出现"代码默认值 30 / 配置里写了 20"这种没人会发现的漂移——它只会表现为
#: "扩窗比预期少了几条"，**不报错**。
#:
#: ⚠ 半径的单位是 **QA 对**：`±1` 拿回前后各**一整对**（最多 4 条消息），
#: 不是各一条消息——**扩窗比 message 粒度时更贵**（值见 config-reference §6）。


@dataclass(frozen=True, slots=True)
class SelectedMemory:
    """进入打包候选的一条记忆。**它可能是检索来的，也可能是扩窗补的。**"""

    memory_id: str
    pair: QaPair
    #: **真实**的 rerank 名次（0-based）。`None` = **扩窗补进来的，从没被检索器选中**。
    #: ⚠ **绝不给它编一个数**——见模块 docstring 的线 2。
    rerank_rank: int | None
    #: 是否由扩窗补进来（诊断用：段里有几条是"上下文"而不是"命中"）。
    is_neighbor: bool

    @property
    def pair_idx(self) -> int:
        return self.pair.pair_idx

    @property
    def session_key(self) -> tuple[str, str]:
        return (self.pair.user_id, self.pair.session_id)


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    """扩窗的产出，外加**两个必须被看见的计数**。"""

    selected: tuple[SelectedMemory, ...]
    """去重后的全部记忆：**全部 rerank 候选 + 新扩出来的邻居**。"""

    neighbors_added: int
    """真正新加进来的邻居条数（**去重之后**的净值）。"""

    missing_rows: int
    """在 Qdrant 里被选中、但**真源里查不到**的候选条数。

    ⚠ 非 0 说明**索引与真源脱钩了**（Qdrant 是派生索引，`id` 应与 SQLite 一一对应）。
    正常情况下恒为 0；**不该静默**——调用方要能看到它（与 `package` 的 `dropped_missing`
    是同一条纪律）。
    """


def expand_neighbors(
    ranked: list[Candidate],
    *,
    store: SqliteStore,
    seed_limit: int = DEFAULT_EXPANSION_SEED_LIMIT,
    radius: int = DEFAULT_RADIUS,
) -> ExpansionResult:
    """把 `ranked` 全部保留，并对**前 `seed_limit` 条**扩窗 `± radius`。

    **一次 `store.read()` 里做完所有查询**：短生命周期连接模型下每次开连接都要付一次
    `connect`（D17），而这里有 1 次批量取正文 + 至多 `seed_limit` 次邻域查询。

    ⚠ 读操作**不拿写锁**（D17：不要为读拿 `BEGIN IMMEDIATE`）。
    """
    if seed_limit < 0:
        raise ValueError(f"seed_limit 不得为负：{seed_limit}")
    if radius < 0:
        raise ValueError(f"radius 不得为负：{radius}")

    # ② 去重：同一个 memory_id 只留名次最好的那一次（`ranked` 已由调用方去重并重编号，
    #    这里的字典在语义上仍是"稳定：先到先得"）。
    by_id: dict[str, Candidate] = {}
    for candidate in ranked:
        by_id.setdefault(candidate.memory_id, candidate)

    if not by_id:
        return ExpansionResult(selected=(), neighbors_added=0, missing_rows=0)

    # ① 先把**全部** rerank 候选取正文并放进去——注意是【全部】，不是前 30。
    #    rank31+ 只是**不扩展**，它们仍然是候选（用户口径：不要删除它们）。
    with store.read() as conn:
        pairs = store.fetch_pairs_by_ids(conn, list(by_id))
        pair_by_id: dict[str, QaPair] = {p.id: p for p in pairs}

        selected: dict[str, SelectedMemory] = {}
        for memory_id, candidate in by_id.items():
            pair = pair_by_id.get(memory_id)
            if pair is None:
                continue  # 真源缺行 ⇒ 记进 missing_rows，不补造
            selected[memory_id] = SelectedMemory(
                memory_id=memory_id,
                pair=pair,
                rerank_rank=candidate.rank,
                is_neighbor=False,
            )

        # ③ 只对前 `seed_limit` 条扩窗。**按名次取，不是按列表位置**——
        #    名次已经是连续 0-based（`dedup_candidates` 保证），两者等价；
        #    但按名次写更贴着"种子数"这个规格。
        seeds = [c for c in by_id.values() if c.rank < seed_limit]
        seeds.sort(key=lambda c: c.rank)

        added = 0
        for seed in seeds:
            anchor = selected.get(seed.memory_id)
            if anchor is None:
                continue  # 种子自己的真源行缺失 ⇒ 连它的 session/pair_idx 都不知道，扩不了
            window = store.fetch_pairs_by_idx_range(
                conn,
                anchor.pair.user_id,
                anchor.pair.session_id,
                anchor.pair_idx - radius,
                anchor.pair_idx + radius,
            )
            for pair in window:
                # 已经在里面（它本身就是候选，或已被前一个种子扩到）⇒ **跳过**
                if pair.id in selected:
                    continue
                selected[pair.id] = SelectedMemory(
                    memory_id=pair.id,
                    pair=pair,
                    rerank_rank=None,  # ★ 不造排名
                    is_neighbor=True,
                )
                added += 1

    return ExpansionResult(
        selected=tuple(selected.values()),
        neighbors_added=added,
        missing_rows=len(by_id) - len(pair_by_id),
    )


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """一段**连续**对话（§11.2 / §11.3）。**它是 v1 打包与预算的原子单位。**"""

    source_memory_ids: tuple[str, ...]
    """段内全部记忆的 `id`，**按 `pair_idx` 升序**（= 会话顺序）。"""

    user_id: str
    session_id: str
    start_pair_idx: int
    end_pair_idx: int

    anchor_memory_id: str
    """**段优先级的来源**：段内真实名次最好的那条候选的 `id`。
    最终响应的 `id` 用它（§11.2 的组间顺序 = 种子名次序）。"""

    best_rank: int
    """段内**最小的真实 rerank 名次**。**邻居不参与**。"""

    anchor_event_time: int | None
    """锚点的 `event_time` ⇒ 最终响应的 `created_at`（UTC 日粒度）。"""

    content: str
    """`common/render` 拼好的正文（`render` 逐对 + `render_segment` 连接）。"""

    token_count: int
    """**对 `content` 这个真实字符串**数的 token（§6.4）——不是各对 token 之和。"""

    rerank_member_count: int
    """段里有几条是真实候选（其余是扩窗补的）——诊断用。"""

    @property
    def length(self) -> int:
        """段里有几个 QA 对。"""
        return len(self.source_memory_ids)


def merge_segments(
    selected: tuple[SelectedMemory, ...] | list[SelectedMemory],
    *,
    counter: TokenCounter,
    inject_abs_time: bool = False,
) -> list[ContextSegment]:
    """把选中的记忆按"连续 `pair_idx`"合并成段，并按 `best_rank` 升序返回。

    **算法是一次线性扫描**：按 `(user_id, session_id)` 分组 → 组内按 `pair_idx` 升序 →
    遍历维护"当前段"，`pair_idx == end + 1` 就**直接延长**，否则收尾当前段、新建一段。

    ⚠ **不同 session 永远不能合并**：分组键里就带着 `session_id`，所以
    跨 session 合并**在结构上做不到**，而不是靠一句判断。

    ⚠ **只有 `pair_idx` 严格相邻才延长**。`end + 1` 而不是 `end + k`：
    中间缺一个对就说明那段对话**不完整**，硬拼成一整段会让模型读到自己以为连续、
    实际缺了一环的上下文——而 `content` 里**看不出来**。

    ⚠ 段内顺序 = `pair_idx` 升序（**不把锚点提到最前**，§11.2）。
    """
    # ── 分组：session_key → 该 session 的成员，组内按 pair_idx 升序 ──
    groups: dict[tuple[str, str], list[SelectedMemory]] = {}
    for memory in selected:
        groups.setdefault(memory.session_key, []).append(memory)

    segments: list[ContextSegment] = []
    for (user_id, session_id), members in groups.items():
        members.sort(key=lambda m: m.pair_idx)

        # 一次线性扫描：`runs` 里每个元素是一段连续的成员列表
        runs: list[list[SelectedMemory]] = []
        for memory in members:
            if runs and memory.pair_idx == runs[-1][-1].pair_idx + 1:
                runs[-1].append(memory)  # ★ 直接延长，不新建
            else:
                runs.append([memory])

        for run in runs:
            segments.append(
                _build_segment(
                    run,
                    user_id=user_id,
                    session_id=session_id,
                    counter=counter,
                    inject_abs_time=inject_abs_time,
                )
            )

    # 段间按 best_rank 升序（= 相关性顺序，§11.2 的组间顺序）
    segments.sort(key=lambda s: s.best_rank)
    return segments


def _build_segment(
    run: list[SelectedMemory],
    *,
    user_id: str,
    session_id: str,
    counter: TokenCounter,
    inject_abs_time: bool = False,
) -> ContextSegment:
    """把一段连续的成员收成一个 `ContextSegment`（含渲染与计数）。"""
    # 锚点 = 段内**真实名次最好**的那条。邻居（rerank_rank is None）不参与。
    #
    # ⚠ 扩窗窗口是 `[k-r, k+r]` 的**整段**行，所以种子必与它扩出来的邻居同段 ⇒ 段里
    #    至少有一条真实候选。但 `radius` 可配、`selected` 也可能被别的调用方构造出来，
    #    所以这里**不假设**那个不变量：全是邻居时取会话顺序的第一条当锚点，而不是崩掉。
    ranked_members = [m for m in run if m.rerank_rank is not None]
    # `cast` 不是装饰：上面的 filter 已保证这里没有 None，但**类型系统看不出来**
    # （lambda 的返回类型不会因外层列表推导而收窄）⇒ 这行是"把已知的不变式告诉
    # mypy"，不是忽略一个真实的 None。
    anchor = (
        min(ranked_members, key=lambda m: cast(int, m.rerank_rank)) if ranked_members else run[0]
    )
    best_rank = anchor.rerank_rank if anchor.rerank_rank is not None else _NO_RANK

    # ⚠ 逐对渲染，且**日期口径与索引侧、精排输入侧完全一致**（不变式 I1 / T1）：
    #    `inject_abs_time` 由 `packaging.inject_abs_time` 传下来，三处读的是同一个值。
    pair_texts = [
        render(
            m.pair.question,
            m.pair.answer,
            date=render_date(m.pair.event_time, inject_abs_time=inject_abs_time),
        )
        for m in run
    ]
    content = render_segment(pair_texts)

    return ContextSegment(
        source_memory_ids=tuple(m.memory_id for m in run),
        user_id=user_id,
        session_id=session_id,
        start_pair_idx=run[0].pair_idx,
        end_pair_idx=run[-1].pair_idx,
        anchor_memory_id=anchor.memory_id,
        best_rank=best_rank,
        anchor_event_time=anchor.pair.event_time,
        content=content,
        # ★ 对**拼好的真实字符串**计数，不是把各对的计数加起来
        token_count=counter.count(content),
        rerank_member_count=len(ranked_members),
    )


#: 段里**一条真实候选都没有**时的 `best_rank`。
#:
#: ⚠ 按当前口径**不应该出现**（扩窗窗口是连续区间，种子必与邻居同段）。真出现说明
#: `radius` 或 `selected` 的构造方式变了——给它一个**排在所有真实段之后**的值，
#: 而不是崩掉，也不是静默塞到最前。
_NO_RANK: Final[int] = 1 << 60
