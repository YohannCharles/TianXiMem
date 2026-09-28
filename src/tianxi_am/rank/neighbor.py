"""Neighbor Expansion + Context Segment Merge（§10 / §11.2）。

```text
ranked（已去重、已 rerank 的名次）
  → ① 全部保留进 selected（rerank_rank = 真实名次）
  → ② 只对前 expansion_seed_limit 条扩窗 ± radius（**整段取回，按会话下标切窗口**）
  → ③ 新扩出来的进 selected（rerank_rank = None —— **不造排名**）
  → ④ 按 (user_id, session_id) 分组、组内按 `seq` 升序，**一次线性扫描**合并成段
  → ⑤ 段按 best_rank（段内最小的真实名次）升序
```

## 五条不能越过的线

1. **`rank31+` 只保留、不扩展**（§10 的种子数）
   —— 扩窗代价随种子数线性增长；名次靠后的种子本就可能在预算里被砍掉
2. **不给新扩出来的邻居造排名**
   —— 编个假名次会让"段优先级只看真实名次"失去意义，锚点也会指向一条从没被选中过的记忆
3. **禁止跨 session 扩窗**
   —— 那是跨对话拼接，语义上不成立；而这两个字段是**隔离契约**（§2.2）
4. **只有 `seq` 相邻才合并**
   —— 段的定义就是"连续对话"；硬拼不连续的对会让模型读到**断裂的上下文**，而 `content` 里看不出来
5. **段内顺序只能是 `seq` 升序**
   —— 把种子提到最前会把一段话**拦腰截断**（§11.2），而答案阶段按前缀截断

## D25：相邻性从"整数加一"换成"会话内稠密序 `seq`"

旧口径的位置是 `pair_idx`（session 内连续整数，写时 `MAX+1` 分配），所以"相邻"可以直接
写成 `b == a + 1`、扩窗可以直接写成 `BETWEEN k-r AND k+r`。

D25 把位置换成 `(chunk_ordinal, local_index)`（**请求的纯函数**，见
[`../../../docs/decisions.md`](../../../docs/decisions.md) 的 D25），它**可能有空洞**
（跳号、某批没产出块）⇒ 整数算术**不再等于**"会话里前后各 r 个"。⇒ 改成：

* **扩窗**：取回该 session 的**整段有序列表**，按**列表下标**切 `[i-r, i+r]`
* **合并**：按 `seq`（`ROW_NUMBER()` 现算的稠密序）判相邻

两者都让**缺号不破坏相邻**，而"中间真的少了一块"仍然被抓住（`seq` 有洞）。

## 为什么扩窗要读 SQLite 而不是 Qdrant

§6.3 的分工：**正文不进 Qdrant**。邻域查询走
`UNIQUE(user_id, session_id, chunk_ordinal, local_index)` 建出的索引（不变式 4）
——`WHERE` 命中前两列前缀、`ORDER BY` 命中后两列 ⇒ **不扫全表、不额外排序**。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Final, cast

from tianxi_am.common.annotate import annotate
from tianxi_am.common.config import DEFAULT_EXPANSION_SEED_LIMIT, DEFAULT_RADIUS
from tianxi_am.common.render import event_day, render, render_date, render_segment
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
#: ⚠ 半径的单位是 **记忆块**（D24 前叫 QA 对）：`±1` 拿回前后各**一整块**（最多 4 条消息），
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
    def seq(self) -> int:
        """**会话内稠密序**（0-based 连续，按 `(chunk_ordinal, local_index)` 排）。

        ⚠ 它**不是列**，由 `store.fetch_session_ordered` 的 `ROW_NUMBER()` 现算。
        ⚠ **它必须是真值**：`expand_neighbors` 保证每条 `SelectedMemory` 都拿到它
        （候选那一路由 `fetch_pairs_by_ids` 回来时是 `-1`，那里补过一次）。
        `-1` 流进 `merge_segments` 的相邻判断会让**所有块各自成段**，
        而那看起来只像"检索质量差"。
        """
        return self.pair.seq

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

        # ② 给候选所在的每个 session **取一次有序全段**（带 `seq`）。同时供两处用：
        #      (a) 补上候选自己的 `seq`——`fetch_pairs_by_ids` 给不了（见它的 docstring：
        #          按主键取一批没有会话上下文，窗口函数只会算出"这一批里的第几个"）
        #      (b) 给种子切 `±radius` 窗口
        #    ⚠ **一个 session 只取一次**：种子密集时同一个 session 会被反复扩窗，
        #    重复查询纯属浪费（D17：每次都要 connect）；也让同一 session 的多个种子
        #    看到**同一份**快照。
        session_cache: dict[tuple[str, str], list[QaPair]] = {}
        for row in pair_by_id.values():
            # ⚠ 变量名不与下面那个 `pair` 复用：mypy 会按第一次赋值定死类型，
            #   而下一个 `pair` 是 `QaPair | None`（`.get()` 的返回）
            key = (row.user_id, row.session_id)
            if key not in session_cache:
                session_cache[key] = store.fetch_session_ordered(conn, *key)
        #: `id → 它在会话里的下标`。`ordered` 按 `seq` 升序且 0-based 连续 ⇒ 下标就是 `seq`。
        pos_by_id: dict[str, int] = {
            q.id: index for ordered in session_cache.values() for index, q in enumerate(ordered)
        }

        selected: dict[str, SelectedMemory] = {}
        for memory_id, candidate in by_id.items():
            pair = pair_by_id.get(memory_id)
            if pair is None:
                continue  # 真源缺行 ⇒ 记进 missing_rows，不补造
            selected[memory_id] = SelectedMemory(
                memory_id=memory_id,
                # ★ 补上 `seq`：`merge_segments` 按它排序与判相邻，而按主键取回来的行
                #   `seq` 是 `-1` ⇒ 不补的话**所有块都会各自成段**，
                #   而那看起来只像"检索质量差"。
                pair=replace(pair, seq=pos_by_id[pair.id]),
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
                continue  # 种子自己的真源行缺失 ⇒ 连它在会话里的位置都不知道，扩不了
            ordered = session_cache[(anchor.pair.user_id, anchor.pair.session_id)]
            center = pos_by_id[anchor.memory_id]
            # ⚠ 按**列表下标**切，不是 `pair_idx - r .. + r` 那种整数算术：
            #    位置（`chunk_ordinal`）有空洞时整数窗口会拿到"位置上相邻、
            #    会话里不相邻"的块。这里实现的是**"会话里前后各 r 个"**这个定义本身。
            window = ordered[max(0, center - radius) : center + radius + 1]
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
    """段内全部记忆的 `id`，**按 `seq` 升序**（= 会话顺序）。"""

    user_id: str
    session_id: str
    start_seq: int
    end_seq: int

    anchor_memory_id: str
    """**段优先级的来源**：段内真实名次最好的那条候选的 `id`。
    最终响应的 `id` 用它（§11.2 的组间顺序 = 种子名次序）。"""

    best_rank: int
    """段内**最小的真实 rerank 名次**。**邻居不参与**。"""

    anchor_event_time: int | None
    """锚点的 `event_time` ⇒ 最终响应的 `created_at`（UTC 日粒度）。"""

    content: str
    """`common/render` 拼好的正文（`render` 逐块 + `render_segment` 连接）。"""

    token_count: int
    """**对 `content` 这个真实字符串**数的 token（§6.4）——不是各对 token 之和。"""

    rerank_member_count: int
    """段里有几条是真实候选（其余是扩窗补的）——诊断用。"""

    @property
    def length(self) -> int:
        """段里有几个记忆块。"""
        return len(self.source_memory_ids)


def merge_segments(
    selected: tuple[SelectedMemory, ...] | list[SelectedMemory],
    *,
    counter: TokenCounter,
    inject_abs_time: bool = False,
    seed_placement: str = "keep",
    annotate_relatives: bool = False,
) -> list[ContextSegment]:
    """把选中的记忆按"连续 `seq`"合并成段，并按 `best_rank` 升序返回。

    **算法是一次线性扫描**：按 `(user_id, session_id)` 分组 → 组内按 `seq` 升序 →
    遍历维护"当前段"，`seq == end + 1` 就**直接延长**，否则收尾当前段、新建一段。

    ⚠ `seq` 是**读时的会话内稠密序**（D25），不是 `chunk_ordinal`：所以"中间少了一个块"
    仍然被抓住（`seq` 有洞 ⇒ 中间真的少了一块），而"chunk 序号跳号"（AML 那边的批次编号）
    **不会**被误判成断裂。

    ⚠ **不同 session 永远不能合并**：分组键里就带着 `session_id`，所以
    跨 session 合并**在结构上做不到**，而不是靠一句判断。

    ⚠ **只有 `seq` 严格相邻才延长**。`end + 1` 而不是 `end + k`：
    中间缺一个块就说明那段对话**不完整**，硬拼成一整段会让模型读到自己以为连续、
    实际缺了一环的上下文——而 `content` 里**看不出来**。

    ⚠ 段内顺序 = `seq` 升序（**不把锚点提到最前**，§11.2）。
    """
    # ── 分组：session_key → 该 session 的成员，组内按 seq 升序 ──
    groups: dict[tuple[str, str], list[SelectedMemory]] = {}
    for memory in selected:
        groups.setdefault(memory.session_key, []).append(memory)

    segments: list[ContextSegment] = []
    for (user_id, session_id), members in groups.items():
        members.sort(key=lambda m: m.seq)

        # 一次线性扫描：`runs` 里每个元素是一段连续的成员列表
        runs: list[list[SelectedMemory]] = []
        for memory in members:
            if runs and memory.seq == runs[-1][-1].seq + 1:
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
                    seed_placement=seed_placement,
                    annotate_relatives=annotate_relatives,
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
    seed_placement: str = "keep",
    annotate_relatives: bool = False,
) -> ContextSegment:
    """把一段连续的成员收成一个 `ContextSegment`（含渲染与计数）。"""
    # 锚点 = 段内**真实名次最好**的那条。邻居（rerank_rank is None）不参与。
    #
    # ⚠ 扩窗窗口是"种子前后各 r 个"的**连续区间**，所以种子必与它扩出来的邻居同段 ⇒ 段里
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

    # ── 段内顺序（§11.2 的"组内顺序"，明文列为**可消融项**）─────────────────────
    # 默认 `keep`：纯 `seq` 时间序。它当初被选中的理由是"窗口是一段连续对话，
    # 按时间序读才成立"——所以另两个变体**都要用数据说话**，不能凭直觉换。
    # ⚠ 它们**只改渲染顺序**：`source_memory_ids` / `start_seq` / 锚点 / `best_rank`
    #   一个都不动（否则"段优先级只看真实候选"这条不变式就破了）。
    ordered = run
    echoed: list[SelectedMemory] = []
    if seed_placement == "front" and anchor in run:
        ordered = [anchor, *(m for m in run if m is not anchor)]
    elif seed_placement == "echo" and anchor in run:
        echoed = [anchor]  # 段首重复一份；下面的时间序块**原样保留**

    # ⚠ 逐块渲染，且**日期口径与索引侧、精排输入侧完全一致**（不变式 I1 / T1）：
    #    `inject_abs_time` 由 `packaging.inject_abs_time` 传下来，三处读的是同一个值。
    #
    # ⚠ 而 `annotate_relatives`（`packaging.annotate_relatives`）**只在这一处生效**——
    #    它是那条不变式的**一个声明式例外**：索引侧与精排输入仍然只认 `render_pair`，
    #    所以开它**不改 embedding 输入**（⇒ 不用重建索引、不用换集合）。
    #    代价是 `content` 不再逐字等于被索引的文本，而是"被索引的文本 + 一层纯注解"
    #    ——那层注解是 `annotate()` 的确定性输出，可逆、可测（见 `tests/test_annotate.py`）。
    def _text(member: SelectedMemory) -> str:
        text = render(
            member.pair.question,
            member.pair.answer,
            date=render_date(member.pair.event_time, inject_abs_time=inject_abs_time),
        )
        if not annotate_relatives:
            return text
        anchor_day = event_day(member.pair.event_time)
        # 没有 `event_time` 就没有锚点 ⇒ **一个字都不动**（编一个日期比不注解更糟）
        return text if anchor_day is None else annotate(text, anchor_day)

    content = render_segment([*(_text(m) for m in echoed), *(_text(m) for m in ordered)])

    return ContextSegment(
        source_memory_ids=tuple(m.memory_id for m in run),
        user_id=user_id,
        session_id=session_id,
        start_seq=run[0].seq,
        end_seq=run[-1].seq,
        anchor_memory_id=anchor.memory_id,
        best_rank=best_rank,
        anchor_event_time=anchor.pair.event_time,
        content=content,
        # ★ 对**拼好的真实字符串**计数，不是把各块的计数加起来
        token_count=counter.count(content),
        rerank_member_count=len(ranked_members),
    )


#: 段里**一条真实候选都没有**时的 `best_rank`。
#:
#: ⚠ 按当前口径**不应该出现**（扩窗窗口是连续区间，种子必与邻居同段）。真出现说明
#: `radius` 或 `selected` 的构造方式变了——给它一个**排在所有真实段之后**的值，
#: 而不是崩掉，也不是静默塞到最前。
_NO_RANK: Final[int] = 1 << 60
