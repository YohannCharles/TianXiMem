"""Neighbor Expansion + Context Segment Merge（§10 / §11.2）。

```text
ranked（已去重、已 rerank 的名次）
  → ① 全部保留进 selected（rerank_rank = 真实名次）
  → ② 只对前 expansion_seed_limit 条扩窗 ± radius
       **沿 `prev_memory_id` / `next_memory_id` 跳**（一跳 = 一个完整 QA）
  → ③ 新扩出来的进 selected（rerank_rank = None —— **不造排名**）
  → ④ 按 `(user_id, session_id, request_id)` 分组、组内按 `local_index` 扫描，
       只把 **`prev.next == 当前 id`**（= 显式相邻）的那些合并成段
  → ⑤ 段按 best_rank（段内最小的真实名次）升序
```

## 六条不能越过的线

1. **`rank31+` 只保留、不扩展**（§10 的种子数）
   —— 扩窗代价随种子数线性增长；名次靠后的种子本就可能在预算里被砍掉
2. **不给新扩出来的邻居造排名**
   —— 编个假名次会让"段优先级只看真实名次"失去意义，锚点也会指向一条从没被选中过的记忆
3. **禁止跨 session 扩窗**
   —— 那是跨对话拼接，语义上不成立；而这两个字段是**隔离契约**（§2.2）
4. **禁止跨 Add 扩窗 / 合并**（**D28**）
   —— 跨 Add 不存在可信的全局顺序，硬拼起来会让模型读到**它以为连续、实际不连续**的
   上下文，而 `content` 里**看不出来**
5. **只有显式相邻才合并**
   —— 判据是 `前一条.next_memory_id == 这一条.id`，不是"下标加一"、更不是"同一 session"
6. **段内顺序只能是 `local_index` 升序**
   —— 把种子提到最前会把一段对话**拦腰截断**（§11.2），而答案阶段按前缀截断

## 邻接是**存储里的显式指针**，不是算出来的下标（D28）

`qa_pairs` 上存着 `prev_memory_id` / `next_memory_id`，它们在**一次 Add 内**按
"只连完整 QA"的规则算好、**写下时就是最终形状**（没有 UPDATE、没有回填）。

⇒ 扩窗 = **沿指针跳**，合并 = **看指针是否相接**。两处都不再有
"`pair_idx ± 1`"或"会话内稠密序 `seq`"这类**推断出来的**相邻性——
那种推断在 D25 的位置模型下还能自洽，在真实（不透明）`request_id` 上则根本没有依据。

## 为什么扩窗要读 SQLite 而不是 Qdrant

§6.3 的分工：**正文不进 Qdrant**。邻接指针也在 SQLite（`prev` / `next` 是列），
所以扩窗天然是"回真源查"——Qdrant 只负责"谁和查询像"。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

from tianximem.common.annotate import annotate
from tianximem.common.config import DEFAULT_EXPANSION_SEED_LIMIT, DEFAULT_RADIUS
from tianximem.common.render import event_day, render, render_date, render_segment
from tianximem.common.tokens import TokenCounter
from tianximem.retrieve.fusion import Candidate
from tianximem.store.sqlite_store import QaPair, SqliteStore

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
#: ⚠ **D28 起它就是"沿链跳几跳"**：链上只有完整 QA，所以 A-only / Q-only 永远不作为
#: 邻居出现（它们没有链）。


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
    def local_index(self) -> int:
        """它在**那一次 Add** 的块列表里的序号。

        ⚠ 只在一次 Add 内有意义（D28）——所以它**只**用来给同一条链上的成员排序，
        任何"跨 Add 比大小"的用法都是错的。
        """
        return self.pair.local_index

    @property
    def session_key(self) -> tuple[str, str]:
        return (self.pair.user_id, self.pair.session_id)

    @property
    def chain_key(self) -> tuple[str, str, str]:
        """**邻接链的作用域** = 一次 Add（D28）。

        ⚠ 段合并必须按它分组：换成 `session_key` 就会把**两次 Add** 的记忆放进同一组，
        而"它们恰好下标相接"这件事在下标是**每批各从 0 开始**的数时**毫无意义**。
        """
        return (self.pair.user_id, self.pair.session_id, self.pair.request_id)


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
    `connect`（D17）。跳 `radius` 跳 ⇒ 至多 `radius` 次批量查询（每跳把所有前沿节点
    的 `prev` / `next` 收成**一次** `IN`），不是"每个种子各跳各的"。

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

    with store.read() as conn:
        # ① 先把**全部** rerank 候选取正文并放进去——注意是【全部】，不是前 30。
        #    rank31+ 只是**不扩展**，它们仍然是候选（用户口径：不要删除它们）。
        pairs = store.fetch_pairs_by_ids(conn, list(by_id))
        pair_by_id: dict[str, QaPair] = {p.id: p for p in pairs}

        #: `id → QaPair`：候选与逐跳扩出来的邻居都登记在这里。链的指针也从这里读。
        known: dict[str, QaPair] = dict(pair_by_id)
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
        seeds = sorted((c for c in by_id.values() if c.rank < seed_limit), key=lambda c: c.rank)
        added = 0
        #: 本跳的**前沿**：只从这些节点继续往外跳（不是从全部已选节点）。
        frontier: set[str] = {c.memory_id for c in seeds if c.memory_id in selected}
        for _ in range(radius):
            wanted: dict[str, None] = {}  # 有序去重，顺便稳定
            for memory_id in sorted(frontier):
                pair = known.get(memory_id)
                if pair is None:
                    continue
                for neighbour in (pair.prev_memory_id, pair.next_memory_id):
                    # ⚠ `prev` / `next` 只会在**同一次 Add 内**连过（D28）⇒
                    #    "不跨 Add 扩窗"是**结构性**的，这里不需要再判断 request_id。
                    if neighbour is not None and neighbour not in selected:
                        wanted.setdefault(neighbour, None)
            if not wanted:
                break  # 链到头了（或这一跳全是已选中的）⇒ 停止，不"猜相邻的补上"

            rows = store.fetch_pairs_by_ids(conn, list(wanted))
            for pair in rows:
                known[pair.id] = pair
                if pair.id in selected:
                    continue
                selected[pair.id] = SelectedMemory(
                    memory_id=pair.id,
                    pair=pair,
                    rerank_rank=None,  # ★ 不造排名
                    is_neighbor=True,
                )
                added += 1
            frontier = {pair.id for pair in rows}

    return ExpansionResult(
        selected=tuple(selected.values()),
        neighbors_added=added,
        missing_rows=len(by_id) - len(pair_by_id),
    )


@dataclass(frozen=True, slots=True)
class ContextSegment:
    """一段**连续**对话（§11.2 / §11.3）。**它是 v1 打包与预算的原子单位。**

    ⚠ **"连续"的判据是显式指针**（D28）：段内相邻两条必须满足
    `前一条.next_memory_id == 后一条.id`。所以一个段**整体落在一次 Add 内**，
    自带它的 `request_id`。
    """

    source_memory_ids: tuple[str, ...]
    """段内全部记忆的 `id`，**按 `local_index` 升序**（= 那一次 Add 内的顺序）。"""

    user_id: str
    session_id: str
    request_id: str
    """这一段来自**哪一次 Add**（D28）。段是 Add-local 的，所以这个字段是**唯一值**。"""

    start_local_index: int
    end_local_index: int
    """段首 / 段尾在**那一次 Add** 里的 `local_index`。**不是**跨 Add 的序号。"""

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
    """把选中的记忆按**显式相邻**合并成段，并按 `best_rank` 升序返回。

    **算法是一次线性扫描**：按 `(user_id, session_id, request_id)` 分组 →
    组内按 `local_index` 升序 → 遍历维护"当前段"，
    `前一条.next_memory_id == 这一条.memory_id` 就**直接延长**，否则收尾当前段、新建一段。

    ⚠ **分组键里带着 `request_id`，而且判据读的是存储里的指针**——两道都在保证
    "段不出一次 Add"。只留判据的话，一次 Add 内部的下标为 0 的行会被误判成
    "接着上一次 Add 的末尾"；只留分组的话，一次 Add 里**缺了一块**（比如只有第 0 和第 2 块）
    会被硬拼成一段。**两条都要**。

    ⚠ **只有严格相邻才延长**（`prev.next == cur.id`，不是
    `cur.local_index == prev.local_index + 1`）：
    中间缺一个块就说明那段对话**不完整**，硬拼成一整段会让模型读到自己以为连续、
    实际缺了一环的上下文——而 `content` 里**看不出来**。

    ⚠ 段内顺序 = `local_index` 升序（**不把锚点提到最前**，§11.2）。
    """
    # ── 分组：chain_key → 该 Add 的成员，组内按 local_index 升序 ──
    groups: dict[tuple[str, str, str], list[SelectedMemory]] = {}
    for memory in selected:
        groups.setdefault(memory.chain_key, []).append(memory)

    segments: list[ContextSegment] = []
    for (user_id, session_id, request_id), members in groups.items():
        members.sort(key=lambda m: m.local_index)

        # 一次线性扫描：`runs` 里每个元素是一段连续的成员列表
        runs: list[list[SelectedMemory]] = []
        for memory in members:
            previous = runs[-1][-1] if runs else None
            if previous is not None and previous.pair.next_memory_id == memory.memory_id:
                runs[-1].append(memory)  # ★ 直接延长，不新建
            else:
                runs.append([memory])

        for run in runs:
            segments.append(
                _build_segment(
                    run,
                    user_id=user_id,
                    session_id=session_id,
                    request_id=request_id,
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
    request_id: str,
    counter: TokenCounter,
    inject_abs_time: bool = False,
    seed_placement: str = "keep",
    annotate_relatives: bool = False,
) -> ContextSegment:
    """把一段连续的成员收成一个 `ContextSegment`（含渲染与计数）。"""
    # 锚点 = 段内**真实名次最好**的那条。邻居（rerank_rank is None）不参与。
    #
    # ⚠ 扩窗是"沿链跳"的，跳出来的邻居与种子同链 ⇒ 段里至少有一条真实候选。
    #    但 `radius` 可配、`selected` 也可能被别的调用方构造出来，
    #    所以这里**不假设**那个不变量：全是邻居时取段内第一条当锚点，而不是崩掉。
    ranked_members = [m for m in run if m.rerank_rank is not None]
    # `cast` 不是装饰：上面的 filter 已保证这里没有 None，但**类型系统看不出来**
    # （lambda 的返回类型不会因外层列表推导而收窄）⇒ 这行是"把已知的不变式告诉
    # mypy"，不是忽略一个真实的 None。
    anchor = (
        min(ranked_members, key=lambda m: cast(int, m.rerank_rank)) if ranked_members else run[0]
    )
    best_rank = anchor.rerank_rank if anchor.rerank_rank is not None else _NO_RANK

    # ── 段内顺序（§11.2 的"组内顺序"，明文列为**可消融项**）─────────────────────
    # 默认 `keep`：纯位置（`local_index`）时间序。它当初被选中的理由是"窗口是一段连续对话，
    # 按时间序读才成立"——所以另两个变体**都要用数据说话**，不能凭直觉换。
    # ⚠ 它们**只改渲染顺序**：`source_memory_ids` / `start_local_index` / 锚点 / `best_rank`
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
        request_id=request_id,
        start_local_index=run[0].local_index,
        end_local_index=run[-1].local_index,
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
#: ⚠ 按当前口径**不应该出现**（扩窗沿链跳，种子必与它扩出来的邻居同链 ⇒ 同段）。
#: 真出现说明 `radius` 或 `selected` 的构造方式变了——给它一个**排在所有真实段之后**的值，
#: 而不是崩掉，也不是静默塞到最前。
_NO_RANK: Final[int] = 1 << 60
