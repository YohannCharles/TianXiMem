"""`rank/neighbor.py` —— 扩窗与段合并（§10 / §11.2），以及**它们在整条链上的位置**。

```text
ranked（已去重、已 rerank）
  → expand_neighbors：全部保留 + 只对前 N 条**沿 prev/next 跳 ±radius 跳**
  → merge_segments  ：`前一条.next == 这一条.id` 才合并（**同一次 Add 内**）
```

> **段模型是 `Search` 链上的一环**——排在 `hybrid → checker` 之后。段的三个概念
> （锚点、`best_rank`、连续性）**只由它定义**，链上别处不存在第二处——见
> [`../src/tianximem/rank/CLAUDE.md`](../src/tianximem/rank/CLAUDE.md)。

⚠ **D28：邻接只在一次 Add 内**。所以本文件的脚手架分两种：

* `_line(store, ...)` —— 在**一次 Add 内**落满 `0..max(indices)`，返回选中那几个的 `id`。
  **相邻性来自这一次 Add 的链**（库里存着 `prev` / `next`）。
* `_seed(...)`（`conftest.seed_pair_in`）—— 每个 `index` 是**一次独立的 Add**，互不相邻。
  需要"两条互不相邻的候选"时用它。

⚠ 这里**不测** rerank（在 [`test_reranker.py`](./test_reranker.py)），
也不测打包的 `score` / 预算（在 [`test_packaging.py`](./test_packaging.py)）。

⚠ 真源一律是**真的 `SqliteStore`**，不 mock——扩窗的正确性**全在 SQL 的过滤条件上**，
mock 掉就等于没测。
"""

from __future__ import annotations

import pytest
from tests.conftest import FakeCounter, Wired, seed_line

from tianximem.common.render import render, render_segment
from tianximem.rank import expand_neighbors, merge_segments
from tianximem.rank.neighbor import SelectedMemory
from tianximem.retrieve import Candidate
from tianximem.store.sqlite_store import SqliteStore, make_pair_id

#: 脚手架用的 `request_id`——同一条"线"上的块都在这**一次 Add** 里。
_LINE = "r-line"

# ── 脚手架 ──────────────────────────────────────────────────────────────


def _line(
    store: SqliteStore,
    indices: range | list[int],
    *,
    user_id: str = "u1",
    session_id: str = "s1",
) -> list[str]:
    """**在一次 Add 内**落满 `0..max(indices)`，返回 `indices` 那几个块的 `memory_id`。

    ⚠ 想造"对话里中间缺了一块"，**不要**在这里少落一块——那样链是断的（那正是"缺块"）。
    想造"存在但没被选中"，就把它**落下来、但别放进 `ranked`**。
    """
    return seed_line(store, indices, user_id=user_id, session_id=session_id, request_id=_LINE)


def _ranked(*memory_ids: str) -> list[Candidate]:
    """按给定顺序造一批候选（名次 = 位置，0-based 连续）。"""
    return [Candidate(mid, rank) for rank, mid in enumerate(memory_ids)]


def _by_id(selected: tuple[SelectedMemory, ...]) -> dict[str, SelectedMemory]:
    return {m.memory_id: m for m in selected}


def _expand(
    store: SqliteStore, ranked: list[Candidate], *, radius: int = 1, **kwargs
) -> tuple[SelectedMemory, ...]:
    """⚠ **半径在这里显式给（默认 1），不要吃 `DEFAULT_RADIUS`。**

    本文件测的**就是**扩窗，而产品默认是 **`radius=0`（= 不扩窗）**（**D31**）——
    继承它会让整个文件**静默地什么都不测**：返回的段照样合法、`id` 照样对，
    只有"邻居有没有进来"这一件事没了。要关扩窗的用例**自己写 `radius=0`**。
    """
    return expand_neighbors(ranked, store=store, radius=radius, **kwargs).selected


def _idx(store: SqliteStore, session_id: str, index: int, *, user_id: str = "u1") -> str:
    """按位置**直接算出** `memory_id`——用来断言"某一行**不在**结果里"。

    ⚠ 位置是 `(request_id=_LINE, local_index=index)`（与 `_line` 同一个约定）。
    """
    return make_pair_id(user_id, session_id, _LINE, index)


def _set_qa(store: SqliteStore, memory_id: str, question: str, answer: str) -> None:
    """改写一对的正文。

    ⚠ 生产代码里**没有**"更新正文"的入口（正文只在配对时写入一次），所以这里直接
    `UPDATE`。顺带它还证明了一件事：`content` 是从真源**现读**的，不是候选里带来的。
    """
    with store.transaction() as conn:
        conn.execute(
            "UPDATE qa_pairs SET question = ?, answer = ? WHERE id = ?",
            (question, answer, memory_id),
        )


def _set_event_time(store: SqliteStore, memory_id: str, event_time: int) -> None:
    """给一对设上 `event_time`——锚点的 `created_at` 靠它（§11.3）。"""
    with store.transaction() as conn:
        conn.execute("UPDATE qa_pairs SET event_time = ? WHERE id = ?", (event_time, memory_id))


# ══ 一、保留全部精排结果 + 只扩前 N 条 ═══════════════════════════════════


def test_all_reranked_candidates_are_kept(store: SqliteStore) -> None:
    """**`R1…RN` 一条都不许删**，且每条的 `rerank_rank` 必须是真的。

    ⚠ 最容易写错的版本是"只保留前 30 条"——那样 `top_k=100` 永远填不满，
    而且**不会报错**：返回的每一条看起来都合法，只是少了一大截。
    """
    ids = _line(store, range(40))

    got = expand_neighbors(_ranked(*ids), store=store, seed_limit=30, radius=1)

    assert len(got.selected) == 40  # 一条不少
    by_id = _by_id(got.selected)
    for rank, mid in enumerate(ids):
        assert by_id[mid].rerank_rank == rank
        assert by_id[mid].is_neighbor is False


def test_only_the_top_seed_limit_candidates_expand(store: SqliteStore) -> None:
    """种子数**精确等于** `seed_limit`——第 `seed_limit + 1` 名之后**不扩**。

    构造：0…9 连续落库（**一条 Add 内的链**），候选只挑 `4`（名次 0）、`5`（名次 1）、
    `0`（名次 2）。此时每颗种子能扩出什么完全由 `seed_limit` 决定。
    """
    ids = _line(store, range(10))
    ranked = _ranked(ids[4], ids[5], ids[0])

    expectations = [
        (0, ()),  # 一颗种子都不扩
        (1, (3,)),  # 只有名次 0（块 4）⇒ ±1 ⇒ 补进块 3
        (2, (3, 6)),  # 加上名次 1（块 5）⇒ 再补块 6
        (3, (1, 3, 6)),  # 加上名次 2（块 0）⇒ 再补块 1
    ]
    for seed_limit, expected_pairs in expectations:
        got = expand_neighbors(ranked, store=store, seed_limit=seed_limit, radius=1)
        neighbors = {m.local_index for m in got.selected if m.is_neighbor}
        assert neighbors == set(expected_pairs), f"seed_limit={seed_limit}"
        assert got.neighbors_added == len(expected_pairs)
        # 不论扩不扩，三条候选**始终在里面**
        assert {ids[4], ids[5], ids[0]} <= set(_by_id(got.selected))


def test_candidates_after_the_seed_limit_are_kept_but_not_expanded(store: SqliteStore) -> None:
    """`rank31+`：**保留但不扩**——它们的邻域不许被拉进来（§10 的种子数）。

    构造：0…63 连续落库（一条链）。候选 = 块 20…49（名次 0…29，种子）
    + 块 0…9（名次 30…39，尾巴）。两组邻域不相交。
    """
    ids = _line(store, range(64))
    seeds, tail = ids[20:50], ids[0:10]

    got = expand_neighbors(_ranked(*(seeds + tail)), store=store, seed_limit=30, radius=1)
    by_id = _by_id(got.selected)

    # ① 尾巴一条不少，名次是真的，且都标着"不是邻居"
    for offset, mid in enumerate(tail):
        assert by_id[mid].rerank_rank == 30 + offset
        assert by_id[mid].is_neighbor is False

    # ② 尾巴的邻居**没有**被拉进来：块 10 只可能来自块 9 或块 11
    assert _idx(store, "s1", 10) not in by_id

    # ③ 对照（阳性）：种子那侧的邻居**在**里面 —— 否则上面那条断言是空过的
    edge = by_id[_idx(store, "s1", 19)]
    assert edge.is_neighbor is True
    assert edge.rerank_rank is None


def test_neighbor_already_a_candidate_is_not_added_twice(store: SqliteStore) -> None:
    """`neighbor.memory_id` 已在 `selected` 里 ⇒ **跳过，不覆盖**。

    ⚠ 覆盖的后果不是"多一条"，而是**那条候选的真实名次被抹成 `None`**——
    它会从段优先级里消失，于是段被排到后面去，而**看起来只是"排序怪怪的"**。

    构造：候选 = 块 2（名次 0）、块 3（名次 1）、块 0（名次 2）。
    块 3 落在块 2 的邻域里 ⇒ 它必须**保住名次 1**，不能被降级成邻居。
    """
    ids = _line(store, range(5))
    ranked = _ranked(ids[2], ids[3], ids[0])

    got = expand_neighbors(ranked, store=store, seed_limit=30, radius=1)
    by_id = _by_id(got.selected)

    pairs = [m.local_index for m in got.selected]
    assert sorted(pairs) == [0, 1, 2, 3, 4]
    assert len(pairs) == len(set(pairs))  # 结构上就不可能重复
    assert got.neighbors_added == 2  # 块 1 与块 4（块 3 是候选，不是新增）

    # ★ 落在别人邻域里的那条候选，名次**没有被抹掉**
    assert by_id[ids[3]].rerank_rank == 1
    assert by_id[ids[3]].is_neighbor is False
    # 同理：被两个种子同时扩到的块 1 只加了一次，且它是邻居
    assert by_id[_idx(store, "s1", 1)].is_neighbor is True
    assert by_id[ids[2]].rerank_rank == 0
    assert by_id[ids[0]].rerank_rank == 2


def test_overlapping_neighborhoods_are_deduplicated(store: SqliteStore) -> None:
    """两个种子的邻域重叠 ⇒ **合并成一份**，`neighbors_added` 数的是**净值**。"""
    ids = _line(store, range(6))
    ranked = _ranked(ids[1], ids[3])  # 邻域 {0,1,2} 与 {2,3,4}，在块 2 上重叠

    got = expand_neighbors(ranked, store=store, seed_limit=30, radius=1)

    pairs = sorted(m.local_index for m in got.selected)
    assert pairs == [0, 1, 2, 3, 4]  # 块 2 只出现一次
    assert len(got.selected) == len(set(_by_id(got.selected)))  # 结构上就不可能重复
    assert got.neighbors_added == 3  # 0 / 2 / 4


def test_new_neighbors_carry_no_rank(store: SqliteStore) -> None:
    """新扩出来的邻居 `rerank_rank = None`——**绝不给人造排名**。

    编一个假名次（比如"接着最后一名往下排"）会让段优先级指向一条
    **从没被检索器选中过**的记忆，锚点也跟着错——而 `best_rank` 看起来完全正常。
    """
    ids = _line(store, range(3))
    got = expand_neighbors(_ranked(ids[1]), store=store, seed_limit=30, radius=1)

    neighbors = [m for m in got.selected if m.is_neighbor]
    assert len(neighbors) == 2
    assert all(m.rerank_rank is None for m in neighbors)
    assert {m.local_index for m in neighbors} == {0, 2}


def test_radius_widens_the_hop_count(store: SqliteStore) -> None:
    """`radius` 的**单位是记忆块**：`±2` 沿链前后各跳两跳（§10 的粒度提醒）。"""
    ids = _line(store, range(9))
    got = expand_neighbors(_ranked(ids[4]), store=store, seed_limit=30, radius=2)

    assert {m.local_index for m in got.selected} == {2, 3, 4, 5, 6}


def test_radius_zero_means_no_expansion(store: SqliteStore) -> None:
    ids = _line(store, range(3))
    got = expand_neighbors(_ranked(ids[1]), store=store, seed_limit=30, radius=0)

    assert [m.local_index for m in got.selected] == [1]
    assert got.neighbors_added == 0


def test_expansion_stops_at_a_break_in_the_chain(store: SqliteStore) -> None:
    """**链上缺一环就停**（D28）：`0` 的 `next` 指着 `1`，而 `1` 不在库里。

    ⚠ 这正是"读指针、不数下标"的正面证据：`0` 与 `2` 的下标看着像只差 2，
    但它们在**对话结构上**不相邻——中间少的那块是**永远补不回来的**。
    """
    ids = _line(store, range(3))
    with store.transaction() as conn:
        conn.execute("DELETE FROM qa_pairs WHERE id = ?", (ids[1],))

    got = expand_neighbors(_ranked(ids[0]), store=store, seed_limit=30, radius=5)

    assert {m.memory_id for m in got.selected} == {ids[0]}  # 一步都跳不出去
    assert got.neighbors_added == 0


# ══ 二、隔离与边界：跨 session / 跨 Add 是硬红线，越界不许炸 ═══════════════


def test_expansion_never_crosses_sessions_or_users(store: SqliteStore) -> None:
    """**邻接只在一次 Add 内**（D28）⇒ 跨 session / 跨 user 的**结构上就到不了**。

    构造：同一个 `request_id`、同一个 `local_index` 的链在三个 (user, session) 组合下
    各落一条。三条候选各自只该看到**自己那条链**上的邻居。
    """
    a = _line(store, range(3), user_id="u1", session_id="s1")
    b = _line(store, range(3), user_id="u1", session_id="s2")
    c = _line(store, range(3), user_id="u2", session_id="s1")

    got = expand_neighbors(_ranked(a[0], b[0], c[0]), store=store, seed_limit=30, radius=1)
    got_ids = set(_by_id(got.selected))

    assert got_ids == {a[0], a[1], b[0], b[1], c[0], c[1]}  # 各自只扩到自己的链上
    assert {m.pair.session_id for m in got.selected if m.memory_id in set(a)} == {"s1"}
    assert {m.pair.user_id for m in got.selected if m.memory_id in set(a)} == {"u1"}


def test_expansion_never_crosses_adds(store: SqliteStore) -> None:
    """**不同 Add 之间永不相邻**——即使 user / session 都一样、`local_index` 也接得上。

    ⚠ 这是 D28 的核心断言：`Add1` 的末尾与 `Add2` 的开头**在位置上看不出边界**
    （各自从 0 起），但链上就是断的。老的"下标 ±1"写法会在这里静默跨过去。
    """
    add1 = _line(store, range(3), session_id="s1")  # r-line
    add2 = seed_line(store, range(3), session_id="s1", request_id="r-other")

    got = expand_neighbors(_ranked(add1[1]), store=store, seed_limit=30, radius=5)

    assert {m.memory_id for m in got.selected} == set(add1)  # 一步都不越界
    assert not (set(got.selected) & set(add2))


def test_expansion_at_chain_boundaries(store: SqliteStore) -> None:
    """链的头尾：走到 `prev=None` / `next=None` 就停——**都不许出事**。"""
    ids = _line(store, range(3))

    head = expand_neighbors(_ranked(ids[0]), store=store, seed_limit=30, radius=1)
    tail = expand_neighbors(_ranked(ids[2]), store=store, seed_limit=30, radius=1)

    assert sorted(m.local_index for m in head.selected) == [0, 1]
    assert sorted(m.local_index for m in tail.selected) == [1, 2]
    assert all(m.local_index >= 0 for m in head.selected + tail.selected)


def test_deleted_middle_row_is_a_break_not_a_bridge(store: SqliteStore) -> None:
    """被删掉的那一行**不会**让两侧"接上"——指针指着的是一个不存在的 id。"""
    ids = _line(store, range(4))
    with store.transaction() as conn:
        conn.execute("DELETE FROM qa_pairs WHERE id = ?", (ids[1],))
        conn.execute("DELETE FROM qa_pairs WHERE id = ?", (ids[2],))

    got = expand_neighbors(_ranked(ids[0], ids[3]), store=store, seed_limit=30, radius=3)

    assert {m.memory_id for m in got.selected} == {ids[0], ids[3]}
    # 合并那一侧同样断开
    segments = merge_segments(got.selected, counter=FakeCounter())
    assert [s.length for s in segments] == [1, 1]


def test_missing_source_row_is_counted_not_fabricated(store: SqliteStore) -> None:
    """Qdrant 给了 `id`、真源里查不到 ⇒ **丢掉并计数**（`missing_rows`）。

    非 0 说明**索引与真源脱钩了**（Qdrant 是派生索引、`id` 与 SQLite 一一对应）。
    静默丢掉会让"少了哪一条"永远查不出来。
    """
    a = _line(store, range(1))[0]
    ghost = make_pair_id("u1", "s1", "r-ghost", 0)  # 库里没有这一行

    got = expand_neighbors(_ranked(a, ghost), store=store, seed_limit=30, radius=1)

    assert got.missing_rows == 1
    assert len(got.selected) == 1
    assert got.selected[0].memory_id == a


def test_empty_candidates_short_circuit(store: SqliteStore) -> None:
    """没有候选 ⇒ 空结果，**且不查库**（`by_id` 为空时直接返回）。"""
    got = expand_neighbors([], store=store, seed_limit=30, radius=1)
    assert got.selected == ()
    assert got.neighbors_added == 0
    assert got.missing_rows == 0


@pytest.mark.parametrize(("seed_limit", "radius"), [(-1, 1), (30, -1)])
def test_negative_parameters_are_rejected(store: SqliteStore, seed_limit: int, radius: int) -> None:
    """负数参数**响亮失败**——它不会报错，只会让扩窗"永远不触发"或"到处越界"。"""
    with pytest.raises(ValueError, match="不得为负"):
        expand_neighbors([], store=store, seed_limit=seed_limit, radius=radius)


# ══ 三、Context Segment Merge：显式相邻才合并 ════════════════════════════


def test_the_spec_example_merges_into_three_segments(
    store: SqliteStore, counter: FakeCounter
) -> None:
    """**规格里那个例子**：块 `8,9,10,11,15,16,30` ⇒ `8..11` / `15..16` / `30`。

    ⚠ 断言的是一条**规则**（`前一条.next == 这一条.id`），不是一个实现：
    中间缺一个块就说明那段对话**不完整**，硬拼成一整段会让模型读到自己以为连续、
    实际缺了一环的上下文——而 `content` 里**看不出来**。

    ⚠ **缺口靠"存在但没被选中"表达**：这里把 0..30 全部落进**同一条链**，
    只把 8..11 / 15,16 / 30 放进 `ranked`。
    """
    all_ids = _line(store, range(31))
    picked = [all_ids[i] for i in (8, 9, 10, 11, 15, 16, 30)]

    # `radius=0`：本用例测的是**合并**，不测扩窗（扩窗会把缺口填上）
    segments = merge_segments(_expand(store, _ranked(*picked), radius=0), counter=counter)

    assert [(s.start_local_index, s.end_local_index) for s in segments] == [
        (8, 11),
        (15, 16),
        (30, 30),
    ]
    assert [s.length for s in segments] == [4, 2, 1]
    assert [s.source_memory_ids for s in segments] == [
        tuple(picked[0:4]),
        tuple(picked[4:6]),
        (picked[6],),
    ]


@pytest.mark.parametrize("n", [2, 3, 4, 12])
def test_contiguous_runs_of_any_length_merge_into_one_segment(
    store: SqliteStore, counter: FakeCounter, n: int
) -> None:
    """连续 2 / 3 / 4 / N 条 ⇒ **恰好一段**（N 段被一条条 `append` 成一段）。"""
    ids = _line(store, range(n))
    segments = merge_segments(_expand(store, _ranked(*ids)), counter=counter)

    assert len(segments) == 1
    assert segments[0].length == n
    assert segments[0].start_local_index == 0
    assert segments[0].end_local_index == n - 1
    assert segments[0].source_memory_ids == tuple(ids)


def test_a_gap_starts_a_new_segment(store: SqliteStore, counter: FakeCounter) -> None:
    """**不相邻就新建一段**——差 1 是相邻，差 2 就断开。"""
    all_ids = _line(store, range(7))
    picked = [all_ids[i] for i in (0, 1, 3, 4, 6)]  # 2 与 5 也在链上，只是没被选中
    segments = merge_segments(_expand(store, _ranked(*picked), radius=0), counter=counter)

    assert [(s.start_local_index, s.end_local_index) for s in segments] == [
        (0, 1),
        (3, 4),
        (6, 6),
    ]


def test_adjacent_member_extends_the_previous_segment_rather_than_creating_one(
    store: SqliteStore, counter: FakeCounter
) -> None:
    """相邻的那一条**直接 append 进前一段**，不新建。

    ⚠ 与上一条的区别在于"新建了几段"：同样是 4 条记忆，连续的合并后**段数更少**。
    段数少了，`top_k` 名额与 token 预算都省下来——这正是合并的意义。
    """
    contiguous = _line(store, range(4), session_id="s1")
    split_line = seed_line(store, range(42), session_id="s2", request_id="r-split")
    split = [split_line[i] for i in (30, 31, 40, 41)]  # 32..39 也在链上，只是没选中

    c = merge_segments(_expand(store, _ranked(*contiguous), radius=0), counter=counter)
    s = merge_segments(_expand(store, _ranked(*split), radius=0), counter=counter)

    assert [seg.length for seg in c] == [4]  # 一段
    assert [seg.length for seg in s] == [2, 2]  # 两段


def test_two_adds_in_one_session_never_merge(store: SqliteStore, counter: FakeCounter) -> None:
    """**同一 session 的两次 Add 永不合并**（D28）——即使各自的 `local_index` 接得上。

    ⚠ 分组键里带着 `request_id`、而判据读的是存储里的指针——两道都在保证这件事。
    两次 Add 的块**必然**在下标上"接得上"（各自从 0 起），
    所以这条边界**每次**都被顶到脸上，不需要特意构造。
    """
    first = _line(store, range(2), session_id="s1")  # r-line
    second = seed_line(store, range(2), session_id="s1", request_id="r-second")

    segments = merge_segments(_expand(store, _ranked(*(first + second))), counter=counter)

    assert len(segments) == 2
    assert all(s.length == 2 for s in segments)
    assert {s.request_id for s in segments} == {"r-line", "r-second"}
    assert len({s.session_id for s in segments}) == 1  # 同一个 session，**照样分两段**


def test_different_sessions_never_merge(store: SqliteStore, counter: FakeCounter) -> None:
    """**不同 session 永远不能合并**。"""
    s1 = _line(store, range(2), session_id="s1")
    s2 = _line(store, range(2), session_id="s2")

    segments = merge_segments(_expand(store, _ranked(*(s1 + s2))), counter=counter)

    assert len(segments) == 2
    assert all(s.length == 2 for s in segments)
    assert {s.session_id for s in segments} == {"s1", "s2"}


def test_different_users_never_merge(store: SqliteStore, counter: FakeCounter) -> None:
    """跨 user 同理——那是 §2.2 的隔离红线，不是排序问题。"""
    u1 = _line(store, range(2), user_id="alice", session_id="same")
    u2 = _line(store, range(2), user_id="bob", session_id="same")

    segments = merge_segments(_expand(store, _ranked(*(u1 + u2))), counter=counter)

    assert len(segments) == 2
    assert {s.user_id for s in segments} == {"alice", "bob"}


def test_members_within_a_segment_are_ordered_by_local_index(
    store: SqliteStore, counter: FakeCounter
) -> None:
    """段内 = **`local_index` 升序**（那一次 Add 的顺序），**不把种子提到最前**（§11.2）。

    ⚠ 种子在段中间是常态（它前面的邻居才是上下文）。
    把种子提到最前会把一段对话**拦腰截断**，而答案阶段按前缀截断。
    """
    ids = _line(store, range(4))
    # 名次故意与位置反过来：种子是第 2 块（名次 0）与第 1 块（名次 1）
    ranked = _ranked(ids[2], ids[1])

    segments = merge_segments(_expand(store, ranked), counter=counter)

    assert segments[0].source_memory_ids == (ids[0], ids[1], ids[2], ids[3])
    assert segments[0].start_local_index == 0
    # 种子**不在**段首——它在它本来该在的位置上
    assert segments[0].anchor_memory_id == ids[2]
    assert segments[0].source_memory_ids[0] != segments[0].anchor_memory_id


# ══ 四、段优先级与锚点 ══════════════════════════════════════════════════


def test_best_rank_is_the_minimum_among_real_candidates(store: SqliteStore) -> None:
    """第 9 块邻居 / 第 10 块名次 2 / 第 11 块名次 57 / 第 12 块邻居
    ⇒ `best_rank = 2`，锚点 = 第 10 块。

    ⚠ 段里**有多条真实候选时取最小的那个**——取最后一条、或取段内第一条，
    都会让"段间按相关性排序"变成"段间按某个别的东西排序"。
    """
    ids = _line(store, range(13))
    picked = [ids[i] for i in (9, 10, 11, 12)]
    # 刻意造一个跨过 seed_limit 的远名次（57）：它是真候选，只是排在很后面
    ranked = [Candidate(picked[1], 2), Candidate(picked[2], 57)]

    # seed_limit 给足以便两条候选都能扩（本用例测的是合并语义，不是种子数）
    expansion = expand_neighbors(ranked, store=store, seed_limit=100, radius=1)
    segments = merge_segments(expansion.selected, counter=FakeCounter())

    assert len(segments) == 1
    seg = segments[0]
    assert (seg.start_local_index, seg.end_local_index) == (9, 12)
    assert seg.best_rank == 2
    assert seg.anchor_memory_id == picked[1]
    assert seg.length == 4
    assert seg.rerank_member_count == 2  # 9/12 是邻居
    assert {m.local_index for m in expansion.selected if m.is_neighbor} == {9, 12}


def test_neighbors_do_not_change_segment_priority(store: SqliteStore) -> None:
    """**邻居不参与排名**：把扩窗关掉，段的 `best_rank` 与锚点**一点不变**。

    这就是 §13"开关只影响它命名的那一件事"在扩窗上的落点：
    `radius=0`（关掉扩窗）只该**少几条上下文**，**不该动排序**。
    """
    ids = _line(store, range(13))
    picked = [ids[i] for i in (9, 10, 11, 12)]
    ranked = [Candidate(picked[1], 2), Candidate(picked[2], 57)]

    # ⚠ `seed_limit` 两次都给足（本用例要变的**只有** `radius` 这一个旋钮）
    without = merge_segments(
        _expand(store, ranked, seed_limit=100, radius=0), counter=FakeCounter()
    )
    with_neighbors = merge_segments(
        _expand(store, ranked, seed_limit=100, radius=1), counter=FakeCounter()
    )

    assert len(without) == len(with_neighbors) == 1
    assert without[0].length == 2  # 只有那两条真实候选
    assert with_neighbors[0].length == 4  # 加上扩出来的第 9 / 12 块

    # ★ 关掉扩窗只减少上下文：优先级与锚点原封不动
    assert without[0].best_rank == with_neighbors[0].best_rank == 2
    assert without[0].anchor_memory_id == with_neighbors[0].anchor_memory_id == picked[1]
    assert without[0].anchor_event_time == with_neighbors[0].anchor_event_time


def test_anchor_comes_from_the_best_ranked_candidate_not_the_first_member(
    store: SqliteStore,
) -> None:
    """锚点 = **真实名次最好**的那条候选；`anchor_event_time` 跟着它走。

    ⚠ 锚点决定了响应的 `id` 与 `created_at`（§11.2 的组间顺序 = 种子名次序）。
    取段内第一条会得到一个"这段对话里其实没被选中"的 `id`。
    """
    ids = _line(store, range(3))
    # 只有块 1 有 event_time——锚点若不是它，日期就会变空串
    _set_event_time(store, ids[1], 1683525360000)

    segments = merge_segments(_expand(store, _ranked(ids[1], ids[0])), counter=FakeCounter())

    assert len(segments) == 1
    seg = segments[0]
    assert seg.anchor_memory_id == ids[1]
    assert seg.anchor_event_time == 1683525360000
    assert seg.best_rank == 0
    assert seg.source_memory_ids[0] != seg.anchor_memory_id  # 第 0 块才是第一条


def test_segments_are_sorted_by_best_rank(store: SqliteStore, counter: FakeCounter) -> None:
    """**段间按 `best_rank` 升序**——相关性顺序，不是位置顺序，也不是段长。

    ⚠ 段内是"Add 内顺序"、段间是"相关性顺序"（§11.2 的两条规则）。
    两边搞反不会报错：返回的每一段都完整、合法，只是**最好的一段不在最前面**——
    而答案阶段按前缀截断，于是最相关的那段可能整个被丢掉。
    """
    late = _line(store, range(2), session_id="s_late")
    mid = _line(store, range(1), session_id="s_mid")
    early = _line(store, range(3), session_id="s_early")

    # 名次 0 / 1 / 2 分别给 mid / late / early 的首条，其余按序排在其后
    ranked = _ranked(mid[0], late[0], early[0], early[1], early[2])
    segments = merge_segments(_expand(store, ranked), counter=counter)

    assert [s.best_rank for s in segments] == [0, 1, 2]
    assert [s.session_id for s in segments] == ["s_mid", "s_late", "s_early"]
    assert [s.length for s in segments] == [1, 2, 3]  # 长的那段反而排最后


def test_segment_content_is_the_rendered_run_in_order(store: SqliteStore) -> None:
    """段的 `content` = 段内各对**按 Add 内顺序**渲染后、用 `SEGMENT_SEP` 连起来。

    ⚠ 走 [`common/render`](../../src/tianximem/common/render.py)——它是渲染的**唯一实现**
    （不变式 I1）。这里断言的是"段这一层没有自己拼字符串"。
    """
    ids = _line(store, range(2), session_id="s1")
    _set_qa(store, ids[0], "Q0", "A0")
    _set_qa(store, ids[1], "Q1", "A1")

    segments = merge_segments(_expand(store, _ranked(ids[0], ids[1])), counter=FakeCounter())

    assert segments[0].content == render_segment([render("Q0", "A0"), render("Q1", "A1")])
    assert segments[0].content == "Q: Q0\nA: A0\nQ: Q1\nA: A1"
    assert segments[0].token_count == len(segments[0].content)  # 对**拼好的串**数的


def test_segment_token_count_is_not_the_sum_of_parts(store: SqliteStore) -> None:
    """`token_count` 对**段这一整串**计数——其中包括**连接符**。

    ⚠ 两种算法在字符计数器下**恰好相差一个 `SEGMENT_SEP`**（本用例钉的就是那一个），
    而真分词器下它们**根本不等**：BPE 的合并可以跨越拼接边界。
    漏掉连接符会让 100 个段的输出比预算多出 99 个 token，
    而那正好把 AML 的前缀截断点推进一段证据的中间。
    """
    ids = _line(store, range(2), session_id="s1")
    _set_qa(store, ids[0], "Q0", "A0")
    _set_qa(store, ids[1], "Q1", "A1")

    selected = _expand(store, _ranked(ids[0], ids[1]))  # 两对都进了 selected
    seg = merge_segments(selected, counter=FakeCounter())[0]

    parts = sum(len(render(m.pair.question, m.pair.answer)) for m in selected)
    assert seg.content == "Q: Q0\nA: A0\nQ: Q1\nA: A1"
    assert seg.token_count == len(seg.content) == 23
    assert seg.token_count == parts + 1  # ★ 多出来的正是那个 SEGMENT_SEP


def test_every_segment_has_a_real_candidate(store: SqliteStore, counter: FakeCounter) -> None:
    """**每段都至少有一条真实候选**——扩窗沿链跳，种子必与它扩出来的邻居同链。

    这条是 `_build_segment` 里"锚点一定存在"那个假设的可观测形式。
    真出现无候选的段，说明邻接的构造方式变了（那时 `best_rank` 会是 `_NO_RANK`）。
    """
    ids = _line(store, range(12))
    for seed_limit in (1, 5, 12):
        selected = _expand(store, _ranked(*ids), seed_limit=seed_limit)
        for seg in merge_segments(selected, counter=counter):
            assert seg.rerank_member_count >= 1


# ══ 五、整条链上的位置：`top_k` 在合并**之后** ═══════════════════════════


def test_raw_memory_count_can_exceed_top_k(wired_expanding: Wired) -> None:
    """扩窗会把 raw 数抬到 `top_k` 之上——**`top_k` 约束的是段数，不是 raw 数**。

    这里 1 个候选扩出 2 个邻居 = 3 条 raw，`top_k=1` 仍然合法地返回 1 段。
    若在扩窗**之前**按 raw 数截断，邻居会被砍掉，返回的段就**比该有的少**。
    """
    wired = wired_expanding
    ids = seed_line(wired.store, range(3))
    wired.qdrant.by_user["u1"] = [ids[1]]  # 只有一个候选

    got = wired.search(top_k=1)

    assert got.count == 1
    assert got.items[0].content == "Q: q0\nA: a0\nQ: q1\nA: a1\nQ: q2\nA: a2"
    assert got.items[0].id == ids[1]  # 锚点 = 唯一那条真实候选


def test_top_k_applies_to_segments_after_merging(wired: Wired) -> None:
    """`top_k` 在**段合并之后**生效：10 条相邻的候选只该占 **1** 个名额。

    ⚠ 这就是"不要在扩窗阶段按 raw memory 数截断"的正面证据。
    """
    ids = seed_line(wired.store, range(10))
    wired.qdrant.by_user["u1"] = ids  # 10 个候选，但它们在**同一次 Add 里连续**

    got = wired.search(top_k=10)

    assert got.count == 1  # 不是 10
    assert got.considered_segments == 1
    assert got.items[0].content.count("\n") == 19  # 10 对 × 2 行 + 9 个连接符
    assert got.items[0].id == ids[0]


def test_anchor_drives_id_and_created_at_end_to_end(wired_expanding: Wired) -> None:
    """端到端：响应的 `id` 与 `created_at` 都来自**锚点**（= `best_rank` 那条候选）。

    锚点不是段里第一条——第 0 块是扩出来的邻居，它**没有** `event_time`；
    日期必须来自第 1 块（真实候选、名次 0）。
    """
    wired = wired_expanding
    ids = seed_line(wired.store, range(3))
    _set_event_time(wired.store, ids[1], 1683525360000)
    wired.qdrant.by_user["u1"] = [ids[1]]

    item = wired.search(top_k=1).items[0]

    assert item.id == ids[1]
    assert item.created_at == "2023-05-08"
    assert item.content.startswith("Q: q0")  # 段的第一条是邻居


def test_token_budget_applies_to_the_merged_output(wired: Wired) -> None:
    """预算对**合并后的最终串**生效——段少了，同样的预算就能多装内容。

    本用例只断言"预算没被破"与"段是原子的"，具体算术在 `test_packaging.py`。
    """
    ids = seed_line(wired.store, range(6))
    wired.qdrant.by_user["u1"] = ids

    got = wired.search(top_k=10)

    assert got.output_tokens <= wired.budget_tokens
    assert got.count <= 10
    # 段是原子单位：返回的那一段的内容与段本身逐字相等（没有被裁掉半条）
    assert got.items[0].content.startswith("Q: q0\nA: a0")
    assert got.items[0].content.endswith("Q: q5\nA: a5")


def test_a_budget_too_small_for_the_first_segment_yields_no_data(
    wired: Wired, monkeypatch: pytest.MonkeyPatch
) -> None:
    """预算小到装不下**第一段** ⇒ 返回空 `data`，**而不是半个段**。

    ⚠ 空结果是合法的（§2.1）；半个段不合法——它会让模型读到半截对话。
    ⚠ 这里直接改私有字段：`wired` 的预算**刻意给得极大**（见 fixture 的说明），
    好让它自己的用例测别的东西。要测预算就得**在边界上**设一个值。
    """
    ids = seed_line(wired.store, range(4))
    wired.qdrant.by_user["u1"] = ids
    monkeypatch.setattr(wired.services.search, "_budget_tokens", 3)  # 一段至少 11 个字符

    got = wired.search(top_k=10)

    assert got.items == ()
    assert got.truncated_by_budget is True
    assert got.considered_segments == 1


def test_expanded_content_is_identical_to_a_direct_render(wired_expanding: Wired) -> None:
    """**不变式 I1 的端到端版**：邻居进段走的也是 `common/render`，
    与直接渲染同一对的结果**逐字相同**。

    两处渲染一旦分叉，"检索命中的是什么"与"模型读到的是什么"就漂移了，
    而**这种漂移不会报错**。
    """
    wired = wired_expanding
    ids = seed_line(wired.store, range(3))
    wired.qdrant.by_user["u1"] = [ids[1]]

    item = wired.search(top_k=1).items[0]

    assert item.content == render_segment([render(f"q{i}", f"a{i}") for i in range(3)])


# ── 段内顺序的三个变体（§11.2 的"组内顺序"，可消融项）──────────────────────
def _segment(store: SqliteStore, ids: list[str], placement: str):
    """**只读**：以第 1 块为唯一候选合并出那一段（种子落在段中间）。"""
    ranked = _ranked(ids[1])
    got = expand_neighbors(ranked, store=store, seed_limit=30, radius=1)
    segments = merge_segments(got.selected, counter=FakeCounter(), seed_placement=placement)
    assert len(segments) == 1
    return segments[0]


def test_seed_placement_keep_is_the_default_and_time_ordered(store: SqliteStore) -> None:
    """`keep`（默认）＝纯位置序，种子在它本来的位置——**既有行为一个字节都不许变**。"""
    ids = _line(store, range(3))
    segment = _segment(store, ids, "keep")

    lines = segment.content.split("\n")
    assert lines[0].endswith("q0") and lines[-1].endswith("a2")
    assert segment.source_memory_ids == tuple(ids)


def test_seed_placement_front_moves_only_the_text(store: SqliteStore) -> None:
    """`front`：种子排到段首，**但 `id` / 段数 / 锚点 / 名次全不变**。

    ⚠ 这条纯度是关键：段优先级只看真实候选（`best_rank` / 锚点），
    若换序顺带改了它们，§13 的对照就不成立了——而**看起来只是"顺序变了"**。
    """
    ids = _line(store, range(3))
    keep = _segment(store, ids, "keep")
    front = _segment(store, ids, "front")

    seed_text = "Q: q1\nA: a1"  # 种子（第 1 块）渲染出来的那两行
    assert front.content.startswith(seed_text)  # 种子在最前
    assert front.content != keep.content  # 顺序确实变了
    # ★ 其余一切逐字相同
    assert front.source_memory_ids == keep.source_memory_ids
    assert front.anchor_memory_id == keep.anchor_memory_id
    assert front.best_rank == keep.best_rank
    assert front.start_local_index == keep.start_local_index
    assert front.end_local_index == keep.end_local_index
    # 段内的对**一个不多一个不少**（只是重排）
    assert sorted(front.content.split("\n")) == sorted(keep.content.split("\n"))


def test_seed_placement_echo_repeats_the_seed_and_keeps_chronology(store: SqliteStore) -> None:
    """`echo`：种子在段首重复一份，**下面完整的顺序块原样保留**（时间不断）。

    它是"既要显眼、又不打破时序"的那条路——代价是多一对的 token。
    """
    ids = _line(store, range(3))
    keep = _segment(store, ids, "keep")
    echo = _segment(store, ids, "echo")

    # 段首是种子那对的副本，其后**逐字**是原顺序块（⇒ 时序一次都没被打断）
    assert echo.content == "Q: q1\nA: a1" + "\n" + keep.content
    assert echo.anchor_memory_id == keep.anchor_memory_id
    assert echo.best_rank == keep.best_rank
