"""精排探针 —— `make probe-reranker`。

**接上真端点之后的第一件事不是看效果，是看它到底通不通。** 本脚本回答四个问题，
**四个都不涉及"结果好不好"**：

| # | 问题 | 判据 |
| --- | --- | --- |
| 1 | 端点能通吗 | 一次真请求拿到 200 |
| 2 | 线格式对得上吗 | 响应能过 `RemoteReranker._parse` 的四条集合校验 |
| 3 | 顺序**确实可能改变**吗 | 同一条链跑两遍（开/关精排），比较输出的 id 序 |
| 4 | 扩窗用的是**精排后**的名次吗 | `seed_limit=1` 时，被拉进来的邻居是谁 |

## 为什么检索那一段是**桩**

本探针要验的是 **rerank**，所以它把上游**故意排坏**：假 Qdrant 返回的候选序是
"最明显相关的那条排最后"。这不是偷懒，是**唯一能看见精排效果的方法**——
如果上游顺序本来就对，reranker 原样返回它也是对的，于是"调用了但没用结果"
与"调用了且结果一致"在输出上**一模一样**。

⇒ **本探针不评价检索**（那是 `contract-check` 与 `eval/harness` 的事），
也不消耗任何 Smoke 配额。

## 与 `check_env.py` / `contract-check` 的分工

| 工具 | 看什么 |
| --- | --- |
| `tools/check_env.py` | 环境通不通（Qdrant / 三段模型端点 / 密钥） |
| `eval/smoke/preflight.py` | 契约合不合规（14 条，打真 HTTP 的全链） |
| **本脚本** | **精排这一环的连通性 + 它有没有真的在链上起作用** |

⚠ **退出码**：`0` 端点与线格式都正常 · `1` 端点不可用或线格式不符 ·
`2` 前置条件不满足（没配 reranker）。
**顺序有没有变不影响退出码**——那是信息，不是判据（一个诚实但保守的 reranker
完全可能给出与 RRF 相同的顺序，而那不该让探针失败）。
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Final

from tianximem.common.config import ConfigError, load_config
from tianximem.common.render import render
from tianximem.rank import RemoteReranker, RerankUnavailable
from tianximem.retrieve import DenseArm, EvidenceChecker, HybridRetriever
from tianximem.service.app import build_reranker
from tianximem.service.pipeline import SearchPipeline
from tianximem.store.qdrant_store import ScoredMemoryId
from tianximem.store.sqlite_store import SqliteStore, make_pair_id

#: 本探针全部数据都落在这个 user 下（隔离字段，不影响别的数据）。
_USER: Final[str] = "probe-u"

EXIT_OK = 0
EXIT_PROBE_FAILED = 1
EXIT_PRECONDITION_FAILED = 2

#: 探针用的查询。**刻意让正确答案一眼可辨**——不是为了评效果，
#: 而是为了"顺序该不该变"有一个**可核对**的预期。
QUERY: Final[str] = "what time does the night train leave"

#: 明显相关的那一对。它落在 `s-match` 里，**并且那个 session 还有第二对**——
#: 第二对**不在检索候选里**，只有扩窗够得到 ⇒ "种子是不是它"可以直接看出来。
MATCH_QUESTION: Final[str] = "what time does the night train leave"
MATCH_ANSWER: Final[str] = "[assistant] 23:40 from platform 3"

#: 只有扩窗才够得到的那一对（**不是检索候选**）。
CONTEXT_QUESTION: Final[str] = "is the night train usually busy"
CONTEXT_ANSWER: Final[str] = "[assistant] never on sundays"

#: 干扰项：**每条各占一个 session**、每对只有一对。
#:
#: ⚠ **这是关键**：同一 session 里位置相邻的对会**合并成一段**
#: （D25 起"位置相邻" = 读时稠密序 `seq` 相邻，
#: 于是 6 条候选在输出里只剩 1 项——"顺序变没变"就**没有可观测的地方**了
#: （那样写的探针两遍输出一模一样，其实什么也没证明）。
#: 各占一个 session ⇒ 每对自成一段 ⇒ 段间顺序 = `best_rank` 顺序 = **精排名次的直接读数**。
DISTRACTORS: Final[list[tuple[str, str]]] = [
    ("do you like apples", "[assistant] yes, especially the green ones"),
    ("what is the weather like today", "[assistant] cloudy with a light breeze"),
    ("remind me to buy milk", "[assistant] added to your list"),
    ("how long is the flight to oslo", "[assistant] about two hours"),
    ("my favourite colour is teal", "[assistant] noted"),
    ("did the package arrive", "[assistant] it arrived on tuesday"),
]


def _rule(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


# ── 桩：本探针不看检索 ─────────────────────────────────────────────────


class _StubEmbedder:
    """假 embedder：只满足 `DenseArm` 的协议（**不打远程网关**）。"""

    dim = 8

    def encode(self, texts):  # noqa: ANN001, ANN201
        return [[0.0] * self.dim for _ in texts]


class _StubQdrant:
    """假 Qdrant：只满足 `HybridRetriever` 的两个调用，**按给定顺序回候选**。

    ⚠ 顺序是**倒着给的**（最相关的排最后）——理由见模块 docstring。

    ⚠ **分数必须两两不同**：给成同一个值会触发 V13 的并列 tie-break
    （`(-score, memory_id)`），检索层于是**按 id 重排**、桩给的顺序不再等于
    "RRF 顺序"——而探针下面还印着"RRF 给的顺序：[…]"。那是**探针自己的前提不成立**
    （第一次实测就是这么发生的：关精排那一臂的顺序既不是桩序、也没报错）。
    """

    #: 首名与末名的分数差——只要两两不同即可，量级无关（RRF 分数**不是**校准量）。
    _STEP = 1e-6

    def __init__(self, memory_ids: list[str]) -> None:
        self._ids = list(memory_ids)
        self.queries: list[str] = []

    def exists(self) -> bool:
        return True

    def hybrid_search(self, *, user_id: str, query_text: str, dense_vector, top_k: int):  # noqa: ANN001, ANN201
        self.queries.append(query_text)
        return [
            ScoredMemoryId(mid, 1.0 - rank * self._STEP)
            for rank, mid in enumerate(self._ids[:top_k])
        ]


# ── 探针 1：端点本身 ───────────────────────────────────────────────────


def probe_endpoint(reranker: RemoteReranker) -> bool:
    """打一次真请求，并把**分数**打出来。**线格式不对会当场抛 `RerankUnavailable`。**"""
    _rule("① 端点与线格式")
    documents = [render(MATCH_QUESTION, MATCH_ANSWER)] + [render(q, a) for q, a in DISTRACTORS]
    print(f"POST {reranker.url}")
    print(f"model（我们声明的）: {reranker.name}")
    print(f"documents         : {len(documents)} 篇")
    try:
        scores = reranker.score(query=QUERY, documents=documents)
    except RerankUnavailable as exc:
        print(f"\n✗ 端点不可用：{exc}")
        return False

    for i, score in sorted(enumerate(scores), key=lambda kv: -kv[1]):
        mark = "★" if i == 0 else " "
        text = MATCH_QUESTION if i == 0 else DISTRACTORS[i - 1][0]
        print(f"  {mark} {score:>10.6f}  {text[:52]}")
    print(f"\n✓ 200 + 线格式通过（{reranker.calls} 次调用，{reranker.last_latency_ms:.0f} ms）")
    print("★ = 与查询逐字相同的那篇：**它是信息，不是判据**——本探针不评效果。")
    return True


# ── 探针 2：它在链上真的起作用吗 ───────────────────────────────────────


def _build_pipeline(
    *, store: SqliteStore, qdrant: _StubQdrant, reranker: RemoteReranker | None
) -> SearchPipeline:
    return SearchPipeline(
        store=store,
        qdrant=qdrant,  # type: ignore[arg-type]
        retriever=HybridRetriever(
            store=qdrant,  # type: ignore[arg-type]
            dense=DenseArm(_StubEmbedder()),
        ),
        checker=EvidenceChecker(),
        counter=_CharCounter(),
        budget_tokens=1_000_000,  # 探针不看预算，别让它提前停
        seed_limit=1,  # ★ 只让**精排后的名次 0** 扩窗 —— 这样"种子是谁"一目了然
        radius=1,
        reranker=reranker,
    )


class _CharCounter:
    """1 字符 = 1 token。**探针不测预算**，所以不需要真分词器（也就不需要联网取 BPE）。"""

    name = "probe-char"

    def count(self, text: str) -> int:
        return len(text)


def probe_pipeline(reranker: RemoteReranker, *, top_k: int, radius: int) -> bool:
    """同一条链跑两遍（关精排 / 开精排），比较**段顺序**与**谁被扩窗**。"""
    _rule("② 精排有没有真的作用到链上")
    workdir = Path(tempfile.mkdtemp(prefix="tianxi-rerank-probe-"))
    try:
        store = SqliteStore.open(workdir / "probe.db")
        match_id, context_id, distractors = _seed_corpus(store)
        # RRF 顺序 = **故意排坏**：6 条干扰项在前，最相关的那条排在**最后**
        rrf_order = [*distractors, match_id]
        qdrant = _StubQdrant(rrf_order)
        print(f"检索候选（桩，**最相关的排最后**）：{_short(rrf_order)}")
        print(f"其中 ★ 是：{match_id[:6]}（`{MATCH_QUESTION[:40]}`）")
        print(f"只有扩窗够得到的那条：{context_id[:6]}（它在 ★ 的同一个 session 里）")

        baseline = _build_pipeline(store=store, qdrant=qdrant, reranker=None)
        without = baseline.run(user_id=_USER, query=QUERY, top_k=top_k)
        _show("关精排", without, note=f"rerank_disabled={baseline.rerank_disabled}")

        probe = _build_pipeline(store=store, qdrant=qdrant, reranker=reranker)
        with_rerank = probe.run(user_id=_USER, query=QUERY, top_k=top_k)
        _show(
            "开精排",
            with_rerank,
            note=f"rerank_calls={probe.rerank_calls}、degraded={probe.rerank_degraded}",
        )

        if probe.rerank_degraded:
            print("\n✗ 这一次降级了 —— 端点在这一跳失败了，先看 ① 的输出")
            return False
        if probe.rerank_calls != 1:
            print(f"\n✗ 一次 Search 调用了 {probe.rerank_calls} 次精排（应当恰好 1 次）")
            return False

        before = [item.id for item in without.items]
        after = [item.id for item in with_rerank.items]
        print(f"\n· 段顺序{'变了' if before != after else '没变'}")
        print(
            f"  ★ 那条在输出里的位置：关精排 = 第 {_position(before, match_id)} 位，"
            f"开精排 = 第 {_position(after, match_id)} 位"
        )
        # ⚠ 判据是**正文**，不是 id：`memory_id` 是哈希，它当然不会出现在 content 里。
        #    扩窗的产物是"邻居的正文进了某一段"，所以只能按正文找。
        was = CONTEXT_QUESTION in _all_content(without)
        now = CONTEXT_QUESTION in _all_content(with_rerank)
        print(f"· ★ 的邻居（{context_id[:6]}，`{CONTEXT_QUESTION[:34]}`）出现了吗")
        print(f"      关精排：{'是' if was else '否'}    开精排：{'是' if now else '否'}")
        print(f"  ⇒ 它**只可能由 Seed=★ 扩出来**（`seed_limit=1`、半径 ±{radius}）；")
        print("    关精排时种子是 RRF 名次 0（一条干扰项），它的 session 里没有别的对。")
        return True
    finally:
        _rmtree(workdir)


def _show(label: str, response, *, note: str) -> None:  # noqa: ANN001
    print(f"\n{label}（{note}）")
    print(f"  段数     : {response.count}（候选 {len(DISTRACTORS) + 1} 条）")
    print(f"  段顺序   : {_short([item.id for item in response.items])}")
    for i, item in enumerate(response.items[:3]):
        print(f"  第 {i + 1} 段   : {_one_line(item.content)}")


def _position(memory_ids: list[str], target: str) -> str:
    return str(memory_ids.index(target) + 1) if target in memory_ids else "不在里面"


def _all_content(response) -> str:  # noqa: ANN001
    return "\n".join(item.content for item in response.items)


def _seed_corpus(store: SqliteStore) -> tuple[str, str, list[str]]:
    """落进真源，返回 `(★ 的 id, 只有扩窗够得到的那个 id, 干扰项的 id)`。

    布局：

    ```text
    s-match   : pair0 = ★（明显相关）      pair1 = 只有扩窗够得到的那条
    s-distract-<i>: 每条干扰项**各占一个 session**（理由见 DISTRACTORS 的注释）
    ```

    ⚠ 干扰项的 id **在这里收集**，不去按 session 前缀回查：它们的 `session_id` 是
    `s-distract-<i>`，拿一个共同前缀去查**查不到任何东西**（那些查询要求 session 相等），
    于是"候选 7 条"实际只有 1 条，两遍输出一模一样
    ——**探针自己把自己测成了空过的**。

    ⚠ **★ 与 context 必须在同一次 Add 内相邻**（D28）：扩窗只沿 `prev` / `next` 走，
    分两次 Add 的话 context **永远扩不到**——那会让本探针测不到"扩窗把 ★ 提前"。
    """
    match_id = make_pair_id(_USER, "s-match", "probe-match", 0)
    context_id = make_pair_id(_USER, "s-match", "probe-match", 1)
    with store.transaction() as conn:
        match = store.insert_pair(
            conn,
            user_id=_USER,
            session_id="s-match",
            request_id="probe-match",
            local_index=0,
            prev_memory_id=None,
            next_memory_id=context_id,
            question=MATCH_QUESTION,
            answer=MATCH_ANSWER,
            status="complete",
            event_time=None,
            pair_id=match_id,
        )
        context = store.insert_pair(
            conn,
            user_id=_USER,
            session_id="s-match",
            request_id="probe-match",
            local_index=1,
            prev_memory_id=match_id,
            next_memory_id=None,
            question=CONTEXT_QUESTION,
            answer=CONTEXT_ANSWER,
            status="complete",
            event_time=None,
            pair_id=context_id,
        )
        distractors: list[str] = []
        for idx, (question, answer) in enumerate(DISTRACTORS):
            # 干扰项**各占一次 Add**（各一块）——它们之间不需要任何邻接
            pair = store.insert_pair(
                conn,
                user_id=_USER,
                session_id=f"s-distract-{idx}",
                request_id=f"probe-d{idx}",
                local_index=0,
                prev_memory_id=None,
                next_memory_id=None,
                question=question,
                answer=answer,
                status="complete",
                event_time=None,
            )
            distractors.append(pair.id)
    return match.id, context.id, distractors


def _short(memory_ids: list[str]) -> str:
    """`memory_id` 是 64 位哈希 ⇒ 只打前 6 位（够区分，且一行放得下）。"""
    return "[" + " ".join(mid[:6] for mid in memory_ids) + "]"


def _one_line(text: str) -> str:
    return text.replace("\n", " ⏎ ")[:96]


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path)
    except OSError:
        print(f"⚠ 临时目录没能删掉（多半还被占着）：{path}", file=sys.stderr)


# ── 入口 ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="精排探针（打真网关，不消耗 Smoke 配额）")
    parser.add_argument("--top-k", type=int, default=10, help="传给 Search 的 top_k")
    parser.add_argument("--radius", type=int, default=1, help="扩窗半径（对）")
    args = parser.parse_args(argv)

    _rule("TianXiMem 精排探针")
    try:
        config = load_config()
    except ConfigError as exc:
        print(f"✗ 配置不完整：{exc}")
        return EXIT_PRECONDITION_FAILED

    print(f"profile     : {config.profile}")
    print(f"rerank 开关 : {config.rerank.enabled}")
    print(f"超时        : {config.rerank.timeout_seconds}s")
    # ⚠ **单独印出来**：它是"填错就静默降级"的那一类，而本探针存在的理由正是
    #   把这类失败变成看得见的一行（两端各有自己的拒绝码：vLLM 400 / 自研封装 422）。
    print(f"信封        : {config.rerank.envelope}（对面那个网关收的字段名）")

    reranker = build_reranker(config)
    if reranker is None:
        print("\n✗ 没有可用的 reranker（`TIANXIMEM_RERANKER_BASE_URL` / `_API_KEY` 没填全，")
        print("  或 `configs/*.yaml` 的 `rerank.enabled` 是 false）。")
        print("  ⇒ 先填 `.env`（见 .env.example 的那一段），再跑一次。")
        return EXIT_PRECONDITION_FAILED

    try:
        if not probe_endpoint(reranker):
            return EXIT_PROBE_FAILED
        if not probe_pipeline(reranker, top_k=args.top_k, radius=args.radius):
            return EXIT_PROBE_FAILED
    finally:
        reranker.close()

    _rule("结论")
    print("端点通、线格式对、链上真的用了它的名次 —— 精排这一环可以进对照实验了。")
    print("⚠ 本探针**不评价效果**（§13 的 E3 才回答'值不值'，那要跑代理评测）。")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
