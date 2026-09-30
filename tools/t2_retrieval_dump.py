"""T2 的**机器半边**：133 道 multi-session 题的**纯 BM25** 检索结果 → 待填的标注表。

> **T2 是一次"归因"实验**（§13）：失败样本要由人分成三类——
> (i) 没召回 · (ii) 召回了但被 top-100 截断 · (iii) 在里面但排序靠后。
> 本工具只负责把"人需要看的那几个事实"摆出来（**证据 session 的名次**），
> **分类留空**——判读是人的动作，机器代填等于把结论伪造出来。

## 为什么是"直查 Qdrant"而不是打服务

T2 的口径是**纯 BM25**，而 D15 决定**服务只有混合检索一种形态**（没有裸 BM25 模式）。
给服务加一个单路查询是 [roadmap](../../docs/roadmap.md) 里**登记过但未实现**的切片
（它顺带能拿到 A4 的反事实分布），本工具刻意不走那条路：

* 它只服务一次归因分析，**不是被判分的 arm**——`eval/experiments/CLAUDE.md` 明说
  "纯 BM25 检索是本目录的 T2 手段，不是一条被评分的 arm"；
* 走服务意味着**改 service 与检索层**，而那两处是稳定代码。

⚠ 本工具在 `tools/` 下 ⇒ **可以** `import tianximem`（"打 HTTP"那条边界只管
[`eval/`](../eval/CLAUDE.md)；`tools/` 里的探针本来就直连组件）。

## 用法

```bash
uv run python tools/t2_retrieval_dump.py                     # 用 configs/*.yaml 的集合
uv run python tools/t2_retrieval_dump.py --collection memories_dev
```

⚠ **前提：语料已经喂进去了**（`make eval` 跑过一轮，或 `--limit` 那轮跑过同一批题）。
此题在库里查不到任何 session 时，本工具会**直说**，而不是输出一张"全是没召回"的表。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final

from tianximem.common.config import load_config
from tianximem.store.qdrant_store import (
    BM25_MODEL,
    KEY_MEMORY_ID,
    KEY_SESSION_ID,
    KEY_USER_ID,
    SPARSE_VECTOR,
    QdrantStore,
)

EXIT_OK: Final[int] = 0
EXIT_FAILED: Final[int] = 1
EXIT_PRECONDITION_FAILED: Final[int] = 2

#: T2 的目标类别（§13）。LongMemEval 的 `question_type` 里就是这个字符串。
MULTI_SESSION: Final[str] = "multi-session"

#: 取多深的名次。**要深到"没进这个表"就等于"根本没召回"**：LongMemEval 每个 haystack
#: 只有十几个 session（几十到几百个 QA 对），所以 2000 足以覆盖整个用户的语料。
#: ⚠ 若哪天数据或切批变了、`n_retrieved` 撞到这个上限，说明"没召回"这一类不再可信。
DEFAULT_DEPTH: Final[int] = 2000

#: 标注表的落点。**它要提交进 git**（`eval/experiments/CLAUDE.md`：133 行人工劳动，
#: 丢了要重做）——与 `eval/reports/**` 的忽略规则不冲突，那是另一个目录。
DEFAULT_OUT: Final[Path] = Path("eval/experiments/t2_annotations.jsonl")

#: `label` 的取值域——**由人来填**，这里只是把词表钉住，免得 133 行里出现五种写法。
LABELS: Final[tuple[str, ...]] = ("not_retrieved", "truncated", "ranked_low", "ok")


def bm25_rankings(
    client: Any, *, collection: str, user_id: str, query: str, depth: int
) -> list[dict]:
    """**单路 BM25**（Qdrant 的 `qdrant/bm25`，§7.1）取该用户的前 `depth` 条。

    ⚠ `user_id` 过滤**加在这一路查询上**（与 `hybrid_search` 同一条纪律）：
    过滤放晚了会让别的用户的候选先占名次（§2.2 的隔离）。
    """
    from qdrant_client import models

    result = client.query_points(
        collection_name=collection,
        query=models.Document(text=query, model=BM25_MODEL),
        using=SPARSE_VECTOR,
        limit=depth,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(key=KEY_USER_ID, match=models.MatchValue(value=user_id)),
            ]
        ),
        with_payload=[KEY_MEMORY_ID, KEY_SESSION_ID],
    )
    return [dict(p.payload or {}) for p in result.points]


def dump(*, collection: str, out_path: Path, depth: int, bench_dir: Path) -> int:
    from eval.datasets import load_longmemeval

    samples = load_longmemeval(bench_dir)
    targets = [
        (sample, question)
        for sample in samples
        for question in sample.questions
        if question.category == MULTI_SESSION
    ]
    if not targets:
        print("一道 multi-session 题都没加载到——数据集加载层出问题了？", file=sys.stderr)
        return EXIT_FAILED

    config = load_config()
    # 与 `service/app.py` 的装配一致：本机 Qdrant 不带鉴权，**不传 `api_key`**。
    store = QdrantStore(
        url=config.storage.qdrant.url,
        collection=collection,
    )
    if not store.exists():
        print(
            f"集合 {collection} 不存在——先把语料喂进去（`make eval`，见本文件顶部）",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION_FAILED

    rows: list[dict] = []
    empty = 0
    for sample, question in targets:
        hits = bm25_rankings(
            store.client,
            collection=collection,
            user_id=sample.user_id,
            query=question.question,
            depth=depth,
        )
        if not hits:
            empty += 1
        # 该题的证据是 **session 级** id（`answer_session_ids`）；payload 里的 `session_id`
        # 与它同源（`longmemeval._sessions` 直接用 `haystack_session_ids`）⇒ 可直接比。
        first_rank: dict[str, int] = {}
        for rank, payload in enumerate(hits):
            session_id = str(payload.get(KEY_SESSION_ID) or "")
            first_rank.setdefault(session_id, rank)
        rows.append(
            {
                "qid": question.qid,
                "question": question.question,
                "is_abstention": question.is_abstention,
                "evidence_sessions": list(question.evidence),
                # 名次是 **0-based**；缺键 = 没进前 `depth`（= 没召回）
                "evidence_ranks": {
                    sid: first_rank[sid] for sid in question.evidence if sid in first_rank
                },
                "n_retrieved": len(hits),
                "in_top_100": sum(1 for r in first_rank.values() if r < 100),
                "label": "",
                "note": "",
            }
        )

    if empty == len(rows):
        print(
            "所有题都查不到任何命中——语料不在这个集合里（或 user_id 对不上）。",
            file=sys.stderr,
        )
        return EXIT_PRECONDITION_FAILED

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    abstention = sum(1 for row in rows if row["is_abstention"])
    print(f"{len(rows)} 道 multi-session 题（其中 {abstention} 道拒答）→ {out_path}")
    print(f"（{empty} 道在该集合里查不到任何命中）")
    print(
        "\n下一步是**人**的活：把每行 `label` 填成 "
        f"{' / '.join(f'`{label}`' for label in LABELS)} 之一（口径见 docs/experiments.md 的 T2）。"
    )
    print("填完跑 `make t2` 看分布。⚠ 别用脚本代填——T2 的结论全靠这一步没被伪造。")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="t2_retrieval_dump.py", description="T2 的纯 BM25 检索转储"
    )
    parser.add_argument("--collection", default=None, help="缺省用 configs/*.yaml 里的集合名")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="取多深的名次")
    args = parser.parse_args(argv)

    from eval.datasets import benchmark_dir

    collection = args.collection or load_config().storage.qdrant.collection
    return dump(
        collection=collection, out_path=Path(args.out), depth=args.depth, bench_dir=benchmark_dir()
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
