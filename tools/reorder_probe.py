"""把某轮某几类的**错题**拿出来，换几种**注入顺序**重新判一遍——看哪一档会翻正。

## 三个策略（都**只改顺序**：不改内容、不改集合）

| 策略 | 判据 |
| --- | --- |
| `keyword` | 正文里出现 **query 的原词**（词边界、大小写无关）的段提到最前 |
| `stem` | 同上，但按**最小词干**匹配（`dogs`→`dog`、`running`→`run`）
——"小狗/狗"那类同形词 |
——「小狗/狗」那类同形词 |

## ⚠ 必须带 `control` 臂

段是**重新从服务取回来的**（原轮的 `input.jsonl` 里没有段的边界，拼回去不可能）。
而"重取"与"原轮"的检索结果**未必逐字相同**（V13 的并列次序修复就改过排序）。
⇒ **`control`（重取的段、按服务给的顺序原样注入）是这几条策略的共同对照**——
不设它，任何一个"翻正"都可能只是重取带来的，而不是重排带来的。

## 用法

```bash
python -m tools.reorder_probe --run ours-lme60 --limit 60 --spread \\
    --category temporal-reasoning --category multi-session \\
    --base-url http://127.0.0.1:8000 --modes control,keyword,stem,rerank
```

⚠ **需要服务在跑**（要重取段）。它**只重跑判分那两步**，不再投喂、不再检索。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from eval.datasets import benchmark_dir  # noqa: E402
from eval.harness import pipeline_for, run_judge  # noqa: E402
from tools.truncation_probe import categories_of, load_run  # noqa: E402

__all__ = ["MIN_SUFFIXES", "reorder"]

#: **最小词干化**用的后缀——只去这几个，不引第三方依赖。
#: ⚠ 它不是 Porter、也不假装是：目的只是"`dogs` 与 `dog` 算同一个词"那类同形词。
#: 顺序有意义：长的先试（`ing` 要在 `s` 之前，否则 `runing` 会先被削成 `runin`）。
MIN_SUFFIXES: tuple[str, ...] = ("ings", "ing", "ies", "es", "ed", "s")

_WORD = re.compile(r"[A-Za-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text)]


def _stem(token: str) -> str:
    """最小词干：**只去后缀**（去完至少留 3 个字符），**再收掉双写辅音**。

    ⚠ 第二步不是装饰：`running` 去 `ing` 得到 `runn`，而查询里的 `run` 就是它——
    **这一臂存在的意义（"`dogs` 与 `dog` 算同一个词"）全靠它对上**。
    """
    for suffix in MIN_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    # 双写辅音收一个（`runn`→`run`、`stopp`→`stop`）；`ll`/`ss`/`zz` 不收（英文里本来就双写）
    if len(token) >= 4 and token[-1] == token[-2] and token[-1] not in "lsz":
        token = token[:-1]
    return token


def _hits(text: str, wanted: set[str], *, stem: bool) -> int:
    tokens = _tokens(text)
    if stem:
        return sum(1 for t in tokens if _stem(t) in wanted)
    return sum(1 for t in tokens if t in wanted)


def reorder(segments: list[str], query: str, *, mode: str) -> list[str]:
    """按 `mode` 重排；`control` 原样返回。**排序稳定**：同分的保持服务给的原序。"""
    if mode == "control":
        return list(segments)
    if mode in ("keyword", "stem"):
        wanted = {_stem(t) for t in _tokens(query)} if mode == "stem" else set(_tokens(query))
        # 命中多的在前；同分保持原序（`sorted` 稳定）
        return sorted(
            segments,
            key=lambda s: -_hits(s, wanted, stem=(mode == "stem")),
        )
    raise ValueError(f"未知重排策略 {mode!r}——只有 control / keyword / stem / rerank")


def _rerank_order(segments: list[str], query: str) -> list[str]:
    """用远端 reranker 打分后重排（分数降序）。**不发请求就抛**——宁可炸也不静默降级。"""
    from tianxi_am.rank.reranker import RemoteReranker

    base = os.environ.get("TIANXI_RERANKER_BASE_URL", "")
    key = os.environ.get("TIANXI_RERANKER_API_KEY", "")
    model = os.environ.get("TIANXI_RERANKER_MODEL", "Qwen3-Reranker-4B")
    if not base or not key:
        raise SystemExit("重排臂要 `TIANXI_RERANKER_BASE_URL` / `_API_KEY`（.env 里有）")
    reranker = RemoteReranker(base_url=base, api_key=key, model=model, timeout=120.0)
    scores = reranker.score(query=query, documents=segments)
    order = sorted(range(len(segments)), key=lambda i: -scores[i])
    return [segments[i] for i in order]


def _fetch_segments(base_url: str, *, user_id: str, query: str, top_k: int) -> list[str]:
    """重取一轮的段（**只取 `content`**，与原轮注入同口径）。"""
    response = httpx.post(
        f"{base_url.rstrip('/')}/search",
        json={"request_id": "reorder-probe", "user_id": user_id, "query": query, "top_k": top_k},
        timeout=120.0,
    )
    response.raise_for_status()
    return [str(item["content"]) for item in response.json()["data"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reorder_probe", description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--dataset", default="longmemeval-s")
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--modes", default="control,keyword,stem,rerank")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None, help="要与产生该 run 时一致")
    parser.add_argument("--spread", action="store_true", help="同上")
    parser.add_argument("--reports-dir", default="eval/reports")
    args = parser.parse_args(argv)

    reports = Path(args.reports_dir)
    items, verdicts = load_run(args.run, reports)
    categories = categories_of(args.dataset, limit=args.limit, spread=args.spread)
    wanted = set(args.category or [])
    failing = [
        qid
        for qid, ok in verdicts.items()
        if not ok and (not wanted or categories.get(qid) in wanted)
    ]
    modes = [m.strip() for m in args.modes.split(",")]
    print(f"{args.run}：错题 {len(failing)} 道；策略 {modes}\n", flush=True)

    fetched: dict[str, list[str]] = {}
    for qid in failing:
        fetched[qid] = _fetch_segments(
            args.base_url, user_id=f"lme-{qid}", query=str(items[qid]["question"]), top_k=args.top_k
        )
    print("段已重取\n", flush=True)

    probe_dir = reports / "runs" / f"{args.run}__reorder"
    pipeline = pipeline_for(benchmark_dir(), args.dataset)
    flips: dict[str, list[str]] = {qid: [] for qid in failing}
    for mode in modes:
        batch = []
        for qid in failing:
            segments = fetched[qid]
            ordered = (
                _rerank_order(segments, str(items[qid]["question"]))
                if mode == "rerank"
                else reorder(segments, str(items[qid]["question"]), mode=mode)
            )
            item = dict(items[qid])
            item["speaker_1_memories"] = "\n".join(ordered)
            batch.append(item)
        results = run_judge(pipeline, batch, probe_dir / mode, dataset=args.dataset, max_tokens=256)
        got = {r.qid: r.is_correct for r in results}
        print(f"  {mode:<8}：{sum(got.values())}/{len(failing)} 判对   "
              f"翻正：{', '.join(q for q in failing if got.get(q)) or '（无）'}", flush=True)
        for qid in failing:
            if got.get(qid):
                flips[qid].append(mode)

    print("\n各题在哪些策略下翻正：", flush=True)
    for qid in failing:
        print(f"   {qid}：{'、'.join(flips[qid]) if flips[qid] else '——'}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
