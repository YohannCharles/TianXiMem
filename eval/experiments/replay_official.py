#!/usr/bin/env python3
"""按采集的 timeline 重放官方 add/search 并判分——**README §6.2 的线①**。

## 它和 `run.py` 的分工

`run.py` 是**通用** runner：加载 → 整份投喂 → 逐题检索 → 裁判。官方采集这份数据
**套不进那个形状**（[`../datasets/official_capture.py`](../datasets/official_capture.py)
的模块 docstring 讲了为什么）：

| | `run.py` | 本脚本 |
| --- | --- | --- |
| Add 的 payload | 我们 shape + 切批 | **采集原文**，一条 add 一投，`request_id` 用官方那个 |
| 顺序 | 先灌完整个 sample，再检索 | **按 timeline 交错**（否则 41% 的题会读到未来） |

⇒ 它是 `run.py` 的**平行实现**，只共用零件：`ServiceClient`（HTTP）、
`build_official_items`（注入）、`run_judge`（裁判）、`build_record`（记录）。

## 三个可续跑的落点

```
eval/reports/runs/<run_id>/
    <user_id>/hits.jsonl     检索快照（seq → 命中的 memory_id/content…）
    <user_id>/input.jsonl    注入后的输入项（每次重写）
    <user_id>/answers.jsonl  模型答案（**追加**，按 id 跳过）
    <user_id>/labels.jsonl   判分结果（**按 id 合并**，已判的跳过）
```

⇒ 中断后重跑：**已判完的 user 整户跳过**（`labels.jsonl` 覆盖全部题）；
判到一半的 user 只补缺的题（检索有 `hits.jsonl` 也不重跑）。
⚠ 唯一的例外是 `--force`：显式要求重判。

## 用法

```bash
# 冒烟：1 个 user、最多 3 题
uv run python eval/experiments/replay_official.py --users 1 --max-questions 3

# 正式：可评分题最多的 261 个 user（约 5,500 题、14,000 条 add）
uv run python eval/experiments/replay_official.py --users 261 --run-id official-capture-top261
```

⚠ **跑之前服务得起着**（`make serve`），且**一次只跑一条链**——网关在 Cloudflare 后面，
并发会把 524 从"不会发生"变成"随机发生"（[`CLAUDE.md`](./CLAUDE.md) 的执行纪律）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
from eval.datasets.aml.plan import AddEvent, SearchEvent
from eval.datasets.official_capture import UserPlan, replay_plan
from eval.datasets.prepare import PreparationError, ensure_dataset
from eval.datasets.registry import benchmark_dir, capture_dir, file_fingerprint
from eval.harness import ServiceClient, build_official_items, pipeline_for, run_judge
from eval.harness.api_config import ANSWER_MODEL
from eval.harness.plan_driver import send_event
from eval.harness.run_record import build_record, write_record

__all__ = ["main", "replay_user", "load_hits", "save_hits"]

DATASET = "official-capture"

#: 采集的族名 → 归档清单里的 slug。**只有 `docpp` 一处不同**：
#: 套件里它叫 `docpp`（`extra_pipeline._judge_one` 也按这个名字分派），
#: 而 D35 之后的归档清单（`eval/datasets/prepare.py`）叫 `doc-pp`。
_ARCHIVE_SLUG = {"docpp": "doc-pp"}
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]


def _append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def save_hits(path: Path, hits_by_qid: dict[str, list]) -> None:
    """检索快照——**judge 侧要的字段一个不少**，重判时不必再检索一遍。"""
    with path.open("w", encoding="utf-8") as handle:
        for qid, hits in hits_by_qid.items():
            handle.write(
                json.dumps(
                    {
                        "qid": qid,
                        "hits": [
                            {
                                "id": hit.id,
                                "content": hit.content,
                                "created_at": hit.created_at,
                                "score": hit.score,
                            }
                            for hit in hits
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_hits(path: Path):
    """读回检索快照（`SearchHit` 对象——`render_memories` 要它）。"""
    from eval.harness import SearchHit

    return {row["qid"]: [SearchHit(**hit) for hit in row["hits"]] for row in _read_jsonl(path)}


def replay_user(
    plan: UserPlan,
    *,
    client: ServiceClient,
    pipeline: Path,
    out_dir: Path,
    top_k: int = 100,
    date_mode: str = "none",
    annotate_mark: str = "paren",
    max_tokens: int = 1024,
    timeout: float | None = None,
    force: bool = False,
) -> list:
    """一个 user：**按 timeline 交错**地投 add / 检索，然后判分。

    返回 `JudgeResult` 列表；`labels.jsonl` 不存在或不全时会真的打 HTTP。
    """
    user_dir = out_dir / plan.user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    labels_path = user_dir / "labels.jsonl"
    hits_path = user_dir / "hits.jsonl"
    wanted = {str(row["seq"]) for row in plan.questions}

    if not force and labels_path.exists():
        rows = _read_jsonl(labels_path)
        if {str(row["id"]) for row in rows} >= wanted:
            print(f"    （已判完，跳过——{len(rows)} 题）", flush=True)
            return _results_from(rows, user_dir)

    hits_by_qid = {} if force else load_hits(hits_path)
    missing = {str(row["seq"]) for row in plan.questions} - set(hits_by_qid)
    if missing:
        for event in plan.events:
            if event.kind == "add":
                # ⚠ **一批投不进去不能打死整轮**（实测 2026-10-06）：官方 43,272 条 add 里
                #   有 **1 条**的某个记忆块（连续 7 条 assistant，53,185 字符 ≈ 13k token）
                #   超过 embedding 的 8,192 token 上限 ⇒ 服务端**稳定 500**（重试也一样）。
                #   线上那一批就是丢了（AML 重试再多次也过不去），所以这里**跳过并记账**，
                #   而不是让一个 40 小时的跑批死在第 10 个 user 上。
                try:
                    send_event(
                        client,
                        plan.user_id,
                        AddEvent(
                            event.session_id, event.messages, event.request_id, batch_ready=True
                        ),
                    )
                except httpx.HTTPStatusError as error:
                    _record_skip(
                        out_dir,
                        {
                            "user_id": plan.user_id,
                            "seq": event.seq,
                            "request_id": event.request_id,
                            "session_id": event.session_id,
                            "n_messages": len(event.messages),
                            "chars": sum(len(m.get("content") or "") for m in event.messages),
                            "status": error.response.status_code,
                            "detail": (error.response.text or "")[:300],
                        },
                    )
                    print(
                        f"    ⚠ /add 被服务端拒（{error.response.status_code}），"
                        f"**跳过这一批**（seq={event.seq}）——见 skipped.jsonl",
                        flush=True,
                    )
            elif str(event.seq) in missing:
                # ⚠ **检索偶发 500 也不许打死整轮**（2026-10-06 实测：一轮死在一次
                #   `/search` 500 上，而服务本身健康、同一请求随后就成功）。
                #   ⇒ 记账 + **不把这一题算进去**（判了就会以"没有记忆"的形状被计算，
                #   那比缺一个样本更糟）⇒ 下一次重跑会把它补上（labels 不全 ⇒ 不跳过整户）。
                try:
                    hits_by_qid[str(event.seq)] = send_event(
                        client,
                        plan.user_id,
                        SearchEvent(str(event.seq), event.question["query"]),
                        top_k=top_k,
                    )
                except httpx.HTTPStatusError as error:
                    if error.response.status_code < 500:
                        raise
                    _record_skip(
                        out_dir,
                        {
                            "kind": "search",
                            "user_id": plan.user_id,
                            "seq": event.seq,
                            "status": error.response.status_code,
                            "detail": (error.response.text or "")[:300],
                        },
                    )
                    print(
                        f"    ⚠ /search 被服务端拒（{error.response.status_code}）——"
                        f"**这一题这次不判**（seq={event.seq}，下次重跑补）",
                        flush=True,
                    )
        save_hits(hits_path, hits_by_qid)
    else:
        print(f"    （检索快照已存在，{len(missing)} 题待补）", flush=True)

    # 只判**这一轮真的检索到了**的题——检索失败的那些留到下次重跑补（见上面的 try）。
    asked = [row for row in plan.questions if str(row["seq"]) in hits_by_qid]
    items = build_official_items(
        asked, hits_by_qid, date_mode=date_mode, annotate_mark=annotate_mark
    )
    return run_judge(
        pipeline, items, user_dir, dataset=DATASET, max_tokens=max_tokens, timeout=timeout
    )


def _record_skip(out_dir: Path, row: dict) -> None:
    """把被服务端拒掉的 add 写进 `skipped.jsonl`（**逐条留痕**，不静默丢）。

    它是"这一轮的语料少了哪几批"的唯一记录——`data_fingerprint.note` 里只报总数。
    """
    path = out_dir / "skipped.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _results_from(rows: list[dict], user_dir: Path) -> list:
    """把已落盘的 `labels.jsonl` 读成 `JudgeResult`（跳过裁判那一步时用）。"""
    from eval.harness import JudgeResult

    answers = _read_jsonl(user_dir / "answers.jsonl")
    generated = {row["id"]: row.get("generated_answer", "") for row in answers}
    return [
        JudgeResult(
            qid=str(row["id"]),
            is_correct=bool(row["is_correct"]),
            label=str(row["label"]),
            judge_response=str(row.get("judge_response") or ""),
            generated_answer=generated.get(str(row["id"]), ""),
            partial=row.get("partial"),
        )
        for row in rows
    ]


def data_fingerprint_for(capture: Path, *, plans: list[UserPlan], note: str) -> dict:
    """套件本身进指纹——**重放的可比性取决于那三份文件没被换过**。"""
    files = [
        capture / name
        for name in ("official-eval-kit.jsonl", "official-adds.jsonl", "official-timeline.jsonl")
    ]
    return {
        "dataset": DATASET,
        "benchmark_dir": str(capture),
        "files": [file_fingerprint(path) for path in files if path.exists()],
        "n_samples": len(plans),
        "n_questions": sum(len(plan.questions) for plan in plans),
        "batching": "verbatim（采集原文，不重切批）",
        "add_shape": "verbatim（采集原文，不加前缀）",
        "note": note,
    }


def _write_partial_record(
    args: argparse.Namespace,
    capture: Path,
    plans: list[UserPlan],
    samples: list,
    results: list,
    run_id: str,
):
    """把**当前已经判到的**结果落成一份 run record（每个 user 之后刷一次）。

    这一轮是几十小时的长跑：没有它，中途被打断时手上只有散落的 `labels.jsonl`，
    而 `overall` / 分类明细要等整轮结束才存在。⚠ **它是滚动快照**：
    `data_fingerprint.note` 里写清"这一轮还在跑"，免得被当成跑完的数。
    """
    skipped = _read_jsonl(Path(args.reports_dir) / "runs" / run_id / "skipped.jsonl")
    note = (
        f"**滚动快照**：这一刻已判 {len(samples)}/{len(plans)} 个 user"
        f"（{len(results)} 题）"
        + (
            f"；⚠ **{len(skipped)} 条 add 被服务端拒**（语料少了那几批，见 skipped.jsonl）"
            if skipped
            else ""
        )
        + f"。{args.notes}"
    ).strip()
    record = build_record(
        run_id=run_id,
        step=args.step,
        profile=args.profile,
        bench_dir=capture,
        samples=samples,
        results=results,
        data_fingerprint=data_fingerprint_for(capture, plans=plans, note=note),
        models={
            "embedder": "<未声明：见 configs/*.yaml 的 models.embedder>",
            "llm": ANSWER_MODEL or "<未声明>",
            "reranker": "<按服务的启动配置>",
        },
        configs_dir=Path(args.configs_dir),
        notes=note,
    )
    write_record(record, Path(args.reports_dir))
    return record


def _pick_per_family(plans: list[UserPlan], per_family: int) -> list[UserPlan]:
    """每族取前 N 个 user（族按该 user 的**多数题**归属——一个 user 的题基本同族）。

    **为什么要有这一档**：缺省顺序是"题多的先跑"，而题多的永远是那几个大 user
    （实测前 15 个 user 就占 3,985 题）。一整晚的跑批要的是**每一族都有分**，
    而不是"最大的那一族跑完了、其余的还没轮到"。
    """
    from collections import Counter

    grouped: dict[str, list[UserPlan]] = {}
    for plan in plans:  # plans 已按题数降序 ⇒ 组内顺序天然是降序
        counter = Counter(row["dataset"] for row in plan.questions)
        grouped.setdefault(counter.most_common(1)[0][0], []).append(plan)
    picked: list[UserPlan] = []
    for family in sorted(grouped):
        picked += grouped[family][:per_family]
    return picked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--offline", action="store_true", help="禁止下载裁判依赖，缺失就退出")
    parser.add_argument("--users", type=int, default=None, help="取可评分题最多的前 N 个 user")
    parser.add_argument(
        "--per-family",
        type=int,
        default=None,
        help=(
            "**每个数据集族取前 N 个 user**（按该 user 的题数降序）——一晚上的跑批要的是"
            "「每一族都有分」，而不是「先把最大的那一族跑完」。与 `--users` 互斥"
        ),
    )
    parser.add_argument("--user-id", default=None, help="只跑一个 user（冒烟用）")
    parser.add_argument("--max-questions", type=int, default=None, help="每个 user 最多几题")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--top-k", type=int, default=100, help="§2.2 固定的 100")
    parser.add_argument("--capture-dir", default=None, help="采集目录（缺省见 CAPTURE_DIR）")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--reports-dir", default="eval/reports")
    parser.add_argument("--step", default="step-0")
    parser.add_argument("--profile", default="local")
    parser.add_argument("--configs-dir", default="configs")
    parser.add_argument("--memory-date", default="none", help="注入时怎么带日期（同 run.py）")
    parser.add_argument("--annotate-mark", default="paren")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--judge-timeout", type=float, default=None)
    parser.add_argument("--force", action="store_true", help="重判已有 labels 的题（默认跳过）")
    parser.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="每 N 个 user 刷一次 run record（滚动快照；长跑被打断时至少有一份能看的数）",
    )
    parser.add_argument("--notes", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    capture = Path(args.capture_dir) if args.capture_dir else capture_dir()
    if not (capture / "official-eval-kit.jsonl").exists():
        print(
            f"✗ 采集目录里没有套件：{capture}\n"
            "  ⇒ 先跑 `uv run python tools/build_official_kit.py`（它要采集目录里的三份原文）",
            file=sys.stderr,
        )
        return 2

    plans = replay_plan(capture, only_users=[args.user_id] if args.user_id else None)
    if not plans:
        print(f"✗ {capture} 里没有可评分的 user", file=sys.stderr)
        return 2
    if args.users is not None:
        plans = plans[: args.users]
    if args.per_family is not None:
        plans = _pick_per_family(plans, args.per_family)
    if args.max_questions is not None:
        plans = replay_plan(
            capture,
            only_users=[plan.user_id for plan in plans],
            max_questions=args.max_questions,
        )
        plans = [plan for plan in plans if plan.questions]

    try:
        for family in sorted({row["dataset"] for plan in plans for row in plan.questions}):
            ensure_dataset(
                _ARCHIVE_SLUG.get(family, family),
                benchmark_dir(),
                offline=args.offline,
                purpose="judge",
            )
    except PreparationError as error:
        print(f"数据准备失败：{error}", file=sys.stderr)
        return 2

    run_id = args.run_id or f"{DATASET}-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%S')}"
    out_dir = Path(args.reports_dir) / "runs" / run_id
    pipeline = pipeline_for(benchmark_dir(), DATASET)
    if not pipeline.exists():
        print(f"✗ 分派 pipeline 不在：{pipeline}", file=sys.stderr)
        return 2

    total_questions = sum(len(plan.questions) for plan in plans)
    print(
        f"{run_id}：{len(plans)} 个 user / {total_questions} 题 / "
        f"{sum(plan.add_count for plan in plans)} 条 add → {out_dir}",
        flush=True,
    )

    results: list = []
    samples: list = []
    # ⚠ **不传 `add_shape`**：本脚本直接调 `client.add()` 发采集原文，
    #   那条渲染路径（`shape_batch`）根本不参与——传一个值只会让人以为它生效了。
    client = ServiceClient(args.base_url)
    try:
        for index, plan in enumerate(plans, start=1):
            print(
                f"  [{index}/{len(plans)}] {plan.user_id[:14]}："
                f"{plan.add_count} 条 add / {len(plan.questions)} 题",
                flush=True,
            )
            results += replay_user(
                plan,
                client=client,
                pipeline=pipeline,
                out_dir=out_dir,
                top_k=args.top_k,
                date_mode=args.memory_date,
                annotate_mark=args.annotate_mark,
                max_tokens=args.max_tokens,
                timeout=args.judge_timeout,
                force=args.force,
            )
            samples.append(plan.to_sample())
            # ★ **每个 user 之后都落一次记录**：这一轮要跑几十小时，被 Ctrl-C / 断网 /
            #   重启打断是常态——没有这一行，中途拿到的只有一堆散落的 labels，
            #   而 `overall` 要等整轮跑完才存在（"跑到一半什么都看不见"）。
            if index % args.record_every == 0 or index == len(plans):
                _write_partial_record(args, capture, plans[:index], samples, results, run_id)
    except KeyboardInterrupt:
        print("\n⚠ 中断——已判完的 user 整户跳过，重跑本命令即可续上", file=sys.stderr)
    finally:
        client.close()

    if not results:
        print("✗ 一道题都没判——什么都没落盘", file=sys.stderr)
        return 1

    record = _write_partial_record(args, capture, plans, samples, results, run_id)
    path = Path(args.reports_dir) / "runs" / f"{run_id}.json"
    total = record.breakdown.get("n", "?")
    print(f"\n{record.run_id}：overall={record.scores['overall']}（n={total}）")
    for category, entry in record.breakdown.items():
        print(f"  {category}: {entry}")
    print(f"\nrun record → {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
