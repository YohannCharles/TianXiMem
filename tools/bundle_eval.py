#!/usr/bin/env python3
"""Memory600 合并包的评测入口：**一次 Add、两臂 Search、按族分派 answer / judge**。

```text
PYTHONIOENCODING=utf-8 uv run python tools/bundle_eval.py \
    --bundle var/exports/memory600-20261013 \
    --out   eval/reports/runs/bundle600-<日期> \
    --arms  e0,e2
```

## 为什么不用 `eval/experiments/run.py`

`run.py` 的 `dataset` 是**单值**（判分口径按它挑 pipeline），而这份包**混了 7 个族**。
本工具按 **space** 分组（一个 space 只属于一个族，`frozen_bundle` 的校验钉着这条），
逐 space 调 `run_judge(dataset=<族名>)` ⇒ 每道题走它自己族的答案 prompt 与裁判。

## 两条**声明**（不是默认，改动前先读）

1. **两臂共用一份语料**：`/add` 的行为与 `router` / `agentic` 无关（那两者只影响 Search）
   ⇒ 本工具**只 Add 一次**，两臂各自跑检索。⚠ 若哪天臂的定义变成"连 Add 都不一样"，
   这条声明就失效了（`--skip-ingest` 与 `--out` 的续跑语义也跟着变）。
2. **一个包一套库/集合**：600 包复用了 500 包的 283 个 `user_id`（前缀 `memory500-v1/…`）
   ⇒ **两个包不许进同一个 SQLite / Qdrant 集合**，否则同一批记忆会被投两遍
   （`applied_batches` 按 `request_id` 判重，第二次的 `request_id` 不同 ⇒ 拦不住）。
   本工具用 `--out` 的目录名派生临时库与集合，天然隔离。

## 交付物

```text
<out>/
├── report.json                 ← 每臂 × 每族的 答对/题数/准确率 + 覆盖
├── <臂>-hits.json              ← qid → 检索结果（逐题落盘，可续跑）
├── <臂>-labels.json            ← qid → 判分结果
├── <臂>/<族>/<user_id>/        ← 每 space 的 input/answers/labels（run_judge 的产物）
└── bundle-eval.db + 集合        ← 语料（**两臂共用**）
```
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (_REPO_ROOT, _REPO_ROOT / "tools"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from dataclasses import dataclass  # noqa: E402

from eval.datasets.frozen_bundle import iter_space_records  # noqa: E402
from eval.datasets.preprocess import Message, Question, Sample, Session  # noqa: E402
from eval.harness.driver import SearchHit, ServiceClient  # noqa: E402
from eval.harness.judge import build_input_items, pipeline_for, run_judge  # noqa: E402
from tools.coq_eval import _served  # noqa: E402

from tianximem.common.config import AppConfig, load_config  # noqa: E402
from tianximem.service.app import build_services, create_app  # noqa: E402

#: 两条臂。形状是 `(label, router, agentic, force, mode)`。
#:
#: **E0**：默认原始路径，不带 Router、不带 Agentic。
#:
#: **E2**：完整 Router + Plan Agentic；`force=False` 保留 Router 的二分类，
#: `MULTI_HOP` 才进入 Plan，其余路由继续走原有路径。
ARMS = {
    "e0": ("E0 默认原始路径（不带 Router / Agentic）", False, False, False, "react"),
    "e2": ("E2 Router + Plan Agentic", True, True, False, "plan"),
}

#: bundle 里的 `source_dataset` → harness 的族名（`pipeline_for` / `run_judge` 认的那个）。
#: ⚠ 两者的命名**不完全一样**（`locomo-refined` vs `locomo`），必须显式映射。
FAMILY = {
    "musique": "musique",
    "mquake-remastered": "mquake-remastered",
    "hybridqa": "hybridqa",
    "longmemeval-s": "longmemeval-s",
    "locomo-refined": "locomo-refined",
    "halumem": "halumem",
    "personamem-v2": "personamem-v2",
}


@dataclass(frozen=True, slots=True)
class Space:
    """一个隔离空间 + 绑在它上面的题（就是 harness 的 `Sample` 加一个族名）。"""

    sample: Sample
    family: str

    @property
    def user_id(self) -> str:
        return self.sample.user_id


def load_spaces(
    bundle: Path,
    *,
    families: list[str] | None = None,
    limit_spaces: int | None = None,
    per_family: int | None = None,
) -> list[Space]:
    """把包读成 `Space` 列表。**只读**，不切片、不调服务。"""
    out: list[Space] = []
    for space, questions in iter_space_records(bundle):
        if not questions:
            continue  # 包校验里已经禁止孤儿 space；这里只是不把空 space 当成错
        # ⚠ **族名取自题上的 `source_dataset`**（那是 harness 认的名字），不是 space 上那个
        #   短名（`mquake` / `locomo` / `longmemeval` / `personamem`——它们是**另一种写法**）。
        #   一个 space 只属于一个族，这条不一致就是包坏了 ⇒ 响亮失败。
        raw = {str(q["source_dataset"]) for q in questions}
        if len(raw) != 1:
            raise ValueError(f"space {space['space_id']} 混了两个族：{sorted(raw)}")
        family = FAMILY.get(raw.pop())
        if family is None:
            raise ValueError(f"bundle 里出现了没登记的族：{space['dataset']!r}")
        if families and family not in families:
            continue
        sessions = tuple(
            Session(
                session["session_id"],
                tuple(
                    Message(m["role"], m["content"], m.get("timestamp"))
                    for m in session["messages"]
                ),
            )
            for session in space["sessions"]
        )
        sample = Sample(
            user_id=space["user_id"],
            dataset=family,
            sessions=sessions,
            questions=tuple(
                Question(
                    qid=q["id"],
                    question=q["question"],
                    gold=q["gold"],
                    category=str(q.get("source_category") or ""),
                    evidence=(),
                    is_abstention=bool(q.get("is_abstention")),
                )
                for q in questions
            ),
            speaker_names=tuple(space.get("speaker_names") or ()),
        )
        out.append(Space(sample=sample, family=family))
    out.sort(key=lambda s: (s.family, s.user_id))
    if per_family is not None:
        # 冒烟用：每族取**语料最小**的 N 个 space（Add 的墙钟与消息数成正比）。
        picked: list[Space] = []
        by_family: dict[str, list[Space]] = defaultdict(list)
        for space in out:
            by_family[space.family].append(space)
        for items in (by_family[name] for name in sorted(by_family)):
            items.sort(key=lambda s: sum(len(m.messages) for m in s.sample.sessions))
            picked.extend(items[:per_family])
        picked.sort(key=lambda s: (s.family, s.user_id))
        return picked
    return out[:limit_spaces] if limit_spaces is not None else out


def _arm_config(
    base: AppConfig, key: str, *, sqlite: Path, collection: str, metrics: Path
) -> AppConfig:
    """一条臂的配置。**只动实验开关 + mode + 隔离/观测**。"""
    _, router, agentic, force, mode = ARMS[key]
    return replace(
        base,
        storage=replace(
            base.storage,
            sqlite=replace(base.storage.sqlite, path=str(sqlite)),
            qdrant=replace(base.storage.qdrant, collection=collection),
        ),
        # ⚠ 开发期的 `local.yaml` **刻意关掉精排**（省墙钟）⇒ 这里不看 profile，直接钉住。
        rerank=replace(base.rerank, enabled=True),
        router=replace(base.router, enabled=router),
        agentic=replace(base.agentic, enabled=agentic, force=force, mode=mode),
        # ⛔ 本工具发的不是官方流量，别污染 S6 的采集文件。
        capture=replace(base.capture, enabled=False),
        metrics_path=str(metrics),
    )


@contextmanager
def _service(config: AppConfig):
    """按配置起服务，yield 一个 `ServiceClient`；退出时优雅关停。"""
    services = build_services(config)
    try:
        with _served(create_app(services)) as base_url:
            yield ServiceClient(base_url)
    finally:
        services.close()


def cmd_ingest(
    spaces: list[Space],
    *,
    out_dir: Path,
    collection: str,
    top_k: int,
    base: AppConfig,
    workers: int,
) -> None:
    """**一次 Add**：整包的语料（两臂共用）。逐 space 落盘批次数，可续跑看进度。"""
    if workers < 1:
        raise ValueError("--ingest-workers 必须 >= 1")
    sqlite = out_dir / "bundle-eval.db"
    config = _arm_config(
        base, "e0", sqlite=sqlite, collection=collection, metrics=out_dir / "ingest-metrics.json"
    )
    done_path = out_dir / "ingest-done.json"
    done = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()
    started = time.monotonic()
    total_msgs = sum(len(x.messages) for s in spaces for x in s.sample.sessions)
    batches = 0
    pending = [
        (index, space)
        for index, space in enumerate(spaces, start=1)
        if space.user_id not in done
    ]

    def persist_done() -> None:
        """原子替换检查点；中断时最多丢当前正在完成的一个 space。"""
        temporary = done_path.with_name(f"{done_path.name}.tmp")
        temporary.write_text(json.dumps(sorted(done), ensure_ascii=False), encoding="utf-8")
        temporary.replace(done_path)

    def ingest_one(base_url: str, item: tuple[int, Space]) -> tuple[int, Space, int]:
        index, space = item
        # 每个并发任务使用自己的 HTTP client；服务端仍是同一个单进程实例。
        with ServiceClient(base_url) as client:
            return index, space, client.ingest(space.sample)

    with _service(config) as server_client:
        base_url = server_client.base_url
        print(
            f"    ADD 并发度 = {workers}，待处理 {len(pending)}/{len(spaces)} 个 space",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="bundle-add") as pool:
            futures = [pool.submit(ingest_one, base_url, item) for item in pending]
            for future in as_completed(futures):
                index, space, space_batches = future.result()
                batches += space_batches
                done.add(space.user_id)
                elapsed = time.monotonic() - started
                persist_done()
                print(
                    f"    ADD [{index}/{len(spaces)}] {space.family}/{space.user_id.split('/')[-1]}"
                    f" ⇒ 累计 {batches} 批，{elapsed / 60:.1f} 分钟",
                    flush=True,
                )
    print(
        f"    ADD 完：{len(done)}/{len(spaces)} 个 space、{total_msgs} 条消息、{batches} 批",
        flush=True,
    )


def run_arm(
    spaces: list[Space],
    *,
    key: str,
    out_dir: Path,
    collection: str,
    top_k: int,
    base: AppConfig,
    max_questions: int | None,
    endpoint_index: int | None,
) -> dict:
    """一条臂：逐 space 检索 → 按族装配答案输入 → 判分。返回汇总。"""
    label = ARMS[key][0]
    sqlite = out_dir / "bundle-eval.db"
    config = _arm_config(
        base, key, sqlite=sqlite, collection=collection, metrics=out_dir / f"{key}-metrics.json"
    )
    hits_path, labels_path = out_dir / f"{key}-hits.json", out_dir / f"{key}-labels.json"
    hits: dict[str, list] = (
        json.loads(hits_path.read_text(encoding="utf-8")) if hits_path.exists() else {}
    )
    labels: dict[str, dict] = (
        json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    )
    print(f"  ══ {key} · {label} ══", flush=True)
    with _service(config) as client:
        for index, space in enumerate(spaces, start=1):
            questions = list(space.sample.questions)
            if max_questions is not None:
                questions = questions[:max_questions]
            pending = [q for q in questions if q.qid not in hits]
            for question in pending:
                try:
                    hits[question.qid] = [
                        # ⚠ `SearchHit` 是 `slots=True` 的 dataclass ⇒ **没有 `__dict__`**；
                        #   用 `asdict` 落盘（它的四个字段就是 §2.1 的四项，原样存）。
                        asdict(h)
                        for h in client.search(
                            user_id=space.user_id, query=question.question, top_k=top_k
                        )
                    ]
                except Exception as exc:  # noqa: BLE001 —— 一题失败不该打死整轮
                    hits[question.qid] = []
                    print(f"    ⚠ SEARCH 失败 {question.qid}：{type(exc).__name__}", flush=True)
            if pending:
                hits_path.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
            todo = [q for q in questions if q.qid not in labels]
            if not todo:
                continue
            sample = replace(space.sample, questions=tuple(todo))
            hits_by_qid = {q.qid: [SearchHit(**h) for h in hits.get(q.qid, [])] for q in todo}
            items = build_input_items(sample, hits_by_qid)
            rows = run_judge(
                pipeline_for(_bench_dir(), space.family),
                items,
                out_dir / key / space.family / space.user_id.split("/")[-1],
                dataset=space.family,
                endpoint_index=endpoint_index,
            )
            for row in rows:
                labels[row.qid] = {
                    "is_correct": bool(row.is_correct),
                    "label": row.label,
                    "family": space.family,
                    "user_id": space.user_id,
                }
            labels_path.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8")
            print(
                f"    [{index}/{len(spaces)}] {space.family}/{space.user_id.split('/')[-1]}"
                f"：{sum(1 for r in rows if r.is_correct)}/{len(rows)}",
                flush=True,
            )
    per_family: dict[str, dict] = defaultdict(lambda: {"n": 0, "correct": 0})
    for row in labels.values():
        per_family[row["family"]]["n"] += 1
        per_family[row["family"]]["correct"] += int(row["is_correct"])
    total = sum(v["n"] for v in per_family.values())
    correct = sum(v["correct"] for v in per_family.values())
    return {
        "label": label,
        "questions": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else None,
        "per_family": {
            k: dict(v) | {"accuracy": round(v["correct"] / v["n"], 4)}
            for k, v in sorted(per_family.items())
        },
    }


def _bench_dir() -> Path:
    return Path(os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--arms", default="e0,e2", help="逗号分隔；可选 e0 / e2")
    parser.add_argument("--families", default=None, help="逗号分隔；默认全部 7 族")
    parser.add_argument("--limit-spaces", type=int, default=None, help="只跑前 N 个 space")
    parser.add_argument(
        "--per-family",
        type=int,
        default=None,
        help="每族取 N 个**语料最小**的 space（冒烟用；Add 墙钟 ∝ 消息数）",
    )
    parser.add_argument(
        "--max-questions", type=int, default=None, help="每个 space 只跑前 N 题（冒烟）"
    )
    parser.add_argument("--skip-ingest", action="store_true", help="复用已有语料（续跑第二臂）")
    parser.add_argument(
        "--ingest-workers",
        type=int,
        default=1,
        help="Add 的并发 space 数；续跑时可提高并发度（默认 1）",
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--endpoint-index", type=int, default=None)
    args = parser.parse_args(argv)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"未知臂 {unknown}——只有 {sorted(ARMS)}")
    families = [f.strip() for f in args.families.split(",")] if args.families else None

    spaces = load_spaces(
        args.bundle, families=families, limit_spaces=args.limit_spaces, per_family=args.per_family
    )
    if not spaces:
        raise SystemExit("没有选中任何 space——检查 --families / --limit-spaces")
    questions = sum(len(s.sample.questions) for s in spaces)
    messages = sum(len(m.messages) for s in spaces for m in s.sample.sessions)
    print(
        f"包 = {args.bundle}\n空间 {len(spaces)} 个 / 题 {questions} 道 / 消息 {messages} 条"
        f"\n臂 = {arms}",
        flush=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    collection = f"bundle600-{args.out.name}"
    base = load_config()

    if not args.skip_ingest:
        cmd_ingest(
            spaces,
            out_dir=args.out,
            collection=collection,
            top_k=args.top_k,
            base=base,
            workers=args.ingest_workers,
        )

    report_path = args.out / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    for key in arms:
        report[key] = run_arm(
            spaces,
            key=key,
            out_dir=args.out,
            collection=collection,
            top_k=args.top_k,
            base=base,
            max_questions=args.max_questions,
            endpoint_index=args.endpoint_index,
        )
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n════ 汇总 ════")
    print(f"{'臂':34s} {'答对':>5s} {'题数':>5s} {'准确率':>7s}")
    for key in arms:
        entry = report[key]
        head = f"{key + ' ' + entry['label']:34s}"
        print(f"{head} {entry['correct']:5d} {entry['questions']:5d} {entry['accuracy']!s:>7s}")
        for family, row in entry["per_family"].items():
            print(f"    {family:20s} {row['correct']:4d}/{row['n']:<4d} = {row['accuracy']}")
    print(f"\n产物：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
