"""路由分类 ＋ 多跳拆解 的**只读诊断测量**（共享固定题单，1170 题）。

```bash
uv run --no-sync python tools/route_decompose_probe.py --limit 5      # 冒烟
uv run --no-sync python tools/route_decompose_probe.py                # 全量
```

## 它量什么

对共享固定题单的每一题，跑**真的 Search**，但**只走到三件事就停**：

1. **共同取证判断**（确定性的那一支，`retrieval.grounded_evidence`）；
2. **路由分类**（`MULTI_HOP` / `DEFAULT`）；
3. **多跳拆解**（`MULTI_HOP` 那批的 `{{sN}}` 计划）。

**不跑**多跳的逐轮检索、**不做** answer / judge。

## 为什么必须包住服务内部

这三个决定都发生在服务**内部**，而 `/search` 的响应契约
（`{id, content, created_at, score}`）**不能改** ⇒ 只能包住装配好的对象取出来。
`tools/` 允许 `import src/`（只有 [`../eval/harness/`](../eval/harness/) 不许），
形状照抄 [`coq_eval.py`](./coq_eval.py) 的第 971-972 行。

## 三个包装点，各自的理由

| 包谁 | 拿什么 | 为什么要包 |
| --- | --- | --- |
| `services.search._router` | `RouteDecision`（含 **`.raw`**） | 服务自己把 `.raw` **丢掉**（`pipeline._route` 不写进 trace）⇒ 只有这里拿得到模型原文 |
| `services.search._agentic` | 拆解出来的 `Plan` | **只调 `_decompose` 就返回**，于是这一轮一次检索都不发生 |
| `services.search._retriever` | "默认路径有没有去取候选" | ⛔ 直接**返回空**：`DEFAULT` 那批按要求**不处理**（否则每题要付一次 embedding ＋ Qdrant ＋ 可能的精排） |

**为什么"只拆解"能真的不检索**（`service/pipeline.py:669-689`）：候选是在**路由之后**
才取的——`MULTI_HOP` 走 `_agentic.search(...)`、其余走 `_retriever.search(...)`。
⇒ 执行器返回**零候选且 `fallback_reason=None`** ⇒ 进 `_rank_and_package(())`，
那里对空列表安全（`_maybe_rerank` 空即早返回、扩窗空即早返回、打包空即早返回）。

## 串行，不并发

记录是**按请求切片**归属到题上的（请求前后各取一次下标）。⇒ 必须**一次只有一个请求在飞**，
否则两条请求的记录会互相插队、**错配到别的题上**（而且不会报错）。
服务的 `/search` 是 `def`（FastAPI 下线程池），所以记录器仍然加锁。

## 不改什么

- ⛔ **不发 `Add`**、**不重灌语料**——用的是**已经 add 好**的那个 collection。
- ⛔ **不改 `src/`、不改任何 `configs/*.yaml`**。
- ⛔ **不把题面原文写进产物**（`rows.jsonl` 只放路由/拆解与 qid）。

⚠ **一个真实的写入**：共同取证那一支在用户的**事实索引未覆盖**时会懒补派生事实
（`retrieval.fact_backfill_limit`）。那是**派生表、可由正文重建、幂等**
（首见即冻结 ＋ 覆盖守卫），且正式跑已经补过 ⇒ 重复跑近乎空转。
"不动库"的准确含义是"**不动真源与语料**"。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

# ⚠ 仓库根要上 `sys.path`：`eval.*` 是仓库里的一个包，而 `uv run python tools/x.py`
#   的 `sys.path[0]` 是 `tools/`（同 `coq_eval.py` / `ab_answer_prompt.py`）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# `_served` / `_free_port` / `_force_utf8_stdio` **复用**同一份实现
# （`bundle_eval.py` 已经这么 import：不抄第二份起服务的代码）。
from coq_eval import _force_utf8_stdio, _served  # noqa: E402
from eval.datasets.registry import benchmark_dir, capture_dir  # noqa: E402
from eval.experiments.selection_suite import build_plans, load_selection  # noqa: E402
from eval.harness import ServiceClient  # noqa: E402

from tianximem.agent.plan import PlanResult, PlanStats  # noqa: E402
from tianximem.common.config import load_config  # noqa: E402
from tianximem.service.app import build_services, create_app  # noqa: E402

TOP_K = 100

DEFAULT_MANIFEST = Path("eval/experiments/selections/memory600-official10-v1.tsv")
DEFAULT_OUT = Path("eval/reports/runs/route-decompose-20261015")

#: 这个标记进 `PlanStats.stop_reason` ⇒ 会在 `trace.agentic_stop_reason` 里露出一处。
#: 起个**一眼认得出不是真结果**的名字，避免被读成"多跳真的这样停了"。
STOP_REASON_MARKER = "decompose_probe"


# ── 记录器 ────────────────────────────────────────────────────────────────────


class _Recorder:
    """按请求切片认领记录。**锁必须加**（服务的路由跑在线程池里）。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict] = []

    def add(self, kind: str, payload: dict) -> None:
        with self._lock:
            self._events.append({"kind": kind, **payload})

    def mark(self) -> int:
        with self._lock:
            return len(self._events)

    def since(self, start: int) -> list[dict]:
        with self._lock:
            return list(self._events[start:])


def _enum_name(value: object) -> str | None:
    if value is None:
        return None
    for attr in ("value", "name"):
        got = getattr(value, attr, None)
        if isinstance(got, str):
            return got
    return str(value)


# ── 三个包装器 ────────────────────────────────────────────────────────────────


class _RecordingRouter:
    """包住查询路由器：记 `RouteDecision`（服务自己会丢掉 `.raw`）。"""

    def __init__(self, inner, recorder: _Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def route(self, query: str):
        decision = self._inner.route(query)
        self._recorder.add(
            "route",
            {
                "route": _enum_name(getattr(decision, "route", None)),
                "status": _enum_name(getattr(decision, "status", None)),
                "reason": getattr(decision, "reason", None),
                "raw": getattr(decision, "raw", None),
                "latency_ms": getattr(decision, "latency_ms", None),
            },
        )
        return decision

    def __getattr__(self, name: str):  # 其余属性（含 close/连接池）转发给真身
        return getattr(self._inner, name)


def _step_dict(step) -> dict:
    return {
        "id": step.id,
        "template": step.query,
        "depends_on": list(step.depends_on),
        "placeholders": list(step.placeholders),
        "declared_dependencies": list(step.declared_dependencies),
        "unused_dependencies": list(step.unused_dependencies),
        "dependencies_repaired": step.dependencies_repaired,
    }


def _empty_result(plan) -> PlanResult:
    """零候选的 `PlanResult`：让编排层**跳过整条取候选的路**（一次检索都不发生）。

    ⚠ `PlanStats` 的**前九个字段全必填**——漏一个就在 `_record_agentic` 里抛，
    那正是我们要避免的 500。
    """
    steps = tuple(plan.steps) if plan is not None else ()
    return PlanResult(
        candidates=(),
        rerank_query=None,
        stats=PlanStats(
            rounds=(),
            stop_reason=STOP_REASON_MARKER,
            fallback_reason=None,
            candidate_count=0,
            unique_candidate_count=0,
            selected_evidence_count=0,
            joint_query_count=0,
            llm_calls=1,  # 拆解那一次
            rerank_calls=0,
            latency_ms=0.0,
            plan=steps,
            decompose_llm_calls=1,
            step_llm_calls=0,
            plan_steps=len(steps),
            steps_executed=0,
        ),
        plan=steps,
    )


class _DecomposeOnlyAgent:
    """包住多跳执行器：**只拆解**，然后交出零候选。"""

    def __init__(self, inner, recorder: _Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def search(self, *, user_id: str, query: str, top_k: int) -> PlanResult:
        started = time.monotonic()
        decompose = getattr(self._inner, "_decompose", None)
        if decompose is None:
            # `react` 模式没有"先拆一份计划"这一步 —— 明确记下来，别假装拆了。
            self._recorder.add(
                "decompose",
                {
                    "ok": False,
                    "failure": "executor_has_no_decompose_step",
                    "steps": [],
                    "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
                },
            )
            return _empty_result(None)
        plan, failure = decompose(query)
        self._recorder.add(
            "decompose",
            {
                "ok": plan is not None,
                "failure": failure,
                "steps": [_step_dict(s) for s in (plan.steps if plan is not None else ())],
                "latency_ms": round((time.monotonic() - started) * 1000.0, 3),
            },
        )
        return _empty_result(plan)

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


class _NoFetchRetriever:
    """包住混合检索器：**记一笔然后返回空**。

    `DEFAULT` 那批（以及任何回退到混合链的情况）⇒ 按要求**不处理**：
    否则每题要付一次 embedding ＋ Qdrant prefetch（`rerank.enabled=true` 时再加一次精排）。
    ⚠ 其余属性（`close` 等）转发给真身，免得关停时缺方法。
    """

    def __init__(self, inner, recorder: _Recorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def search(self, *, user_id: str, query: str, top_k: int) -> list:
        self._recorder.add("retriever", {"top_k": top_k})
        return []

    def __getattr__(self, name: str):
        return getattr(self._inner, name)


# ── 驱动 ──────────────────────────────────────────────────────────────────────


def _collapse(events: list[dict]) -> dict:
    """把一次请求里的若干条事件折成一题的记录。"""
    route = next((e for e in events if e["kind"] == "route"), None)
    decompose = next((e for e in events if e["kind"] == "decompose"), None)
    return {
        "routed": route is not None,
        "route": None if route is None else route["route"],
        "route_status": None if route is None else route["status"],
        "route_reason": None if route is None else route["reason"],
        "route_raw": None if route is None else route["raw"],
        "route_latency_ms": None if route is None else route["latency_ms"],
        "is_multi_hop": route is not None and route["route"] == "MULTI_HOP",
        # 没走到路由 ⇒ 共同取证那一支已经给了结果（或空白/top_k 短路，见 `error`）。
        "grounded_short_circuit": route is None,
        "default_fetch": any(e["kind"] == "retriever" for e in events),
        "decomposition": None if decompose is None else decompose,
    }


def _load_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # 半行（上次被强杀）⇒ 那题重做
        done.add((row["user_id"], row["qid"]))
    return done


def _summarize(rows: list[dict], meta: dict) -> dict:
    routes = Counter(r["route"] or "<not-routed>" for r in rows)
    mh = [r for r in rows if r["is_multi_hop"]]
    steps_hist: Counter = Counter()
    failures: Counter = Counter()
    payloads: Counter = Counter()
    repaired = 0
    for r in mh:
        dec = r["decomposition"] or {}
        if dec.get("ok"):
            n = len(dec.get("steps") or [])
            steps_hist[str(n)] += 1
            for s in dec["steps"]:
                if s["placeholders"]:
                    payloads["steps_with_placeholders"] += 1
                if s["dependencies_repaired"]:
                    repaired += 1
        else:
            failures[str(dec.get("failure"))] += 1

    by_dataset: dict[str, dict] = defaultdict(lambda: {"questions": 0, "MULTI_HOP": 0, "DEFAULT": 0, "not_routed": 0})
    for r in rows:
        slot = by_dataset[r["dataset"]]
        slot["questions"] += 1
        if r["route"] == "MULTI_HOP":
            slot["MULTI_HOP"] += 1
        elif r["route"] == "DEFAULT":
            slot["DEFAULT"] += 1
        else:
            slot["not_routed"] += 1

    def pct(values: list[float]) -> dict:
        if not values:
            return {}
        vs = sorted(values)
        return {
            "p50": round(vs[len(vs) // 2], 1),
            "p90": round(vs[min(len(vs) - 1, int(len(vs) * 0.9))], 1),
            "max": round(vs[-1], 1),
        }

    return {
        **meta,
        "questions_recorded": len(rows),
        "route_distribution": dict(routes),
        "route_status_counts": dict(Counter(r["route_status"] for r in rows if r["routed"])),
        "route_reason_counts": dict(Counter(r["route_reason"] for r in rows if r["routed"])),
        "decompose": {
            "attempted": len(mh),
            "ok": sum(1 for r in mh if (r["decomposition"] or {}).get("ok")),
            "failures": dict(failures),
            "steps_histogram": dict(steps_hist),
            "mean_steps": round(statistics.fmean([len((r["decomposition"] or {}).get("steps") or []) for r in mh]), 2)
            if mh
            else None,
            # ⚠ `steps_with_placeholders` 是本脚本**数出来的**（原生没有这个字段）；
            #   `dependencies_repaired` 是原生的（`PlanStep.dependencies_repaired`）。
            "steps_with_placeholders": payloads["steps_with_placeholders"],
            "dependencies_repaired_total": repaired,
        },
        "by_dataset": {k: dict(v) for k, v in sorted(by_dataset.items())},
        "latency": {
            "route_ms": pct([r["route_latency_ms"] for r in rows if r["route_latency_ms"] is not None]),
            "decompose_ms": pct(
                [
                    r["decomposition"]["latency_ms"]
                    for r in mh
                    if isinstance((r["decomposition"] or {}).get("latency_ms"), (int, float))
                ]
            ),
        },
        "errors": sum(1 for r in rows if r["error"]),
    }


def _render_md(summary: dict) -> str:
    lines = [
        "# 路由分类 ＋ 多跳拆解（只读诊断）",
        "",
        f"题数 **{summary['questions_recorded']}** · profile `{summary['profile']}` · "
        f"collection `{summary['collection']}`",
        "",
        "## 路由分布",
        "",
        "| 结果 | 题数 |",
        "|---|---:|",
    ]
    for k, v in sorted(summary["route_distribution"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    dec = summary["decompose"]
    lines += [
        "",
        "## 多跳拆解",
        "",
        f"尝试 **{dec['attempted']}** · 成功 **{dec['ok']}** · 平均步数 **{dec['mean_steps']}**",
        "",
        f"步数分布：`{dec['steps_histogram']}` · 拆解失败：`{dec['failures']}`",
        f"含占位的步数：**{dec['steps_with_placeholders']}** · `depends_on` 被规范化：**{dec['dependencies_repaired_total']}**",
        "",
        "## 按数据集",
        "",
        "| 数据集 | 题数 | MULTI_HOP | DEFAULT | 未路由 |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in summary["by_dataset"].items():
        lines.append(f"| {k} | {v['questions']} | {v['MULTI_HOP']} | {v['DEFAULT']} | {v['not_routed']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()  # ⚠ 必须在任何 print 之前（Windows 控制台是 GBK）

    parser = argparse.ArgumentParser(description="路由分类 + 多跳拆解 的只读诊断测量")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-root", type=Path, default=benchmark_dir())
    parser.add_argument("--capture-dir", type=Path, default=capture_dir())
    parser.add_argument("--profile", default="default", help="⚠ 必须让路由与多跳都开着")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（冒烟用）")
    parser.add_argument(
        "--no-retriever-shortcut",
        action="store_true",
        help="不短路检索器（默认短路：DEFAULT 那批不处理）",
    )
    args = parser.parse_args(argv)

    # ⚠ `load_config()` **没有 profile 参数**——profile 走环境变量，且真实环境变量
    #   压过 `.env`（`.env` 排在它之下）⇒ 这一行就能把它钉住，不依赖 `.env` 当前值。
    os.environ["TIANXIMEM_PROFILE"] = args.profile

    manifest = load_selection(args.manifest)
    jobs = build_plans(
        manifest, args.data_root, args.capture_dir, None, f"route-decompose:{args.manifest.stem}"
    )

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    metrics_path = out / "metrics.json"

    base = load_config()
    config = replace(
        base,
        # ⛔ 强制关掉采集：`default.yaml` 现在是 `true`，而本工具发的**不是官方流量**，
        #   落进那份文件会把"官方请求原文"污染成混合体（同 `coq_eval._arm_config`）。
        capture=replace(base.capture, enabled=False),
        # §14 的出口：不给它路径就是 `NullMetricsSink`，轮数/停止原因/耗时整块空着。
        metrics_path=str(metrics_path),
    )
    services = build_services(config)

    recorder = _Recorder()
    if services.search.router is None:
        raise SystemExit(f"✗ profile `{args.profile}` 没装查询路由器；换个 profile 或改配置")
    if services.search.agentic is None:
        raise SystemExit(f"✗ profile `{args.profile}` 没装多跳执行器；换个 profile 或改配置")
    services.search._router = _RecordingRouter(services.search.router, recorder)  # noqa: SLF001
    services.search._agentic = _DecomposeOnlyAgent(services.search.agentic, recorder)  # noqa: SLF001
    if not args.no_retriever_shortcut:
        services.search._retriever = _NoFetchRetriever(services.search._retriever, recorder)  # noqa: SLF001

    done = _load_done(rows_path)
    pending = [
        (plan.sample, q)
        for plan, _entries in jobs
        for q in plan.sample.questions
        if (plan.sample.user_id, q.qid) not in done
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    print(
        f"题单 {args.manifest.name} · 待跑 {len(pending)} 题（已记录 {len(done)}）· "
        f"profile={args.profile} collection={config.storage.qdrant.collection}",
        flush=True,
    )

    rows: list[dict] = []
    app = create_app(services)
    with _served(app) as base_url:
        client = ServiceClient(base_url)
        with rows_path.open("a", encoding="utf-8") as handle:
            for index, (sample, question) in enumerate(pending, 1):
                before = recorder.mark()
                error = None
                started = time.monotonic()
                try:
                    # ⚠ 只 Search，**从不 Add**。
                    client.search(user_id=sample.user_id, query=question.question, top_k=TOP_K)
                except Exception as exc:  # noqa: BLE001 —— 一题失败不该拖死整轮
                    error = f"{type(exc).__name__}: {exc}"[:400]
                elapsed = round((time.monotonic() - started) * 1000.0, 1)

                row = {
                    "qid": question.qid,
                    "dataset": sample.dataset,
                    "user_id": sample.user_id,
                    "top_k": TOP_K,
                    "elapsed_ms": elapsed,
                    "error": error,
                    **_collapse(recorder.since(before)),
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 50 == 0:
                    print(f"  {index}/{len(pending)} …", flush=True)

    summary = _summarize(
        rows,
        {
            "manifest": str(args.manifest),
            "profile": args.profile,
            "collection": config.storage.qdrant.collection,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    )
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(_render_md(summary), encoding="utf-8")

    print(json.dumps({k: summary[k] for k in ("questions_recorded", "route_distribution")}, ensure_ascii=False))
    print(f"→ {out / 'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
