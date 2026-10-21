"""多跳检索（Chain-of-Query）的**小规模真实评测**——E0 / E1 / E2 ＋ 每轮精排消融。

```bash
uv run --env-file .env python tools/coq_eval.py \
    --manifest eval/coq-multihop.jsonl \
    --out eval/reports/runs/coq-<YYYYMMDD>
```

## 四臂（`--arms` 可挑，默认全跑）

| 臂 | `router.enabled` | `agentic.enabled` | `agentic.force` | 每轮精排 | 回答的问题 |
| --- | --- | --- | --- | --- | --- |
| **E0** | `false` | `false` | —— | —— | 原混合链（**当前默认行为**，也是基线） |
| **E1** | `false` | `true` | `true` | `true` | 强制走多跳，**绕过路由器** |
| **E2** | `true` | `true` | `false` | `true` | 路由器 ＋ 多跳（完整系统） |
| **A** | `false` | `true` | `true` | `false` | 每轮精排消融 |

⚠ **E1 对应消融表的 B 臂**（"每轮精排"那一档）——两臂配置逐字相同，所以**不重复跑**：
`--arms e0,e1,e2,a` 里的 A 与 E1 构成每轮精排的对照，报告里按 B 记。

⚠ **E1 必须绕过路由器**（`agentic.force`）：不绕过，测出来的就是"路由器 × 多跳"的
合成效果，而"CoQ 本身有没有用"与"路由器分得准不准"就分不开了。

## 它**不**做什么

* ⛔ **不生成答案、不改判分口径**：答案与裁判都走既有的
  [`../eval/harness/`](../eval/harness/)（`build_input_items` + `run_judge`）。
* ⛔ **不把标准答案或问题当记忆写进去**：只 `ingest` 对话。
* ⛔ **不复用开发库**：默认新建一个隔离 collection ＋ 隔离的 SQLite 真源，
  跑完**保留**并由你决定删不删。

## 它与既有基线的口径关系

语料 = LoCoMo-Refined 的**前 3 段对话**（`--limit 3` 的那三段），
但题目只取 `category == "1"` 的多跳题。因此这里的 overall 不能与全量题集的
overall 直接比较。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

# ⚠ **仓库根要上 `sys.path`**：`eval.*` 是仓库里的一个包，而 `uv run python tools/x.py`
#   的 `sys.path[0]` 是 `tools/`。同 [`ab_answer_prompt.py`](./ab_answer_prompt.py)。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasets.locomo import load_locomo  # noqa: E402
from eval.datasets.musique import (  # noqa: E402
    load_musique_haystack,
    supporting_snippets,
)
from eval.harness import (  # noqa: E402
    SearchHit,
    ServiceClient,
    build_input_items,
    pipeline_for,
    run_judge,
)
from eval.harness.add_shape import DEFAULT_ADD_SHAPE, shape_batch  # noqa: E402

from tianximem.common.config import AppConfig, load_config  # noqa: E402
from tianximem.service.app import build_services, create_app  # noqa: E402

#: `top_k` 由 AML 固定为 100。**这个工具不拿它当旋钮。**
TOP_K = 100


@dataclass(frozen=True, slots=True)
class Arm:
    """一条实验臂。⚠ **每一项差异都必须是"只影响它命名的那一件事"**。"""

    key: str
    label: str
    router: bool
    agentic: bool
    force: bool
    round_rerank: bool


ARMS: tuple[Arm, ...] = (
    Arm("e0", "E0 原混合链（基线）", router=False, agentic=False, force=False, round_rerank=True),
    Arm(
        "e1", "E1 强制多跳（绕过路由器）", router=False, agentic=True, force=True, round_rerank=True
    ),
    Arm("e2", "E2 路由器 + 多跳", router=True, agentic=True, force=False, round_rerank=True),
    Arm("a", "消融 A · 每轮不精排", router=False, agentic=True, force=True, round_rerank=False),
)


#: 本工具支持的两个题池。⚠ **形状正好相反**，所以 `_select` 要分开：
#: LoCoMo 是 `N 题 : 1 段对话`，musique-haystack 是 `N 题 : 1 个 haystack`。
DATASETS: dict[str, dict[str, str]] = {
    "locomo-refined": {
        "manifest": "eval/coq-multihop.jsonl",
        "judge": "locomo-refined",
        "note": "只取 LoCoMo 的 `category == \"1\"`。该子集的 overall 不能与全量题集直接比较；"
        "部分题需要人工复核链式依赖强度。",
    },
    "musique-haystack": {
        "manifest": "eval/coq-musique.jsonl",
        "judge": "musique",
        "note": "MuSiQue 清单中的严格链式多跳题（题数以本次加载结果为准），"
        "语料是这些题自己的全部候选段落并成的**一个** haystack"
        "⇒ 池子远大于 `top_k`，检索真的有得选。"
        "⛔ 这一档**没有拒答题**，答案 prompt 也不给拒答出口。",
    },
}

DEFAULT_DATASET = "locomo-refined"


def _degraded_arms(report: dict, chosen, mode: str = "react") -> list[tuple[str, int, dict]]:
    """哪些臂的多跳循环**被降级过** → `[(标签, 降级题数, 停止原因分布), ...]`。

    ⛔ **多跳循环被优雅降级时，返回的可能是普通混合检索结果**，而汇总表仍会给出
    一个看起来正常的准确率。判据必须显式暴露这种降级，打印与告警共用它。

    ⚠ **判据是 `fallback_reason`，不是某一个停止原因**：
    `fallback_reason is not None` 的**定义**就是"这一题的多跳没跑成、回退走了混合检索"
    ——那正是这一列要说的事。只数 `llm_unavailable` 会漏掉
    `llm_error` / `retrieval_error` / Plan 的 `plan_invalid`：它们**同样是没跑成**。
    ⚠ Plan 模式尤其要看它：拆解失败 ⇒ `plan_invalid` ⇒ 那一题其实是混合检索的结果。
    """
    out: list[tuple[str, int, dict]] = []
    for arm in chosen:
        arm_key, arm_label = _arm_identity(arm, mode)
        entry = report.get(arm_key, {})
        count = entry.get("agentic_degraded", 0)
        if count:
            out.append((arm_label, count, entry.get("agentic_stop_reasons", {})))
    return out


def _plan_calls(calls: list[dict]) -> list[dict]:
    """真的跑过 Plan、且知道题号的那些请求。**答案输入的判据都以它为分母。**"""
    return [c for c in calls if c.get("mode") == "plan" and c.get("qid")]


def _recovery_calls(calls: list[dict]) -> list[dict]:
    """⭐ 进了 `unresolved` 恢复分支的那些请求（见 `PlanRecovery`）。

    ⚠ 判据是 `entered`——**"触发但池子已经够了（没检索）"也算**：那种题的最终答题输入
    同样**不**带执行记录（用户口径：中间步骤失败，不代表原题答不出来）。
    """
    return [c for c in _plan_calls(calls) if (c.get("recovery") or {}).get("entered")]


def _plan_record_texts(calls: list[dict]) -> dict[str, str]:
    """`qid → 前置到答案输入的那一段`。

    ⭐ **恢复分支的题不前置**：那些题的最终答案是**拿原题**回答的，
    带上"某一步 unresolved"的执行记录只会把失败过程塞进上下文。
    ⛔ 记录**没有丢**——它照旧完整躺在 `<臂>-rounds.json` 里。
    """
    return {
        call["qid"]: _execution_record_text(call)
        for call in _plan_calls(calls)
        if not (call.get("recovery") or {}).get("entered")
    }


def _require_plan_records(
    calls: list[dict], qids: set[str], *, default_routed: int = 0
) -> dict[str, str]:
    """Plan 那一臂的执行记录：**每一道"该带记录"的题都要有**，缺一条就响亮失败。

    ⚠ **分母要把恢复分支的题排除**：那些题的答题输入按设计
    **只带原题 ＋ 精排后的真实记忆**（见 `_plan_record_texts`）——把它们算进分母，
    "恢复生效"就会被读成"记录丢了"。在 LoCoMo 那 25 题上就有 9 题进了恢复分支，
    于是这条守卫报了一条**假的** `16/25`，而真正该拦的（`--reuse-hits` 那种"一条记录
    都没有"）反而被淹没在噪声里。

    ⚠ **还要把"被判 `DEFAULT` 的题"排除**：E2 的**路由器是开着的**，
    判成 `DEFAULT` 的题**根本不会调执行器** ⇒ 按定义就没有 plan 记录。
    ⛔ **这一条不能靠"跳过守卫"绕过**：执行记录是**前置进答案输入**的
    （见调用点的 `build_input_items(..., execution_record=...)`），少了它判的就是
    **另一种配置**，而分数看起来完全正常——那正是这条守卫要防的事。
    ⇒ 判据是"缺的条数**不超过**被路由到 `DEFAULT` 的条数"（`default_routed`），
    超过就说明**真的有记录丢了**，照旧响亮失败。
    """
    record_text = _plan_record_texts(calls)
    recovery_qids = {call["qid"] for call in _recovery_calls(calls) if call.get("qid")}
    expected = set(qids) - recovery_qids
    missing = sorted(expected - set(record_text))
    if missing and len(missing) > default_routed:
        raise SystemExit(
            f"✗ Plan 那一臂只拿到 {len(record_text)}/{len(expected)} 题的执行记录"
            f"（缺 {missing[:5]}，共 {len(missing)} 条）。\n"
            "  ⚠ 恢复分支（`recovery.entered`）与**被判 `DEFAULT`**（执行器不会跑）的题"
            "都在分母之外，缺的不是它们。\n"
            f"  ⚠ 本轮 `router.routes['DEFAULT']` = {default_routed}，"
            "解释不了这个缺口。\n"
            "  原因通常是 `--reuse-hits`（没重跑 Search ⇒ 没有 plan_history）。\n"
            "  ⇒ 去掉 `--reuse-hits`，或用一个会真的跑 Search 的 `--out`。"
        )
    if missing:
        # ⭐ 预期内的那一种：被路由到 DEFAULT 的题本就没有记录。
        print(
            f"    ⚠ {len(missing)} 题被路由器判为 DEFAULT（执行器本就不该跑）⇒ 没有执行记录，属预期",
            flush=True,
        )
    return record_text


def _plan_summary(calls: list[dict]) -> dict:
    """Plan 那一臂的过程汇总。

    ⚠ **它是从 `plan_history` 里数出来的**（不是从 `metrics` 快照）：
    这里要的是"计划几步 / 真的跑了几步 / 哪一步停的"，而那些只有逐条记录有。
    """
    planned = Counter()
    executed = Counter()
    stop_reasons = Counter()
    repaired = 0
    quotes_unverified = 0
    recall_total = 0
    recalled_rounds = 0
    answered_steps = 0
    total_steps = 0
    recovery_statuses = Counter()
    fallback_recalled = 0
    fallback_added = 0
    fallback_ms = 0.0
    rerank_modes = Counter()
    with_record = 0
    without_record = 0
    for call in calls:
        if call.get("mode") != "plan":
            continue
        rec = call.get("recovery") or {}
        recovery_statuses[rec.get("status", "not_triggered")] += 1
        fallback_recalled += rec.get("recalled", 0)
        fallback_added += rec.get("added_unique", 0)
        fallback_ms += rec.get("latency_ms", 0.0)
        rerank_modes[rec.get("rerank_query_mode", "joint")] += 1
        if call.get("answer_input_has_record", True):
            with_record += 1
        else:
            without_record += 1
        planned[call["plan_steps"]] += 1
        executed[call["steps_executed"]] += 1
        repaired += call.get("dependencies_repaired", 0)
        stop_reasons[call["stop_reason"]] += 1
        for step in call["plan_history"]:
            total_steps += 1
            if step["status"] == "answered":
                answered_steps += 1
            quotes_unverified += step["quotes_unverified"]
            if step["documents"]:
                recall_total += step["recalled"]
                recalled_rounds += 1
    return {
        "plan_steps": dict(sorted(planned.items())),
        "steps_executed": dict(sorted(executed.items())),
        "stop_reasons": dict(stop_reasons),
        "dependencies_repaired": repaired,
        #: ⭐ 恢复分支（`unresolved`）的四态分布：`not_triggered` / `success` /
        #: `error` / `skipped_pool_full`。**"触发但池子已经够了"与"真的跑了兜底"
        #: 就在这里分开**（前者 `skipped_pool_full`、后者 `success`）。
        "recovery_statuses": dict(recovery_statuses),
        "fallback_recalled": fallback_recalled,
        "fallback_added_unique": fallback_added,
        "fallback_latency_ms": round(fallback_ms, 1),
        "rerank_query_modes": dict(rerank_modes),
        "answer_input_without_record": without_record,
        "answer_input_with_record": with_record,
        "quotes_unverified": quotes_unverified,
        "answered_steps": answered_steps,
        "total_step_records": total_steps,
        "mean_recalled_per_round": (
            round(recall_total / recalled_rounds, 3) if recalled_rounds else None
        ),
    }


def _arm_identity(arm: Arm, mode: str) -> tuple[str, str]:
    """`(报告键, 标签)`。

    ⚠ `react` **不带后缀**：它是默认模式，直接使用臂键。⚠ `plan` 必须带后缀——
    两模式的产物要在**同一个 run 目录**里并存，否则 `report[arm_key] = entry`
    会互相覆盖，而**汇总表照样打印**。
    """
    if mode == "react":
        return arm.key, arm.label
    return f"{arm.key}-{mode}", f"{arm.label} · {mode}"


def _load_manifest(path: Path) -> dict[str, list[str]]:
    """`user_id → [question_id]`。**只读 id**——题目原文从 `dataset/` 现取。"""
    wanted: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = json.loads(line)
        wanted.setdefault(row["user_id"], []).append(row["question_id"])
    if not wanted:
        raise SystemExit(f"✗ {path} 里一条题都没有")
    return wanted


def _select(bench_dir: Path, wanted: dict[str, list[str]], *, dataset: str) -> list:
    """把清单落成 `Sample`（**只保留清单里那几道题**）。

    两个数据集的**形状正好相反**，所以这里要分开：

    * **LoCoMo-Refined**：`N 题 : 1 段对话`。⚠ 用 `replace` 缩 `questions` 而不是筛掉整个
      sample——ADD 要的是**整段对话**（题是长在对话上的）。
    * **musique-haystack**：`N 题 : 1 个 haystack`。清单里**所有题共用一个 `user_id`**，
      而语料是这些题**自己的段落并起来**的（见
      [`../eval/datasets/musique.py`](../eval/datasets/musique.py) 的 `load_musique_haystack`）
      ——**不能**退回 `load_musique`，那样每题只剩 20 段，
      池子小于 `top_k` ⇒ 检索无事可做。
    """
    if dataset == "musique-haystack":
        qids = [qid for qids in wanted.values() for qid in qids]
        return [load_musique_haystack(bench_dir, qids)]

    by_user = {sample.user_id: sample for sample in load_locomo(bench_dir)}
    selected = []
    for user_id, qids in wanted.items():
        sample = by_user.get(user_id)
        if sample is None:
            raise SystemExit(f"✗ 清单里的 {user_id} 不在数据集里（清单与 dataset/ 对不上了？）")
        index = {q.qid: q for q in sample.questions}
        missing = [qid for qid in qids if qid not in index]
        if missing:
            raise SystemExit(f"✗ {user_id} 的清单里有 {len(missing)} 题不在数据集里：{missing[:3]}")
        selected.append(replace(sample, questions=tuple(index[qid] for qid in qids)))
    return selected


def _require_source_rows(samples, sqlite_path: Path, *, add_shape: str = DEFAULT_ADD_SHAPE) -> None:
    """真源须非空；MuSiQue 须逐条匹配本次候选语料，否则在 Search 前失败。

    ⚠ `--skip-add` 复用 **Qdrant 集合**，而 **SQLite 真源按 `--out` 目录定位**。
    换一个 `--out` 再 `--skip-add` ⇒ 集合里有 id、真源里没正文
    ⇒ 检索"成功"、返回却全是缺行的空结果 ⇒ **`data: []`，而它看起来像"这题没有相关记忆"**。

    LoCoMo 只检查用户非空；MuSiQue 是全 user 段落，逐条独立落库，可直接核对
    shape_batch 生成的全部载荷（包括长段切片）。不读 gold；计数相等但正文被替换、
    漏段或混入旧语料也必须失败。
    """
    import sqlite3

    if not sqlite_path.exists():
        raise SystemExit(f"✗ {sqlite_path} 不存在——`--skip-add` 要**同一个 `--out`**")
    with sqlite3.connect(sqlite_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        for sample in samples:
            user_id = sample.user_id
            found = conn.execute(
                "SELECT COUNT(*) FROM qa_pairs WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            if not found:
                raise SystemExit(
                    f"✗ 真源 {sqlite_path} 里没有 {user_id!r} 的任何记忆。\n"
                    "  `--skip-add` 只跳过 ADD，**不会**把真源建出来——"
                    "要么去掉 `--skip-add`，要么用它所属的那个 `--out`。\n"
                    "  ⚠ 不管这一条的话，检索会安静地返回 `data: []`，"
                    "而它看起来像「这题没有相关记忆」。"
                )
            if sample.dataset == "musique":
                expected = Counter()
                for session in sample.sessions:
                    payloads = shape_batch(
                        session.messages,
                        dataset=sample.dataset,
                        speaker_names=sample.speaker_names,
                        shape=add_shape,
                    )
                    if any(p["role"] != "user" for p in payloads):
                        raise SystemExit("✗ MuSiQue 语料完整性校验只接受全 user 的候选段落")
                    expected.update(
                        (session.session_id, p["content"].strip(), None) for p in payloads
                    )
                actual = Counter(
                    conn.execute(
                        "SELECT session_id, question, answer FROM qa_pairs WHERE user_id = ?",
                        (user_id,),
                    )
                )
                missing = sum((expected - actual).values())
                extra = sum((actual - expected).values())
                if missing or extra:
                    raise SystemExit(
                        f"✗ MuSiQue 真源语料不匹配：应有 {sum(expected.values())} 条载荷，"
                        f"缺少 {missing} 条，多出或正文不符 {extra} 条。\n"
                        "  停止 Search；使用与本次清单及语料一致的隔离 `--out` 重新 Add，"
                        "不要复用修复前的库。"
                    )


def _arm_config(
    base: AppConfig,
    arm: Arm,
    *,
    collection: str,
    sqlite_path: Path,
    metrics_path: Path,
    rerank: bool,
    per_round_top_k: int | None = None,
    max_rounds: int | None = None,
    mode: str | None = None,
) -> AppConfig:
    """把一条臂压进配置。**只动那四个开关**——其余一切（语料 / 索引 / 预算 / 判分）逐字相同。

    四处**隔离 / 可观测 / 口径对齐**的改动（它们不是"开关"，见各自的理由）：

    * SQLite 真源与 Qdrant collection 都换成这一次实验专用的；
    * ⛔ **`capture` 强制关掉**。它在 `.env` 里可能是开的，
      而本工具发的 `/add` 与 `/search` **不是官方流量** —— 让它们落进那份文件会把
      "官方请求原文"污染成"官方 + 自测"的混合体，而**采集文件本身没有区分标记**。
    * **`metrics_path` 指向本次实验的产物目录**。⚠ 缺了它 `.env` 里多半**没有**
      `TIANXIMEM_METRICS_PATH` ⇒ `build_metrics_sink` 给的是 `NullMetricsSink`
      ⇒ **轮数 / 停止原因 / LLM 与精排调用数 / 耗时全都不记**，
      而报告里那些"多跳到底花了多少"的数字会**整块空着**——且不会报错。
    * ⛔ **`rerank.enabled` 显式打开**（`--no-rerank` 才能关）：
      [`../configs/local.yaml`](../configs/local.yaml) 为开发期省时间默认关闭精排。
      若沿用 profile，四臂都不会发生精排（`rerank.calls = 0`），每轮精排对照
      会变成相同配置，**分数仍会生成但结论错误**。四臂必须共用同一个
      `Reranker`，所以这里不看 profile，直接钉住。
    """
    return replace(
        base,
        storage=replace(
            base.storage,
            sqlite=replace(base.storage.sqlite, path=str(sqlite_path)),
            qdrant=replace(base.storage.qdrant, collection=collection),
        ),
        rerank=replace(base.rerank, enabled=rerank),
        router=replace(base.router, enabled=arm.router),
        agentic=replace(
            base.agentic,
            enabled=arm.agentic,
            force=arm.force,
            round_rerank_enabled=arm.round_rerank,
            # ⚠ `None` = **不覆盖**（用配置里的值）。只有显式传了数才换——
            #   于是"这一臂到底用了哪个 per_round_top_k"永远能从传进来的参数看出来。
            **({} if per_round_top_k is None else {"per_round_top_k": per_round_top_k}),
            **({} if max_rounds is None else {"max_rounds": max_rounds}),
            # ⚠ 同上：`--mode` 不传就不覆盖 ⇒ **不填 mode 的配置逐字跑 React**。
            **({} if mode is None else {"mode": mode}),
        ),
        capture=replace(base.capture, enabled=False),
        metrics_path=str(metrics_path),
    )


@contextmanager
def _served(app):
    """把 app 起成**真 uvicorn**，yield 它的 `base_url`；退出时**优雅关停**。

    ⚠ 用真服务器而不是 `httpx.ASGITransport`：后者**只有异步实现**
    （`handle_async_request`），而 `ServiceClient` 是同步的。
    ⚠ 用真 HTTP 而不是直接调 `SearchPipeline`：`/search` 的请求校验、错误处理、
    响应形状都是**契约的一部分**，绕过去就测不到它们了。
    ⚠ 也因此**复用** `ServiceClient.ingest` 的切批与 `request_id` 口径——那是
    "现有 ADD 流程"的定义所在，重写一遍必然漂。
    ⚠ **不在这里 `services.close()`**：`create_app` 的 lifespan 会在关停时做那件事，
    两处都做就是关两次连接池。所以退出时是"让服务器自己停"。
    """
    import threading

    import uvicorn

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", access_log=False)
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 30.0
        while not server.started:
            if time.monotonic() > deadline:
                raise SystemExit("✗ 服务在 30s 内没起来")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=30.0)


def _free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _force_utf8_stdio() -> None:
    """把 stdout/stderr 切到 UTF-8 + `errors="replace"`。

    ⚠ **这不是洁癖，是一个到的崩溃**（就在写这个工具的当下）：本文件的输出通篇是中文，
    而 Windows 控制台默认 GBK、`sys.stdout` 的 error handler 是 `strict`
    ⇒ 结尾那几行带 `⚠` 的说明在**打印的时候**抛 `UnicodeEncodeError`，
    于是**跑完了几十分钟的实验，最后在汇报那一行崩掉**。
    （同 [`../eval/smoke/preflight.py`](../eval/smoke/preflight.py) 的 `_force_utf8_stdio`。）
    """
    import contextlib

    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):  # 被重定向到非文本流时会抛
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def _ingest(samples, client: ServiceClient) -> int:
    total = 0
    for sample in samples:
        batches = client.ingest(sample)
        total += batches
        print(f"    ADD {sample.user_id}: {batches} 批", flush=True)
    return total


def _write_hits(path: Path, hits: dict[str, list]) -> None:
    """逐题落盘：**id + created_at + score + 正文**（⚠ 含数据集原文 ⇒ runs/ gitignored）。"""
    path.write_text(
        json.dumps(
            {
                qid: [
                    {"id": h.id, "created_at": h.created_at, "score": h.score, "content": h.content}
                    for h in hs
                ]
                for qid, hs in hits.items()
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )


def _reuse_hits(out_dir: Path, arm_key: str) -> dict[str, list] | None:
    """复用已保存的 Search 结果（`--reuse-hits`）；**没有正文就响亮拒绝**。

    ⚠ 不完整的 `hits` 若只存 `{id, created_at}`，则**不能**安全复用：答案步需要正文。
    这种情况下**报错**，而不是从 SQLite 反推一份"看起来一样"的正文——
    那正是"结果依赖它拿到的到底是哪份文本"这类静默错误的温床。
    """
    path = out_dir / f"{arm_key}-hits.json"
    if not path.exists():
        print(f"    （--reuse-hits：没有 {path.name}，照常检索）", flush=True)
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    missing = [qid for qid, items in raw.items() if any("content" not in it for it in items)]
    if missing:
        raise SystemExit(
            f"✗ {path.name} 是旧格式（缺正文，共 {len(missing)} 题）⇒ **不能安全复用**。\n"
            "  答案步要用正文拼 retrieved_context，从库里反推会得到'看起来一样'的另一份输入。\n"
            "  ⇒ 去掉 `--reuse-hits` 重跑这一臂（Search 会重算，那是忠实的）。"
        )
    print(f"    （--reuse-hits：复用 {len(raw)} 题的检索结果，**不重跑 Search**）", flush=True)
    fields = ("id", "content", "created_at", "score")
    return {
        qid: [SearchHit(**{k: item[k] for k in fields}) for item in items]
        for qid, items in raw.items()
    }


def _search_all(samples, client: ServiceClient, recorder: _RecordingAgent | None = None):
    """逐题 Search。`recorder` 给了就在每次调用后**把 qid 贴到那条记录上**。

    ⚠ **qid 必须由这里贴**（而不是让 recorder 自己猜）：recorder 包在执行器上，
    它看到的是 Query 文本，而"这条记录属于哪道题"只有遍历题单的人知道。
    按顺序 zip 是**错的**（`--reuse-hits`、跳题、失败重试都会错位），
    贴在一次调用的返回值上才是稳的。
    """
    hits: dict[str, list] = {}
    started = time.monotonic()
    for sample in samples:
        for question in sample.questions:
            hits[question.qid] = client.search(
                user_id=sample.user_id, query=question.question, top_k=TOP_K
            )
            if recorder is not None and recorder.calls:
                recorder.calls[-1].setdefault("qid", question.qid)
    print(
        f"    SEARCH {len(hits)} 题，{time.monotonic() - started:.0f}s",
        flush=True,
    )
    return hits


def _snapshot(services) -> dict:
    sink = services.search.metrics
    return sink.snapshot() if hasattr(sink, "snapshot") else {}


class _RecordingAgent:
    """把执行器**包一层**，逐轮落盘，使实验能从产物重建每一轮。

    ⚠ 为什么是包一层而不是改 `SearchPipeline`：那些 `RoundRecord` **已经在
    `AgenticResult.stats` 里了**，只是编排层把它们**汇总成观测值**之后就不再持有。
    ⇒ 这是**导出**问题，不是记录问题——为一个诊断需求去改编排层的接口不划算，
    而"改测试去摸私有件"在本仓也是常态（`tests/test_router.py` 就换过 `_metrics`）。
    ⚠ 它对结果**完全透明**：原样转发、原样返回。

    ⚠ **两种模式的记录形状不同**（`react` 是 `rounds`、`plan` 是 `plan_history`）：
    它们描述的是两套不同的推进机制，硬塞进同一个键会让读的人以为字段含义相同。
    判据是**结果对象上有没有 `plan_history`**（`PlanResult` 有、`AgenticResult` 没有），
    ⛔ 不是"当前配置里写的是哪个模式"——那样两个事实就分不开了（记录来自**实际跑过的**那一次）。
    """

    def __init__(self, inner: object, mode: str = "react") -> None:
        self._inner = inner
        #: 配置里声明的模式。**只用于对照**：真正的形状由返回的对象决定。
        self.mode = mode
        self.calls: list[dict] = []

    def search(self, *, user_id: str, query: str, top_k: int):
        result = self._inner.search(user_id=user_id, query=query, top_k=top_k)  # type: ignore[attr-defined]
        record: dict = {
            "user_id": user_id,
            "query": query,
            # ⚠ `query` 是**数据集原文** ⇒ 这份产物同样**不进 Git**
            #   （`.gitignore` 已覆盖 `eval/reports/runs/*`）。
            "stop_reason": result.stats.stop_reason,
            "fallback_reason": result.stats.fallback_reason,
            "rerank_query": result.rerank_query,
        }
        plan_history = getattr(result, "plan_history", None)
        if plan_history is None:
            record["mode"] = "react"
            record["rounds"] = [
                {
                    "index": record_.index,
                    "query": record_.query,
                    "recalled": record_.recalled_candidates,
                    "documents": list(record_.document_ids),
                    "candidates": list(record_.candidate_ids),
                    "selected": list(record_.selected_ids),
                    "dropped": record_.dropped_documents,
                    "rerank_status": record_.rerank_status,
                    "is_sufficient": record_.is_sufficient,
                    "confidence": record_.confidence_score,
                    "new_query": record_.new_query,
                    "extraction": record_.extraction,
                    "stop_reason": record_.stop_reason,
                    "retrieval_ms": record_.retrieval_latency_ms,
                    "rerank_ms": record_.rerank_latency_ms,
                    "llm_ms": record_.llm_latency_ms,
                    "input_tokens": record_.input_tokens,
                    "output_tokens": record_.output_tokens,
                }
                for record_ in result.stats.rounds
            ]
        else:
            record["mode"] = "plan"
            record["plan"] = [
                {"id": step.id, "query": step.query, "depends_on": list(step.depends_on)}
                for step in result.plan
            ]
            record["plan_steps"] = result.stats.plan_steps
            record["steps_executed"] = result.stats.steps_executed
            record["decompose_llm_calls"] = result.stats.decompose_llm_calls
            record["step_llm_calls"] = result.stats.step_llm_calls
            record["dependencies_repaired"] = result.stats.dependencies_repaired
            # ⭐ 恢复分支（`unresolved`）的诊断。
            # ⚠ 汇总必须读取实际恢复状态；缺失字段会默认成 `not_triggered`，
            #   使未记录的恢复看起来像"一次都没触发"。
            #   用例 `test_the_recorder_really_writes_the_recovery_fields` 钉住它。
            record["recovery"] = {
                "entered": result.recovery.entered,
                "status": result.recovery.status,
                "executed_rounds": result.recovery.executed_rounds,
                "target_top_k": result.recovery.target_top_k,
                "pool_before": result.recovery.pool_before,
                "recalled": result.recovery.recalled,
                "reranked": result.recovery.reranked,
                "added_unique": result.recovery.added_unique,
                "latency_ms": result.recovery.latency_ms,
                "rerank_query_mode": result.recovery.rerank_query_mode,
            }
            # ⚠ "最终答题输入折不折执行记录"是**评测层**的决定（服务不碰答案输入）
            #   ⇒ 这里把它算出来记上，免得读报告的人要拿两条规则去推。
            record["answer_input_has_record"] = not result.recovery.entered
            # ⚠ `history_text` 是**控制模型那一轮实际收到的历史**，用于重建每一步
            #   ⇒ 与 `query` 一样含数据集原文，只落 `runs/`（gitignored）。
            record["plan_history"] = [
                {
                    "index": step.index,
                    "step_id": step.step_id,
                    "template": step.template,
                    "executed_query": step.executed_query,
                    "recalled": step.recalled_candidates,
                    "documents": list(step.document_ids),
                    "candidates": list(step.candidate_ids),
                    "evidence": list(step.evidence_ids),
                    "quotes": list(step.evidence_quotes),
                    "quotes_unverified": step.quotes_unverified,
                    "status": step.status,
                    "answer": step.answer,
                    "canonical_next_query": step.canonical_next_query,
                    "history": step.history_text,
                    "dropped": step.dropped_documents,
                    "rerank_status": step.rerank_status,
                    "extraction": step.extraction,
                    "stop_reason": step.stop_reason,
                    "retrieval_ms": step.retrieval_latency_ms,
                    "rerank_ms": step.rerank_latency_ms,
                    "llm_ms": step.llm_latency_ms,
                    "input_tokens": step.input_tokens,
                    "output_tokens": step.output_tokens,
                }
                for step in plan_history
            ]
        self.calls.append(record)
        return result


#: 前置到答案输入的那一段的抬头。⛔ **它必须说清"这是模型生成的执行记录"**：
#: 不伪装成记忆、不编造 memory id。见 [`../eval/harness/judge.py`](../eval/harness/judge.py)
#: 的 `execution_record` 参数。
PLAN_RECORD_HEADER: str = (
    "The block below is a MACHINE-GENERATED execution record of an intermediate\n"
    "retrieval plan produced by a model. It is NOT a memory record and it is NOT\n"
    "evidence; its identifiers are step identifiers, not memory identifiers.\n"
    "Use it only as context about what has already been established."
)


def _execution_record_text(call: dict) -> str:
    """`plan_history` → **答案输入前置的那一段**。

    形状固定：抬头（标明是模型生成的执行记录）＋ 完整计划 ＋ 实际执行过的问答。

    ⛔ **不在这里放记忆原文**：后部才是真实记忆（`judge.build_input_items` 负责拼起来）。
    ⛔ **不编造 memory id**：那一段的来源只用 `plan_history` 里记着的稳定 ID，
    而它们本来就来自候选池。
    """
    lines = [PLAN_RECORD_HEADER, "", f"Original question: {call['query']}", "", "Fixed plan:"]
    lines += [f"{step['id']}: {step['query']}" for step in call.get("plan", [])]
    lines.append("")
    lines.append("Executed steps:")
    for step in call.get("plan_history", []):
        lines.append(f"{step['step_id']} question: {step['executed_query']}")
        answer = step["answer"]
        lines.append(f"{step['step_id']} answer: {answer if answer else '(none)'}")
        sources = ", ".join(step["evidence"]) or "(none)"
        lines.append(f"{step['step_id']} sources: {sources}")
        if step["status"] != "answered":
            # ⚠ 停在哪一步、为什么停，必须一起进答案输入：否则模型会把"没有下文"
            #   读成"计划只有这么多"，而那与"这一步没解出来"是两件事。
            lines.append(f"{step['step_id']} status: {step['status']} ({step['stop_reason']})")
    return "\n".join(lines)


def _gold_evidence_texts(bench_dir: Path) -> dict[str, tuple[str, ...]]:
    """`question_id → 金标证据那几条消息的**原文**`。

    ⚠ **为什么要读原始 JSONL**：加载器只留 `Question.evidence`（**消息 id**），
    而那个 id 我们**永远不输出**——我们的 `content` 是一段 QA 块的
    渲染文本，里面没有 LoCoMo 的消息编号。若用消息 id 做覆盖率，会得到错误的 0。
    ⇒ 使用金标消息的**原文**做包含判定：它在我们这边会原样出现在某条
    `Q:` / `A:` 里（`add_shape` 只在前面加 `<Speaker>: `、`annotate` 只在中间插日期，
    都**不删改原文**）。

    ⚠ 读不到那份文件就返回空 ⇒ 覆盖率报"未测"，而不是报 0。
    """
    path = bench_dir / "locomo-refined" / "questions.jsonl"
    if not path.exists():
        return {}
    out: dict[str, tuple[str, ...]] = {}
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        qid = f"{row['sample_id']}#q{int(row['qa_index']):04d}"
        texts = tuple(
            message["text"].strip()
            for message in row.get("evidence_messages") or []
            if message.get("text", "").strip()
        )
        if texts:
            out[qid] = texts
    return out


def _gold_evidence(samples, bench_dir: Path, *, dataset: str) -> dict[str, tuple[str, ...]]:
    """`question_id → 金标证据的**可比较文本**`。**按数据集各取各的**（两份口径不可比）。

    * **LoCoMo**：读原始 `questions.jsonl` 的 `evidence_messages[].text`——那是**原文**。
      ⚠ 不能用 `Question.evidence`：那是**消息 id**，而我们**从不输出**它；
      用 id 匹配覆盖率会产生错误的 0。
    * **musique-haystack**：取支持段落正文的**前 60 字**（`supporting_snippets`）。
      ⚠ **不用 `title`**：它是维基文章名，一文多 chunk ⇒ 召回同文章的另一段也会算命中。

    ⚠ **两个口径不可比**（一个是句级原文包含、一个是段级标题包含）⇒
      `report.json` 里两个数据集的 `coverage` **不要放在一起读**。
    """
    if dataset == "musique-haystack":
        qids = [q.qid for sample in samples for q in sample.questions]
        return supporting_snippets(bench_dir, qids)
    return _gold_evidence_texts(bench_dir)


def _evidence_coverage(samples, hits: dict[str, list], gold: dict[str, tuple[str, ...]]) -> dict:
    """金标证据有没有出现在**返回给 AML 的那份文本**里。

    | 指标 | 含义 |
    | --- | --- |
    | `any_hit` | 至少命中一条金标证据的题数（"有没有捞到"） |
    | `full_hit` | **全部**金标证据都在的题数（"完整证据链"的近似） |

    ⚠ 这是**文本包含**口径，不是 id 映射（见 `_gold_evidence` 的说明）：
    它可能**高估**（同一句话出现在别的记忆块里也算命中）。⇒ 当**上界**读，
    别当精确的召回率。完整链路要的是"消息 id → 我们的 `memory_id`"那层映射，本仓还没有。
    """
    any_hit = full_hit = total = 0
    per_question: dict[str, dict] = {}
    for sample in samples:
        for question in sample.questions:
            total += 1
            texts = gold.get(question.qid)
            if not texts:
                per_question[question.qid] = {"gold": 0, "hit": 0, "status": "no_gold"}
                continue
            returned = "\n".join(hit.content for hit in hits.get(question.qid, []))
            hit = sum(1 for text in texts if text in returned)
            per_question[question.qid] = {"gold": len(texts), "hit": hit, "status": "measured"}
            if hit:
                any_hit += 1
            if hit == len(texts):
                full_hit += 1
    return {
        "any_hit": any_hit,
        "full_hit": full_hit,
        "total": total,
        "measured": sum(1 for v in per_question.values() if v["status"] == "measured"),
        "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    parser = argparse.ArgumentParser(description="多跳检索的小规模真实评测")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, choices=list(DATASETS))
    parser.add_argument("--manifest", default=None, help="默认按 --dataset 取")
    parser.add_argument("--out", required=True, help="产物目录（逐题落盘，可续跑）")
    parser.add_argument("--bench-dir", default=None, help="默认取 TIANXIMEM_BENCHMARK_DIR")
    parser.add_argument("--collection", default=None, help="默认 coq-eval-<--out 的目录名>")
    parser.add_argument("--arms", default=",".join(arm.key for arm in ARMS))
    parser.add_argument("--skip-add", action="store_true", help="复用上一次的隔离真源（续跑）")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="显式关掉精排（默认**打开**——见 `_arm_config` 里那条踩坑记录）",
    )
    parser.add_argument("--skip-judge", action="store_true", help="只跑检索覆盖，不叫裁判")
    parser.add_argument(
        "--reuse-hits",
        action="store_true",
        help="复用本轮 <臂>-hits.json 的检索结果，**不重跑 Search**（答案/判分仍会重做）",
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=None,
        help="覆盖判分的**单批**超时（秒）；不传用 `DEFAULT_JUDGE_TIMEOUT`（1200s）。"
        "多跳/plan 臂每题上下文大 ⇒ 答案步很慢，默认 20 分钟可能不够",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="覆盖 `agentic.max_rounds`（检索轮数上限，含第一轮）；不传就用配置里的值",
    )
    parser.add_argument(
        "--mode",
        choices=("react", "plan"),
        default=None,
        help="覆盖 `agentic.mode`（选哪个多跳执行器）；不传就用配置里的值（`default.yaml` 是 `plan`）",
    )
    parser.add_argument(
        "--per-round-top-k",
        type=int,
        default=None,
        help="覆盖 `agentic.per_round_top_k`（每轮精排后交给 CoQ 的条数）；不传就用配置里的值",
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="**总题数**上限（跨对话累计，用来冒烟；正式跑不要传）",
    )
    args = parser.parse_args(argv)

    bench_dir = Path(args.bench_dir or os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # ⚠ **默认按 `--out` 的目录名取，不是按日期**：
    #    同一天跑两轮（改了语料或口径再跑）会**撞在同一个 collection 上**——
    #    已有向量**不会报错**，只会静默混进这次检索。
    collection = args.collection or f"coq-eval-{Path(args.out).name}"

    dataset = args.dataset
    manifest_path = Path(args.manifest or DATASETS[dataset]["manifest"])
    wanted = _load_manifest(manifest_path)
    samples = _select(bench_dir, wanted, dataset=dataset)
    gold_evidence = _gold_evidence(samples, bench_dir, dataset=dataset)
    if args.limit_questions is not None:
        # ⚠ **跨对话累计的总数**，不是"每段各取 N"——后者在三段对话上会取到 3N 题，
        #   而"取 1 题冒烟"实际跑了 3 题（题数看起来对不上，却不会报错）。
        remaining = args.limit_questions
        capped = []
        for sample in samples:
            if remaining <= 0:
                break  # 题取够了 ⇒ **整段对话都不投喂**（省一次 ADD）
            take = sample.questions[:remaining]
            capped.append(replace(sample, questions=take))
            remaining -= len(take)
        samples = capped
    question_count = sum(len(s.questions) for s in samples)
    print(
        f"清单 {manifest_path}（{dataset}）：{len(samples)} 个 user / {question_count} 题\n"
        f"待投喂语料：{sum(s.message_count for s in samples)} 条源消息\n"
        f"隔离 collection = {collection}",
        flush=True,
    )

    base = load_config()
    sqlite_path = out_dir / "coq-eval.db"
    pipeline = pipeline_for(bench_dir, DATASETS[dataset]["judge"])
    chosen = [arm for arm in ARMS if arm.key in {k.strip() for k in args.arms.split(",")}]
    if not chosen:
        raise SystemExit("✗ --arms 里一个已知的臂都没有")

    # ⚠ **不传就用配置里的值**，并把**实际生效的那个数**写进记录——
    #   否则"这轮的分数是哪套参数跑出来的"以后没法归因（本仓在这类事上栽过多次）。
    per_round_top_k = (
        args.per_round_top_k if args.per_round_top_k is not None else base.agentic.per_round_top_k
    )
    if args.per_round_top_k is not None:
        print(
            f"⚠ `--per-round-top-k {per_round_top_k}` 覆盖了配置"
            f"（`agentic.per_round_top_k` = {base.agentic.per_round_top_k}）",
            flush=True,
        )
    max_rounds = args.max_rounds if args.max_rounds is not None else base.agentic.max_rounds
    if args.max_rounds is not None:
        print(
            f"⚠ `--max-rounds {max_rounds}` 覆盖了配置"
            f"（`agentic.max_rounds` = {base.agentic.max_rounds}）",
            flush=True,
        )
    # ⚠ 与上面两个**同一条口径**：不传就用配置里的值（`default.yaml` 是 `plan`；
    #   代码内置缺省仍是 `react`，见 `DEFAULT_AGENTIC_MODE`），
    #   并把**实际生效的那个模式**写进记录与报告标题。⛔ 不把"两模式的结果变化"
    #   归因给计划机制之前，先要能证明跑的是哪一套。
    mode = args.mode if args.mode is not None else base.agentic.mode
    if args.mode is not None:
        print(
            f"⚠ `--mode {mode}` 覆盖了配置（`agentic.mode` = {base.agentic.mode}）",
            flush=True,
        )

    # ⚠ **续跑要合并，不能覆盖**：`--arms e1,a` 这类部分重跑若从头写 `report.json`，
    #   已保存的 E0 / E2 结果会**静默消失**——而汇总表会照常打印，只是少两行。
    report: dict[str, dict] = {}
    report_path = out_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"（续跑：已有 {sorted(report)} 的结果，本次只覆盖 {args.arms}）", flush=True)

    for position, arm in enumerate(chosen):
        arm_key, arm_label = _arm_identity(arm, mode)
        print(f"\n══ {arm_label} ══", flush=True)
        config = _arm_config(
            base,
            arm,
            collection=collection,
            sqlite_path=sqlite_path,
            metrics_path=out_dir / f"{arm_key}-metrics.json",
            rerank=not args.no_rerank,
            per_round_top_k=per_round_top_k,
            max_rounds=max_rounds,
            mode=mode,
        )
        services = build_services(config)
        # ⚠ 逐轮落盘：包一层，**只在真的接上了执行器时**包。
        recorder = None
        if services.search.agentic is not None:
            recorder = _RecordingAgent(services.search.agentic, mode=mode)
            services.search._agentic = recorder  # noqa: SLF001 — 见 `_RecordingAgent`
        app = create_app(services)
        with _served(app) as base_url:
            client = ServiceClient(base_url)
            try:
                if position == 0 and not args.skip_add:
                    batches = _ingest(samples, client)
                    print(f"    ADD 共 {batches} 批", flush=True)
                elif args.skip_add or position > 0:
                    # ⚠ **不是第一臂就一律不 ADD**：四臂必须读**同一份**记忆状态，
                    #   再灌一遍既浪费又会改变真源（幂等守卫会挡住重复写，但那是白跑一趟）。
                    print("    （记忆已在：不重复 ADD）", flush=True)
                if position == 0:
                    _require_source_rows(samples, sqlite_path, add_shape=client.add_shape)
                hits = _reuse_hits(out_dir, arm_key) if args.reuse_hits else None
                if hits is None:
                    hits = _search_all(samples, client, recorder)
                elif recorder is not None:
                    # ⚠ 复用 hits ⇒ Search 没跑 ⇒ `recorder.calls` 是空的，而 Plan 那一臂的
                    #   答案输入**必须**带上执行记录（见 `_require_plan_records`）。
                    #   记录本身没丢——已保存在 `<臂>-rounds.json`，这里按写盘顺序读回来。
                    rounds_path = out_dir / f"{arm_key}-rounds.json"
                    if rounds_path.exists():
                        recorder.calls = json.loads(rounds_path.read_text(encoding="utf-8"))
                        print(
                            f"    （--reuse-hits：从 {rounds_path.name} 载入 "
                            f"{len(recorder.calls)} 条执行记录）",
                            flush=True,
                        )
                # ⚠ **必须连正文一起存**：答案步要用正文拼 `retrieved_context`。
                #   若只存 `{id, created_at}`，续跑时只能重跑 Search；从 SQLite 反推正文
                #   可能得到"看起来一样、其实不是同一份输入"的结果。
                _write_hits(out_dir / f"{arm_key}-hits.json", hits)
                metrics_snapshot = _snapshot(services)
                if args.reuse_hits:
                    # ⚠ 复用 hits ⇒ Search 没跑 ⇒ 实时快照全是 0。报告要的是**真跑过那一轮**
                    #   的观测值：`router.routes.DEFAULT` 尤其——Plan 那一臂的守卫靠它把
                    #   "被判 `DEFAULT`、执行器根本没跑"的题排除在缺口之外（见
                    #   `_require_plan_records`）。缺了它，那 14 题会被读成"记录丢了"。
                    prev_metrics = out_dir / f"{arm_key}-metrics.json"
                    if prev_metrics.exists():
                        metrics_snapshot = json.loads(prev_metrics.read_text(encoding="utf-8"))
                entry: dict = {
                    "label": arm_label,
                    "switches": {
                        "router.enabled": arm.router,
                        "agentic.enabled": arm.agentic,
                        "agentic.force": arm.force,
                        # ⚠ **模式必须记**：它是这一轮的自变量，不记的话"换模式之后分数变了"
                        #   会被读成模型抖动（本仓在这类事上栽过多次）。
                        "agentic.mode": mode,
                        "agentic.round_rerank_enabled": arm.round_rerank,
                        # ⚠ **每轮交给模型的条数必须记**：它是这一轮真跑的参数。
                        #   ⚠ 它管的是**精排后取几条**，不是召回几条——**两模式在这里相同**
                        #   （都是"召回请求的 `top_k` → 精排 → 取前 N"）。
                        "agentic.per_round_top_k": per_round_top_k,
                        # ⚠ 同上：轮数上限也是"这一轮真跑的参数"，不记就没法归因。
                        "agentic.max_rounds": max_rounds,
                    },
                    "coverage": _evidence_coverage(samples, hits, gold_evidence),
                    "metrics": metrics_snapshot,
                    # ⚠ 把**跑的时候是哪份 profile** 记下来：`local` 与 `default` 的差别
                    #   （精排开关）已经在这里被钉住了，但"其余差别是什么"要能追溯。
                    "profile": base.profile,
                    "rerank_enabled": not args.no_rerank,
                }
                execution_record: dict[str, str] = {}
                if recorder is not None:
                    # 逐轮执行记录。⚠ **它含数据集原文**（Query 与候选 id）
                    #   ⇒ 与 `runs/` 一样**不进 Git**。
                    (out_dir / f"{arm_key}-rounds.json").write_text(
                        json.dumps(recorder.calls, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
                    entry["recorded_requests"] = len(recorder.calls)
                    # ⛔ **降级必须摆到台面上**。
                    #    首轮 `llm_unavailable`（LLM 调用失败）会让多跳循环**优雅降级**，
                    #    返回的其实是**普通混合检索**的结果，而**汇总表仍会给出一个
                    #    看起来正常的准确率**。只写在 `rounds.json` 里等于没有告警：
                    #    降级后的 E1 会被当成 E1。
                    stops = Counter(record["stop_reason"] for record in recorder.calls)
                    entry["agentic_stop_reasons"] = dict(stops)
                    # ⚠ 判据是 `fallback_reason`（= 这一题真的走了混合检索），
                    #   不是某一个停止原因：`llm_error` / `retrieval_error` /
                    #   Plan 的 `plan_invalid` **同样**是"多跳没跑成"。
                    entry["agentic_degraded"] = sum(
                        1 for record in recorder.calls if record["fallback_reason"]
                    )
                    if mode == "plan":
                        # ⛔ **响亮失败**：Plan 那一臂的答案输入**必须**带上执行记录，
                        #   否则判的是"少一段上下文"的另一种配置，而分数看起来正常。
                        #   ⚠ 分母排除两类题——恢复分支的、与**被判 `DEFAULT`** 的
                        #   （后者执行器根本不跑）。见 `_require_plan_records`。
                        default_routed = (
                            (entry["metrics"].get("router") or {}).get("routes", {}).get("DEFAULT", 0)
                        )
                        execution_record = _require_plan_records(
                            recorder.calls, set(hits), default_routed=default_routed
                        )
                        entry["plan"] = _plan_summary(recorder.calls)
                        print(
                            "    Plan 过程：" + json.dumps(entry["plan"], ensure_ascii=False),
                            flush=True,
                        )
                if not args.skip_judge:
                    items = []
                    for sample in samples:
                        # ⚠ **只有 Plan 那一臂前置执行记录**：React 的 `execution_record`
                        #   是空 dict，因此不改变答案输入及截断位置。
                        items += build_input_items(sample, hits, execution_record=execution_record)
                    results = run_judge(
                        pipeline,
                        items,
                        out_dir / arm_key,
                        dataset=DATASETS[dataset]["judge"],
                        timeout=args.judge_timeout,
                    )
                    correct = sum(1 for r in results if r.is_correct)
                    entry["judged"] = len(results)
                    entry["correct"] = correct
                    entry["accuracy"] = round(correct / len(results), 4) if results else None
                    entry["per_question"] = {r.qid: bool(r.is_correct) for r in results}
                    print(f"    判分：{correct}/{len(results)} = {entry['accuracy']}", flush=True)
            finally:
                client.close()
            report[arm_key] = entry
            (out_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    print("\n\n════ 汇总 ════")
    header = (
        f"{'臂':<30} {'答对':>6} {'题数':>5} {'准确率':>8} {'证据≥1':>7} {'证据全':>7} {'降级':>5}"
    )
    print(header)
    degraded_arms = _degraded_arms(report, chosen, mode)
    for arm in chosen:
        arm_key, arm_label = _arm_identity(arm, mode)
        entry = report.get(arm_key, {})
        coverage = entry.get("coverage", {})
        # ⚠ **降级那一列只在这里取**：`_degraded_arms()` 已经按同一判据列过一遍
        #   （⛔ 再 append 一次会让同一臂**打印两遍**）。
        degraded = entry.get("agentic_degraded", 0)
        print(
            f"{arm_label:<28} {entry.get('correct', '-'):>6} {entry.get('judged', '-'):>5} "
            f"{entry.get('accuracy', '-'):>8} "
            f"{coverage.get('any_hit', '-'):>7} {coverage.get('full_hit', '-'):>7} "
            f"{degraded or '-':>5}"
        )
    if degraded_arms:
        # ⛔ **不许让一个降级过的臂看起来像跑过了**。
        print("\n⛔ **这些臂有多跳循环被降级**——上面那一行的准确率**不是**这个臂该有的数：")
        for label, count, stops in degraded_arms:
            print(f"    {label}：{count} 题（停止原因 {stops}）")
        print(
            "  降级 = 这一题的多跳没有执行成功，已回退到普通混合检索。\n"
            "  ⇒ 这些数**只能用来看'服务没崩'，不能用来回答'多跳有没有用'**。\n"
            "  ⚠ 常见原因：端点额度/限流（实测 OpenRouter 免费档 402）、"
            "Plan 的拆解输出非法（`plan_invalid`）。"
        )
    print(f"\n产物：{out_dir}（report.json / <臂>-hits.json / <臂>/ 逐题答案与判分）")
    print(f"⚠ {DATASETS[dataset]['note']}")
    if any(arm.agentic for arm in chosen):
        print(
            "⚠ 多跳臂的用量看 report.json 的 metrics.agentic："
            "`rounds` / `llm_calls` / `rerank_calls` 都是**只有真跑过的那些请求**的和。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
