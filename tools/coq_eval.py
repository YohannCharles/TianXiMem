"""多跳检索（Chain-of-Query）的**小规模真实评测**——E0 / E1 / E2 ＋ 每轮精排消融。

```bash
uv run --env-file .env python tools/coq_eval.py \
    --manifest eval/coq-multihop.jsonl \
    --out eval/reports/runs/coq-20261010
```

## 四臂（`--arms` 可挑，默认全跑）

| 臂 | `router.enabled` | `agentic.enabled` | `agentic.force` | 每轮精排 | 回答的问题 |
| --- | --- | --- | --- | --- | --- |
| **E0** | `false` | `false` | —— | —— | 原混合链（**当前默认行为**，也是基线） |
| **E1** | `false` | `true` | `true` | `true` | 强制走多跳，**绕过路由器** |
| **E2** | `true` | `true` | `false` | `true` | 路由器 ＋ 多跳（完整系统） |
| **A** | `false` | `true` | `true` | `false` | §12.4 的消融：每轮**不**精排 |

⚠ **E1 就是消融表的 B 臂**（"每轮精排"那一档）——两臂配置逐字相同，所以**不重复跑**：
`--arms e0,e1,e2,a` 里的 A 与 E1 构成 §12.4 要求的那对对照，报告里按 B 记。

⚠ **E1 必须绕过路由器**（`agentic.force`）：不绕过，测出来的就是"路由器 × 多跳"的
合成效果，而"CoQ 本身有没有用"与"路由器分得准不准"就分不开了（§12.3）。

## 它**不**做什么

* ⛔ **不生成答案、不改判分口径**：答案与裁判都走既有的
  [`../eval/harness/`](../eval/harness/)（`build_input_items` + `run_judge`）。
* ⛔ **不把标准答案或问题当记忆写进去**：只 `ingest` 对话（§12.2 第 2 条）。
* ⛔ **不复用开发库**：默认新建一个隔离 collection ＋ 隔离的 SQLite 真源，
  跑完**保留**并由你决定删不删（§12.2 的"隔离测试数据"）。

## 它与既有基线的口径关系

语料 = LoCoMo-Refined 的**前 3 段对话**（`--limit 3` 的那三段）⇒ 与
[`../eval/reports/ledger.md`](../eval/reports/ledger.md) 里那条 0.7775 同源。
但**题目不同**：这里只取 `category == "1"`（多跳）的 25 道 ⇒
**不要把这里的 overall 与 346 题那条直接比**。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
    ServiceClient,
    build_input_items,
    pipeline_for,
    run_judge,
)

from tianximem.common.config import AppConfig, load_config  # noqa: E402
from tianximem.service.app import build_services, create_app  # noqa: E402

#: §2.2：`top_k` 由 AML 固定为 100。**这个工具不拿它当旋钮。**
TOP_K = 100


@dataclass(frozen=True, slots=True)
class Arm:
    """一条实验臂。⚠ **每一项差异都必须是"只影响它命名的那一件事"**（§13）。"""

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
        "note": "只取 LoCoMo 的 `category == \"1\"`、25 道 ⇒ **不要与 346 题那条 overall 直接比**。"
        "⚠ 人工复核发现这 25 道里真正的链式依赖几乎没有（见 coq-multihop-20261010.md §5）。",
    },
    "musique-haystack": {
        "manifest": "eval/coq-musique.jsonl",
        "judge": "musique",
        "note": "MuSiQue 的 **50 道严格链**（4 跳 29 + 3 跳 21，2026-10-12 起；"
        "旧的 25 条见 `eval/reports/coq-musique-20261011.md`），"
        "语料是这 50 道题自己的段落并成的**一个** haystack（822 段）"
        "⇒ 池子远大于 `top_k`，检索真的有得选。"
        "⛔ 这一档**没有拒答题**，答案 prompt 也不给拒答出口。",
    },
}

DEFAULT_DATASET = "locomo-refined"


def _load_manifest(path: Path) -> dict[str, list[str]]:
    """`user_id → [question_id]`。**只读 id**——题目原文从 `dataset/` 现取（§12.5）。"""
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


def _require_source_rows(samples, sqlite_path: Path) -> None:
    """**真源里必须有行**，否则响亮失败。

    ⚠ 这一条是**踩出来的**：`--skip-add` 复用的是 **Qdrant 集合**，而 **SQLite 真源是按
    `--out` 目录走的**。换一个 `--out` 再 `--skip-add` ⇒ 集合里有 id、真源里没正文
    ⇒ 检索"成功"、返回却全是缺行的空结果 ⇒ **`data: []`，而它看起来像"这题没有相关记忆"**。
    （2026-10-11 实测：同一个集合，一个 out 目录 25/25 命中，另一个 0/25。）

    判据取"用户里有没有行"而不是"行数对不对"——前者足够挡住这个陷阱，后者要重算一遍配对。
    """
    import sqlite3

    users = [sample.user_id for sample in samples]
    if not sqlite_path.exists():
        raise SystemExit(f"✗ {sqlite_path} 不存在——`--skip-add` 要**同一个 `--out`**")
    with sqlite3.connect(sqlite_path) as conn:
        for user_id in users:
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


def _arm_config(
    base: AppConfig,
    arm: Arm,
    *,
    collection: str,
    sqlite_path: Path,
    metrics_path: Path,
    rerank: bool,
) -> AppConfig:
    """把一条臂压进配置。**只动那四个开关**——其余一切（语料 / 索引 / 预算 / 判分）逐字相同。

    四处**隔离 / 可观测 / 口径对齐**的改动（它们不是"开关"，见各自的理由）：

    * SQLite 真源与 Qdrant collection 都换成这一次实验专用的；
    * ⛔ **`capture` 强制关掉**。它在 `.env` 里可能是开的（S6 的官方请求采集），
      而本工具发的 `/add` 与 `/search` **不是官方流量** —— 让它们落进那份文件会把
      "官方请求原文"污染成"官方 + 自测"的混合体，而**采集文件本身没有区分标记**。
    * **`metrics_path` 指向本次实验的产物目录**。⚠ 缺了它 `.env` 里多半**没有**
      `TIANXIMEM_METRICS_PATH` ⇒ `build_metrics_sink` 给的是 `NullMetricsSink`
      ⇒ **§14 那一段（轮数 / 停止原因 / LLM 与精排调用数 / 耗时）全都不记**，
      而报告里那几个"多跳到底花了多少"的数字会**整块空着**——且不会报错。
    * ⛔ **`rerank.enabled` 显式打开**（`--no-rerank` 才能关）。⚠ 这一条是**踩过坑补的**：
      `.env` 里 `TIANXIMEM_PROFILE=local` ⇒ [`../configs/local.yaml`](../configs/local.yaml)
      **刻意把精排关掉**（开发期省时间）。于是第一轮四臂**一次精排都没发生**
      （`rerank.calls = 0`），而 §12.4 要测的"每轮精排值不值"那对对照**两个臂一模一样**
      ——**分数照样出得来，结论却完全错**。方案 §12.3 要求四臂共用同一个 `Reranker`，
      所以这里不看 profile，直接钉住。
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

    ⚠ **这不是洁癖，是一个实测到的崩溃**（就在写这个工具的当下）：本文件的输出通篇是中文，
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


def _search_all(samples, client: ServiceClient) -> dict[str, list]:
    hits: dict[str, list] = {}
    started = time.monotonic()
    for sample in samples:
        for question in sample.questions:
            hits[question.qid] = client.search(
                user_id=sample.user_id, query=question.question, top_k=TOP_K
            )
    print(
        f"    SEARCH {len(hits)} 题，{time.monotonic() - started:.0f}s",
        flush=True,
    )
    return hits


def _snapshot(services) -> dict:
    sink = services.search.metrics
    return sink.snapshot() if hasattr(sink, "snapshot") else {}


class _RecordingAgent:
    """把执行器**包一层**，逐轮落盘（方案 §11 的"实验模式下应能重建每轮"）。

    ⚠ 为什么是包一层而不是改 `SearchPipeline`：那些 `RoundRecord` **已经在
    `AgenticResult.stats` 里了**，只是编排层把它们**汇总成观测值**之后就不再持有。
    ⇒ 这是**导出**问题，不是记录问题——为一个诊断需求去改编排层的接口不划算，
    而"改测试去摸私有件"在本仓也是常态（`tests/test_router.py` 就换过 `_metrics`）。
    ⚠ 它对结果**完全透明**：原样转发、原样返回。
    """

    def __init__(self, inner: object) -> None:
        self._inner = inner
        self.calls: list[dict] = []

    def search(self, *, user_id: str, query: str, top_k: int):
        result = self._inner.search(user_id=user_id, query=query, top_k=top_k)  # type: ignore[attr-defined]
        self.calls.append(
            {
                "user_id": user_id,
                "query": query,
                # ⚠ `query` 是**数据集原文** ⇒ 这份产物同样**不进 Git**
                #   （`.gitignore` 已覆盖 `eval/reports/runs/*`）。
                "stop_reason": result.stats.stop_reason,
                "fallback_reason": result.stats.fallback_reason,
                "rerank_query": result.rerank_query,
                "rounds": [
                    {
                        "index": record.index,
                        "query": record.query,
                        "recalled": record.recalled_candidates,
                        "documents": list(record.document_ids),
                        "candidates": list(record.candidate_ids),
                        "selected": list(record.selected_ids),
                        "dropped": record.dropped_documents,
                        "rerank_status": record.rerank_status,
                        "is_sufficient": record.is_sufficient,
                        "confidence": record.confidence_score,
                        "new_query": record.new_query,
                        "extraction": record.extraction,
                        "stop_reason": record.stop_reason,
                        "retrieval_ms": record.retrieval_latency_ms,
                        "rerank_ms": record.rerank_latency_ms,
                        "llm_ms": record.llm_latency_ms,
                        "input_tokens": record.input_tokens,
                        "output_tokens": record.output_tokens,
                    }
                    for record in result.stats.rounds
                ],
            }
        )
        return result


def _gold_evidence_texts(bench_dir: Path) -> dict[str, tuple[str, ...]]:
    """`question_id → 金标证据那几条消息的**原文**`。

    ⚠ **为什么要读原始 JSONL**：加载器只留 `Question.evidence`（形如 `D2:8` 的
    **消息 id**），而那个 id 我们**永远不输出**——我们的 `content` 是一段 QA 块的
    渲染文本，里面没有 LoCoMo 的消息编号。第一版覆盖率就是这么写的，结果四臂
    一律 `0/25`，看起来像"一条证据都没检索到"，**而那是假的**。
    ⇒ 改用金标消息的**原文**做包含判定：它在我们这边会原样出现在某条
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
      ⚠ 不能用 `Question.evidence`：那是 `D2:8` 这种**消息 id**，而我们**从不输出**它
      （第一版覆盖率拿 id 去匹配，四臂一律 0/25，看着像"一条证据都没检索到"，**而那是假的**）。
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
    """金标证据有没有出现在**返回给 AML 的那份文本**里（§12.5 的两个指标）。

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
    parser.add_argument("--collection", default=None, help="默认 coq-eval-<日期>")
    parser.add_argument("--arms", default=",".join(arm.key for arm in ARMS))
    parser.add_argument("--skip-add", action="store_true", help="复用上一次的隔离真源（续跑）")
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="显式关掉精排（默认**打开**——见 `_arm_config` 里那条踩坑记录）",
    )
    parser.add_argument("--skip-judge", action="store_true", help="只跑检索覆盖，不叫裁判")
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="**总题数**上限（跨对话累计，用来冒烟；正式跑不要传）",
    )
    args = parser.parse_args(argv)

    bench_dir = Path(
        args.bench_dir or os.environ.get("TIANXIMEM_BENCHMARK_DIR", "dataset")
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    collection = args.collection or f"coq-eval-{time.strftime('%Y%m%d')}"

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
        f"隔离 collection = {collection}",
        flush=True,
    )

    base = load_config()
    sqlite_path = out_dir / "coq-eval.db"
    pipeline = pipeline_for(bench_dir, DATASETS[dataset]["judge"])
    chosen = [arm for arm in ARMS if arm.key in {k.strip() for k in args.arms.split(",")}]
    if not chosen:
        raise SystemExit("✗ --arms 里一个已知的臂都没有")

    # ⚠ **续跑要合并，不能覆盖**：`--arms e1,a` 这类部分重跑若从头写 `report.json`，
    #   上一次跑出来的 E0 / E2 会**静默消失**——而汇总表会照常打印，只是少两行。
    report: dict[str, dict] = {}
    report_path = out_dir / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"（续跑：已有 {sorted(report)} 的结果，本次只覆盖 {args.arms}）", flush=True)

    for position, arm in enumerate(chosen):
        print(f"\n══ {arm.label} ══", flush=True)
        config = _arm_config(
            base,
            arm,
            collection=collection,
            sqlite_path=sqlite_path,
            metrics_path=out_dir / f"{arm.key}-metrics.json",
            rerank=not args.no_rerank,
        )
        services = build_services(config)
        # ⚠ 逐轮落盘（§11）：包一层，**只在真的接上了执行器时**包。
        recorder = None
        if services.search.agentic is not None:
            recorder = _RecordingAgent(services.search.agentic)
            services.search._agentic = recorder  # noqa: SLF001 — 见 `_RecordingAgent`
        app = create_app(services)
        with _served(app) as base_url:
            client = ServiceClient(base_url)
            try:
                if position == 0 and not args.skip_add:
                    batches = _ingest(samples, client)
                    print(f"    ADD 共 {batches} 批", flush=True)
                elif args.skip_add or position > 0:
                    # ⚠ **不是第一臂就一律不 ADD**：四臂必须读**同一份**记忆状态（§12.3），
                    #   再灌一遍既浪费又会改变真源（幂等守卫会挡住重复写，但那是白跑一趟）。
                    print("    （记忆已在：不重复 ADD）", flush=True)
                if position == 0:
                    _require_source_rows(samples, sqlite_path)
                hits = _search_all(samples, client)
                (out_dir / f"{arm.key}-hits.json").write_text(
                    json.dumps(
                        {
                            qid: [{"id": h.id, "created_at": h.created_at} for h in hs]
                            for qid, hs in hits.items()
                        },
                        ensure_ascii=False,
                        indent=1,
                    ),
                    encoding="utf-8",
                )
                entry: dict = {
                    "label": arm.label,
                    "switches": {
                        "router.enabled": arm.router,
                        "agentic.enabled": arm.agentic,
                        "agentic.force": arm.force,
                        "agentic.round_rerank_enabled": arm.round_rerank,
                    },
                    "coverage": _evidence_coverage(samples, hits, gold_evidence),
                    "metrics": _snapshot(services),
                    # ⚠ 把**跑的时候是哪份 profile** 记下来：`local` 与 `default` 的差别
                    #   （精排开关）已经在这里被钉住了，但"其余差别是什么"要能追溯。
                    "profile": base.profile,
                    "rerank_enabled": not args.no_rerank,
                }
                if recorder is not None:
                    # §11 的逐轮过程。⚠ **它含数据集原文**（Query 与候选 id）
                    #   ⇒ 与 `runs/` 一样**不进 Git**。
                    (out_dir / f"{arm.key}-rounds.json").write_text(
                        json.dumps(recorder.calls, ensure_ascii=False, indent=1), encoding="utf-8"
                    )
                    entry["recorded_requests"] = len(recorder.calls)
                if not args.skip_judge:
                    items = []
                    for sample in samples:
                        items += build_input_items(sample, hits)
                    results = run_judge(
                        pipeline,
                        items,
                        out_dir / arm.key,
                        dataset=DATASETS[dataset]["judge"],
                    )
                    correct = sum(1 for r in results if r.is_correct)
                    entry["judged"] = len(results)
                    entry["correct"] = correct
                    entry["accuracy"] = round(correct / len(results), 4) if results else None
                    entry["per_question"] = {r.qid: bool(r.is_correct) for r in results}
                    print(f"    判分：{correct}/{len(results)} = {entry['accuracy']}", flush=True)
            finally:
                client.close()
            report[arm.key] = entry
            (out_dir / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8"
            )

    print("\n\n════ 汇总 ════")
    header = f"{'臂':<30} {'答对':>6} {'题数':>5} {'准确率':>8} {'证据≥1':>7} {'证据全':>7}"
    print(header)
    for arm in chosen:
        entry = report.get(arm.key, {})
        coverage = entry.get("coverage", {})
        print(
            f"{arm.label:<28} {entry.get('correct', '-'):>6} {entry.get('judged', '-'):>5} "
            f"{entry.get('accuracy', '-'):>8} "
            f"{coverage.get('any_hit', '-'):>7} {coverage.get('full_hit', '-'):>7}"
        )
    print(f"\n产物：{out_dir}（report.json / <臂>-hits.json / <臂>/ 逐题答案与判分）")
    print(
        f"⚠ {DATASETS[dataset]['note']}"
    )
    if any(arm.agentic for arm in chosen):
        print(
            "⚠ 多跳臂的用量看 report.json 的 metrics.agentic："
            "`rounds` / `llm_calls` / `rerank_calls` 都是**只有真跑过的那些请求**的和。"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
