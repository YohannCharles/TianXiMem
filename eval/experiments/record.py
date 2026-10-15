"""run record 的组装与打印——"这一轮跑的是什么、结果长什么样"（§13 / §12.1 R1）。

本模块是 `run.py` 拆出来的记录半（2026-10-15）：数据指纹之外那几样"描述这次 run"的东西——
截断警示语（`truncation_note`）、模型与消融开关的声明（`models_fingerprint` /
`_parse_switches`）、§14 服务指标的读取与精排降级的提示（`_read_service_metrics` /
`_warn_about_rerank`），以及跑完之后屏幕上那几行（`_print_result`）。
**一轮怎么跑不在本模块**（那在 [`runner.py`](./runner.py)），命令行与薄 façade 在
[`run.py`](./run.py)。

⚠ 测试/消费方若 `monkeypatch` 本模块的名字，**要打在真实现所在的这里**
（`eval.experiments.record`）——打在 `run.py` 的 façade 上对搬走的实现不生效。

⚠ **"打 HTTP、不 import `src/`"这条边界在 [`../CLAUDE.md`](../CLAUDE.md)，本文件不重复。**
"""

from __future__ import annotations

import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eval.datasets import Sample, data_fingerprint
from eval.datasets.aml import INPUT_CONTRACT
from eval.harness import build_record
from eval.harness.api_config import ANSWER_MODEL, ENDPOINTS
from eval.harness.scheduling import worker_counts


def derive_run_id(dataset: str, *, now: datetime | None = None) -> str:
    """`<dataset>-<UTC 时间戳>`——**一次 run 一个 id，且 id 里带得动数据集**。

    不用随机串：`eval/reports/runs/<run_id>.json` 是要被人翻的，
    而"这个文件是跑哪份数据的"应当从文件名就能读出来。
    """
    stamp = (now or datetime.now(tz=UTC)).strftime("%Y%m%dT%H%M%S")
    return f"{dataset}-{stamp}"


def models_fingerprint(*, embedder: str, llm: str, reranker: str) -> dict[str, str]:
    """§12.1 R1 的模型指纹：**换模型会让所有阈值与权重失效**。

    ⚠ **这三行是"声明"，不是"探测"**（与 `.env.example` 里 reranker 那条同一口径）：
    端点本身不回可信的模型名（reranker 回的永远是服务端路径）。
    `src/` 那边的真值在 `configs/*.yaml` 的 `models.embedder`，而 **harness 不解析 yaml**
    ——解析会与 `common/config/` 抢同一份知识
    （[`../harness/run_record.py`](../harness/run_record.py)
    的 `config_fingerprint` 只逐字节哈希它，理由在那个 docstring 里）。
    ⇒ 所以这里由调用方声明，缺了就**显式写成"没说"**，而不是编一个看起来合理的值。
    """
    return {
        "embedder": embedder or "<未声明：传 --embedder 或设 TIANXIMEM_EMBED_MODEL>",
        "llm": llm or "<未声明：传 --llm>",
        "reranker": reranker or "disabled",
    }


def _default_llm() -> str:
    """归档 pipeline 的模型名从 `api_config` 来——**那是它们唯一的配置源**。

    `--model` 之类的 CLI 参数是死的（`answer()` / `evaluate()` 在协程开头用
    `api_config` 的常量重写 `args.*`），所以 harness 侧能读的也只有同一个名字。
    """
    return ANSWER_MODEL or ""


def _parse_switches(raw: str | None) -> dict[str, Any]:
    """`--switches '{"neighbor.radius": 0}'`——**消融臂显式声明自己改了哪几项**。

    它与配置快照 hash 是两回事：快照回答"配置文件长什么样"，这里回答"这次跑的是哪一组消融"
    （[`../harness/run_record.py`](../harness/run_record.py) 的 `config_fingerprint`）。
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise SystemExit(f"--switches 不是合法 JSON：{error}") from None
    if not isinstance(parsed, dict):
        raise SystemExit("--switches 必须是一个 JSON 对象（键=开关名，值=该臂的取值）")
    return parsed


def truncation_note(
    limit: int | None, *, max_questions: int | None = None, spread: bool = False
) -> str:
    """`--limit` 的警示语——写进数据指纹的 `note`（§13）。

    **这是本文件唯一一处"截断是否发生过"的判断**：跑了一小撮与跑完了的数字
    **在 run record 里长得一模一样**，不写下来就会有人拿它们比。
    ⚠ **抽样方式也要写**：`spread` 时取的是**跨类**的题，不写就成了"另一种前 N 题"。
    """
    head = ""
    if limit is not None:
        if spread:
            head = (
                f"⚠ **截断跑（分层抽样）**：按 `question_type` 按比例取了约 {limit} 题"
                "（**不是前 N 题**）——不可与全量比"
            )
        else:
            head = f"⚠ **截断跑**：只加载了前 {limit} 个 sample——不可与全量比"
    if max_questions is not None:
        tail = (
            f"⚠ **题目截断**：每个 sample 只判了前 {max_questions} 题（语料是整份）——不可与全量比"
        )
        return f"{head}\n{tail}" if head else tail
    return head


def _count_questions(samples: list[Sample]) -> int:
    return sum(len(sample.questions) for sample in samples)


def _assemble_record(args, *, run_id: str, bench_dir: Path, samples, results, note: str):
    """把这一轮的**配置指纹 + 数据指纹 + 结果**装成 run record（§13）。"""
    if getattr(args, "input_contract", "native") == INPUT_CONTRACT:
        path = Path(args.reports_dir) / "runs" / run_id / "input-manifest.json"
        fingerprint = json.loads(path.read_text(encoding="utf-8"))["data_fingerprint"]
        fingerprint = fingerprint | {
            "dataset": args.dataset,
            "benchmark_dir": str(bench_dir),
            "add_shape": args.add_shape,
            "note": note,
        }
    else:
        fingerprint = data_fingerprint(
            bench_dir,
            args.dataset,
            n_samples=len(samples),
            n_questions=_count_questions(samples),
            add_shape=args.add_shape,
            note=note,
        )
    return build_record(
        run_id=run_id,
        step=args.step,
        profile=args.profile,
        bench_dir=bench_dir,
        samples=samples,
        results=results,
        data_fingerprint=fingerprint,
        models=models_fingerprint(
            embedder=args.embedder,
            llm=args.llm or _default_llm(),
            reranker=args.reranker,
        ),
        switches=_parse_switches(args.switches),
        configs_dir=Path(args.configs_dir),
        metrics=_read_service_metrics(args)
        | {
            "evaluation_scheduling": {
                "ingest_workers": worker_counts(
                    args.ingest_workers,
                    args.judge_workers,
                    users=len(samples),
                    endpoints=len(ENDPOINTS),
                )[0],
                "judge_workers": worker_counts(
                    args.ingest_workers,
                    args.judge_workers,
                    users=len(samples),
                    endpoints=len(ENDPOINTS),
                )[1],
            }
        },
        notes=args.notes,
    )


def _read_service_metrics(args) -> dict[str, Any]:
    """读服务侧写的 §14 指标快照（`--metrics`，缺省取 `TIANXIMEM_METRICS_PATH`）。

    **它不是 `counters=`**：那份是 §6.5 的 pending 计数器，`schema.validate()` 要求
    `pending_orphaned_*` 两个键**必须在场**（D24 之后恒为 `None`）。§14 的指标住
    `metrics=`（[`../reports/CLAUDE.md`](../reports/CLAUDE.md) 的字段表就是这么分的）。

    ⚠ **读不到就返回空 dict，并且说出来**：空 `metrics` 是**可见的**
    （记录里没有 `rerank` 键），但它有两种来源——"服务侧没配 `TIANXIMEM_METRICS_PATH`"
    与"这一轮真的没精排"。不说出来，第二种会被读成第一种。
    """
    raw = args.metrics or os.environ.get("TIANXIMEM_METRICS_PATH", "")
    if not raw:
        return {}
    path = Path(raw)
    if not path.exists():
        print(
            f"⚠ 指标快照不存在：{path}——服务侧没配 `TIANXIMEM_METRICS_PATH`，"
            "或那个进程这一轮没跑过 Search。这一轮的 metrics 为空。",
            file=sys.stderr,
        )
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _warn_about_rerank(metrics: dict[str, Any]) -> None:
    """**V12**："精排到底有没有在跑"此前没有任何门禁看得见——降级后的响应与精排成功的
    响应**逐字同样合法**，`contract-check` 的 14 条全部照样通过。

    所以这一条要念出来。**它只警告，不让 run 失败**：端点挂了是**已实现的降级路径**（D12），
    不是这一轮跑错了——但"这一轮的排序其实退回了 RRF 顺序"必须写在记录旁边。
    """
    rerank = metrics.get("rerank") or {}
    searches = metrics.get("searches", 0)
    if rerank.get("degraded"):
        print(
            f"\n⚠ 精排降级 {rerank['degraded']} 次（共 {searches} 次 Search）"
            "——**这些请求的排序退回了 RRF 顺序**（D12 / V12）。"
        )
    elif rerank.get("disabled"):
        print(
            f"\n⚠ 这个 run 全程没走精排（`rerank.disabled={rerank['disabled']}`）"
            "——排序是融合名次。是刻意的还是漏配了，看服务启动那几行。"
        )


def _print_result(record, path: Path) -> None:
    print(
        f"\n{record.run_id}：overall={record.scores['overall']}"
        f"（n={record.breakdown.get('abstention', {}).get('n', 0)} 道拒答另计）"
    )
    if record.scores.get("dataset_score"):
        print("  dataset_score=" + json.dumps(record.scores["dataset_score"], ensure_ascii=False))
    for category, entry in record.breakdown.items():
        print(f"  {category}: {entry}")
    print(f"\nrun record → {path}")
    _warn_about_rerank(record.metrics)
    print("⚠ 数字只有 eval/reports/ 一个家：结论写进 ledger.md，不要复制别处。")
