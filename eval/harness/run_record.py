"""把一次 run 组装成 run record——**配置指纹 + 数据指纹 + 结果**（§13）。

> **数字只有 [`../reports/`](../reports/) 一个家**（`schema.py` 定形状，本模块组装）。
> 别处引用数字一律指回去。

## 三个指纹各防什么

| 指纹 | 不记的后果 |
| --- | --- |
| **配置**（profile + 开关 + 快照 hash） | 改过开关的两次 run 被当成同一次，**无法归因** |
| **数据**（数据集 + 版本 + 切批口径） | 归档被换掉、或 `limit=` 截断过，数字却看着可比 |
| **模型**（embedder / LLM / reranker） | §12.1 R1：换模型会让阈值与权重**全部失效** |

## 七维子分：**代理评测产不出，就不填**

`architecture.md` §3 里七个维度是**由设计机制回应**的，不是由题面回应。
所以这里填的是"证据在哪"，不是一个编出来的数——理由见
[`schema.PROXY_SCORE_NOTE`](../reports/schema.py)。

**真正可用的信号是 `breakdown`**：按数据集自己的分类统计准确率。
这正是"哪一类在掉"的输入，而 §14 砍掉的是**按问题类型做收益归因**那种论文式分析。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from eval.datasets import Sample
from eval.reports.schema import DIMENSIONS, PROXY_SCORE_NOTE, RunRecord

from .judge import JudgeResult

__all__ = ["build_record", "config_fingerprint", "summarize", "write_record"]


def config_fingerprint(
    profile: str,
    switches: dict[str, Any] | None = None,
    *,
    configs_dir: Path | None = None,
) -> dict[str, Any]:
    """配置指纹（§13）：profile + 开关状态 + **配置快照 hash**。

    ## 开关的家在 `configs/<profile>.yaml`（③-d）

    本模块**不解析**那些 yaml——解析会与 `common/config.py` 抢同一份知识，
    而 harness 的边界是"打 HTTP、**不 import `src/`**"。⇒ 这里只**逐字节哈希**
    `configs/default.yaml` + `configs/<profile>.yaml`（后者覆盖前者，见 `configs/CLAUDE.md`）。
    **"改了配置但没人注意到"这件事，靠哈希就够发现了。**

    `switches` 留给**消融 arm 显式声明自己改了哪几项**（`checker.*` / `neighbor.*` /
    `rerank.*` / `agent.*` / `budget.*`）——那些键的**落点已定、消费方尚未接线**，
    所以今天传进来的多半是空字典。**它与快照 hash 是两回事**：快照回答"配置文件长什么样"，
    `switches` 回答"这次跑的是哪一组消融"。
    """
    switches = dict(switches or {})
    canonical = json.dumps(switches, sort_keys=True, ensure_ascii=False)

    directory = Path(configs_dir) if configs_dir is not None else Path("configs")
    files = [p for p in (directory / "default.yaml", directory / f"{profile}.yaml") if p.exists()]
    hashes = {p.name: _file_hash(p) for p in files}

    return {
        "profile": profile,
        "switches": switches,
        "switches_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16],
        "snapshot_files": sorted(hashes),
        "snapshot_hashes": hashes,
        "note": (
            "快照 hash 覆盖 configs/default.yaml 与 configs/<profile>.yaml 的**逐字节**内容；"
            "消融开关（checker.* / neighbor.* / rerank.* / agent.* / budget.*）的落点已定、"
            "**消费方尚未接线**，所以 `switches` 今天通常是空的——不代表没有开关。"
            if files
            else f"{directory} 下没有找到配置文件——**这个 run 的配置指纹是不完整的**"
        ),
    }


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def summarize(results: list[JudgeResult], samples: list[Sample]) -> dict[str, Any]:
    """算总分与分类明细。

    **拒答题单列**：`question_id` 以 `_abs` 结尾的题**行为与其他题不同**
    （正确答案是拒答），混进总分会让一个分类的数字失真（§12.2）。
    """
    category_of = {q.qid: q.category for s in samples for q in s.questions}
    abstention_of = {q.qid: q.is_abstention for s in samples for q in s.questions}

    by_category: dict[str, list[bool]] = {}
    abstention: list[bool] = []
    overall: list[bool] = []
    for result in results:
        overall.append(result.is_correct)
        if abstention_of.get(result.qid, False):
            abstention.append(result.is_correct)
            continue
        by_category.setdefault(category_of.get(result.qid, "unknown"), []).append(result.is_correct)

    return {
        "overall": _accuracy(overall),
        "n": len(overall),
        "breakdown": {
            category: {"accuracy": _accuracy(values), "n": len(values)}
            for category, values in sorted(by_category.items())
        },
        "abstention": {"accuracy": _accuracy(abstention), "n": len(abstention)},
    }


def _accuracy(values: list[bool]) -> float | None:
    """**空集合返回 `None` 而不是 0**——`0.0` 会被读成"全错"，而它其实是"没测到"。"""
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def build_record(
    *,
    run_id: str,
    step: str,
    profile: str,
    bench_dir: Path,
    samples: list[Sample],
    results: list[JudgeResult],
    data_fingerprint: dict[str, Any],
    models: dict[str, str],
    switches: dict[str, Any] | None = None,
    configs_dir: Path | None = None,
    metrics: dict[str, Any] | None = None,
    counters: dict[str, Any] | None = None,
    archive: str = "",
    notes: str = "",
) -> RunRecord:
    """组装一条 run record。**七维子分一律 `None` + 理由**（代理评测产不出它们）。"""
    summary = summarize(results, samples)
    scores: dict[str, Any] = {
        "overall": summary["overall"],
        "by_dimension_note": PROXY_SCORE_NOTE,
        **{dimension: None for dimension in DIMENSIONS},
    }
    record = RunRecord(
        run_id=run_id,
        step=step,
        profile=profile,
        config_fingerprint=config_fingerprint(profile, switches, configs_dir=configs_dir),
        data_fingerprint=data_fingerprint,
        models=models,
        scores=scores,
        breakdown=summary["breakdown"] | {"abstention": summary["abstention"]},
        metrics=metrics or {},
        counters=counters or _unavailable_counters(),
        archive=archive,
        notes=notes,
    )
    record.validate()
    return record


def _unavailable_counters() -> dict[str, Any]:
    """§6.5 那三个计数器的**当前状态**：**D24（2026-09-27）起它们已不存在**。

    它们量的是"一个 QA 对是新建 / 被补全 / 落单"，而 D24 取消了跨 Add 合并
    ⇒ 块在写下那一刻就是最终形状，**没有"补全"这种归宿**，发射端
    `pairing/instrument.py` 也已删除。

    **保留字段、不填 0**：字段留着是为了**旧 run record 仍然读得回来**
    （[`schema.py`](../reports/schema.py) 的校验要求它存在）；而 `None` 比 `0` 诚实——
    `0` 会被读成"没有 pending"，事实是"**这个量已经不存在**"。
    """
    return {
        "pending_created": None,
        "pending_completed": None,
        "pending_orphaned_real": None,
        "pending_orphaned_misjudged": None,
        "note": "D24 起 pending 概念已取消（发射端已删除）——这个量**不存在**，不是 0",
    }


def write_record(record: RunRecord, reports_dir: Path) -> Path:
    """落到 `eval/reports/runs/<run_id>.json`。

    ⚠ `eval/reports/**/*.json` 已被 `.gitignore` 忽略（体积大、可重跑，且逐题结果可能
    含受许可约束的数据内容，§12.5）。**结论进 `ledger.md`，原始产出进 `runs/`。**
    """
    return record.write(Path(reports_dir) / "runs" / f"{record.run_id}.json")
