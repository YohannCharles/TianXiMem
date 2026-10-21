"""run record 的 **schema**——"一次 run 记什么"的唯一声明处。

> **数字只有这一个家**：其他文档引用运行数字时一律指回 [`reports/`](./)。
> 本文件定义**形状**，组装在 [`../harness/run_record.py`](../harness/run_record.py)。

字段覆盖运行记录、服务指标、兼容计数器、七个评测维度、模型标识和 SQLite 归档。
**少一组，两次 run 就不可比。**
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "DIMENSION_MECHANISM",
    "DIMENSIONS",
    "PROXY_SCORE_NOTE",
    "RunRecord",
]


#: 官方七个维度——**逐字采用**。改了名字就没法跨 run 比。
DIMENSIONS: tuple[str, ...] = (
    "Explicit fact recall",
    "Relational and multi-hop reasoning",
    "Temporal and event understanding",
    "Memory governance",
    "Personalization and care",
    "Rules and process execution",
    "Epistemic safety and privacy",
)

#: 必须**在设计里显式回应**的两维——它们的证据**不是代理分数，是机制与单测**
#: （映射见 [`../../docs/architecture.md`](../../docs/architecture.md)）。
#: 代理评测只有 LoCoMo-Refined + LongMemEval 两套题面，**产不出七个维度子分**——
#: 所以这两维在 run record 里填的是**证据在哪**，不是一个编出来的数。
DIMENSION_MECHANISM: dict[str, str] = {
    "Memory governance": (
        "批次级幂等守卫 + 填空追加（§6.1 / §6.5）——"
        "由 `tests/test_idempotency.py` 验证；**它测的是写入是否诚实，检索变差不代表它坏了**"
    ),
    "Epistemic safety and privacy": (
        "`user_id` 唯一隔离 + `Search` 不生成答案（§2.2 / §2.1）——"
        "由 `tests/test_isolation.py` 按路径逐个验证"
    ),
}

#: 代理评测产不出七维子分时，`scores` 里那份说明**必须**带上——空值不写理由，
#: 就和"跑了但没分"长得一样，等同于没有测量。
PROXY_SCORE_NOTE = (
    "代理评测的七个维度子分**不可得**：LoCoMo-Refined + LongMemEval 的题面只有"
    "数据集自己的分类，不携带 §3.2 的维度标签。真正的信号是 `breakdown`（按数据集分类），"
    "以及第 4/7 维的机制证据（见 `dimension_mechanism`）。**不要用分类去凑这七个数。**"
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """一次 run 的全部可追溯信息。

    **三个指纹缺一不可**：配置指纹（改了什么）、数据指纹（跑的哪份数据、怎么切的批）、
    模型标识（换模型会让所有阈值与权重失效，不记就归因不了分数变化）。

    `scores["overall"]` 是**端到端总分**，重点指标不是 Recall@K；
    缺口在"留哪些、按什么顺序留"。
    """

    run_id: str
    step: str
    profile: str
    config_fingerprint: dict[str, Any]
    data_fingerprint: dict[str, Any]
    models: dict[str, str]
    scores: dict[str, Any]
    #: **代理评测真正可用的信号**：按数据集自己的分类统计准确率
    #: （LoCoMo 的 1–4 类是 multihop/temporal/open-domain/single-hop，LME 是六种
    #: `question_type`）。这里的分类用于定位"哪一类在掉"，不做收益归因。
    breakdown: dict[str, Any] = field(default_factory=dict)
    #: run 的服务指标。**只留会触发动作的**——不改变判断的信息不记。
    metrics: dict[str, Any] = field(default_factory=dict)
    #: 兼容字段：当前写入模型不使用 pending 阶段，因此四个字段为 `None`。
    #: 字段保留使已有 run record 仍按同一 schema 读取。
    counters: dict[str, Any] = field(default_factory=dict)
    #: 机制维度对应的代码与测试证据。
    dimension_mechanism: dict[str, str] = field(default_factory=lambda: dict(DIMENSION_MECHANISM))
    #: 本次 run 的 SQLite 副本。**它是唯一的不可重建物**。
    archive: str = ""
    notes: str = ""
    #: 代理评测结果**不可线性外推**——这个范围声明跟着每个数字走。
    scope: str = "proxy: LoCoMo-Refined + LongMemEval only; relative comparisons only"

    def validate(self) -> None:
        """结构性校验：**必填项一个都不能空**。

        `scores` 里的七维值允许为 `None`（代理评测产不出它们），但**必须带理由**
        （`scores["by_dimension_note"]`）——`None` 与"跑了但没分"在报告里长得一样，
        等同于没有测量。**用数据集分类去凑这七个数是本文件唯一防的事。**
        """
        if not self.run_id or not self.step or not self.profile:
            raise ValueError("run_id / step / profile 必填")
        for key in ("config_fingerprint", "data_fingerprint", "models"):
            if not getattr(self, key):
                raise ValueError(f"{key} 为空——§13 要求三个指纹齐全，缺一个两次 run 就不可比")
        if "overall" not in self.scores:
            raise ValueError("scores 缺 `overall`（端到端总分，§13）")
        missing = [d for d in DIMENSIONS if d not in self.scores]
        if missing:
            raise ValueError(f"scores 缺维度：{missing}——七个维度必须逐维记录（§3.2）")
        if any(self.scores[d] is None for d in DIMENSIONS) and not self.scores.get(
            "by_dimension_note"
        ):
            raise ValueError(
                "七维里有空值却没说理由——空值必须写清为什么（代理评测产不出它们），"
                "否则与「跑了但没分」无法区分（§3.2）"
            )
        for orphaned in ("pending_orphaned_real", "pending_orphaned_misjudged"):
            if orphaned not in self.counters:
                # 当前这两个字段**恒为 None**（量已不存在），但键必须在场：
                # 少了它，记录之间形状不一致，读的人分不清「没写」与「没有这个量」。
                raise ValueError(
                    f"counters 缺 `{orphaned}`——当前计数不再产生这个量，"
                    "但**键仍必须显式为 None**（写 `0` 会被读成「没有 pending」）"
                )

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path
