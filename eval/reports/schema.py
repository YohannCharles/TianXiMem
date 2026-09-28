"""run record 的 **schema**——"一次 run 记什么"的唯一声明处。

> **数字只有这一个家**：`docs/experiments.md` 的结论列是**链接**，
> 其余文档引用数字一律指回 [`reports/`](./)。本文件定义**形状**，
> 组装在 [`../harness/run_record.py`](../harness/run_record.py)。

字段清单来自 §13（记录）+ §14（指标）+ §6.5（计数器）+ §3.2（七个维度）+
§12.1 R1（模型标识）+ §15（SQLite 随 run 归档）。**少一组，两次 run 就不可比。**
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


#: 官方七个维度——**逐字采用**（§3.2）。改了名字就没法跨 run 比。
DIMENSIONS: tuple[str, ...] = (
    "Explicit fact recall",
    "Relational and multi-hop reasoning",
    "Temporal and event understanding",
    "Memory governance",
    "Personalization and care",
    "Rules and process execution",
    "Epistemic safety and privacy",
)

#: §3.2 要求**在设计里显式回应**的两维——它们的证据**不是代理分数，是机制与单测**
#: （映射见 [`../../docs/architecture.md`](../../docs/architecture.md) §3）。
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
#: 就和"跑了但没分"长得一样，而那正是 §3.2 说的"等于没测"。
PROXY_SCORE_NOTE = (
    "代理评测的七个维度子分**不可得**：LoCoMo-Refined + LongMemEval 的题面只有"
    "数据集自己的分类，不携带 §3.2 的维度标签。真正的信号是 `breakdown`（按数据集分类），"
    "以及第 4/7 维的机制证据（见 `dimension_mechanism`）。**不要用分类去凑这七个数。**"
)


@dataclass(frozen=True, slots=True)
class RunRecord:
    """一次 run 的全部可追溯信息（§13）。

    **三个指纹缺一不可**：配置指纹（改了什么）、数据指纹（跑的哪份数据、怎么切的批）、
    模型标识（§12.1 R1——切模型会让所有阈值与权重失效，不记就归因不了分数变化）。

    `scores["overall"]` 是**端到端总分**——§12.2 / §14 明确：重点指标是端到端，
    **不是 Recall@K**（召回已接近天花板，缺口在"留哪些、按什么顺序留"）。
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
    #: `question_type`）。§14 只砍掉"按问题类型的**收益归因**"那种论文式分析，
    #: 而"哪一类在掉"正是决定下一步改什么的输入。
    breakdown: dict[str, Any] = field(default_factory=dict)
    #: §14 的四个指标。**只留会触发动作的**——"看着有信息量但不改变下一步动作"的不记。
    metrics: dict[str, Any] = field(default_factory=dict)
    #: §6.5 的三个计数器。**D24（2026-09-27）起这个量已不存在**（`pending` 概念被取消，
    #: 发射端 `pairing/instrument.py` 已删除）⇒ 新记录的四个字段**恒为 `None`**。
    #: 字段保留是为了**旧 run record 还读得回来**；校验也跟着保留，只是理由变了。
    counters: dict[str, Any] = field(default_factory=dict)
    #: 第 4/7 维的**机制证据在哪**（§3.2 要求显式回应，代理分数回应不了这两维）。
    dimension_mechanism: dict[str, str] = field(default_factory=lambda: dict(DIMENSION_MECHANISM))
    #: 本次 run 的 SQLite 副本（§15）。**它是唯一的不可重建物**。
    archive: str = ""
    notes: str = ""
    #: 代理评测**不可线性外推**（§12.4 / P3）——这句话跟着每个数字走。
    scope: str = "proxy: LoCoMo-Refined + LongMemEval only; relative comparisons only"

    def validate(self) -> None:
        """结构性校验：**必填项一个都不能空**。

        `scores` 里的七维值允许为 `None`（代理评测产不出它们），但**必须带理由**
        （`scores["by_dimension_note"]`）——`None` 与"跑了但没分"在报告里长得一样，
        而那正是 §3.2 说的"等于没测"。**用数据集分类去凑这七个数是本文件唯一防的事。**
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
                # D24 之后这两个字段**恒为 None**（量已不存在），但键必须在场：
                # 少了它，新记录与旧记录的形状就对不上，读的人分不清「没写」与「没有这个量」。
                raise ValueError(
                    f"counters 缺 `{orphaned}`——D24 之后这个量已不存在，"
                    "但**键仍必须显式为 None**（写 `0` 会被读成「没有 pending」）"
                )

    def to_json(self) -> str:
        self.validate()
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path
