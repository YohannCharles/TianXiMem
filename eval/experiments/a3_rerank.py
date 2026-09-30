"""A3 —— **Rerank 开 / 关**（§13 / §11.2）。

```text
on    rerank.enabled = true    ← 精排（`qwen3-reranker-4b`，100 篇/题 5–12s）
off   rerank.enabled = false   ← 直接用融合名次
```

**它决定什么**：排序值不值这份算力（§11 主线的验证）。若不值，把算力挪去别处。

## ⚠ 两臂**共用一套向量**——这是 A3 与 T1 最大的不同

`rerank.enabled` 只改**排名**，不改正文 ⇒ 索引、embedding 输入、缓存全都一样。
所以两臂**可以而且应该**用同一个集合（`memories_dev`）；只有 A3 与 T1 之间才需要分集合
（T1 改的是正文）。⇒ **跑 A3 不需要重建索引**，开关一翻即可，代价只是再跑一轮评测。

## 用法

```bash
# 1) 冻结两臂配置（跑之前做；之后改 configs/*.yaml 不影响已冻结的臂）
uv run python -m eval.experiments.a3_rerank --freeze

# 2) 两个服务，各带一份快照（**同一个集合**，端口不同）
TIANXIMEM_CONFIG_DIR=configs/runs/a3-on  TIANXIMEM_PROFILE=local make serve   # :8000
TIANXIMEM_CONFIG_DIR=configs/runs/a3-off TIANXIMEM_PROFILE=local make serve   # :8001

# 3) 两臂各跑一轮（同一份数据、同一批题）
uv run python -m eval.experiments.a3_rerank --execute \
    --on-url http://127.0.0.1:8000 --off-url http://127.0.0.1:8001

# 4) 比一比（都是相对比较，别外推）
uv run python -m eval.experiments.a3_rerank --compare <run-id-on> <run-id-off>
```

⚠ **`--limit` 只给冒烟用**：截断跑会在数据指纹里留警示，两臂的数字不能拿它定论。
⚠ 两臂若跑在**不同时刻**，注意 `rerank.enabled` 之外的配置必须一样——`--freeze` 保证这一点。

⇒ 本文件只声明两臂；冻结、核对、驱动、比较都在 [`arms.py`](./arms.py)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from eval.experiments.arms import Arm as ArmBase
from eval.experiments.arms import Spec
from eval.experiments.arms import base_drift as base_drift
from eval.experiments.arms import freeze as freeze
from eval.experiments.arms import main as main
from eval.experiments.arms import verify as verify

SPEC: Final[Spec] = Spec(
    label="A3",
    generator="a3_rerank.py",
    note=(
        "#   ↑ 两臂**共用一套向量**（rerank 只改排名）⇒ 集合不用换；变的是排序与耗时。\n"
        "#     这也是 A3 与 T1 最大的不同：T1 改正文，那边必须分集合。\n"
    ),
    prog="a3_rerank.py",
    description="§13 的 A3 实验脚手架",
    compare_labels=("on", "off"),
    url_flags=(
        ("--on-url", "http://127.0.0.1:8000"),
        ("--off-url", "http://127.0.0.1:8001"),
    ),
    plan_hint="两臂快照齐备。要跑的话（**同一个集合**，两个服务）：",
    execute_hint="两臂共用一套向量（rerank 只改排名）⇒ 集合不用重建，开关一翻即可。",
    compare_warning=(
        "这是**代理评测上的相对比较**（§12.4 / P3）：两份数据集共用契约，其余四份不适用。"
        "结论写进 eval/reports/ledger.md 与 docs/experiments.md 的结论列。"
    ),
)


@dataclass(frozen=True, slots=True)
class Arm(ArmBase):
    """一条臂 = 一份冻结配置。**两臂只差 `rerank.enabled`。**"""

    SPEC: ClassVar[Spec] = SPEC

    name: str
    rerank_enabled: bool

    def overrides(self) -> dict:
        """相对基线**只动这一件事**——A3 的对照内容就是它。"""
        return {"rerank": {"enabled": self.rerank_enabled}}

    def switches(self) -> dict:
        return {"rerank.enabled": self.rerank_enabled}


ARM_ON: Final[Arm] = Arm(name="a3-on", rerank_enabled=True)
ARM_OFF: Final[Arm] = Arm(name="a3-off", rerank_enabled=False)
ARMS: Final[tuple[Arm, ...]] = (ARM_OFF, ARM_ON)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(ARMS))
