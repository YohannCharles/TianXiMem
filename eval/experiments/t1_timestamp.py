"""T1 —— 日期前缀 **带 / 不带**。

```text
带（默认）  packaging.inject_abs_time = true
不带        packaging.inject_abs_time = false   ← 消融臂
```

**它决定什么**：`content` 的渲染方式。日期写在每一对旁边
能让日期锚点紧邻对应语句，因此当前作为默认口径；关闭时仍保留作消融臂。

**这不是"验一条结论"，而是量化一个开关的影响**：开关会改变**检索命中的内容**
（见下），所以必须使用隔离的 A/B 运行。

## ⚠ 这是"贵"消融项：两臂必须跑在**两个集合**上

正文一改，**embedding 输入就改**：同一份渲染既作为索引输入，也作为 `content`，
所以：

| 后果 | 为什么不能省 |
| --- | --- |
| 向量要**重算** | 缓存键是渲染后文本的哈希 ⇒ 自动失效 |
| 两臂要**分集合** | 共用一个集合时，"检索到的是哪一臂的向量"**看不出来**——而两臂的向量不一样 |
| 两臂各有一份**冻结配置** | 对比必须记录每臂实际使用的 `configs/runs/<arm>/` |

⇒ 本文件只声明两臂；冻结、核对、驱动、比较都在 [`arms.py`](./arms.py)。

## 用法

```bash
# 1) 冻结两臂的配置快照（**跑之前**做，之后改 configs/*.yaml 不影响已冻结的臂）
uv run python -m eval.experiments.t1_timestamp --freeze

# 2) 两个服务（各带一份快照）——**服务自己读配置，脚本不代管进程**
TIANXIMEM_CONFIG_DIR=configs/runs/t1-plain TIANXIMEM_PROFILE=local make serve   # :8000
TIANXIMEM_CONFIG_DIR=configs/runs/t1-dated TIANXIMEM_PROFILE=local make serve   # 另一个端口

# 3) 跑两臂（同一份数据、同一批题）
uv run python -m eval.experiments.t1_timestamp --execute \
    --plain-url http://127.0.0.1:8000 --dated-url http://127.0.0.1:8001

# 4) 比一比
uv run python -m eval.experiments.t1_timestamp --compare <run-id-plain> <run-id-dated>
```

⚠ **`--limit` 只给冒烟用**：截断跑会在数据指纹里留警示，**两臂的数字不能拿它定论**。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Final

from eval.experiments.arms import Arm as ArmBase
from eval.experiments.arms import Spec
from eval.experiments.arms import freeze as freeze
from eval.experiments.arms import main as main
from eval.experiments.arms import verify as verify

SPEC: Final[Spec] = Spec(
    label="T1",
    generator="t1_timestamp.py",
    note=("#   ↑ 集合必须分开：两臂的向量不同，混在一个集合里检索到的是哪一臂**看不出来**。\n"),
    prog="t1_timestamp.py",
    description="§13 的 T1 实验脚手架",
    compare_labels=("plain", "dated"),
    url_flags=(
        ("--plain-url", "http://127.0.0.1:8000"),
        ("--dated-url", "http://127.0.0.1:8001"),
    ),
    plan_hint="两臂快照齐备。要跑的话（服务各带一份快照）：",
    execute_hint="两臂**必须分集合**（已写进各自快照），且都跑完整数据集。",
    compare_warning=(
        "这只是**一次**两臂之差。差异要归因，得看逐题判语"
        "（`runs/<run_id>/*/labels.jsonl`）——总分分不开「日期前缀」与「检索变了」这两件事"
        "（正文进了索引，§11.3）。逐题判语保存在 run 产物中。"
    ),
)


@dataclass(frozen=True, slots=True)
class Arm(ArmBase):
    """一条臂 = 一份冻结配置 + 一个集合。**两臂只差 `inject_abs_time`。**"""

    SPEC: ClassVar[Spec] = SPEC

    name: str
    inject_abs_time: bool
    collection: str

    def overrides(self) -> dict:
        """这份快照相对基线**必动的两项**（其余全部继承 `default.yaml`）。"""
        return {
            "storage": {"qdrant": {"collection": self.collection}},
            "packaging": {"inject_abs_time": self.inject_abs_time},
        }

    def switches(self) -> dict:
        """只记**开关**：集合名是「必须分集合」的落地，不是这次对照的自变量。"""
        return {"packaging.inject_abs_time": self.inject_abs_time}


#: 两臂。**顺序即「对照在前、变体在后」**。
ARM_PLAIN: Final[Arm] = Arm(name="t1-plain", inject_abs_time=False, collection="memories_t1_plain")
ARM_DATED: Final[Arm] = Arm(name="t1-dated", inject_abs_time=True, collection="memories_t1_dated")
ARMS: Final[tuple[Arm, ...]] = (ARM_PLAIN, ARM_DATED)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(ARMS))
