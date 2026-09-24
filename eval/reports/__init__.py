"""数字的唯一落点（§13 / §14 / §3.2 / §15）。

| 文件 | 是什么 | 入库 |
| --- | --- | --- |
| [`schema.py`](./schema.py) | run record 的**形状** | ✅ |
| `ledger.md` | 结果台账（人工维护，**结论本身**） | ✅ |
| `runs/**` | 每次 run 的原始产出 | ❌ gitignored（体积大、可重跑、可能含受许可约束的内容，§12.5） |
"""

from .schema import DIMENSION_MECHANISM, DIMENSIONS, PROXY_SCORE_NOTE, RunRecord

__all__ = ["DIMENSION_MECHANISM", "DIMENSIONS", "PROXY_SCORE_NOTE", "RunRecord"]
