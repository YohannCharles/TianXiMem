"""运行数字与 run record 的唯一落点。

| 文件 | 是什么 | 入库 |
| --- | --- | --- |
| [`schema.py`](./schema.py) | run record 的**形状** | ✅ |
| `runs/**` | 每次 run 的原始产出 | ❌ gitignored（体积大、可重跑、可能含受许可约束的内容） |
"""

from .schema import DIMENSION_MECHANISM, DIMENSIONS, PROXY_SCORE_NOTE, RunRecord

__all__ = ["DIMENSION_MECHANISM", "DIMENSIONS", "PROXY_SCORE_NOTE", "RunRecord"]
