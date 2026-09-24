"""数据加载与 schema 落差预处理层（§12.2 / §12.3 第 9 条）。

```python
from eval.datasets import load_locomo, load_longmemeval, benchmark_dir

samples = load_locomo(benchmark_dir())
```

**加载器只服务两个计分数据集**（LoCoMo-Refined + LongMemEval）——它们是六份里
唯一共用同一套契约的（§12.4）。其余四份的契约差异**不在这里加载**。

> **边界（D16）**：只有 [`locomo`][eval.datasets.locomo] /
> [`longmemeval`][eval.datasets.longmemeval] / [`registry`][eval.datasets.registry]
> 三个模块允许知道数据集的文件名。`pairing/` 与 `store/` 只能看见
> [`preprocess`][eval.datasets.preprocess] 归一化后的形状。
"""

from .locomo import load_locomo
from .longmemeval import load_longmemeval
from .preprocess import Message, Question, Sample, Session
from .registry import BATCHING_LOCAL, benchmark_dir, data_fingerprint, file_fingerprint

__all__ = [
    "BATCHING_LOCAL",
    "Message",
    "Question",
    "Sample",
    "Session",
    "benchmark_dir",
    "data_fingerprint",
    "file_fingerprint",
    "load_locomo",
    "load_longmemeval",
]
