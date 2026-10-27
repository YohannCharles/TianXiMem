"""排序、邻居扩窗、段合并与打包。

**本目录是"排序 → 扩窗 → 打包"三连环的落地点，顺序不可换。**

```text
Hybrid Retrieval → RRF → memory_id 稳定去重
  → 【rerank（恰好一次）】      ← rank/reranker.py：远端客户端
  → Neighbor Expansion         ← rank/neighbor.py：前 N 条扩 ±radius
  → Context Segment Merge      ← rank/neighbor.py：沿显式链相接的块合成段
  → Token Budget               ← rank/packaging.py + common/tokens.py
  → Final Packaging            ← rank/packaging.py
```

端点不可用时降级回 RRF 顺序，**不报错**（请求格式见 [`reranker.py`](./reranker.py)）。
排序约定见 [`packaging.py`](./packaging.py) 与 [`neighbor.py`](./neighbor.py)。
"""

from tianximem.rank.neighbor import (
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_RADIUS,
    ContextSegment,
    ExpansionResult,
    SelectedMemory,
    expand_neighbors,
    merge_segments,
)
from tianximem.rank.packaging import (
    PackagedResponse,
    ResponseItem,
    day_granularity,
    package,
    placeholder_score,
)
from tianximem.rank.reranker import RemoteReranker, Reranker, RerankUnavailable

__all__ = [
    "DEFAULT_EXPANSION_SEED_LIMIT",
    "DEFAULT_RADIUS",
    "ContextSegment",
    "ExpansionResult",
    "PackagedResponse",
    "RemoteReranker",
    "RerankUnavailable",
    "Reranker",
    "ResponseItem",
    "SelectedMemory",
    "day_granularity",
    "expand_neighbors",
    "merge_segments",
    "package",
    "placeholder_score",
]
