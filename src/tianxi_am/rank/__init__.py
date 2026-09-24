"""排序、扩窗、打包（§10 / §11 主线）。

**本目录是"排序 → 扩窗 → 打包"三连环的落地点，顺序不可换。**

## 现在的阶段（2026-09-24）

```text
Hybrid Retrieval → RRF → memory_id 稳定去重
  → 【rerank（恰好一次）】      ← rank/reranker.py：**远端客户端已接**
  → Neighbor Expansion         ← rank/neighbor.py：前 N 条扩 ±radius
  → Context Segment Merge      ← rank/neighbor.py：连续 pair_idx 合成段
  → Token Budget               ← rank/packaging.py + common/tokens.py
  → Final Packaging            ← rank/packaging.py
```

**三连环全部落地**。`RemoteReranker` 打主网关的 `/rerank`（线格式实测过，
见 [`reranker.py`](./reranker.py) 的表格）；端点不可用时按 D12 降级回 RRF 顺序，
**不报错**。两个阶段的完整对照写在 [`CLAUDE.md`](./CLAUDE.md)，**改动之前先读那一节。**
"""

from tianxi_am.rank.neighbor import (
    DEFAULT_EXPANSION_SEED_LIMIT,
    DEFAULT_RADIUS,
    ContextSegment,
    ExpansionResult,
    SelectedMemory,
    expand_neighbors,
    merge_segments,
)
from tianxi_am.rank.packaging import (
    PackagedResponse,
    ResponseItem,
    day_granularity,
    package,
    placeholder_score,
)
from tianxi_am.rank.reranker import RemoteReranker, Reranker, RerankUnavailable

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
