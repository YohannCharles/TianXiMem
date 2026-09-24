"""排序、扩窗、打包（§11 主线）。

**本目录是"排序 → 扩窗 → 打包"三连环的落地点，顺序不可换。**

## ⚠ 目前只有最小切片

`packaging.py` 是 **Step 1 的最小切片**，为的是让 `Search` 能产出合法响应、
通过契约校验。**它不是最终的 Search Pipeline**——Rerank（Step 3）与
Neighbor Expansion（Step 2）都**插在它上游**。

两个阶段的完整对照写在 [`CLAUDE.md`](./CLAUDE.md)，**改动之前先读那一节。**
"""

from tianxi_am.rank.packaging import (
    DEFAULT_TZ,
    PackagedResponse,
    ResponseItem,
    day_granularity,
    package,
    placeholder_score,
)

__all__ = [
    "DEFAULT_TZ",
    "PackagedResponse",
    "ResponseItem",
    "day_granularity",
    "package",
    "placeholder_score",
]
