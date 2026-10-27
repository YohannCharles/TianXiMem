"""`Add` 与 `Search` 的**编排**——把各层的调用串成真能跑的两条路径。

> **为什么编排在本目录**：本层"不做检索、不做配对、不碰存储"指的是**不重新实现**那些
> 逻辑（全部往下调用）；而"**按什么顺序调**"必须有人拥有——Search 的链条横跨 `retrieve/`
> 与 `rank/`，Add 的链条横跨 `pairing/`、`store/`、`embed/`，**没有任何单个下层模块能拥有
> 整条链**。所以顺序在这里，逻辑在下面。

**两条链分别住在哪两个文件**（本文件是薄 façade，只重新导出下列名字、没有自己的逻辑）：

* **Add 链**：`AddOutcome` / `AddPipeline` → [`add_pipeline.py`](add_pipeline.py)
* **Search 链**：`AgenticSearch` / `SearchPipeline`（含私有的 `_SearchTrace`）→
  [`search_pipeline.py`](search_pipeline.py)
"""

from __future__ import annotations

from tianximem.service.add_pipeline import AddOutcome, AddPipeline
from tianximem.service.search_pipeline import AgenticSearch, SearchPipeline

__all__ = ["AddOutcome", "AddPipeline", "AgenticSearch", "SearchPipeline"]
