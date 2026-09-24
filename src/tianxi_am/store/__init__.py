"""存储层：SQLite 真源 + Qdrant 派生索引。

（原标题写"（Step 2 的）Qdrant 派生索引"，**已作废**——D15 把 Qdrant 与 dense / Weighted RRF
一起归 **Step 1**，代码也确实在 Step 1 落地；Step 2 只剩 Neighbor Expansion 与双预算截断。）

**这是唯一接触 SQLite 与 Qdrant 的目录**（§6.3 的分工表只有在读写收口到一处时才守得住）。
"""
