"""词法那一路：**Qdrant 原生 `qdrant/bm25`**（§7.1）。

## 这一路没有本地计算

它是**服务端推理**：我们把**文本**交给 Qdrant，它在服务端算 sparse 向量与 IDF。
所以本模块不是"一个 BM25 实现"，而是**这一路的策略与前提**：

* 查询侧**原样送**（与 dense 那一路同一条规格，§7.2）
* **不使用任何学出来的稀疏权重**（learned sparse）——它既不是 BM25，而且提交时的
  `text-embedding-v4` 不提供该能力 ⇒ 任何依赖它的代码在提交时都是**死重**（§7.1 / D2）

## ⚠ 一条还没验的假设：它的分词器

"真 BM25 更强"这个结论**对分词器高度敏感**（§7.1，BGE-M3 论文 Table 11：Lucene
Analyzer 下 BM25 反超学出来的稀疏权重，XLM-R 下明显落败）。所以选 `qdrant/bm25` 的理由
**不是**"BM25 天然更强"，而是"**配正经分词器的** BM25 更强"。而**它用哪个分词器，
我们手里的文档没写** ⇒ **要实测一次**，别默认等价于 Lucene Analyzer
（`docs/open-questions.md` **V1**）。

## 常量的单数来源

`qdrant/bm25` 这个标识符与 `bm25` 这个向量名**都不在本模块重定义**，而是从
[`../store/qdrant_store.py`](../store/qdrant_store.py) 转出——那是**执行**那一侧的定义处。
"""

from __future__ import annotations

from tianxi_am.store.qdrant_store import BM25_MODEL, SPARSE_VECTOR

__all__ = [
    "BM25_MODEL",
    "SPARSE_VECTOR",
    "lexical_query",
]


def lexical_query(query: str) -> str:
    """词法那一路的查询输入 = **原样**。

    这个函数是**故意的恒等映射**：它把"查询侧不做改写"这条规格变成**可断言的对象**，
    而不是一句写在文档里的约定（§7.2 / §5）。v1 没有 Query Analyzer，
    检索之前那一次独立的 query 改写属 v2。

    ⚠ 它也**不**去除停用词、不转小写、不做词干化——那些都算改写，而分词是
    **Qdrant 服务端**的事（这正是 V1 要实测的那一条）。
    """
    if not query.strip():
        raise ValueError("查询不得为空（§2.1：query 是必填字段）")
    return query
