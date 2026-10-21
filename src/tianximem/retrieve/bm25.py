"""词法检索：**Qdrant 原生 `qdrant/bm25`**。

## 这一路没有本地计算

它是**服务端推理**：我们把**文本**交给 Qdrant，它在服务端算 sparse 向量与 IDF。
所以本模块不是"一个 BM25 实现"，而是**这一路的策略与前提**：

* 查询侧**原样发送**，与 dense 路径一致
* **不使用 learned sparse 权重**：它不等同于 BM25，提交模型 `text-embedding-v4`
  也不提供该能力

Qdrant 在服务端负责 BM25 分词。检索实现不假定其分词结果与 Lucene 等分析器等价。

## 常量的单数来源

`qdrant/bm25` 这个标识符与 `bm25` 这个向量名**都不在本模块重定义**，而是从
[`../store/qdrant_store.py`](../store/qdrant_store.py) 转出——那是**执行**那一侧的定义处。
"""

from __future__ import annotations

from tianximem.store.qdrant_store import BM25_MODEL, SPARSE_VECTOR

__all__ = [
    "BM25_MODEL",
    "SPARSE_VECTOR",
    "lexical_query",
]


def lexical_query(query: str) -> str:
    """词法那一路的查询输入 = **原样**。

    这个函数是**故意的恒等映射**，使"查询侧不做改写"成为可断言的接口行为。
    服务不运行独立的预检索查询改写；多跳执行器只在自己的循环中生成新的检索查询。

    ⚠ 它也**不**去除停用词、不转小写、不做词干化——那些都算改写，而分词是
    **Qdrant 服务端**的事。
    """
    if not query.strip():
        raise ValueError("查询不得为空（§2.1：query 是必填字段）")
    return query
