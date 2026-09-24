"""混合检索的**策略与参数所有权**（§7.3）。

## 本模块与 `store/` 的分界（2026-09-24 定）

| 谁 | 内容 |
| --- | --- |
| **本模块（策略与参数）** | `prefetch_limit` / `weights` / `k` / `top_k` 的取值、**校验**、标定 |
| **`store/qdrant_store.py`（执行）** | 翻成 `prefetch` + `rrf` 的 Qdrant 调用 |

⇒ **本模块说"用什么参数"，`store/` 说"怎么发给 Qdrant"。** 一份实现不许两边都写。

## 参数里有一个不是旋钮

`rrf_k` 是**正确性常量**：Qdrant 的默认 `k` 是 **2**（官方文档逐字："k is a constant
(set to 2 by default)"），而 RRF 文献是 1-based 的 `1/(60 + rank)`。Qdrant 的秩是
0-based、公式 `1/(rank + k)`，所以**只有 `k = 61`** 才等价于文献的 60
（`1/(0+61) = 1/(1+60)`）。**填错不会报错**——两种写法产出的名次都"看起来正常"，
只有分数差一个数量级（已在 v1.17.0 上实测：`2/61 = 0.032786883` vs 默认 `2/2 = 0.5`）。

因此本模块**拒绝**任何 `k != 61` 的配置，而不是"警告后照用"。见 D5。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tianxi_am.retrieve.bm25 import lexical_query
from tianxi_am.store.qdrant_store import HybridParams, QdrantStore

__all__ = [
    "DEFAULT_PREFETCH_LIMIT",
    "DEFAULT_WEIGHTS",
    "RRF_K",
    "Candidate",
    "HybridRetriever",
    "dedup_candidates",
    "make_hybrid_params",
]

#: 每路进入 RRF 融合的候选池大小（§7.3 的 `N`）。
#: ⚠ **不是**种子数（§10 的 20）、**不是** Top-K（§2.2 的 100）——三个不同的量。
DEFAULT_PREFETCH_LIMIT: Final[int] = 200

#: 权重初值（§7.3 的 `[w_bm25, w_dense]`，顺序与 `prefetch` 顺序一一对应）。
#: **任何调整都必须有 ablation 数据支撑**——Qdrant 官方明确警告"无评测集时手调权重
#: 不太可能稳定优于默认值"。
DEFAULT_WEIGHTS: Final[tuple[float, float]] = (0.5, 0.5)

#: **正确性常量，不是可调项**（D5）。
RRF_K: Final[int] = 61


class RetrievalParamError(ValueError):
    """检索参数不合法。

    ⚠ 这类错误**必须响亮**：`rrf_k` 填错、`top_k` 被写死，都属于"不报错但结果错"的一类。
    """


@dataclass(frozen=True, slots=True)
class Candidate:
    """一个检索候选。**这是 `retrieve/` 交给 `rank/` 的东西。**

    * `rank` 是 **0-based 名次**——**唯一有意义的量**
    * **没有 `score`，这是故意的。** 融合分数**不是校准量**（§8：Qdrant 官方明确警告
      不要把单路阈值用到根级 `score_threshold`），而响应里的 `score` 必须是
      **最终名次的函数**（`1/(rank+1)`，由 `rank/` 现算）。

    ⇒ **上游手里根本没有那个值，也就漏不出去**——这是**结构性**防止透传，
    而不是靠一句"记得别传"。Rerank 接入后名次会变，届时 `score` 也随之重算。
    """

    memory_id: str
    rank: int


def make_hybrid_params(
    *,
    prefetch_limit: int = DEFAULT_PREFETCH_LIMIT,
    weights: Sequence[float] = DEFAULT_WEIGHTS,
    rrf_k: int = RRF_K,
) -> HybridParams:
    """校验并构造交给 `store/` 执行的参数对象。

    **校验在这里，不在 `store/`**——参数的可接受域是检索策略的一部分。
    """
    if prefetch_limit <= 0:
        raise RetrievalParamError(f"prefetch_limit 必须为正：{prefetch_limit}")
    if len(weights) != 2:
        raise RetrievalParamError(
            f"weights 必须是两个（先后对应 bm25、dense，见 §7.3 的 prefetch 顺序）：{weights!r}"
        )
    w_bm25, w_dense = float(weights[0]), float(weights[1])
    if w_bm25 < 0 or w_dense < 0:
        raise RetrievalParamError(f"权重不得为负：{(w_bm25, w_dense)!r}")
    if w_bm25 == 0 and w_dense == 0:
        raise RetrievalParamError("两个权重不能同时为 0——那等于不检索")
    if rrf_k != RRF_K:
        raise RetrievalParamError(
            f"rrf_k 必须是 {RRF_K}，收到 {rrf_k}。\n"
            "  它是【正确性常量】不是可调项：Qdrant 的 k 默认是 2、文献是 60，"
            f"而 Qdrant 的秩 0-based，只有 {RRF_K} 才等价于文献的 60（D5）。\n"
            "  填错【不会报错】——两种写法的名次都看起来正常，只有分数差一个数量级。"
        )
    return HybridParams(
        prefetch_limit=prefetch_limit,
        weights=(w_bm25, w_dense),
        rrf_k=rrf_k,
    )


def dedup_candidates(candidates: Sequence[Candidate]) -> list[Candidate]:
    """按 `memory_id` **稳定去重**，并把名次**重新编成连续的 0-based**。

    **"稳定"的含义**：同一个 `memory_id` 出现多次时，**保留名次最好的那一次**
    （第一次出现），而不是最后一次——融合结果里不可能有两条完全相同的记忆，
    真出现时只有"最早的那条"能与 RRF 的名次语义对上。

    ⚠ **必须重新编号**，不能只删元素：下游有"**只对前 N 条**做扩窗"这类判据
    （§10 的种子数），而它读的是名次。留着空洞会让"前 30 条"实际只剩 27 条，
    而**不会报错**——只是扩得比预期少。

    ⚠ 这一步在 **rerank 之前**（§10 的流水线）：留着重复项会让同一份证据
    被 rerank 两次、在预算里占两个名额。
    """
    seen: dict[str, int] = {}
    out: list[Candidate] = []
    for candidate in candidates:
        if candidate.memory_id in seen:
            continue
        seen[candidate.memory_id] = len(out)
        out.append(Candidate(memory_id=candidate.memory_id, rank=len(out)))
    return out


@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """两路 `prefetch` → Weighted RRF 的**编排**。

    它只做三件事：把查询交给 dense 那一路算一次向量、把参数交给 `store/` 执行、
    把结果编上 0-based 名次。**不重排、不扩窗、不打包**——那些在 `rank/`。
    """

    store: QdrantStore
    dense: object  # 见 dense.py 的 DenseArm；用鸭子类型避免循环导入
    params: HybridParams

    def search(self, *, user_id: str, query: str, top_k: int) -> list[Candidate]:
        """按 `top_k` 取候选。

        ⚠ **`top_k` 来自请求，不写死 100**（§7.3）：AML 说它固定是 100，但契约字段就是
        `top_k`——写死会在它传更小值时变成"返回超限"，那是**契约错误**而不是截断。
        """
        if top_k <= 0:
            return []
        dense_vector = self.dense.encode_query(query)  # 每查询【恰好 1 次】
        hits = self.store.hybrid_search(
            user_id=user_id,
            # 词法那一路的输入策略在这里落地（bm25.lexical_query 是恒等映射，
            # 把"查询侧不做改写"变成**可断言的对象**，而不是一句文档约定）。
            query_text=lexical_query(query),
            dense_vector=dense_vector,
            top_k=top_k,
        )
        # ⚠ 这里【丢掉】store 返回的融合分数：它不是校准量，且响应里的 `score`
        #    必须是最终名次的函数（由 `rank/` 现算）。丢掉是**结构性**的，
        #    不是"记得别传"——见 Candidate 的 docstring。
        return [Candidate(memory_id=h.memory_id, rank=rank) for rank, h in enumerate(hits)]
