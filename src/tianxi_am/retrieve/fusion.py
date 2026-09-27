"""混合检索的**策略与参数所有权**（§7.3）。

**本模块说"用什么参数"（取值 / 校验 / 标定），`store/qdrant_store.py` 说"怎么发给
Qdrant"**（翻成 `prefetch` + `rrf`）。**一份实现不许两边都写**——分工表见
[`CLAUDE.md`](./CLAUDE.md)。

## 参数里有一个不是旋钮

`rrf_k` 是**正确性常量**：Qdrant 的默认 `k` 是 **2**，而 RRF 文献是 1-based 的
`1/(60 + rank)`；Qdrant 的秩是 0-based、公式 `1/(rank + k)`，所以**只有 `k = 61`**
才等价于文献的 60（`1/(0+61) = 1/(1+60)`）。**填错不会报错**——两种写法产出的名次都
"看起来正常"，只有分数差一个数量级。

因此本模块**拒绝**任何 `k != 61` 的配置，而不是"警告后照用"（D5；值见 config-reference §3）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from tianxi_am.common.config import DEFAULT_PREFETCH_LIMIT, DEFAULT_WEIGHTS
from tianxi_am.retrieve.bm25 import lexical_query
from tianxi_am.retrieve.dense import DenseArm
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

#: ⚠ **`DEFAULT_PREFETCH_LIMIT` / `DEFAULT_WEIGHTS` 的初值住在
#: [`../common/config.py`](../common/config.py)**（理由与 `DEFAULT_EXPANSION_SEED_LIMIT`
#: 同一条：`retrieve/` 与 `store/` 都要用，而 `common/` 是最底层）。本模块**只 import**，
#: 不再各写一份字面量。§7.3 的两条注意事项（`prefetch_limit` **不是**种子数/Top-K；
#: 权重调整需 ablation 数据）也写在那里，**不在这里重复**——重复描述就是漂移的开始。

#: **正确性常量，不是可调项**（D5）。
#:
#: ⚠ 与 `common/config.py` 的 `RRF_K` **各有一份**，这是**有意的**：那边是配置层的拒绝依据，
#: 这里是**策略层对直接调用方的防御**——不能只靠"配置层已经查过"。两处相等由
#: `tests/test_config.py::test_rrf_k_is_the_same_constant_in_both_places` 钉住。
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

    **"稳定"= 保留名次最好的那一次**（第一次出现）：只有"最早的那条"能与 RRF 的
    名次语义对上。

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

    它只做两件事：把查询交给 dense 那一路算一次向量、把结果编上 0-based 名次。
    **不重排、不扩窗、不打包**——那些在 `rank/`。

    ⚠ **融合参数（`prefetch_limit` / `weights` / `rrf_k`）不在本类里**：它们由
    [`make_hybrid_params`](#make_hybrid_params) 校验后，**在装配处注入 `QdrantStore`**
    ——因为**执行**它们的是 `store/qdrant_store.hybrid_search()`。

    ⚠ **别把 `params` 字段加回本类**：参数需要一个能被执行处读到的地方，而不是编排者
    手里——留着它却从不传给 store（装配处也不传 `hybrid=`），改 yaml 里的
    `retrieval.prefetch_limit` / `rrf.weights` 就会**静默无效**（两边默认值恰好相等时
    尤其看不出来）。回归用例：
    `tests/test_retrieve.py::test_build_services_forwards_retrieval_params_to_the_store`。
    """

    store: QdrantStore
    #: ⚠ **不要退回 `dense: object`**：[`dense.py`](./dense.py) 不 import 本模块
    #: （`Candidate` 定义在本模块，依赖单向 `fusion → dense`），所以这里标注具体类型。
    #: 标成 `object` 会让 `.encode_query()` 变成"`object` 上没有的属性"——接口断了
    #: 也只在运行时才炸，而 mypy 连报都报不出来。
    dense: DenseArm

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
        # ★ V13（2026-09-26）：**并列分数的次序必须在这里定死。**
        #   RRF 的分数是离散的（`1/(61+名次)`），一趟里几十条并列是**常态**；而
        #   Qdrant 对**同分项**的返回顺序在两次相同请求之间都可能变
        #   （实测：同一请求连打 6 次，前 3 名 6 次全同，**第 7 名起有 3 次不同，
        #   而两侧的 `score` 完全相同**）。名次是**位置派生**的（下面的 `enumerate`），
        #   所以不排序的话，整条链（种子 → 扩窗 → 打包顺序）会跟着抖——
        #   表现为"**同配置重跑差 ±3pt**"，而**没有任何东西会报错**。
        #   ⇒ 主键 `score` 降序、**次级键 `memory_id` 升序**：纯确定性，不带排序意图。
        #     （更讲究的一档是拿 dense 分当次级键——同类实现有这么做——但那要求
        #     拿得到**分路**分数，而融合整个交给 Qdrant 是我们既定的形态，见 D5。）
        #   ⚠ 分数在这里**只用于排序**，排完照旧丢掉（它不是校准量，见下）。
        ordered = sorted(hits, key=lambda h: (-h.score, h.memory_id))
        # ⚠ 这里【丢掉】store 返回的融合分数：它不是校准量，且响应里的 `score`
        #    必须是最终名次的函数（由 `rank/` 现算）。丢掉是**结构性**的，
        #    不是"记得别传"——见 Candidate 的 docstring。
        return [Candidate(memory_id=h.memory_id, rank=rank) for rank, h in enumerate(ordered)]
