"""模拟 AML 的切批（§6.5 / §12.3 第 5 条）——**本地只能复现"20 条消息"那一路**。

## 为什么只有一路

AML 按"**20 条消息** 或 **2,000 个 Adapter 计数的词**"切分，而
**"Adapter" 官方从未定义**（§6.5）。⇒ 词数那一路本地无从复现，
**所以这里也只实现 20 条那一路**——写一个自造的词数近似只会让人以为复现了两路。

**后果必须记住**：本地测出的 `pending` 埋点数与**线上必然对不上**。
那三个计数器只能用来验证**续接逻辑自身是否自洽**，**不能用来判断配对质量**
（[`../../src/tianxi_am/observability/CLAUDE.md`](../../src/tianxi_am/observability/CLAUDE.md)）。
这也是 §17.1 的 **S2**，只能靠 Smoke 清掉。

> **切批口径是常量、不是旋钮**——它记录在数据指纹里（`registry.BATCHING_LOCAL`）。
> 把它做成可调的会立刻产生一个诱人的错误：拿不同的切法比分数。

## 批次边界也是"我们造的"

线上批次边界由 AML 决定，本地由本模块决定。所以
**"续接逻辑自洽"能在本地验证，"切批会不会打断一个 QA 对"不能**——
后者要在 Smoke 上用真实的 20 条复现（§17.1 S2）。
"""

from __future__ import annotations

from collections.abc import Sequence

from eval.datasets import Message

__all__ = ["MAX_MESSAGES_PER_BATCH", "batches", "request_id_for"]

#: §6.5 的本地可复现口径。**常量**，理由见模块 docstring。
MAX_MESSAGES_PER_BATCH = 20


def batches(
    messages: Sequence[Message], *, max_messages: int = MAX_MESSAGES_PER_BATCH
) -> list[tuple[Message, ...]]:
    """按**源序**切批——消息顺序就是 `pair_idx` 的最终依据，**绝不重排**。

    **切批只在 session 内发生**（跨 session 的边界由 `session_id` 隔开，
    而每个 session 单独投喂）——沿用 §6.5 的形状。
    """
    if max_messages < 1:
        raise ValueError(f"max_messages 必须 >= 1，收到 {max_messages}")
    return [
        tuple(messages[start : start + max_messages])
        for start in range(0, len(messages), max_messages)
    ]


def request_id_for(user_id: str, session_id: str, index: int) -> str:
    """**确定性** id——同一 session 的第 `index` 批在每次 run 里拿到同一个 `request_id`。

    这不是好看而已：§2.2 规定 Add 的**重试沿用同一个 `request_id` 与 payload**，
    所以幂等守卫（`applied_batches`）能不能命中，取决于这里是否稳定。
    随机 id 会让"重跑一遍"变成"把同一份数据写第二遍"，而**库里不会报错**。
    """
    return f"{user_id}|{session_id}|{index}"
