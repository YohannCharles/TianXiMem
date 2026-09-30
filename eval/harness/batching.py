"""模拟 AML 的切批（§6.5 / §12.3 第 5 条）——**本地只能复现"20 条消息"那一路**。

**规则原文**（`docs/contract.md` §7.1，二手外部来源）："平台每个来源会话默认调用一次 Add；
超过 **20 条消息或 2000 词**时在**最近的完整消息或句子边界**分段。"
⇒ 消息一般不会被从中间切开，`pairing/` 收到的是整条消息——这条**支持**本模块的"消息为原子"。

**但"词"怎么数仍然没有定义**（§17.1 **S2** 的另一半正卡在这里：官方从未定义 "Adapter 计的词"）
⇒ 词数那一路本地无从复现，**所以只实现 20 条那一路**：写一个自造的词数近似只会让人以为
复现了两路——而"词"的计数口径正是 §17.1 **S2** 里仍然没清的那一半，本模块**不是线上的等价物**。

**这一路在哪份数据上够用，已经量过（全量）**：

| | 最长单条 | 首批被 20 条切开 | 首批被 2000 词切开 |
| --- | --- | --- | --- |
| LoCoMo-Refined | 87 词 | **100%** | **0** |
| LongMemEval | 11,661 词 | 60% | **40%** |

⇒ **LoCoMo 上两条路径完全重合**（消息太短，永远先撞 20 条）；**LongMemEval 上不重合**。
**别用 LoCoMo 的"完全重合"外推 LongMemEval。**

**后果必须记住**：本地复现的批次形状与线上**不保证一致**。D24（2026-09-27）取消了跨 Add
合并之后，这条不影响任何指标，但它仍然影响**组合结果本身**：
批界落在哪里，决定了有多少 QA 对被切成两个半块。⇒ 拿本地分数比线上时，**这条要算进不确定性**。

> **切批口径是常量、不是旋钮**——它记录在数据指纹里（`registry.BATCHING_LOCAL`）。
> 做成可调会立刻产生一个诱人的错误：拿不同的切法比分数。

批次边界也是"我们造的"：线上由 AML 决定、本地由本模块决定，所以**"切批会不会打断一个
QA 对"只能在 Smoke 上用真实的 20 条复现**（§17.1 S2）。
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
    """按**源序**切批——`request_id` 里带着批序号，**绝不重排**。

    ⚠ **批边界是"声明"的，不是靠到达顺序推断的**：`request_id_for(user, session, index)`
    把它写进 id，而服务端位置 = `(request_id, local_index)`（D28）——`local_index` 是块
    **在那一次 Add 里**的序号。⇒ 重排调用顺序**不会**翻转会话顺序，
    但**重排会让同一 `request_id` 对应另一批消息**——那才是要防的。

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
