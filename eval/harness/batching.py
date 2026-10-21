"""模拟 AML 的**打包**——
**两条预算：20 条消息 + 2,000 词**（官方那条规则的两半）。

> **两条预算都要做**：只做“20 条消息”那一路不够——CL-Bench 那种 158,789 字符的正文切完
> 仍可能形成超限的长块；词预算会把它们像平台那样**摊进多条 Add**。
> ⚠ **词数用空白分词近似**（平台 Adapter 不在本仓），所以本模块**不是线上行为的
> 精确等价物**。批次边界必须进入数据指纹，不能把本地切法当成平台事实。

**规则原文**（二手外部来源）："平台每个来源会话默认调用一次 Add；
超过 **20 条消息或 2000 词**时在**最近的完整消息或句子边界**分段。"

⚠ **消息可能被从中间切开。** 官方全量数据中，单条消息硬上限约为 8,000 字符；
超长消息会先被切成连续的多条同 role 消息，每片重新带 `<标签>: ` 前缀，
拼回去仍是全文，片再落进不同的 Add。

⇒ **单条切分在
[`add_shape.py`](./add_shape.py) 的 `MAX_MESSAGE_CHARS`（`_split_labelled`）里做，
且 driver **先 shape、再切批**——**本模块拿到的已是切好的片**；加上本模块的词预算，
clbench / beam 端到端跑通。**别把本地批次读成"线上就是这样"**——残留的差异只剩"词怎么数"。

**"词"怎么数仍然没有定义**
⇒ **词数那一路按空白分词近似实现**（[`MAX_BATCH_WORDS`][...]）：近似只能让批次形状**接近**
线上、不等于线上——"词怎么数"在平台侧仍未定义，本模块**不是线上的等价物**。

**这条路径在不同数据上的覆盖范围**：

| | 最长单条 | 首批被 20 条切开 | 首批被 2000 词切开 |
| --- | --- | --- | --- |
| LoCoMo-Refined | 87 词 | **100%** | **0** |
| LongMemEval | 11,661 词 | 60% | **40%** |

⇒ **LoCoMo 上两条路径完全重合**（消息太短，永远先撞 20 条）；**LongMemEval 上不重合**。
**别用 LoCoMo 的"完全重合"外推 LongMemEval。**

**后果必须记住**：本地复现的批次形状与线上**不保证一致**。这不改变指标定义，
但会影响**组合结果本身**：
批界落在哪里，决定了有多少 QA 对被切成两个半块。⇒ 拿本地分数比线上时，**这条要算进不确定性**。

⚠ **输入是"渲染后的 payload"，不是 `Message`**：单条消息的超长切分
住在 [`add_shape.shape_batch`](./add_shape.py)（上限是**连标签一起**算的，所以必须先渲染），
而切完条数会变 ⇒ **先 shape、再切批**。[`driver.ingest`](./driver.py) 就是这个顺序。

> **切批口径是常量、不是旋钮**——它记录在数据指纹里（`registry.BATCHING_LOCAL`）。
> 做成可调会立刻产生一个诱人的错误：拿不同的切法比分数。

批次边界也是"我们造的"：线上由 AML 决定、本地由本模块决定，所以**"切批会不会打断一个
QA 对"只能在 Smoke 上用真实的 20 条复现**。
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["MAX_BATCH_WORDS", "MAX_MESSAGES_PER_BATCH", "batches", "request_id_for"]

#: 本地可复现的切批口径。**常量**，理由见模块 docstring。
MAX_MESSAGES_PER_BATCH = 20

#: **每条 Add 的正文词数上限**——**官方那条规则就是它**：
#: 「达到 20 条消息**或 2,000 个词**中的任一边界时确定性分段」。
#:
#: ⚠ **计数是近似**：官方说"词数按**冻结 Adapter** 计数"，而那个 Adapter 是平台侧组件、
#: 本仓没有 ⇒ 这里用**空白分词**（`str.split()`）代替。**它是个代理，不是等价物**，
#: 所以进数据指纹（`registry.BATCHING_LOCAL` 那一格旁边）。
#: 旁证：官方真实语料 41,270 条 add 的每-add 空白词数 **p99 = 2,056**——与 2,000 对得上。
#:
#: ⚠ **它不是可选的**：没有它，一条 158,789 字符的正文（CL-Bench 里真有）切完仍是
#: 21 条**同 role** 消息 ⇒ 按  合并成**一个 160,273 字符的块** ⇒ 嵌入直接 400。
#: 加上它，那些片才会像官方那样**摊进多条 add**。
MAX_BATCH_WORDS = 2_000


def _content_words(item: object) -> int:
    """一条 payload 的正文词数（**空白分词**，见 [`MAX_BATCH_WORDS`][...] 那条注记）。

    只认 `content`——官方那条合计口径算的就是它。⚠ **必须是渲染后的 payload**：
    正文在渲染时会加 `<标签>: ` 前缀、超长的还会被切片，拿 `Message` 来算会少算。
    """
    if not isinstance(item, dict):
        raise TypeError(f"切批的输入必须是渲染后的 payload 字典，收到 {type(item).__name__}")
    return len((item.get("content") or "").split())


def batches[T](
    messages: Sequence[T],
    *,
    max_messages: int = MAX_MESSAGES_PER_BATCH,
    max_words: int = MAX_BATCH_WORDS,
) -> list[tuple[T, ...]]:
    """按**源序**切批——`request_id` 里带着批序号，**绝不重排**。

    **对元素类型不做任何假设**（只切片）——传进来的通常是
    [`add_shape.shape_batch`](./add_shape.py) 渲染好的 payload 字典。

    ⚠ **批边界是"声明"的，不是靠到达顺序推断的**：`request_id_for(user, session, index)`
    把它写进 id，而服务端位置 = `(request_id, local_index)`——`local_index` 是块
    **在那一次 Add 里**的序号。⇒ 重排调用顺序**不会**翻转会话顺序，
    但**重排会让同一 `request_id` 对应另一批消息**——那才是要防的。

    **切批只在 session 内发生**（跨 session 的边界由 `session_id` 隔开，
    而每个 session 单独投喂）。
    """
    if max_messages < 1:
        raise ValueError(f"max_messages 必须 >= 1，收到 {max_messages}")
    if max_words < 1:
        raise ValueError(f"max_words 必须 >= 1，收到 {max_words}")

    out: list[tuple[T, ...]] = []
    current: list[T] = []
    size = 0
    for item in messages:
        length = _content_words(item)
        # **单条就超预算时照样自成一批**：它是已经切过片的（≤`add_shape.MAX_MESSAGE_CHARS`），
        # 再切是另一层的事——在这里"等下一个"只会把两批都撑坏。
        if current and (len(current) >= max_messages or size + length > max_words):
            out.append(tuple(current))
            current, size = [], 0
        current.append(item)
        size += length
    if current:
        out.append(tuple(current))
    return out


def request_id_for(user_id: str, session_id: str, index: int) -> str:
    """**确定性** id——同一 session 的第 `index` 批在每次 run 里拿到同一个 `request_id`。

    这不是好看而已：Add 契约要求**重试沿用同一个 `request_id` 与 payload**，
    所以幂等守卫（`applied_batches`）能不能命中，取决于这里是否稳定。
    随机 id 会让"重跑一遍"变成"把同一份数据写第二遍"，而**库里不会报错**。
    """
    return f"{user_id}|{session_id}|{index}"
