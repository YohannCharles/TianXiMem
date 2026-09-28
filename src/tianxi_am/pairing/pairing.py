"""记忆块的组合规则（**D24 起**：一次 `Add` = 组合的唯一边界）。

```text
一次 Add 的消息（按原序）
  → ① 连续同 role 合并成一个 RoleBlock
  → ② 一个 UserBlock + 紧随其后的**全部**非 UserBlock 配成一个 MemoryBlock
  → ③ 开头的非 UserBlock（前面没有 user）各自独立成块
  → 每个 MemoryBlock 恰好一次 embedding
```

> **为什么 ② 吃掉的是"全部"而不是"一个"**：§6.2 承认"一条 user 后跟多条 assistant
> 消息（工具调用等）"，那几条必须留在**同一个**块里。只吃一个的话
> `Q0 / assistant / tool / assistant` 会碎成 `[Q0+A0] [T0] [A1]`——把一段回答劈开。
> 在**只有 user / assistant** 的输入上（所有真实数据集），两种写法**逐字相同**。
> ⚠ 反过来，**开头的非 user 段不合并**：`assistant assistant` 作为 Add 的开头就是两个
> 独立的块（"连续同 role 合并"只发生在 ① 里，那个合并按 role 走的正是这里）。

**不同 Add 之间永不组合**：不拼 QA、不合并连续 assistant、不等下一个 chunk、
不 repair、不重新 embedding。即使 A 与 B 属于同一 session，也各自独立成块。

⚠ **这条推翻了 §6.2 / §6.5 的字面与 D20**（那两条的配对作用域是**整个 session**，
一个 QA 对可以跨批次）。理由、实测代价与边界见
[`../../../docs/decisions.md`](../../../docs/decisions.md) **D24**。要点：跨批接缝在
真实数据上确实存在（LoCoMo 全量 3,075 对里有 **63 对**会被批界切成两半），
换来的是"组合是纯函数、无跨 Add 状态、乱序到达不影响结果"。

本模块是**纯函数**：不认识 SQLite、不认识 Qdrant、不认识任何数据集、**没有任何跨 Add 状态**。
它只接受 §2.1 的 canonical Add 契约形状（`role` / `content` / 可选 `timestamp`）。
落库在 [`apply.py`](./apply.py)。
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from tianxi_am.common.config import DEFAULT_CHUNK_ORDINAL_PATTERN

__all__ = [
    "MemoryBlock",
    "Message",
    "RoleBlock",
    "compose_memory_blocks",
    "encode_answer",
    "is_user",
    "join_question",
    "parse_chunk_ordinal",
]


@dataclass(frozen=True, slots=True)
class Message:
    """一条消息，**只保留 canonical Add 契约的三个字段**（§2.1）。

    ⚠ 这里**不允许**出现任何数据集专属字段（`dia_id` / `speaker` / `has_answer` …）。
    加载层负责把各数据集的形状归一化到这三个字段——**核心系统不认识任何数据集**（D16）。
    """

    role: str
    content: str
    timestamp: int | None = None  # Unix 毫秒，可选

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("role 不得为空")
        # 首尾空白在**进入核心之前**去掉：
        # AML 只用 "\n".join(...) 拼接返回项、不插分隔符（§11.3），
        # 所以任何一项首尾留白都会让拼接处粘连（契约自查清单 §4）。
        object.__setattr__(self, "content", self.content.strip())


def is_user(message: Message) -> bool:
    """**唯一判据**：这条消息的 `role` 是不是 `user`（§6.2）。

    ⚠ **不要枚举 role 白名单。** AML 传入的取值域**没有文档**，
    白名单会在遇到没见过的 role 时**静默丢消息**。其他 role
    （`assistant` / `system` / 工具输出…）**一律按"非 user"处理**——
    它们进同一个 block，且**每条都带 `[role]` 标记**（见 `encode_answer`）。
    """
    return message.role == "user"


def parse_chunk_ordinal(
    request_id: str, pattern: str = DEFAULT_CHUNK_ORDINAL_PATTERN
) -> int:
    """从 `request_id` 里取出**这一批的序号**（D25）——位置模型的一半。

    平台实发形如 `eval:<run_id>:locomo_refined:conv-0:chunk-3`；本仓 harness 用
    `<user_id>|<session_id>|3`。默认正则两个都认（`re.search`，取第 1 个捕获组）。

    ## 取不到就**响亮失败**，没有回退

    ⚠ **这里绝不回退到"`MAX+1` 现算一个位置"**。回退会把 D25 要修的那个 bug
    偷偷带回来：位置一旦由到达顺序决定，`chunk-1` 先于 `chunk-0` 提交就会拿到更小的位置，
    而 §10 扩窗、§11.2 段合并把"位置相邻"当成"会话相邻"⇒ **把对话顺序读反，且不报错**。

    ⇒ 宁可让这一批**非 200**（AML 会重试；格式真变了就该立刻被人看见），
    也不要安静地跑完一整场、留下一份顺序错乱的记忆。**这正是 Smoke 该验的那类未知。**

    ⚠ 模式**可配置**（`ingest.chunk_ordinal_pattern`）⇒ 平台换了形状**不必改代码**。
    这一对"可配置 + 响亮失败"就是 §12.1 R1 对冲 3 要的形态。
    """
    match = re.search(pattern, request_id)
    if match is None:
        raise ValueError(
            f"request_id 里取不出 chunk 序号：{request_id!r}\n"
            f"  当前模式：{pattern!r}（配置项 `ingest.chunk_ordinal_pattern`）\n"
            "  ⚠ 位置由它派生（D25），**取不到就不能落库**——回退到按到达顺序分配位置，"
            "会在乱序/并发到达时**静默把对话顺序读反**。"
        )
    return int(match.group(1))


def encode_answer(messages: Sequence[Message]) -> str:
    """把一个块里的非 user 消息编码成 `answer` 列的文本。

    §11.3：一个块里有多条非 user 消息时，**每条带 role 标记**——否则这几段在模型
    眼里连成一片，分不清哪句是助手说的、哪句是工具输出。渲染模板是 `A: {answer}`，
    所以**标记必须已经存在 `answer` 里**：单个 TEXT 无法在渲染时重建消息边界。

    ⚠ **标记无条件加，即使只有一条非 user 消息**——形状统一，读的人不必分情况。

    ⚠ **空 content 不丢弃**（§2.1 保证非空，这里是防御）：渲染成 `[role]`。
    丢掉它是"静默丢消息"，正是本项目要避免的失败模式。
    """
    return "\n".join(f"[{m.role}] {m.content}" if m.content else f"[{m.role}]" for m in messages)


def join_question(messages: Sequence[Message]) -> str:
    """把一段**连续 user 消息**拼成一个 `question`。

    **与 `encode_answer` 相反：这里不加 role 标记。** 两条理由：这段文本是同一个人的
    发言、role 均一（role 由渲染的 `Q: ` 前缀给出）；而拼接的首要用途是**把被物理切开
    的一条原消息拼回去**（AML 可能按句边界拆超长 message），只有不加标记才与原消息
    **逐字接近**。标记还会**一并进 embedding**（§7.2 要求同一份渲染）⇒ 直接改变检索输入。

    §11.3 要求"每条带 role 标记"的只有 `answer` 那一侧（一个块里可能混着 `assistant` /
    `system` / 工具输出，不标就分不清哪句是谁说的）。
    """
    return "\n".join(m.content for m in messages)


@dataclass(frozen=True, slots=True)
class RoleBlock:
    """一段**连续同 role** 的消息（组合的第 ① 步产物）。

    ⚠ **连续同 role 只在本次 Add 内合并**——跨 Add 的连续 assistant 永远是两块。
    """

    role: str
    messages: tuple[Message, ...]
    source_idxs: tuple[int, ...]
    """这些消息在**本次 Add 的消息列表**里的下标（保留原始来源，便于调试）。"""

    @property
    def is_user(self) -> bool:
        return self.role == "user"


@dataclass(frozen=True, slots=True)
class MemoryBlock:
    """**检索单元**：一个 user 段 +（若有）紧随其后的非 user 段。

    落库后它就是 `qa_pairs` 的一行；**每个 MemoryBlock 恰好一次 embedding**。
    它内部**不再逐条 message 单独 embedding**。

    `messages` / `source_idxs` 把**原始来源**原样带出来：合并只发生在文本层，
    没有任何一条消息因为合并而丢失或改变（调试、邻域扩展、未来做来源定位都要靠它）。
    """

    question: str | None
    """连续 user 消息的拼接；`None` = 块里没有 user 消息（如以 assistant 开头的 Add）。"""

    answer: str | None
    """非 user 消息的拼接，每条带 `[role]` 标记；`None` = 块里没有非 user 消息。"""

    event_time: int | None
    """**该块首条消息**的 timestamp（Unix 毫秒，可空）——§6.1 的 `event_time` 口径。"""

    messages: tuple[Message, ...]
    """这个块覆盖的全部原始消息，**按原序**。"""

    source_idxs: tuple[int, ...]
    """它们在**本次 Add 的消息列表**里的下标。"""

    @property
    def roles(self) -> tuple[str, ...]:
        """来源消息的 role，与 `messages` 一一对应。"""
        return tuple(m.role for m in self.messages)

    @property
    def timestamps(self) -> tuple[int | None, ...]:
        """来源消息的 timestamp，与 `messages` 一一对应。"""
        return tuple(m.timestamp for m in self.messages)

    @property
    def is_paired(self) -> bool:
        """user 段与非 user 段**都**在（= 一个完整的 QA 块）。

        `False` 的块是**合法的**、必须照样存——它就是"首尾配不上"的那一类，
        **不要**去等下一个 Add 把它补齐（那正是 D24 取消的东西）。
        """
        return self.question is not None and self.answer is not None


def compose_memory_blocks(messages: Sequence[Message]) -> tuple[MemoryBlock, ...]:
    """**一次 `Add` 的消息 → 它的记忆块**。这是组合规则的唯一实现。

    决定性的、无副作用的纯函数：不碰数据库、不算 embedding、**没有任何跨 Add 状态**
    ⇒ 同一批消息无论第几次调用、无论哪次 Add 到达，产出的块**逐字相同**。

    | 输入 | 输出 |
    | --- | --- |
    | `U A U A` | `[U+A] [U+A]` |
    | `U U A A A U A` | `[UU+AAA] [U+A]` |
    | `A U A` | `[A] [U+A]`（首个 A 配不上，独立成块） |
    | `U A U` | `[U+A] [U]`（末尾的 U 配不上，独立成块） |
    | `A` / `U` | `[A]` / `[U]` |
    | `U assistant tool assistant` | `[U+A0+T+A1]`（工具调用序列**不劈开**） |

    ⚠ **不判断"是不是超长消息截断"**：配不上的块**就是**独立块，原因是什么无关紧要。
    任何"等下一个 Add 来补齐"的设计都会重新引入跨 Add 状态（D24 取消的正是它）。

    ⚠ **空输入返回空元组**——调用方（`apply_batch`）在此之前就该抛错，见那里的注释。
    """
    role_blocks = _merge_consecutive_roles(messages)
    blocks: list[MemoryBlock] = []
    i = 0
    while i < len(role_blocks):
        if not role_blocks[i].is_user:
            # ③ 开头的非 user 段：前面没有 user 可配 ⇒ 独立成块
            blocks.append(_block_from(role_blocks[i]))
            i += 1
            continue
        # ② 一个 user 段 + 紧随其后的全部非 user 段（工具调用序列等）
        end = i + 1
        while end < len(role_blocks) and not role_blocks[end].is_user:
            end += 1
        blocks.append(_block_from(*role_blocks[i:end]))
        i = end
    return tuple(blocks)


def _merge_consecutive_roles(messages: Sequence[Message]) -> tuple[RoleBlock, ...]:
    """第 ① 步：把**连续同 role** 的消息并成一个 `RoleBlock`。"""
    blocks: list[RoleBlock] = []
    for idx, message in enumerate(messages):
        if blocks and blocks[-1].role == message.role:
            last = blocks.pop()
            blocks.append(
                RoleBlock(
                    role=last.role,
                    messages=(*last.messages, message),
                    source_idxs=(*last.source_idxs, idx),
                )
            )
        else:
            blocks.append(RoleBlock(role=message.role, messages=(message,), source_idxs=(idx,)))
    return tuple(blocks)


def _block_from(*role_blocks: RoleBlock) -> MemoryBlock:
    """第 ② / ③ 步：一个 user 段 + 紧随其后的非 user 段（0 个或多个）→ 一个 `MemoryBlock`。

    两个调用形状：`(user, 非user…)`（②）与 `(单独一个任意 role 段,)`（③）。
    """
    messages = tuple(m for rb in role_blocks for m in rb.messages)
    source_idxs = tuple(i for rb in role_blocks for i in rb.source_idxs)
    user_messages = [m for rb in role_blocks if rb.is_user for m in rb.messages]
    other_messages = [m for rb in role_blocks if not rb.is_user for m in rb.messages]
    return MemoryBlock(
        question=join_question(user_messages) or None,
        answer=encode_answer(other_messages) or None,
        # 该块的 **首条**消息的 timestamp（§6.1）——传进来的顺序就是原序
        event_time=messages[0].timestamp if messages else None,
        messages=messages,
        source_idxs=source_idxs,
    )
