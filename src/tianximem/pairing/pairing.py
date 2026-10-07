"""记忆块的组合规则（**D24 起**：一次 `Add` = 组合的唯一边界）。

```text
一次 Add 的消息（按原序）
  → ① 一段连续 user + **紧随其后的一段连续非 user** ⇒ 配成一个 MemoryBlock
       ⚠ 但只有该 user 段的**最后一条**进配对块（**D32**）——前面的每条各自独立
  → ② **其余消息：每条各自独立成块**——同 role 相邻也不并
  → 每个 MemoryBlock 恰好一次 embedding
```

> **只有"配得上对"才合并**：配不上对的，**一条都不并**。
>
> **为什么不能按"连续同 role 相并"来**（2026-10-01 实测）：官方**判分池**的 Add 实测是
> **100% `role: user`、零 assistant** ⇒ 那样会把一次 Add（≤20 条）**整个塌成 1 块**：
> locomo 412 块 → 55 块、medmemorybench
> 1,562 → 200，而且**配对的块一个不剩** ⇒ `link_blocks` 只连配对块 ⇒ **邻接链全空**
> ⇒ 扩窗（§10）与段合并（§11.2）在那个池子上**一行都不执行**。
> 新规则下同一条数据是 4 条各自独立的消息，**粒度与链路都回来了**。
>
> **为什么 ① 只吃"最后一条"**（2026-10-03，**D32**）：平台的切分按 **8,000 字符**、
> 批预算按 **2,000 词**，而嵌入窗口按 **token**——对密集内容（数值 / 表格）三者
> **不成比例**。CL-Bench 某样本 14 条 × 8,000 字符**只算 1,378 词**（远低于 2,000，
> 被放行）⇒ 合成一个 **38,270 token** 的块（`question` 83,641 字符）
> ⇒ **线上静默截断、本地网关 400**。只配最后一条之后，同一份全量实测
> **最大块 5,541 token**，块数只涨 **+2.2%**。
>
> ⚠ **代价是 D20 那个取舍被反过来了**：`q q q a` 现在产出 `[q] [q] [q+a]`，
> 前面那些 `q` **有问无答**。D20 当初合并它们，正是为了让被切碎的文档"问题完整"；
> 而合起来的代价是**块大到嵌不进去**——两条路都错，只是错的地方不同 ⇒ 选"块小"这一侧。
> 实测依据见 [`../../../docs/open-questions.md`](../../../docs/open-questions.md) 的 **S5**。

> **为什么 ② 吃掉的是"全部"而不是"一个"**：§6.2 承认"一条 user 后跟多条 assistant
> 消息（工具调用等）"，那几条必须留在**同一个**块里。只吃一个的话
> `Q0 / assistant / tool / assistant` 会碎成 `[Q0+A0] [T0] [A1]`——把一段回答劈开。
> 在**只有 user / assistant** 的输入上（所有真实数据集），两种写法**逐字相同**。
> ⚠ 反过来，**配不上的段一律不合并**：`assistant assistant` 开头就是**两个**独立的块，
> 一段没有回答的 `user user user` 也是**三个**（合并只在配对时发生，见上）。

**不同 Add 之间永不组合**：不拼 QA、不合并连续 assistant、不等下一个 chunk、
不 repair、不重新 embedding。即使 A 与 B 属于同一 session，也各自独立成块。
**D28 起邻接也只在这一个作用域内**（`link_blocks`）：跨 Add 既不组合、也不建立 prev/next。

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

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "MemoryBlock",
    "Message",
    "RoleBlock",
    "compose_memory_blocks",
    "encode_answer",
    "is_user",
    "join_question",
    "link_blocks",
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
    | `U A` | `[U+A]` |
    | `U U A A A U A` | `[U] [U+AAA] [U+A]` |
    | `A U A` | `[A] [U+A]`（首个 A 配不上，独立成块） |
    | `U A U` | `[U+A] [U]`（末尾的 U 配不上，独立成块） |
    | `A` / `U` | `[A]` / `[U]` |
    | `U assistant tool assistant` | `[U+A0+T+A1]`（工具调用序列**不劈开**） |
    | `U U U`（全 user，**没有任何非 user**） | `[U] [U] [U]`（**各自独立**） |
    | `A A`（全 assistant） | `[A] [A]`（**各自独立**） |

    ⚠ **合并只在"配得上对"时发生**：一段连续 user 后面**没有**非 user 时，那段**不合并**，
    每条各自成块。理由见模块 docstring（判分池全是 `role: user`）。

    ⚠ **配对时也只配这一段 user 的【最后一条】**（**D32**，2026-10-03）——前面的各自独立。
    这是被**嵌入窗口**逼出来的：一段长 user 全进一个 question 会到 **38,270 token**
    （窗口 8,192）⇒ 线上静默截断、本地 400。**代价**：`q q q a` 里前面的 `q` 变得
    **有问无答**（D20 正是为此才合并的）——两条路都错，D32 选了"块不超窗口"那一侧。

    ⚠ **不判断"是不是超长消息截断"**：配不上的消息**就是**独立块，原因是什么无关紧要。
    任何"等下一个 Add 来补齐"的设计都会重新引入跨 Add 状态（D24 取消的正是它）。

    ⚠ **空输入返回空元组**——调用方（`apply_batch`）在此之前就该抛错，见那里的注释。
    """
    role_blocks = _merge_consecutive_roles(messages)
    blocks: list[MemoryBlock] = []
    i = 0
    while i < len(role_blocks):
        if not role_blocks[i].is_user:
            # 非 user 段：配不上对（前面没有 user 段，或那段已经被配走了）⇒ **各自独立**
            blocks.extend(_standalone(role_blocks[i]))
            i += 1
            continue
        end = i + 1
        while end < len(role_blocks) and not role_blocks[end].is_user:
            end += 1
        if end > i + 1:
            # 后面跟着非 user ⇒ 配对。**但只有这一段 user 的【最后一条】配上**，
            # 前面的各自独立（**D32**，2026-10-03）。
            #
            # 为什么（实测，见 `eval/reports/ledger.md` 的 V18 一节）：平台的 8,000 **字符**
            # 切分会把**一份长文档**切成十几条连续的 user 消息。而整段 user 全进 question
            # 会涨到 **83,641 字符 / 38,270 token**，远超 `text-embedding-v4` 的 **8,192 token**
            # 窗口 ⇒ 线上**静默截断**（尾巴搜不到）、本地开发网关 **400**（跑批直接死）。
            # CL-Bench 全量：**14/12,793 个块超窗口**，最大 **45,160 token**。
            #
            # **只配最后一条**（同一份全量实测）：**最大块 5,541 token**，
            # 块数只涨 **+2.2%**（12,793 → 13,080）。
            #
            # ⚠ **代价是 `q q q a` 产出 `[q] [q] [q+a]`**，前面那些 `q` **有问无答**——
            # 而全进同一个 question 则**块大到嵌不进去**。两条路都错，只是错的地方不同
            # ⇒ 选"块小"这一侧，理由在 D32。
            head = role_blocks[i]
            blocks.extend(
                _standalone(
                    RoleBlock(
                        role=head.role,
                        messages=head.messages[:-1],
                        source_idxs=head.source_idxs[:-1],
                    )
                )
            )
            last = RoleBlock(
                role=head.role,
                messages=(head.messages[-1],),
                source_idxs=(head.source_idxs[-1],),
            )
            blocks.append(_block_from(last, *role_blocks[i + 1 : end]))
        else:
            # 后面没有非 user ⇒ 这段 user 各自独立，**不合并**
            blocks.extend(_standalone(role_blocks[i]))
        i = end
    return tuple(blocks)


def link_blocks(
    blocks: Sequence[MemoryBlock],
) -> tuple[tuple[int | None, int | None], ...]:
    """**Add 内的邻接链**：每个块的 `(prev_local_index, next_local_index)`（D28）。

    ⚠ **只连完整 QA**（`MemoryBlock.is_paired`）。A-only / Q-only 照样存、照样 embedding、
    照样能被检索到，但**不进链**——两侧都是 `None`。它们没有"上下文邻居"可言，
    硬连上去只会让段合并把一段不完整的对话当成连续的。

    ⚠ **这条链只在一次 Add 内成立**：跨 Add 不存在可信的全局序（D28），
    所以这里**只看传进来的这一批块**，不看库、不看别的 Add（与组合规则同一个作用域）。

    ⚠ 不完整块**只可能出现在两端**（组合规则决定：开头的非 user 段配不上 user、
    结尾的 user 段配不上非 user），但实现**不依赖**这条性质——它扫一遍、只把完整的串起来，
    所以将来组合规则变了也不会悄悄错位。

    返回值与 `blocks` **等长同序**：第 i 项是第 i 个块的 `(前一块的 local_index,
    后一块的 local_index)`。调用方把它翻成 `memory_id`（`store.make_pair_id`）。
    """
    links: list[tuple[int | None, int | None]] = [(None, None)] * len(blocks)
    complete = [index for index, block in enumerate(blocks) if block.is_paired]
    for position, index in enumerate(complete):
        prev_index = complete[position - 1] if position else None
        next_index = complete[position + 1] if position + 1 < len(complete) else None
        links[index] = (prev_index, next_index)
    return tuple(links)


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


def _standalone(role_block: RoleBlock) -> tuple[MemoryBlock, ...]:
    """**配不上对的一段 → 每条消息各自一个块**（不合并）。

    这是"合并只在配对时发生"的落点：`A A` 是两块、`U U U` 是三块。
    ⚠ **`source_idxs` 逐条带出来**——合并没了，但"这条块来自哪几条原消息"仍要可追。
    """
    return tuple(
        _block_from(RoleBlock(role=role_block.role, messages=(message,), source_idxs=(index,)))
        for message, index in zip(role_block.messages, role_block.source_idxs, strict=True)
    )


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
