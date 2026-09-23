"""§6.2 的配对判据 + §6.5 三步挂接的**纯逻辑**部分。

**这是全项目逻辑最绕、也最容易静默出错的一层**——出错的表现是"某些记忆永远检索不到"，
而**不会有任何报错**（pairing/README.md）。

本模块是**纯函数**：不认识 SQLite、不认识 Qdrant、不认识任何数据集。
它只接受 §2.1 的 canonical Add 契约形状（`role` / `content` / 可选 `timestamp`），
产出"要落哪些对"的计划；真正落库在 [`continuation.py`](./continuation.py)。

依赖方向：`pairing` → `store`（`Status` 类型来自 DDL 的列域），反向没有依赖。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from tianxi_am.store.sqlite_store import STATUS_COMPLETE, STATUS_PENDING, Status

__all__ = [
    "BatchLimits",
    "BatchPlan",
    "Message",
    "NewPairDraft",
    "ResumeActions",
    "encode_answer",
    "is_user",
    "plan_batch",
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
    """§6.2 的**唯一判据**。

    > 配对只做一种判断：这条消息的 `role` 是不是 `user`。

    ⚠ **不要枚举 role 白名单。** AML 传入的取值域**没有文档**，
    白名单会在遇到没见过的 role 时**静默丢消息**。其他 role
    （`assistant` / `system` / 工具输出…）**一律归入当前对**。
    """
    return message.role == "user"


def encode_answer(messages: Sequence[Message]) -> str:
    """把一个对里的非 user 消息编码成 `answer` 列的文本。

    §11.3：一个对里有多条非 user 消息时，**每条带 role 标记**——否则这几段在模型
    眼里连成一片，分不清哪句是助手说的、哪句是工具输出。渲染模板是 `A: {answer}`，
    所以**标记必须已经存在 `answer` 里**：单个 TEXT 无法在渲染时重建消息边界。

    ⚠ **标记无条件加，即使只有一条非 user 消息。** 因为 `answer` 允许跨批追加；
    若写成"多条时才加标记"，补全时就必须回头改写首条（给它补标记），
    那会**违反 §6.5 的"填空 + 追加、绝不覆盖"**。无条件加是唯一与追加语义相容的写法。

    ⚠ **空 content 不丢弃**（§2.1 保证非空，这里是防御）：渲染成 `[role]`。
    丢掉它是"静默丢消息"，正是本项目要避免的失败模式。
    """
    return "\n".join(f"[{m.role}] {m.content}" if m.content else f"[{m.role}]" for m in messages)


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """批次切分上限（§6.5）。

    ⚠ **必须来自配置，不得硬编码**（§15 / D2 对冲 ③）。默认值只是可用的初值。

    ⚠ "2,000 个 **Adapter 计数的词**"官方**从未定义**（S2），所以词数计数函数
    由调用方注入（见 `plan_batch` 的 `word_counter`），**本地复现不了线上那一路**。
    """

    max_messages: int = 20
    max_words: int = 2000


@dataclass(frozen=True, slots=True)
class NewPairDraft:
    """一个要新建的对。`pair_idx` 已由本批规划分配好。"""

    pair_idx: int
    question: str | None
    answer: str | None
    event_time: int | None
    status: Status


@dataclass(frozen=True, slots=True)
class ResumeActions:
    """本批要对**库里那个既有 `pending` 对**做的事（§6.5 的 3a / 3b / 3d）。"""

    pending_id: str
    append_answer: str | None
    """3a：追加到既有 pending 的 `answer`。None 表示本批开头没有非 user 消息。"""

    close: bool
    """3b：本批出现首个 user 消息 ⇒ 把它标 `complete`。

    ⚠ **最容易漏的一步**：漏了它会永久挂在 pending，而且现象会**伪装成 AML 的切分行为**，
    让人跑去改配对规则——**方向完全错了**（pairing/README.md）。
    """

    final_status: Status | None
    """3d 作用于既有 pending 的情形（**纯接续批**，零条 user 消息）。

    此时被追加的那个 pending 对就是最后一带，同样要标状态，否则它会一直挂到 session 结束。
    它在 3b 已关闭时必为 None。
    """


@dataclass(frozen=True, slots=True)
class BatchPlan:
    resume: ResumeActions | None
    new_pairs: tuple[NewPairDraft, ...]
    session_ended: bool
    message_count: int
    word_count: int
    hit_message_limit: bool
    hit_word_limit: bool


@dataclass(slots=True)
class _Draft:
    """配对过程中的一个"正在接收消息的对"。"""

    pair_idx: int
    question: str | None
    event_time: int | None
    parts: list[Message] = field(default_factory=list)
    from_db: bool = False  # True = 它就是库里那个既有 pending


def plan_batch(
    messages: Sequence[Message],
    *,
    next_idx: int,
    pending_id: str | None,
    limits: BatchLimits | None = None,
    word_counter: Callable[[Message], int] | None = None,
) -> BatchPlan:
    """把一批消息规划成"对既有 pending 的动作 + 要新建的对"（§6.5 第 3 步）。

    三步的落点：

    * **3a** 本批开头、首个 user 消息之前的消息 → 追加到 pending 的 `answer`；
      不存在 pending 时**建一个 `question` 为空的对**（§6.2 承认这种"无问的对"）。
    * **3b** 本批出现首个 user 消息 → 把前一个 pending 标 `complete`。
    * **3c** 其余消息按 §6.2 配对，`pair_idx` 从 `next_idx` 起连续赋值。
    * **3d** 收尾给"涉及的最后一对"定状态：本批两限都未命中 ⇒ session 已结束 ⇒ `complete`。
      **纯接续批（零条 user 消息）也走这一步。**

    配对作用域是**整个 session，不是本批**——所以这里要接 `next_idx` 与 `pending_id`。
    """
    limits = BatchLimits() if limits is None else limits
    counter = _default_word_counter if word_counter is None else word_counter

    word_count = sum(counter(m) for m in messages)
    hit_message_limit = len(messages) >= limits.max_messages
    hit_word_limit = word_count >= limits.max_words
    # §6.5：只要本批命中任一上限，本批的最后一对就是 pending；两限都未命中 ⇒
    # AML 手上已经没有这个 session 的消息了，边界即 session 末端 ⇒ complete。
    session_ended = not (hit_message_limit or hit_word_limit)
    last_status: Status = STATUS_COMPLETE if session_ended else STATUS_PENDING

    idx = next_idx
    drafts: list[_Draft] = []
    # open = 当前正在接收消息的对。既有的 pending 进来时就是"开着的"。
    open_draft: _Draft | None = (
        _Draft(pair_idx=-1, question=None, event_time=None, from_db=True)
        if pending_id is not None
        else None
    )
    appended: list[Message] = []  # 对既有 pending 的 3a 追加
    closing_pending = False  # 3b 是否触发

    for message in messages:
        if is_user(message):
            if open_draft is not None:
                # §6.2：一条 user 消息关闭前一个对。
                if open_draft.from_db:
                    closing_pending = True  # 3b
                open_draft = None
            # 这条 user 消息开一个**新**对
            open_draft = _Draft(
                pair_idx=idx, question=message.content, event_time=message.timestamp
            )
            drafts.append(open_draft)
            idx += 1
        else:
            if open_draft is None:
                # 3a 的后半句：不存在 pending 且本批以非 user 消息开头 ⇒ 建无问的对
                open_draft = _Draft(pair_idx=idx, question=None, event_time=message.timestamp)
                drafts.append(open_draft)
                idx += 1
            if open_draft.from_db:
                appended.append(message)  # 3a
            else:
                open_draft.parts.append(message)

    resume: ResumeActions | None = None
    if pending_id is not None:
        # 既有 pending 被 3b 关闭时，最后一带是本批新建的对，状态已经归它了；
        # 没被关闭（纯接续批）时，最后一带就是它自己，由 3d 定状态。
        resume = ResumeActions(
            pending_id=pending_id,
            append_answer=encode_answer(appended) or None,
            close=closing_pending,
            final_status=None if closing_pending else last_status,
        )

    new_pairs = tuple(
        NewPairDraft(
            pair_idx=d.pair_idx,
            question=d.question,
            answer=encode_answer(d.parts) or None,
            event_time=d.event_time,
            # 只有最后一带可能是 pending；被后续 user 消息关掉的中间对一律 complete
            status=last_status if i == len(drafts) - 1 else STATUS_COMPLETE,
        )
        for i, d in enumerate(drafts)
    )

    return BatchPlan(
        resume=resume,
        new_pairs=new_pairs,
        session_ended=session_ended,
        message_count=len(messages),
        word_count=word_count,
        hit_message_limit=hit_message_limit,
        hit_word_limit=hit_word_limit,
    )


def _default_word_counter(message: Message) -> int:
    """默认词数计数：按空白切分。

    ⚠ **这只是占位**。"2,000 个 **Adapter 计数的词**"官方从未定义（S2 / §12.3 第 5 条），
    所以线上真值与本地必然对不上——`pending` 三个埋点因此只能**相对解读**。
    真正上线时由 `configs/` 注入计数函数；默认值只用于本地自测与单元测试。
    """
    return len(message.content.split())
