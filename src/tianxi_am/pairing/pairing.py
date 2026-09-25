"""§6.2 的配对判据 + §6.5 三步挂接的**纯逻辑**部分。

**这是全项目逻辑最绕、也最容易静默出错的一层**——出错的表现是"某些记忆永远检索不到"，
而**不会有任何报错**（pairing/CLAUDE.md）。

⚠ **连续 user 消息并入同一个 `question`**，而不是把前一个对关成"有问无答"——理由、代价
与影响面见 `plan_batch` 的 docstring 与
[`../../../docs/decisions.md`](../../../docs/decisions.md) D20。

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
    "join_question",
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


def join_question(messages: Sequence[Message]) -> str:
    """把一段**连续 user 消息**拼成一个 `question`。

    **与 `encode_answer` 相反：这里不加 role 标记。** 三条理由：

    * 这段文本是**用户自己的发言**，role 是均一的；`[user]` 标记是在正文里注入源文本没有的
      token，而标记会**一并进 embedding**（§7.2 要求同一份渲染）⇒ 直接改变检索输入。
    * 拼接的首要用途是**把被物理切开的一条原消息拼回去**（AML 可能按句边界拆超长 message）。
      只有不加标记，拼出来的文本才与原消息**逐字接近**。
    * 渲染模板 `Q: {question}` 本身已经标明这是用户侧内容，不需要逐段重复。

    §11.3 要求"每条带 role 标记"的是 `answer` 那一侧（一个对里可能混着 `assistant` /
    `system` / 工具输出，不标就分不清哪句是谁说的）——`question` 侧没有这个问题。
    """
    return "\n".join(m.content for m in messages)


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
    """本批要对**库里那个可续写的对**做的事（§6.5 的 3a / 3b / 3d）。

    ⚠ `open_pair_id` **不一定是 `status = 'pending'` 的那一对**：只要它的 `answer` 为空，
    它就还没写完（判据见 `_Draft.question_is_open`）。名称刻意不叫 `pending_id`——
    那会让人以为续写的对象总是 `pending` 状态，而"不一定是"正是本轮修正的地方。
    """

    open_pair_id: str
    append_question: str | None
    """把本批开头的 user 消息**并入它的 `question`**（append-only）。

    None 表示本批开头没有"续写中的" user 消息。**这是本轮新增的第四个动作**：
    3a 原来只覆盖 `answer` 一侧，而一条被物理切开的 user 消息跨批次时，
    续写的落点是 `question`。
    """

    append_answer: str | None
    """3a：追加到它（既有对）的 `answer`。None 表示本批开头没有非 user 消息。"""

    close: bool
    """3b：本批出现首个 user 消息**且它已有 `answer`** ⇒ 把它标 `complete`。

    ⚠ **最容易漏的一步**：漏了它会永久挂在 pending，而且现象会**伪装成 AML 的切分行为**，
    让人跑去改配对规则——**方向完全错了**（pairing/CLAUDE.md）。

    ⚠ **判据是"已有 `answer`"，不是"出现了 user 消息"**：一个还没收到任何非 user 消息的
    对，它的 `question` 还在写，此时来的 user 消息是**续写**而不是关闭信号。
    """

    final_status: Status | None
    """3d 作用于既有对的情形（**纯接续批**，零条 user 消息）。

    此时被续写的那个对就是最后一带，同样要标状态，否则它会一直挂到 session 结束。
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
    """首条 user 消息的内容。`from_db` 时为 None——那份内容已经在库里，本批只带增量。"""
    event_time: int | None
    parts: list[Message] = field(default_factory=list)
    """要写进这个对的 `answer` 的非 user 消息（`from_db` 时由 `append_answer` 落库）。"""
    question_parts: list[Message] = field(default_factory=list)
    """要**并入**这个对 `question` 的后续 user 消息（`from_db` 时由 `append_question` 落库）。"""
    from_db: bool = False  # True = 它就是库里那个可续写的对
    db_answer_is_null: bool = True
    """仅 `from_db` 时有意义：库里那一对的 `answer` 是否为空。"""

    @property
    def question_is_open(self) -> bool:
        """这个对的 `question` 还在写吗？——**本轮配对规则新增的唯一判据**。

        > **一个对的 `question` = 一段连续 user 消息，直到第一条非 user 消息到达为止。**

        换成可判定的形式就是"**还没有任何非 user 消息进入它**"：

        * 本批新建的对：`parts` 为空 ⇒ 还在写 ⇒ 后续 user 并入 `question`
        * 库里既有、本批要续写的对：它的 `answer` 为空 ⇒ 还在写；
          已有 `answer`（§6.5 的 pending 对，答话跨了批次）⇒ 已定稿 ⇒ 下一条 user 关闭它

        ⚠ 用 `parts`（本批攒下的非 user 消息）而不是只看库里的 `answer`：
        本批若先追加了一条 assistant 消息，这个对的 `answer` 就已经开始了，
        此后再来的 user 必须是**新对**，不能并进 `question`。
        """
        if self.parts:
            return False
        return self.db_answer_is_null if self.from_db else True


def plan_batch(
    messages: Sequence[Message],
    *,
    next_idx: int,
    open_pair_id: str | None,
    open_pair_answer: str | None,
    limits: BatchLimits | None = None,
    word_counter: Callable[[Message], int] | None = None,
) -> BatchPlan:
    """把一批消息规划成"对那个可续写的对的动作 + 要新建的对"（§6.5 第 3 步）。

    三步的落点：

    * **3a** 本批开头、首个 user 消息之前的消息 → 追加到那个既有对的 `answer`；
      不存在既有对时**建一个 `question` 为空的对**（§6.2 承认这种"无问的对"）。
    * **3a′**（**本轮新增**）本批开头的 user 消息，**若那个既有对的 `answer` 还空着** ⇒
      并入它的 `question`，而**不是**关掉它。跨批时由 `append_question` 落库。
    * **3b** 本批出现首个 user 消息**且前一个对已有 `answer`** → 把它标 `complete`。
    * **3c** 其余消息按下面的配对规则配对，`pair_idx` 从 `next_idx` 起连续赋值。
    * **3d** 收尾给"涉及的最后一对"定状态：本批两限都未命中 ⇒ session 已结束 ⇒ `complete`。
      **纯接续批（零条 user 消息）也走这一步。**

    ## 配对规则（§6.2）

    > **一个对的 `question` = 一段连续 user 消息，直到第一条非 user 消息到达为止。**

    它**只在出现连续 user 消息时**与"一条 user 关闭前一个对"不同——此时这段 user 并进
    **同一个** `question`，而不是各自开新对。理由是 **AML 可能按句边界物理切开一条超长
    user 消息**，于是 `q1 q2 q3 a` 这种形状会出现，而它逻辑上仍是一问一答。
    完整论证、代价与"该风险尚未证实"的说明见 D20
    （[`docs/decisions.md`](../../../docs/decisions.md)）。

    配对作用域是**整个 session，不是本批**——所以这里要接 `next_idx` 与那个既有对的状态。

    ⚠ `open_pair_answer` **没有默认值**，这是故意的：`None` 是一个**合法的真实取值**
    （那个对的 `answer` 还空着），所以"忘了传"与"传了 None"必须区分开——
    前者在这里是一个 `TypeError`，后者才走"question 还在写"那条分支。
    `open_pair_id` 为 None 时它也必须传 None。
    """
    limits = BatchLimits() if limits is None else limits
    counter = _default_word_counter if word_counter is None else word_counter

    word_count = sum(counter(m) for m in messages)
    hit_message_limit = len(messages) >= limits.max_messages
    hit_word_limit = word_count >= limits.max_words
    # §6.5：只要本批命中任一上限，本批的最后一对就是 pending；两限都未命中 ⇒
    # AML 手上已经没有这个 session 的消息了，边界即 session 末端 ⇒ complete。
    #
    # ⚠ 两个判据**只看原始消息列表**，与配对规则无关：它们回答的是"AML 为什么在这里切断"，
    #   而那由**传输单位**决定。别把它们改成看合并后的样子（那会让"20 条消息"在连续 user
    #   被折叠后凭空命中上限）。
    session_ended = not (hit_message_limit or hit_word_limit)
    last_status: Status = STATUS_COMPLETE if session_ended else STATUS_PENDING

    idx = next_idx
    drafts: list[_Draft] = []
    # 库里那个可续写的对。它是"开着的"——本批可能是它的续写。
    db_draft: _Draft | None = (
        _Draft(
            pair_idx=-1,
            question=None,
            event_time=None,
            from_db=True,
            # 只有「answer 为空」的对，其 question 才还在写（见 _Draft.question_is_open）
            db_answer_is_null=open_pair_answer is None,
        )
        if open_pair_id is not None
        else None
    )
    open_draft: _Draft | None = db_draft
    closing_pending = False  # 3b 是否触发

    for message in messages:
        if is_user(message):
            if open_draft is not None and open_draft.question_is_open:
                # 3a′：这个对的 question 还没写完 ⇒ 这条 user 是它的**续写**。
                # 关对、开新对都**不做**——那正是旧规则"一条 user 关闭前一个对"的行为。
                open_draft.question_parts.append(message)
            else:
                if open_draft is not None:
                    # 3b：前一个对已经有 answer ⇒ 一条 user 消息关闭它。
                    if open_draft.from_db:
                        closing_pending = True
                    open_draft = None
                # 这条 user 消息开一个**新**对
                open_draft = _Draft(
                    pair_idx=idx, question=message.content, event_time=message.timestamp
                )
                drafts.append(open_draft)
                idx += 1
        else:
            if open_draft is None:
                # 3a 的后半句：不存在可续写的对且本批以非 user 消息开头 ⇒ 建无问的对
                open_draft = _Draft(pair_idx=idx, question=None, event_time=message.timestamp)
                drafts.append(open_draft)
                idx += 1
            # 非 user 一律进 answer——含 `from_db` 的那个对（它是本批对它的 3a 追加）
            open_draft.parts.append(message)

    resume: ResumeActions | None = None
    # ⚠ `open_pair_id is not None` 与 `db_draft is not None` **恒等价**——`db_draft` 正是由
    #   它构造出来的（见上）。写出来不是多一层判断，而是把"有既有对 ⇒ 有它的 id"
    #   这条不变式放到类型层可见的地方。
    if db_draft is not None and open_pair_id is not None:
        # 既有对被 3b 关闭时，最后一带是本批新建的对，状态已经归它了；
        # 没被关闭（纯接续批，或只续写了它的 question）时，最后一带就是它自己，由 3d 定状态。
        resume = ResumeActions(
            open_pair_id=open_pair_id,
            append_question=join_question(db_draft.question_parts) or None,
            append_answer=encode_answer(db_draft.parts) or None,
            close=closing_pending,
            final_status=None if closing_pending else last_status,
        )

    new_pairs = tuple(
        NewPairDraft(
            pair_idx=d.pair_idx,
            question=_merged_question(d),
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


def _merged_question(draft: _Draft) -> str | None:
    """新建对的 `question` = 首条 user 消息 + 本批并入的后续 user 消息。

    `draft.question` 为 None 时（无问的对，或库里既有的对）只返回并入的那部分——
    前者本来就是"无问"，后者走 `append_question` 而不经过这里。
    """
    merged = join_question(draft.question_parts)
    if draft.question is None:
        return merged or None
    return f"{draft.question}\n{merged}" if merged else draft.question


def _default_word_counter(message: Message) -> int:
    """默认词数计数：按空白切分。

    ⚠ **这只是占位**。"2,000 个 **Adapter 计数的词**"官方从未定义（S2 / §12.3 第 5 条），
    所以线上真值与本地必然对不上——`pending` 三个埋点因此只能**相对解读**。
    真正上线时由 `configs/` 注入计数函数；默认值只用于本地自测与单元测试。
    """
    return len(message.content.split())
