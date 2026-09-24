"""§6.5 的批次续接三步 + 落库。

三步**顺序不能换**，且必须落在**同一个事务**里：

```text
1. 幂等守卫（必须最先做）
2. 恢复位置（next_idx / 可续写的对）
3. 挂接本批消息（3a′ / 3a / 3b / 3c / 3d）
   ※ 应用成功时，在同一个事务里向 applied_batches 插入本批这一行
```

⚠ **第 1 步不能省——这是本项目最容易踩的一个陷阱。**

"只填空不覆盖"确实让**内容**写入幂等，但**位置分配不幂等**。若服务在事务提交之后、
响应发出之前崩溃（或响应丢失），AML 会重试同一批（`request_id` 与 payload 不变），
而此时 `MAX(pair_idx)` **已经前移**——重试会把同一批消息重新分配到**新的 `pair_idx`** 上，
落成一份重复记录，**且不会报错**。详见 D4。

**内容幂等与位置幂等解决的是两个不同的问题，别把它们混为一谈。**
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from tianxi_am.pairing.instrument import (
    NullPendingInstrument,
    PendingInstrument,
)
from tianxi_am.pairing.pairing import (
    BatchLimits,
    BatchPlan,
    Message,
    plan_batch,
)
from tianxi_am.store.sqlite_store import STATUS_COMPLETE, STATUS_PENDING, SqliteStore

__all__ = ["AddBatch", "ApplyBatchResult", "apply_batch"]


@dataclass(frozen=True, slots=True)
class AddBatch:
    """一次 `Add` 的 **canonical 契约输入**（§2.1）。

    ⚠ 只有这四样东西。**不得**出现任何数据集专属字段——加载层负责归一化（D16）。
    """

    request_id: str
    user_id: str
    session_id: str
    messages: tuple[Message, ...]


@dataclass(frozen=True, slots=True)
class ApplyBatchResult:
    applied: bool
    """False = 幂等守卫命中（本批已应用过），**没有写任何东西**。"""

    new_pair_count: int
    plan: BatchPlan | None


def apply_batch(
    store: SqliteStore,
    batch: AddBatch,
    *,
    limits: BatchLimits | None = None,
    word_counter: Callable[[Message], int] | None = None,
    instrument: PendingInstrument | None = None,
) -> ApplyBatchResult:
    """应用一批 `Add` 消息。**幂等**：同一 `request_id` 至多被应用一次。

    整个三步在**一个事务**里：崩溃或异常 ⇒ 整批回滚 ⇒ 保持"可重试"，
    不会留下半个批次（contract.md §6）。

    ⚠ **空批次直接抛 `ValueError`。** §2.1 的 `messages` 就是批次的全部内容，
    空批次是**调用方的 bug**（harness 或加载层喂错了），不是 AML 的行为。
    静默当成 no-op 会让"整个 session 一条都没写进去"伪装成成功——
    正是本层最怕的失败模式。代价是它会被重试 32 次，但触发它的是我们自己的代码。
    """
    if not batch.messages:
        raise ValueError("AddBatch.messages 不得为空（§2.1：messages 是批次的全部内容）")

    inst: PendingInstrument = NullPendingInstrument() if instrument is None else instrument
    caller = _default_word_counter if word_counter is None else word_counter

    with store.transaction() as conn:
        # ── 第 1 步：幂等守卫（必须最先做，且必须是查 applied_batches）──
        if store.is_batch_applied(conn, batch.request_id):
            return ApplyBatchResult(applied=False, new_pair_count=0, plan=None)

        # ── 第 2 步：恢复位置 ──
        next_idx = store.next_pair_idx(conn, batch.user_id, batch.session_id)
        # "可续写的对"而不是"pending 对"：判据里除了状态位还有 `answer IS NULL`
        # 这条**内容事实**，理由见 `open_pair` 的 docstring（不依赖我们复现不了的词数计数）。
        open_pair = store.open_pair(conn, batch.user_id, batch.session_id)

        # ── 第 3 步：规划 + 挂接 ──
        plan = plan_batch(
            batch.messages,
            next_idx=next_idx,
            open_pair_id=None if open_pair is None else open_pair.id,
            open_pair_answer=None if open_pair is None else open_pair.answer,
            limits=limits,
            word_counter=caller,
        )

        # 3a′ / 3a / 3b / 3d —— 都作用于那个可续写的对
        if plan.resume is not None:
            resume = plan.resume
            if resume.append_question:
                # 3a′：本批开头的 user 消息是它 question 的续写（跨批的碎片合并）
                store.append_question(conn, resume.open_pair_id, resume.append_question)
            if resume.append_answer:
                store.append_answer(conn, resume.open_pair_id, resume.append_answer)
            if resume.close:
                store.mark_complete(conn, resume.open_pair_id)
            elif resume.final_status == STATUS_COMPLETE:
                # 3d 作用于既有对（纯接续批 + session 结束判定）
                store.mark_complete(conn, resume.open_pair_id)
            # 这一行确实被本批触碰了 ⇒ 更新溯源用的 request_id（§6.1）。
            # ⚠ 它【只用于溯源】：正因为它会被后一批覆盖，幂等守卫不能复用它（D4）。
            store.touch_request_id(conn, resume.open_pair_id, batch.request_id)

        # 3c —— 新建的对，pair_idx 从 next_idx 起连续赋值
        for draft in plan.new_pairs:
            store.insert_pair(
                conn,
                user_id=batch.user_id,
                session_id=batch.session_id,
                pair_idx=draft.pair_idx,
                question=draft.question,
                answer=draft.answer,
                status=draft.status,
                event_time=draft.event_time,
                request_id=batch.request_id,
            )

        # 应用成功时，在【同一个事务】里记下本批
        store.record_batch(conn, batch.request_id, batch.user_id, batch.session_id)

    # 计数（在事务外发，避免事务里的副作用影响回滚语义）
    created = sum(1 for d in plan.new_pairs if d.status == STATUS_PENDING)
    if created:
        inst.record_created(created)
    if plan.resume is not None:
        if plan.resume.close:
            inst.record_completed()
        elif plan.resume.final_status == STATUS_COMPLETE:
            inst.record_orphaned()

    return ApplyBatchResult(applied=True, new_pair_count=len(plan.new_pairs), plan=plan)


def _default_word_counter(message: Message) -> int:
    """默认词数计数（按空白切分）——**与 `pairing._default_word_counter` 同一约定**。

    保留一份独立实现而不是互相 import，是为了让 `continuation` 不依赖 `pairing` 的私有名。
    "Adapter 计数的词"官方从未定义（S2），真正上线时由 `configs/` 注入。
    """
    return len(message.content.split())
