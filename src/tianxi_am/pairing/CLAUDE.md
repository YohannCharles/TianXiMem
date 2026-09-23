# pairing/ — 配对与批次续接

**PRD**：§6.2（一个 QA 对是什么）、§6.5（残缺对与跨批次补全）

## 本模块的构成（**实现已存在**）

```text
pairing.py       本批消息 → QA 对（§6.2 的唯一判据）
continuation.py  批次续接三步 + pending 判定（§6.5）
instrument.py    三个 pending 计数器（计数器的读法与陷阱见 ../observability/）
```

**这是全项目逻辑最绕的一层，也是最容易静默出错的一层。** 出错的表现是"某些记忆永远检索不到"，而**不会有任何报错**。

---

## 配对判据只有一条（§6.2）

> **一个 QA 对 = 从一条 user 消息开始，到（不含）下一条 user 消息之前的全部消息。**

**实现里只做一种判断：这条消息的 `role` 是不是 `user`。**

> ⚠ **不要枚举 role 白名单。** AML 传入的取值域**没有文档**，白名单会在遇到没见过的 role 时**静默丢消息**。其他 role（`assistant` / `system` / 工具输出…）**一律归入当前对**。

这样定义而不是"严格 user + 紧邻 assistant"，是为了覆盖三种真实情况：

| 情况 | 结果 |
| --- | --- |
| 连续多条 user 消息 | 前一对应被下一条 user 关闭 |
| 一条 user 后跟多条 assistant 消息（工具调用等） | 全部归入该对 |
| session 以 assistant 开头 | `question` 为空，构成一个**无问的对** |

**配对的作用域是整个 session，不是一个 Add 批次**（§6.5）。一个 QA 对**可以跨批次**——批次只是 AML 的传输单位。这一点直接决定了 `pair_idx` 必须 session 内连续、新批次要接着数。

**与 ReFind 的 turn 粒度一致**——**与外部基线同粒度，消融对比才干净**（§6.2）。

---

## 批次续接三步，顺序不能换（§6.5）

新批次到达时，服务端**必须先从库里恢复上下文**。

```text
1. 幂等守卫（必须最先做）
   SELECT 1 FROM applied_batches WHERE request_id = ? LIMIT 1
   命中 → 本批已应用过（重试）→ 直接返回 200，不写任何东西
   ※ 应用成功时，在**同一个事务**里向 applied_batches 插入本批这一行

2. 恢复位置
   next_idx = COALESCE(MAX(pair_idx) + 1, 0)        -- 限于该 (user_id, session_id)
   pending  = 该 session 中 status = 'pending' 的那一对
              （至多一个，且必在末尾：ORDER BY pair_idx DESC LIMIT 1）

3. 挂接本批消息
   a. 本批开头、首个 user 消息之前的消息 → 追加到 pending 的 answer
      若此时不存在 pending → 按 §6.2 处理（批次以 assistant 开头，建一个 question 为空的对）
   b. 本批出现首个 user 消息 → 把前一个 pending 对标 complete
      （它是被两条连续 user 消息关掉的，没有内容可填；漏了这一步它会永久挂在 pending）
   c. 其余消息按 §6.2 配对，pair_idx 从 next_idx 起连续赋值
   d. 收尾：给涉及到的最后一对标 status：
      本批两限都未命中 ⇒ session 已结束 ⇒ 标 complete
      ※ **纯接续批（零条 user 消息）也走这一步**
```

> **第 1 步不能省——这是本项目最容易踩的一个陷阱。**
>
> "只填空不覆盖"确实让**内容**写入幂等，但**位置分配不幂等**。若服务在事务提交之后、响应发出之前崩溃（或响应丢失），AML 会重试同一批（`request_id` 与 payload 不变），而此时 `MAX(pair_idx)` **已经前移**——重试会把**同一批消息重新分配到新的 `pair_idx` 上**，落成一份重复记录，**且不会报错**。
>
> 详细论证见 [`../../../docs/decisions.md`](../../../docs/decisions.md) D4。**别把它和内容幂等混为一谈——它们解决的是两个不同的问题。**

> ⚠ **3b 最容易漏，且漏了会伪装成数据问题**（见下）。

---

## `pending` 判定规则（§6.5）

**只要本批命中任一上限（20 条消息 或 2,000 词），本批的最后一对就是 `pending`；两限都未命中，则最后一对是 `complete`**——AML 手上已经没有这个 session 的消息了，边界即 session 末端。

**为什么 `pair_idx` 必须连续**：`event_time` 保证不了 session 内顺序——同一秒的多条消息排序未定义，会让 `±1` 邻域扩展产生抖动，进而**让消融实验不可复现**（§6.1）。两个计分数据集**都没有 per-turn 时间戳**，所以在它们上面这不是"可能发生"而是必然（实证见 [`../eval/datasets/CLAUDE.md`](../../../eval/datasets/CLAUDE.md)）。

### 三个埋点

| 计数器 | 含义 |
| --- | --- |
| `pending_created` | 新建的 `pending` 对 |
| `pending_completed` | 被后续批次补全的对 |
| `pending_orphaned` | 到 session 结束仍是 `pending` 的对 |

**本目录只负责发射**，读法与两个必须分开的来源见 [`../observability/CLAUDE.md`](../observability/CLAUDE.md)。**本地埋点数与线上必然对不上**（切批口径之一无定义）——那条也记在那里。

---

## 写入规则：填空 + 追加，绝不覆盖（§6.5）

```text
question      只在原值为 NULL 时写入
answer        只在原值为 NULL 时写入；跨批续接时允许**追加**（append-only）
status        只允许 pending → complete，不允许反向
```

`answer` 必须允许追加，是因为 §6.2 承认"一条 user 后跟多条 assistant 消息（工具调用等）"——这类对若跨批次，**续接批次带来的 assistant 消息必须并进去，而不是丢掉**。

**追加的安全性由批次级守卫保证**（同一批至多被应用一次），**不需要额外的判重逻辑**。

---

## 写完之后先测什么

测试用例清单见 [`../../tests/CLAUDE.md`](../../../tests/CLAUDE.md)。**本目录的关键用例只有一条要在这里记住**：幂等测试必须模拟"**事务已提交、响应未发出**"的中间态——**只测"重复 POST 两次"抓不到位置重分配**，因为那时 `MAX(pair_idx)` 的状态与崩溃重试时不同。
