# pairing/ — 记忆块组合与落库

**PRD**：§6.2（一个 QA 对是什么）· **D24**（一次 Add = 组合的唯一边界）· **D25**（位置由 `request_id` 的 chunk 序号派生）

## 本模块的构成

```text
pairing.py   本批消息 → 记忆块（组合规则的**唯一实现**，纯函数）
              + `parse_chunk_ordinal()`（从 request_id 取 chunk 序号，D25）
apply.py     幂等守卫 → 组合 → 按 chunk 派生位置写入（一个事务）
```

> ⚠ **D24（2026-09-27）之后没有"批次续接"**：`continuation.py` 已改名 `apply.py`，
> `instrument.py`（三个 `pending` 计数器）、`plan_batch`、`ResumeActions`、`open_pair`、
> `append_question` / `append_answer` / `mark_complete` / `touch_request_id` **全部删除**。
> 理由与实测代价见 [`../../../docs/decisions.md`](../../../docs/decisions.md) **D24**。

---

## 组合规则（D24）

```text
一次 Add 的消息（按原序）
  → ① 连续同 role 合并成一个 RoleBlock
  → ② 一个 UserBlock + 紧随其后的【全部】非 UserBlock 配成一个 MemoryBlock
  → ③ 开头的非 UserBlock（前面没有 user）独立成块
  → 每个 MemoryBlock 恰好一次 embedding
```

| 输入 | 输出 |
| --- | --- |
| `U A U A` | `[U+A] [U+A]` |
| `U U A A A U A` | `[UU+AAA] [U+A]` |
| `A U A` | `[A] [U+A]` |
| `U A U` | `[U+A] [U]` |
| `U assistant tool assistant` | `[U+A0+T+A1]`（工具调用序列不劈开） |

**不同 Add 之间永不组合**：不拼 QA、不合并连续 assistant、不等下一个 chunk、不 repair、
不重新 embedding。即使两个 Add 属于同一 session，也各自独立成块。

**配不上的块直接独立存**——**不判断**它是不是"超长消息被截断"。任何"等下一个 Add
补齐"的设计都会重新引入跨 Add 状态，而那正是 D24 取消的东西。

**② 吃掉的为什么是"全部"而不是"一个"**：§6.2 承认"一条 user 后跟多条 assistant 消息
（工具调用等）"，只吃一个会把 `Q0 / assistant / tool / assistant` 劈成
`[Q0+A0] [T0] [A1]`。在只有 user / assistant 的输入上两种写法**逐字相同**。

### 判据只有一条，且**不枚举 role 白名单**

**只判断这条消息的 `role` 是不是 `user`**（`is_user`）。

> ⚠ AML 传入的 role 取值域**没有文档**。白名单会在遇到没见过的 role 时**静默丢消息**
> ——其他 role（`assistant` / `system` / 工具输出…）**一律按非 user 处理**，
> 且**每条带 `[role]` 标记**。

### 两侧的 role 处理**故意不同**

| 侧 | 写法 | 为什么 |
| --- | --- | --- |
| `question` | `join_question`，**不加标记** | role 均一（由渲染的 `Q: ` 前缀给出）；而拼接要把**被物理切开的一条原消息逐字拼回去** |
| `answer` | `encode_answer`，**每条无条件加** `[role]` | 一个块里可能混着 `assistant` / `system` / 工具输出，不标就分不清哪句是谁说的（§11.3） |

---

## 位置：`(chunk_ordinal, local_index)`（**D25**）

```text
chunk_ordinal  从 request_id 解析（`parse_chunk_ordinal`）——**只能从那儿取**
local_index    这一批组合出的块序号（`compose_memory_blocks` 的输出下标）
id              hash(user_id, session_id, chunk_ordinal, local_index)
```

**两者都是请求的纯函数** ⇒ 没有共享计数器、没有读-改-写 ⇒
**同 session 的 Add 可以并发**（`SessionLocks` 已删），**乱序到达也不翻转会话顺序**。

⚠ **取不到 chunk 序号 ⇒ 响亮失败**（非 200），**没有 `MAX+1` 回退`**——
回退会把"乱序到达静默翻转顺序"那个 bug 带回来。格式住在
`ingest.chunk_ordinal_pattern`（可配置，所以平台换形状不必改代码）。
**这条格式假设的来源是团队告知、不是一手文档**（与 S2/S5 同类）——见 D25。

### 位置对齐 ≠ 组合边界

| | 作用域 |
| --- | --- |
| **组合**（哪些消息进同一个块） | **一次 Add**（D24） |
| **位置的作用域** | `chunk_ordinal` 由**请求**给出 ⇒ 与"哪个 Add 先到"无关（D25） |

**相邻性不走位置**，走 `store.fetch_session_ordered` 现算的稠密序 `seq`
（`ROW_NUMBER() ... - 1`）——所以 chunk 序号**跳号不破坏相邻**，
而"中间真的少了一块"仍然被抓住（见 [`../rank/neighbor.py`](../rank/neighbor.py)）。

---

## 写入形态（D24 之后）

```text
块在写下的那一刻就是最终形状 ⇒ status 恒为 'complete'
没有任何后台任务会回头改这些行（没有 pending、没有 repair、没有重新 embedding）
```

`status` 列保留（§6.1 的 DDL 不改），但**不再有第二个取值**；`'pending'` 只在
"按旧规则写过的库"里还读得到。

**与幂等正交**：守卫仍查 `applied_batches`（D4），`BEGIN IMMEDIATE` 仍保留
（它现在是**数据库级写者串行**的唯一落点——D25 删掉应用层锁之后，并发全在这一处排队）。

⚠ **D25 顺带把失败模式变响了**：位置是纯函数 ⇒ 同一批重放必然算出**同一位置**，
于是重放撞 `UNIQUE` 而**不是**静默落成重复记录。守卫仍然必须留（AML 重试是正常行为，
不能每次都靠撞约束失败）。

---

## 写完之后先测什么

测试用例清单见 [`../../tests/CLAUDE.md`](../../../tests/CLAUDE.md)。**本目录的关键用例只有一条
要在这里记住**：幂等测试必须模拟"**事务已提交、响应未发出**"的中间态（**只测"重复 POST 两次"
太弱**）。D25 之前这一步抓的是"位置重分配"（那次 `MAX(pair_idx)` 的状态与崩溃重试时不同）；
D25 之后位置是纯函数、重放必然算出同一位置，于是这一步抓的是**重放撞 `UNIQUE`**
——守卫若被删掉，表现为**重放把整批写成 500**（`IntegrityError`），而不是静默重复。
