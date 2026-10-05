# pairing/ — 记忆块组合与落库

**PRD**：§6.2（一个 QA 对是什么）· **D24**（一次 Add = 组合的唯一边界）· **D28**（`request_id` 是 opaque string；邻接只在 Add 内）· **D32**（配对只取 user 段的最后一条）

## 本模块的构成

```text
pairing.py   本批消息 → 记忆块（组合规则的**唯一实现**，纯函数）
              + `link_blocks()`（Add 内邻接链：只连完整 QA，D28）
apply.py     幂等守卫（含 payload 指纹）→ 组合 → 连链 → 按位置写入（一个事务）
```

> ⛔ **不实现"批次续接"**：块在写下那一刻就是最终形状，跨 Add 不合并、不补全。
> 理由与实测代价见 [`../../../docs/decisions.md`](../../../docs/decisions.md) **D24**。

---

## 组合规则（D24 → **D32**）

```text
一次 Add 的消息（按原序）
  → ① 一段连续 user + 紧随其后的一段连续非 user ⇒ 配成一个 MemoryBlock
       ⚠ 但只有该段的**最后一条** user 进配对块（**D32**）——前面的各自独立
  → ② 其余消息：每条各自独立成块——**同 role 相邻也不并**
  → 每个 MemoryBlock 恰好一次 embedding
```

| 输入 | 输出 |
| --- | --- |
| `U A U A` | `[U+A] [U+A]` |
| `U U A A A U A` | `[U] [UU+AAA] [U+A]` |
| `A U A` | `[A] [U+A]` |
| `U A U` | `[U+A] [U]` |
| `U assistant tool assistant` | `[U+A0+T+A1]`（工具调用序列不劈开） |
| `U U U U`（全 user，**没有非 user**） | `[U] [U] [U] [U]`（**各自独立**） |
| `A A`（全 assistant） | `[A] [A]`（**各自独立**） |

> ⚠ **这是一条 D24 的修正**（2026-10-01）。旧规则里"连续同 role 合并"是**独立的一条**
> （第 ① 步），于是**判分池那种全 `role: user` 的 Add 会整个塌成一块**。
> 实测：locomo 412 块 → 55 块、medmemorybench 1,562 → 200，且**配对块一个不剩**
> ⇒ `link_blocks` 只连配对块 ⇒ **邻接链全空** ⇒ 扩窗与段合并**一行都不执行**。
> 改成"只有配得上对才合并"之后，同一条数据是 **788 / 3,124 块**（粒度与链路都回来了）。
>
> ⚠ **S5 的那个形状已按 D32 改口径**：`U U U A`（超长消息被切成多片 + 回答）现在产出
> **`[U] [U] [U+A]`**——只有**最后一片**配得上回答，前面那些**有问无答**。
> 这是 D32 明写的取舍（"问题完整" vs "块小到嵌得进窗口"，选了后者）：合并那几片会得到
> **38,270 token** 的块，线上静默截断、本地网关 400。理由与实测见 `decisions.md` **D32**。

**不同 Add 之间永不组合**：不拼 QA、不合并连续 assistant、不等下一个 chunk、不 repair、
不重新 embedding。即使两个 Add 属于同一 session，也各自独立成块。

**配不上的块直接独立存**——**不判断**它是不是"超长消息被截断"。任何"等下一个 Add
补齐"的设计都会重新引入跨 Add 状态，而那正是 D24 取消的东西。

**吃掉的为什么是"全部"非 user 而不是"一个"**：§6.2 承认"一条 user 后跟多条 assistant
消息（工具调用等）"，只吃一个会把 `Q0 / assistant / tool / assistant` 劈成
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

## 位置：`(request_id, local_index)`（**D28**）

```text
request_id   原样来自请求的**整体**（opaque，一个字节都不解析）
local_index  这一批组合出的块序号（`compose_memory_blocks` 的输出下标）
id           hash(user_id, session_id, request_id, local_index)
```

**两者都是请求的纯函数** ⇒ 没有共享计数器、没有读-改-写 ⇒
**同 session 的 Add 可以并发**，而"到达顺序"**根本没有被表达过**。

> ⛔ **`request_id` 是 opaque string，不要从里面取任何东西。**
> 平台实发的是 `r_31156f4174…` 这种不透明 id（2026-09-29 用请求采集抓到的真实请求），
> 任何"按形状解析它"的逻辑在真实流量上都会 **100% 失败**（→ 500 → Add 全挂，
> 见 [`../../../docs/decisions.md`](../../../docs/decisions.md) **D28**）。
> **位置只能来自 `(request_id, local_index)`。**

### 邻接也在这一个作用域内：`link_blocks()`

```text
一次 Add 的块列表 → link_blocks() → 每个块的 (prev_local_index, next_local_index)
                                     ↑ **只连完整 QA**（`MemoryBlock.is_paired`）
```

* A-only / Q-only **照样存、照样 embedding、照样能被检索到**，但**不进链**（两侧 `None`）
* 指针在**写下时**就写死（`insert_pair` 一次 INSERT）——**没有回填、没有 UPDATE**
* ⇒ 跨 Add **永远不相邻**：`Add1` 的末尾与 `Add2` 的开头在位置上看着接得上，
  但链上就是断的（见 [`../rank/neighbor.py`](../rank/neighbor.py) 的合并判据）

| | 作用域 |
| --- | --- |
| **组合**（哪些消息进同一个块） | **一次 Add**（D24） |
| **位置与邻接** | **一次 Add**（D28） |

⇒ **一次 Add 的全部行为只由它自己的 payload 决定**：与到达顺序、与其他 Add、
与 `request_id` 的形状**全都无关**。

---

## 写入形态（D24 之后）

```text
块在写下的那一刻就是最终形状 ⇒ status 恒为 'complete'
没有任何后台任务会回头改这些行（没有 pending、没有 repair、没有重新 embedding）
```

`status` 列保留（§6.1 的 DDL 不改），但**不再有第二个取值**；`'pending'` 只在
"按旧规则写过的库"里还读得到。

**与幂等正交**：守卫仍查 `applied_batches`（D4），`BEGIN IMMEDIATE` 仍保留
（它现在是**数据库级写者串行**的唯一落点——应用层锁已经没有了，并发全在这一处排队）。

⚠ **失败模式是响的**：位置是请求的纯函数 ⇒ 同一批重放必然算出**同一位置**，
于是重放撞 `UNIQUE` 而**不是**静默落成重复记录。守卫仍然必须留（AML 重试是正常行为，
不能每次都靠撞约束失败）。

---

## 写完之后先测什么

测试用例清单见 [`../../tests/CLAUDE.md`](../../../tests/CLAUDE.md)。**本目录的关键用例只有一条
要在这里记住**：幂等测试必须模拟"**事务已提交、响应未发出**"的中间态（**只测"重复 POST 两次"
太弱**）。位置是纯函数、重放必然算出同一位置，所以这一步抓的是**重放撞 `UNIQUE`**
——守卫若被删掉，表现为**重放把整批写成 500**（`IntegrityError`），而不是静默重复。
