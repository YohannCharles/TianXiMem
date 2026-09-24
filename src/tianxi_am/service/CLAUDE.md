# service/ — HTTP 层

**PRD**：§2.1（接口形状）、§2.2（硬性规则）、§15（工程与部署）

## 要写什么

```text
app.py        FastAPI 实例 + 对象图装配（`--factory` 入口）、lifespan
routes.py     POST /add、POST /search（薄路由，只做形状映射）
schemas.py    请求与响应模型（pydantic）= §2.1 的字面翻译
pipeline.py   Add / Search 的**编排**（"按什么顺序调"）
locks.py      按 (user_id, session_id) 的串行化
errors.py     异常 → 保持"可重试"的边界处理
settings.py   路径与密钥（**过渡**：③-d 迁到 common/config.py）
```

> **为什么有 `pipeline.py`**：本层"不做检索、不做配对、不碰存储"指的是**不重新实现**
> 那些逻辑（全部往下调用）。而 Search 的链横跨 `retrieve/` 与 `rank/`、Add 的链横跨
> `pairing/`、`store/`、`embed/`——**没有任何单个下层模块能拥有整条链**，
> 所以"顺序"必须有人拥有，就在这里。路由仍然是薄的（`routes.py` 只有形状映射）。

## 这一层只做三件事

1. **把契约形状钉死**——请求/响应模型就是 §2.1 的字面翻译。
2. **串行化 Add**——按 `(user_id, session_id)` 加锁，保护 `pairing/` 的"读位置 → 写位置"。
3. **保持可重试**——内部异常时**不要吞掉、不要返回部分成功**。

它**不**做检索、不做配对、不碰存储。所有业务都往下走。

> **契约硬约束在根 `CLAUDE.md`（已加载）**——本节只列本层要落地的那几条，不重述 AML 侧的规则原文。

---

## 本层必须守住

| # | 要求 | 出处 |
| --- | --- | --- |
| 1 | **精确计数**：`len(data) <= top_k`，含邻域扩展加进来的槽位 | §2.2 / §10 |
| 2 | **原样回显** `request_id` / `user_id` / `session_id` | §2.1 |
| 3 | **响应前必须持久化完成且立即可搜索**——不允许异步建索引 | §2.1 |
| 4 | **Add 按 `(user_id, session_id)` 串行化**——第 2 步读位置、第 4 步写位置是"读-改-写"，两个并发批次会拿到同一个 `next_idx` | §15 |
| 5 | **非 200 的行为未定义，必须假设 AML 会重试**——内部异常应让本批保持"可重试"（事务未提交），**而不是返回一个"部分成功"** | §15 |
| 6 | **`user_id` 是唯一隔离字段**；`session_id` 不是 Search 的过滤条件 | §2.2 |

---

## 两个容易写错的地方

### 必须 `--workers 1`

第 4 条用的是**进程内锁**。多 worker 会**静默失效**——每个 worker 各有各的锁，两个批次照旧并发。

> **SQLite 的写事务不足以单独解决它**（§15）：两次事务读到的 `MAX(pair_idx)` **会相同**。

> ⚠ **别把 `SessionLocks` 和 SQLite 的 writer 串行化当成一件事**（D17）：前者管同一 session 的**业务顺序**（键 `(user_id, session_id)`、**进程内**），后者管"同时只有一个写事务"（键是整个库文件）。⇒ **不同 session 可以并发进入本层**，它们在 SQLite 处排队。**不要再叠一层应用层写库锁**——职责对照表见 [D17](../../../docs/decisions.md)。

### FastAPI 是 async，但这一层的下游是阻塞的

SQLite 是阻塞调用，本地 reranker 是 GPU 推理——都会**堵住事件循环**。用 FastAPI 不代表要写 `async def` 业务函数：把阻塞部分显式下到线程池（`run_in_threadpool`），**或者干脆把路由写成 `def`**（FastAPI 会自动下线程池）。

**不要**在 `async def` 里直接调阻塞的存储或推理——那会让并发请求互相饿死，而 30 分钟的单请求预算会掩盖这个问题，**直到 Full run 跑 0.5–2 天时才炸**（§15）。

---

## 契约自查

跑 Smoke 之前逐条过 [`../../../docs/contract.md`](../../../docs/contract.md) §4 的清单。**Smoke 次数有限（每轨道 ≤30 次），不要拿它当调试器。**

`make contract-check` 已接通（2026-09-24，指 [`../../eval/smoke/preflight.py`](../../../eval/smoke/preflight.py)）——**尤其是"No.1 精确计数"和"No.3 created_at 始终存在"这两条**，它们最容易在加了邻域扩展之后悄悄破掉。
