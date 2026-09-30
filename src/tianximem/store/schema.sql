-- 真源 DDL —— **D28 起位置 = `(request_id, local_index)`，`request_id` 是 opaque string**
--（这是对 PRD §6.1 的**有意偏离**，逐条记在 docs/decisions.md）
--
-- 两张表：qa_pairs（业务正文）+ applied_batches（唯一旁表 = 批次级幂等守卫）。
--
-- ⚠ 这张 schema 里【没有 CHECK 约束】——不是遗漏，是继承 §6.1 的那条纪律。
--    `status` 的取值域由 Python 侧保证（store/sqlite_store.py 的 Status），数据库层不拦。
--    ⚠ **D24 之后写入恒为 'complete'**（块在写下那一刻就是最终形状，没有 pending、没有 repair）
--    ——`'pending'` 仍然在取值域里只是因为**按旧规则写过的库还能读**。

CREATE TABLE IF NOT EXISTS qa_pairs (
    id             TEXT PRIMARY KEY,     -- hash(user_id, session_id, request_id, local_index)，【位置派生】
    user_id        TEXT NOT NULL,        -- 隔离契约（唯一隔离字段）
    session_id     TEXT NOT NULL,        -- 分组字段；**不是** Search 的过滤条件（§2.2）
    request_id     TEXT NOT NULL,        -- 写下这一行的 Add。**opaque**：只做溯源 + 位置的一半，
                                         -- 一律**不解析**（D28）
    local_index    INTEGER NOT NULL,     -- 它在【那一次 Add 的块列表】里的 0-based 序号（D28）。
                                         -- ⚠ 只在这一次 Add 内有意义，**没有跨 Add 的全局序**
    prev_memory_id TEXT,                 -- Add 内**显式**邻接（D28）：只连"完整 QA"，
    next_memory_id TEXT,                 -- 不完整的块（A-only / Q-only）恒为 NULL，不进链
    question       TEXT,                 -- 可空（块里没有 user 消息：Add 以非 user 开头）；
                                         -- D24 起它是【本次 Add 内】连续 user 消息的拼接
    answer         TEXT,                 -- 可空（块里没有非 user 消息）；D24 起每条非 user 消息
                                         -- 带 [role] 标记，且【同一 Add 内】连续非 user 消息合并
    status         TEXT NOT NULL,        -- 'complete' | 'pending'（D24 之后写入恒为 'complete'）
    event_time     INTEGER,              -- Unix 毫秒，取该块【首条消息】的 timestamp，可空

    -- 不变式 4（D28 重塑）：这个 UNIQUE 建出的索引既保唯一，又是
    -- "按 `local_index` 取回某一次 Add 的全部块"（`fetch_by_request`）的排序键。
    --
    -- ⚠ **相邻性不由这两列出**：跨 Add 不存在可信的全局序（D28），
    --    所以邻接由 `prev_memory_id` / `next_memory_id` **显式**给出，
    --    而它们**只在一次 Add 内**相连。
    UNIQUE(user_id, session_id, request_id, local_index)
);

-- 唯一的旁表：批次级幂等守卫（§6.5 / D4）。每批一行，【只增不改】。
--
-- ⚠ 为什么不能省掉它、改用 qa_pairs.request_id 判重：
--   守卫要回答的是"这批**被应用过吗**"，而 `qa_pairs.request_id` 记的是
--   "哪一批写下了这一行"——两件事。旁表是唯一记着前者的地方，还带着 user_id /
--   session_id / applied_at。
CREATE TABLE IF NOT EXISTS applied_batches (
    request_id   TEXT PRIMARY KEY,       -- AML 的 Add request_id，天然唯一
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    payload_hash TEXT,                   -- 规范化 payload（user_id + session_id + messages）的
                                         -- sha256（D28）：同 id 不同 payload ⇒ 409，而不是静默当重放。
                                         -- ⚠ **NULL = 未知**（D28 之前写下的行没有这一列）⇒ 只能放行
    applied_at   INTEGER NOT NULL        -- Unix 毫秒（与 event_time 同单位）
);
