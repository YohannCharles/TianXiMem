-- 真源 DDL —— **D25 起 `pair_idx` 被 `(chunk_ordinal, local_index)` 取代**
--（原本逐字对应 PRD §6.1；这是对该节的**有意偏离**，逐条记在 docs/decisions.md 的 D25）。
--
-- 两张表：qa_pairs（业务正文）+ applied_batches（唯一旁表 = 批次级幂等守卫）。
--
-- ⚠ 这张 schema 里【没有 CHECK 约束】——不是遗漏，是继承 §6.1 的那条纪律。
--    `status` 的取值域由 Python 侧保证（store/sqlite_store.py 的 Status），数据库层不拦。
--    ⚠ **D24 之后写入恒为 'complete'**（块在写下那一刻就是最终形状，没有 pending、没有 repair）
--    ——`'pending'` 仍然在取值域里只是因为**按旧规则写过的库还能读**。

CREATE TABLE IF NOT EXISTS qa_pairs (
    id            TEXT PRIMARY KEY,      -- hash(user_id, session_id, chunk_ordinal, local_index)，【位置派生】
    user_id       TEXT NOT NULL,         -- 隔离契约（唯一隔离字段）
    session_id    TEXT NOT NULL,
    chunk_ordinal INTEGER NOT NULL,      -- 该块来自【哪一批 Add】：从 request_id 解析（D25）
    local_index   INTEGER NOT NULL,      -- 它在【那一批的块列表】里的 0-based 序号（D25）
    question      TEXT,                  -- 可空（块里没有 user 消息：Add 以非 user 开头）；
                                         -- D24 起它是【本次 Add 内】连续 user 消息的拼接
    answer        TEXT,                  -- 可空（块里没有非 user 消息）；D24 起每条非 user 消息
                                         -- 带 [role] 标记，且【同一 Add 内】连续非 user 消息合并
    status        TEXT NOT NULL,         -- 'complete' | 'pending'（D24 之后写入恒为 'complete'）
    event_time    INTEGER,               -- Unix 毫秒，取该块【首条消息】的 timestamp，可空
    request_id    TEXT NOT NULL,         -- 写该行的 Add；【只用于溯源，不承担幂等职责】

    -- 不变式 4（D25 重塑）：这个 UNIQUE 建出的索引既保唯一，又是
    -- "按会话顺序取整段"（`fetch_session_ordered` 的窗口）的排序键。
    --
    -- ⚠ **顺序语义由这两列给出，不由到达顺序给出**：同一批的并发到达无害
    --    （位置是请求的纯函数）；而 `chunk_ordinal` 的缺失序号（空洞）**不破坏相邻**
    --    ——相邻性走读时的稠密序 `seq`（见 sqlite_store.fetch_session_ordered）。
    UNIQUE(user_id, session_id, chunk_ordinal, local_index)
);

-- 唯一的旁表：批次级幂等守卫（§6.5 / D4）。每批一行，【只增不改】。
--
-- ⚠ 为什么不能省掉它、改用 qa_pairs.request_id 判重：
--   D24 之后"行被后一批改写"这条路已经不存在了，但守卫**仍然不读那一列**——
--   它要回答的是"这批被应用过吗"，而 `qa_pairs.request_id` 记的是"哪一批写下了这一行"，
--   两件事。旁表是唯一记着前者的地方，还带着 user_id / session_id / applied_at。
CREATE TABLE IF NOT EXISTS applied_batches (
    request_id  TEXT PRIMARY KEY,        -- AML 的 Add request_id，天然唯一
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    applied_at  INTEGER NOT NULL         -- Unix 毫秒（与 event_time 同单位）
);
