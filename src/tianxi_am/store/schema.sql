-- 真源 DDL —— 逐字对应 PRD §6.1（与 src/tianxi_am/store/CLAUDE.md 的 DDL 要点一致）。
--
-- 两张表：qa_pairs（业务正文）+ applied_batches（唯一旁表 = 批次级幂等守卫）。
--
-- ⚠ 这张 schema 里【没有 CHECK 约束】——不是遗漏，是为了与 §6.1 的 DDL 保持逐字一致。
--    `status` 的取值域（'complete' | 'pending'）由 Python 侧保证（pairing/pairing.py 的 Status），
--    数据库层不拦。若要改成 CHECK，那是**对 §6.1 的有意偏离**，须先记进 docs/decisions.md。

CREATE TABLE IF NOT EXISTS qa_pairs (
    id          TEXT PRIMARY KEY,        -- hash(user_id, session_id, pair_idx)，【位置派生】
    user_id     TEXT NOT NULL,           -- 隔离契约（唯一隔离字段）
    session_id  TEXT NOT NULL,
    pair_idx    INTEGER NOT NULL,        -- session 内连续，从 0 起
    question    TEXT,                    -- 可空（无问的对：批次以非 user 消息开头）
    answer      TEXT,                    -- 可空（pending 对）；跨批续接时允许【追加】
    status      TEXT NOT NULL,           -- 'complete' | 'pending'
    event_time  INTEGER,                 -- Unix 毫秒，取该对【首条消息】的 timestamp，可空
    request_id  TEXT NOT NULL,           -- 最后触碰该行的 Add；【只用于溯源，不承担幂等职责】

    -- 不变式 4：这个 UNIQUE 建出的索引【正好是】邻域查询的键
    --   (user_id, session_id, pair_idx BETWEEN ? AND ?)
    -- 扩窗从 ±1 改成 ±2 只需改 BETWEEN 的界，不动索引。
    UNIQUE(user_id, session_id, pair_idx)
);

-- 唯一的旁表：批次级幂等守卫（§6.5）。每批一行，【只增不改】。
--
-- ⚠ 为什么不能省掉它、改用 qa_pairs.request_id 判重：
--   那一列会被【后一批覆盖】——若某批唯一触碰过的行随后被下一批改写，
--   这批的指纹就丢了，守卫查不到、重试照旧重复应用。旁表让它永不丢失。
CREATE TABLE IF NOT EXISTS applied_batches (
    request_id  TEXT PRIMARY KEY,        -- AML 的 Add request_id，天然唯一
    user_id     TEXT NOT NULL,
    session_id  TEXT NOT NULL,
    applied_at  INTEGER NOT NULL         -- Unix 毫秒（与 event_time 同单位）
);
