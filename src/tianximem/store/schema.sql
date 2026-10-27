-- Source-of-truth DDL.
--
-- The store has two source tables plus rebuildable grounded-fact and coverage data.
-- Validation of enum-like values is owned by Python.
-- The read path accepts `pending`; the write path creates only `complete`.

CREATE TABLE IF NOT EXISTS qa_pairs (
    id             TEXT PRIMARY KEY,     -- hash(user_id, session_id, request_id, local_index)
    user_id        TEXT NOT NULL,        -- the only Search isolation field
    session_id     TEXT NOT NULL,        -- grouping metadata; never a Search filter
    request_id     TEXT NOT NULL,        -- opaque Add identifier used in identity only
    local_index    INTEGER NOT NULL,     -- zero-based block index within one Add
    prev_memory_id TEXT,                 -- explicit Add-local adjacency for complete QA blocks
    next_memory_id TEXT,                 -- unpaired blocks have NULL links
    question       TEXT,                 -- joined contiguous user messages; nullable
    answer         TEXT,                 -- joined role-labelled non-user messages; nullable
    status         TEXT NOT NULL,        -- `complete` on writes; `pending` remains readable
    event_time     INTEGER,              -- Unix milliseconds from the block's first message

    -- This index enforces position uniqueness and orders fetch_by_request results.
    -- Cross-Add order is undefined; adjacency is represented only by the two link columns.
    UNIQUE(user_id, session_id, request_id, local_index)
);

-- Original Add messages for boundary-preserving derived fact extraction.
-- The parent row remains the merged/searchable memory block.
CREATE TABLE IF NOT EXISTS qa_pair_messages (
    parent_memory_id TEXT NOT NULL REFERENCES qa_pairs(id) ON DELETE CASCADE,
    source_index     INTEGER NOT NULL,
    role             TEXT NOT NULL,
    content          TEXT NOT NULL,
    timestamp        INTEGER,
    PRIMARY KEY(parent_memory_id, source_index)
);
CREATE INDEX IF NOT EXISTS qa_pair_messages_parent
    ON qa_pair_messages(parent_memory_id, source_index);

-- Batch-level idempotency guard. One append-only row records each applied request.
--
-- This cannot be replaced by qa_pairs.request_id: that column records which request
-- produced a source row, while this table records whether an entire batch was applied.
CREATE TABLE IF NOT EXISTS applied_batches (
    request_id   TEXT PRIMARY KEY,       -- AML Add request_id
    user_id      TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    payload_hash TEXT,                   -- sha256 of the canonical user/session/messages payload
                                         -- NULL means no stored hash; such rows are replayable
    applied_at   INTEGER NOT NULL        -- Unix milliseconds
);

-- Atomic facts share one derived table; qualifiers come only from the same source quote.
CREATE TABLE IF NOT EXISTS memory_facts (
    id TEXT PRIMARY KEY,
    parent_memory_id TEXT NOT NULL REFERENCES qa_pairs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    subject TEXT NOT NULL,
    subject_key TEXT NOT NULL,
    relation TEXT NOT NULL,
    relation_key TEXT NOT NULL,
    object TEXT NOT NULL,
    object_key TEXT NOT NULL,
    qualifiers TEXT NOT NULL,
    source_side TEXT NOT NULL CHECK(source_side IN ('question','answer')),
    source_quote TEXT NOT NULL,
    statement TEXT NOT NULL,
    content TEXT NOT NULL,
    event_time INTEGER,
    source_date TEXT NOT NULL,
    version TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_fact_subject
    ON memory_facts(user_id,version,subject_key,relation_key);
CREATE INDEX IF NOT EXISTS memory_fact_object
    ON memory_facts(user_id,version,object_key,relation_key);

-- 空抽取同样记扫描版本；旧版/专用索引的覆盖不能代替本版扫描。
CREATE TABLE IF NOT EXISTS evidence_coverage (
    parent_memory_id TEXT NOT NULL REFERENCES qa_pairs(id) ON DELETE CASCADE,
    user_id TEXT NOT NULL,
    version TEXT NOT NULL,
    PRIMARY KEY(parent_memory_id,version)
);
CREATE INDEX IF NOT EXISTS evidence_coverage_user
    ON evidence_coverage(user_id,version,parent_memory_id);
