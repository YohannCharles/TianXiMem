"""一轮评测怎么跑（§12.1 / §12.3 / §6.5 / §13）。

```python
from eval.harness import ServiceClient, build_input_items, pipeline_for, run_judge

with ServiceClient("http://127.0.0.1:8000") as client:
    for sample in samples:
        client.ingest(sample)                      # 按 20 条切批喂 Add
        hits = {q.qid: client.search(user_id=sample.user_id, query=q.question, top_k=100)
                for q in sample.questions}
    items = build_input_items(sample, hits)
    results = run_judge(pipeline_for(bench_dir, sample.dataset), items, out_dir)
```

**`eval/` 与 `src/tianximem` 之间只有一条通道：HTTP**——harness 不得 `import`
服务内部模块（理由见 [`../CLAUDE.md`](../CLAUDE.md)）。本包只依赖 `httpx` 与标准库。
"""

from .add_shape import (
    ADD_SHAPES,
    DEFAULT_ADD_SHAPE,
    LABEL_RULES,
    MAX_MESSAGE_CHARS,
    shape_batch,
)
from .batching import MAX_MESSAGES_PER_BATCH, batches, request_id_for
from .driver import REQUEST_TIMEOUT_S, SearchHit, ServiceClient
from .judge import (
    MEMORY_FIELD,
    JudgeResult,
    build_input_items,
    build_official_items,
    pipeline_for,
    render_memories,
    run_judge,
)
from .run_record import build_record, config_fingerprint, summarize, write_record

__all__ = [
    "ADD_SHAPES",
    "DEFAULT_ADD_SHAPE",
    "LABEL_RULES",
    "MAX_MESSAGE_CHARS",
    "MAX_MESSAGES_PER_BATCH",
    "MEMORY_FIELD",
    "REQUEST_TIMEOUT_S",
    "JudgeResult",
    "SearchHit",
    "ServiceClient",
    "batches",
    "build_input_items",
    "build_official_items",
    "build_record",
    "config_fingerprint",
    "pipeline_for",
    "render_memories",
    "request_id_for",
    "run_judge",
    "shape_batch",
    "summarize",
    "write_record",
]
