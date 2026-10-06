"""经上游语料清单校验的整表 JSON 和全部 Passage ID 段落。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .. import hybridqa as source
from ..preprocess import Message
from .common import payload
from .plan import AddEvent, InputPlan, SearchEvent

TABLE_PREAMBLE = "Complete table document in JSON. Each header/data cell is [text, link IDs].\n"


def load(root: Path, *, limit: int | None, spread: bool) -> list[InputPlan]:
    samples = source.load_hybridqa(root, limit=limit, spread=spread)
    # 用户 ID 与表标识的映射来自题库元数据，不由答案或 answer-node 选表/段落。
    rows = json.loads((root / source.DATA_DIR / source.QUESTIONS).read_text(encoding="utf-8"))
    tables = {
        f"hybridqa-{row['question_id']}": source.safe_table_id(row["table_id"]) for row in rows
    }
    receipt = json.loads(
        (root / source.DATA_DIR / source.CORPUS_RECEIPT).read_text(encoding="utf-8")
    )
    plans = []
    for sample in samples:
        table_id = tables[sample.questions[0].qid]
        table, passages = source.read_corpus(root, table_id, receipt)
        bodies = [TABLE_PREAMBLE + json.dumps(table, ensure_ascii=False, indent=2)]
        bodies.extend(f"Passage ID: {link}\n\n{body.strip()}" for link, body in passages.items())
        messages = tuple(
            payload(Message("user", body), "Corpus", position=i) for i, body in enumerate(bodies)
        )
        plans.append(
            InputPlan(
                replace(sample, sessions=()),
                (
                    AddEvent("corpus", messages),
                    *(SearchEvent(q.qid, q.question) for q in sample.questions),
                ),
                {
                    "scope": "shared complete table JSON and all linked passages",
                    "table_id": table_id,
                    "time_style": "synthetic",
                },
            )
        )
    return plans
