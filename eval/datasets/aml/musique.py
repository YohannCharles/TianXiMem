"""MuSiQue-Full 的有序候选段落包装与完整检索指令。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .. import musique as source
from ..preprocess import Message
from .common import payload
from .plan import AddEvent, InputPlan, SearchEvent

SEARCH_INSTRUCTION = (
    "Answer using only the supplied paragraphs. If the evidence is insufficient, "
    "respond with INSUFFICIENT_EVIDENCE."
)


def load(root: Path, *, limit: int | None, spread: bool) -> list[InputPlan]:
    rows = source.read_rows(root, limit=limit, spread=spread)
    samples = source.samples_from_rows(rows)
    plans = []
    for row, sample in zip(rows, samples, strict=True):
        messages = tuple(
            payload(
                Message(
                    "user",
                    f"[Paragraph {paragraph['idx']}]\nTitle: {paragraph['title']}\n\n"
                    + source.normalize_content(
                        paragraph["paragraph_text"], where="AML MuSiQue paragraph"
                    ),
                ),
                "Corpus",
                position=position,
            )
            for position, paragraph in enumerate(row["paragraphs"])
        )
        q = sample.questions[0]
        plans.append(
            InputPlan(
                replace(sample, sessions=()),
                (
                    AddEvent("corpus", messages),
                    SearchEvent(q.qid, q.question + "\n\n" + SEARCH_INSTRUCTION),
                ),
                {
                    "scope": "ordered Full-dev paragraphs; answerability variants isolated",
                    "time_style": "synthetic",
                },
            )
        )
    return plans
