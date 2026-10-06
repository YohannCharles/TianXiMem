"""MuSiQue-Full dev：每道题的候选段落组成一个独立用户的记忆语料。

Full 的可答/不可答变体共用 id，user_id/qid 必须带变体标记。只把 title 与
paragraph_text 送进 Add；answer、is_supporting 与 question_decomposition 均不注入。
spread 按跳数和可答性分层。只测本地答案别名精确匹配/拒答，不复现上游支持段落 F1
或成组 answer-sufficiency 指标；与采集中的重采样段落顺序也不等价。
"""

from __future__ import annotations

import json
from pathlib import Path

from .preprocess import Message, Question, Sample, Session, normalize_content
from .sampling import stratified_sample

DATA_DIR = "musique"
JSONL = "musique_full_v1.0_dev.jsonl"
REFUSAL = "INSUFFICIENT_EVIDENCE"
ANSWER_CONTRACT = "musique-local-qa-v1"
USER_PREFIX = "musique-"
SHAPE_NOTE = (
    f"MuSiQue-Full dev（{ANSWER_CONTRACT}）：每题一个独立 user，"
    "语料只含候选段落 title/paragraph_text。"
    "可答/不可答变体用不同 user_id/qid；spread 按跳数和可答性分层。"
    "本地评分为答案别名归一化后精确匹配与拒答标记精确匹配，不是上游答案/支持段落 F1"
    "或成组 sufficiency 指标，也不等价于采集中的重采样语料顺序。"
)


def _variant(row: dict) -> str:
    if not isinstance(row.get("answerable"), bool):
        raise ValueError(f"MuSiQue {row.get('id')}：answerable 必须为 bool")
    return "answerable" if row["answerable"] else "unanswerable"


def load_musique(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """加载 Full dev；limit 数题目，spread 覆盖不同跳数及可答/不可答变体。"""
    path = Path(bench_dir) / DATA_DIR / JSONL
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(str(row["id"]), _variant(row)) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("MuSiQue：同一 id/可答性变体重复，qid 会冲突")
    if limit is not None:
        if limit < 0:
            raise ValueError("MuSiQue：limit 必须非负")
        if limit == 0:
            return []
        rows = (
            stratified_sample(
                rows,
                limit,
                key=lambda row: f"{len(row['question_decomposition'])}-hop/{_variant(row)}",
            )
            if spread
            else rows[:limit]
        )

    samples: list[Sample] = []
    for row in rows:
        variant = _variant(row)
        user_id = f"{USER_PREFIX}{row['id']}-{variant}"
        messages = tuple(
            Message(
                role="user",
                content=normalize_content(
                    f"Title: {paragraph['title']}\n"
                    + normalize_content(
                        paragraph["paragraph_text"], where=f"MuSiQue {user_id} paragraph body"
                    ),
                    where=f"MuSiQue {user_id} paragraph {paragraph['idx']}",
                ),
            )
            for paragraph in row["paragraphs"]
        )
        if not messages:
            raise ValueError(f"MuSiQue {user_id}：候选段落为空")
        answer = row["answer"]
        aliases = row["answer_aliases"]
        if not isinstance(answer, str) or (row["answerable"] and not answer.strip()):
            raise ValueError(f"MuSiQue {user_id}：可答题参考答案为空或类型错误")
        if not isinstance(aliases, list) or any(not isinstance(a, str) for a in aliases):
            raise ValueError(f"MuSiQue {user_id}：answer_aliases 必须为字符串列表")
        question = Question(
            qid=user_id,
            question=normalize_content(row["question"], where=f"MuSiQue {user_id} question"),
            gold={"answer": answer, "answer_aliases": aliases, "answerable": row["answerable"]},
            category=f"{len(row['question_decomposition'])}-hop/{variant}",
            evidence=tuple(str(p["idx"]) for p in row["paragraphs"] if p["is_supporting"]),
            is_abstention=not row["answerable"],
        )
        samples.append(
            Sample(
                user_id=user_id,
                dataset="musique",
                sessions=(Session(session_id=f"{user_id}-corpus", messages=messages),),
                questions=(question,),
            )
        )
    return samples
