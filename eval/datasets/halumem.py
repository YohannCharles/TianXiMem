"""HaluMem-Medium 的 QA 加载层：一个 Sample = 一个用户的一个提问检查点。

每个检查点只包含到当前 session 为止的 dialogue，使用独立 user_id，后续会话
不会污染早期问题。后续检查点会重复投喂历史；Session/Message 对象在加载时共享。
persona_info、memory_points 与 questions/evidence 都是标注，不进入 Add。

本入口只测端到端 QA。裁判复用 official_capture_pipeline 的上游三分类 prompt，
不声称覆盖 HaluMem 的记忆抽取/更新指标，也不声称复现 AML 的输入适配。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from .preprocess import Message, Question, Sample, Session, normalize_content, to_epoch_ms
from .sampling import stratified_sample

DATA_DIR = "halumem"
JSONL = "HaluMem-Medium.jsonl"
USER_PREFIX = "halumem-"
SHAPE_NOTE = (
    "HaluMem-Medium QA：一个 Sample = 一个用户的一个提问检查点，"
    "独立 user_id，只投喂到该 session 为止的原始对话；后续检查点重复投喂历史。"
    "persona_info、memory_points 与题目证据不进入 Add。"
    "判分复用上游 Correct/Hallucination/Omission；不覆盖记忆抽取/更新指标。"
)


def _timestamp(value: str) -> int:
    moment = datetime.strptime(value, "%b %d, %Y, %H:%M:%S").replace(tzinfo=UTC)
    return to_epoch_ms(moment)


def _session(raw: dict, uuid: str, index: int) -> Session:
    messages = tuple(
        Message(
            role=str(turn["role"]),
            content=normalize_content(
                turn["content"], where=f"HaluMem {uuid} session {index + 1} turn {turn_index}"
            ),
            timestamp_ms=_timestamp(turn.get("timestamp") or raw["start_time"]),
        )
        for turn_index, turn in enumerate(raw["dialogue"])
    )
    if not messages:
        raise ValueError(f"HaluMem {uuid} session {index + 1}：对话为空")
    return Session(session_id=f"{uuid}-s{index + 1:03d}", messages=messages)


def load_halumem(
    bench_dir: str | Path, *, limit: int | None = None, spread: bool = False
) -> list[Sample]:
    """limit 数提问检查点；spread 按用户分层，每组至少取一个检查点。"""
    path = Path(bench_dir) / DATA_DIR / JSONL
    with path.open(encoding="utf-8") as handle:
        users = [json.loads(line) for line in handle if line.strip()]
    uuids = [str(user["uuid"]) for user in users]
    if len(set(uuids)) != len(uuids):
        raise ValueError("HaluMem：uuid 重复，无法保证检查点隔离")
    checkpoints = [
        {"user_index": user_index, "uuid": uuids[user_index], "session_index": session_index}
        for user_index, user in enumerate(users)
        for session_index, session in enumerate(user["sessions"])
        if session.get("questions")
    ]
    if limit is not None:
        if limit < 0:
            raise ValueError("HaluMem：limit 必须非负")
        if limit == 0:
            return []
        checkpoints = (
            stratified_sample(checkpoints, limit, key=lambda row: row["uuid"])
            if spread
            else checkpoints[:limit]
        )

    # 只归一化所选检查点需要的前缀，每个用户的 session 在加载时只构造一次。
    prefixes: dict[int, list[Session]] = {}
    samples: list[Sample] = []
    for checkpoint in checkpoints:
        user_index = checkpoint["user_index"]
        uuid = checkpoint["uuid"]
        session_index = checkpoint["session_index"]
        raw_sessions = users[user_index]["sessions"]
        sessions = prefixes.setdefault(user_index, [])
        for index in range(len(sessions), session_index + 1):
            sessions.append(_session(raw_sessions[index], uuid, index))
        user_id = f"{USER_PREFIX}{uuid}-s{session_index + 1:03d}"
        questions = []
        for question_index, raw in enumerate(raw_sessions[session_index]["questions"]):
            if not isinstance(raw.get("answer"), str) or not raw["answer"].strip():
                raise ValueError(f"HaluMem {user_id} question {question_index}：参考回答为空")
            questions.append(
                Question(
                    qid=f"{user_id}-q{question_index:03d}",
                    question=normalize_content(
                        raw["question"], where=f"HaluMem {user_id} question"
                    ),
                    gold=dict(raw),
                    category=str(raw["question_type"]),
                )
            )
        samples.append(
            Sample(
                user_id=user_id,
                dataset="halumem",
                sessions=tuple(sessions[: session_index + 1]),
                questions=tuple(questions),
            )
        )
    return samples
