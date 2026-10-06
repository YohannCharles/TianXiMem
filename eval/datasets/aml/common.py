"""通用包装与持续检查点；合成时钟是本地近似，不声称等于捕获的时间。"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from ..preprocess import Message, Sample
from .plan import AddEvent, InputPlan, SearchEvent, digest

# 捕获使用按消息步进的合成时钟；这里只冻结其形态，不猜测平台的用户偏移。
SYNTHETIC_START_MS = 1704153600000
SYNTHETIC_STEP_MS = 60000


def payload(message: Message, label: str, *, position: int, time_style: str = "synthetic") -> dict:
    body = message.content
    role = "assistant" if message.role == "assistant" else "user"
    if time_style == "inline":
        if message.timestamp_ms is None:
            raise ValueError("AML inline time requires a source timestamp")
        stamp = datetime.fromtimestamp(message.timestamp_ms / 1000, tz=UTC).strftime(
            "%b %d, %Y, %H:%M:%S"
        )
        return {"role": role, "content": f"[Time: {stamp}] {body}"}
    result = {"role": role, "content": f"{label}: {body}"}
    if time_style == "synthetic":
        result["timestamp"] = SYNTHETIC_START_MS + position * SYNTHETIC_STEP_MS
    elif time_style == "source":
        if message.timestamp_ms is not None:
            result["timestamp"] = message.timestamp_ms
    elif time_style != "none":
        raise ValueError(f"Unknown AML time style {time_style!r}")
    return result


def dialogue_plans(samples: list[Sample], *, time_style: str = "synthetic") -> list[InputPlan]:
    plans = []
    for sample in samples:
        events = []
        position = 0
        for session in sample.sessions:
            messages = []
            for message in session.messages:
                messages.append(
                    payload(message, message.role, position=position, time_style=time_style)
                )
                position += 1
            if messages:
                events.append(AddEvent(session.session_id, tuple(messages)))
        events.extend(SearchEvent(q.qid, q.question) for q in sample.questions)
        plans.append(
            InputPlan(replace(sample, sessions=()), tuple(events), {"time_style": time_style})
        )
    return plans


def checkpoint_plans(samples: list[Sample], *, time_style: str = "synthetic") -> list[InputPlan]:
    """用已有前缀检查点恢复持续用户；严格核对前缀，不根据 gold 选会话。"""
    groups: dict[str, list[Sample]] = {}
    for sample in samples:
        # 首个 session 在该 persona 的所有检查点间稳定，避免解析 user_id 的便利命名。
        if not sample.sessions:
            raise ValueError("AML checkpoint: missing source sessions")
        key = sample.sessions[0].session_id
        groups.setdefault(key, []).append(sample)
    plans = []
    for key, checkpoints in groups.items():
        checkpoints.sort(key=lambda sample: len(sample.sessions))
        previous = ()
        events = []
        questions = []
        position = 0
        for checkpoint in checkpoints:
            if checkpoint.sessions[: len(previous)] != previous:
                raise ValueError("AML checkpoint: historical prefixes disagree")
            for session in checkpoint.sessions[len(previous) :]:
                messages = []
                for message in session.messages:
                    messages.append(
                        payload(message, message.role, position=position, time_style=time_style)
                    )
                    position += 1
                if messages:
                    events.append(AddEvent(session.session_id, tuple(messages)))
            events.extend(SearchEvent(q.qid, q.question) for q in checkpoint.questions)
            questions.extend(checkpoint.questions)
            previous = checkpoint.sessions
        sample = replace(
            checkpoints[0],
            user_id=f"checkpoint-{digest(key)[:20]}",
            sessions=(),
            questions=tuple(questions),
        )
        plans.append(
            InputPlan(
                sample,
                tuple(events),
                {
                    "time_style": time_style,
                    "scope": "continuous selected source checkpoints",
                    "source_checkpoints": [s.user_id for s in checkpoints],
                },
            )
        )
    return plans


def corporate_plans(samples: list[Sample]) -> list[InputPlan]:
    if not samples:
        return []
    sessions = samples[0].sessions
    if any(s.sessions != sessions for s in samples):
        raise ValueError("AML CorporateBench: QA subsets must share the same corpus")
    events = []
    position = 0
    for session in sessions:
        for message in session.messages:
            events.append(
                AddEvent(
                    f"document-{position:05d}",
                    (payload(message, "document", position=position, time_style="source"),),
                )
            )
            position += 1
    questions = tuple(q for s in samples for q in s.questions)
    events.extend(SearchEvent(q.qid, q.question) for q in questions)
    return [
        InputPlan(
            replace(samples[0], user_id="corporatebench-zenith", sessions=(), questions=questions),
            tuple(events),
            {"scope": "shared company corpus; one session per document", "time_style": "source"},
        )
    ]
