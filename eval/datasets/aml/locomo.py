"""LoCoMo 对话加公开 LongMemEval 干扰历史；池分配和角色变体显式冻结。

公开题库没有逐题到达时刻或平台反馈写回，不能从答案证据推断它们。
本模式先 Add 完选定历史再 Search；真实交错和写回使用 official-capture。
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..locomo import load_locomo
from ..longmemeval import load_longmemeval
from .common import SYNTHETIC_STEP_MS, dialogue_plans, payload
from .plan import AddEvent, InputPlan, SearchEvent


def load(root: Path, *, limit: int | None, spread: bool, alluser: bool = False) -> list[InputPlan]:
    samples = load_locomo(root)
    if limit is not None:
        samples = samples[:limit]
    if not samples:
        return []
    distractors = load_longmemeval(root, limit=len(samples), spread=False)
    if len(distractors) != len(samples):
        raise ValueError(
            "AML LoCoMo: one declared LME haystack per selected conversation is required"
        )
    plans = []
    for sample, distractor in zip(samples, distractors, strict=True):
        events = []
        position = 0
        for session in sample.sessions:
            messages = []
            for message in session.messages:
                label = sample.speaker_names[message.role == "assistant"]
                item = payload(message, label, position=position)
                if alluser:
                    item["role"] = "user"
                messages.append(item)
                position += 1
            if messages:
                events.append(AddEvent(session.session_id, tuple(messages)))
        for event in dialogue_plans([distractor])[0].events:
            if not isinstance(event, AddEvent):
                continue
            messages = []
            for message in event.messages:
                item = dict(message)
                # LongMemEval 干扰保留自己的角色；alluser 只描述真人 LoCoMo 的形态。
                prefix, _, body = item["content"].partition(": ")
                item["content"] = f"{prefix.title()}: {body}"
                item["timestamp"] += position * SYNTHETIC_STEP_MS
                messages.append(item)
            events.append(
                AddEvent(f"distractor-{distractor.user_id}-{event.session_id}", tuple(messages))
            )
        events.extend(SearchEvent(q.qid, q.question) for q in sample.questions)
        plans.append(
            InputPlan(
                replace(sample, sessions=()),
                tuple(events),
                {
                    "scope": "full source conversation plus one deterministic public LME haystack",
                    "distractor_user_id": distractor.user_id,
                    "time_style": "synthetic",
                    "conversation_roles": "alluser" if alluser else "preserved",
                    "timeline_limit": "no public per-question arrival times or platform feedback",
                },
            )
        )
    return plans
