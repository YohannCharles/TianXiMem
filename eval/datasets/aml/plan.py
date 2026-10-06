"""版本化的输入计划：HTTP 事件与仅供评分的问题分开保存。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace

from ..preprocess import Sample

INPUT_CONTRACT = "aml-v1"


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AddEvent:
    session_id: str
    messages: tuple[dict, ...]
    request_id: str = ""
    batch_ready: bool = False


@dataclass(frozen=True, slots=True)
class SearchEvent:
    qid: str
    query: str


@dataclass(frozen=True, slots=True)
class InputPlan:
    sample: Sample
    events: tuple[AddEvent | SearchEvent, ...]
    metadata: dict = field(default_factory=dict)
    contract: str = INPUT_CONTRACT

    def validate(self) -> None:
        questions = [q.qid for q in self.sample.questions]
        searches = [e.qid for e in self.events if isinstance(e, SearchEvent)]
        if len(set(questions)) != len(questions) or questions != searches:
            raise ValueError("AML plan: questions and ordered Search events must agree uniquely")
        for event in self.events:
            if isinstance(event, SearchEvent):
                if not event.qid or not event.query.strip():
                    raise ValueError("AML plan: empty Search query/id")
                continue
            if not event.session_id or not event.messages:
                raise ValueError("AML plan: empty Add/session")
            for message in event.messages:
                if set(message) - {"role", "content", "timestamp"}:
                    raise ValueError("AML plan: only role/content/timestamp may enter Add")
                if message.get("role") not in {"user", "assistant"}:
                    raise ValueError("AML plan: unsupported role")
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError("AML plan: empty/non-string content")
                if "timestamp" in message and type(message["timestamp"]) is not int:
                    raise ValueError("AML plan: timestamp must be an integer when present")

    def wire(self) -> list[dict]:
        """不包含 gold/category/evidence；身份派生只使用实际请求内容。"""
        return [
            {
                "kind": "add",
                "session_id": event.session_id,
                "request_id": event.request_id,
                "messages": list(event.messages),
            }
            if isinstance(event, AddEvent)
            else {"kind": "search", "qid": event.qid, "query": event.query}
            for event in self.events
        ]

    def fingerprint(self) -> dict:
        return {
            "input_contract": self.contract,
            "dataset": self.sample.dataset,
            "user_id": self.sample.user_id,
            "wire_sha256": digest(self.wire()),
            "evaluation_sha256": digest(
                [
                    {
                        "qid": q.qid,
                        "question": q.question,
                        "gold": q.gold,
                        "category": q.category,
                        "evidence": q.evidence,
                        "is_abstention": q.is_abstention,
                    }
                    for q in self.sample.questions
                ]
            ),
            "metadata": self.metadata,
            "n_questions": len(self.sample.questions),
            "n_adds": sum(isinstance(e, AddEvent) for e in self.events),
            "n_messages": sum(len(e.messages) for e in self.events if isinstance(e, AddEvent)),
        }

    def limit_questions(self, maximum: int | None) -> InputPlan:
        """裁 Search，保留所有 Add 的原有位置；不会把后续记忆提前。"""
        if maximum is None:
            return self
        if maximum < 1:
            raise ValueError("AML plan: max_questions must be positive")
        questions = self.sample.questions[:maximum]
        wanted = {q.qid for q in questions}
        return replace(
            self,
            sample=replace(self.sample, questions=questions),
            events=tuple(e for e in self.events if isinstance(e, AddEvent) or e.qid in wanted),
        )
