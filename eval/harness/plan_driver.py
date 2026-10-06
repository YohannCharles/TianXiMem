"""AML 事件的编译、HTTP 执行和逐事件续跑；不重新检索已经走过的历史时点。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from eval.datasets.aml.plan import AddEvent, InputPlan, SearchEvent, digest

from .add_shape import split_prefixed_payload
from .batching import batches
from .driver import SearchHit, ServiceClient


class PlanStateError(ValueError):
    """输出目录/检查点属于不同输入，或缺少历史 Search 快照。"""


def compile_plan(plan: InputPlan, *, namespace: str) -> InputPlan:
    """先切片/切批，再从完整请求计划派生隔离 ID；金标不参与身份派生。"""
    plan.validate()
    events = []
    for event in plan.events:
        if isinstance(event, SearchEvent) or event.batch_ready:
            events.append(event)
            continue
        messages = [
            piece for message in event.messages for piece in split_prefixed_payload(message)
        ]
        events.extend(
            AddEvent(event.session_id, batch, batch_ready=True) for batch in batches(messages)
        )
    wire_plan = replace(plan, events=tuple(events))
    identity = digest(
        {
            "namespace": namespace,
            "contract": plan.contract,
            "source_user_id": plan.sample.user_id,
            "wire": wire_plan.wire(),
        }
    )
    user_id = f"{plan.contract}-{plan.sample.dataset}-{identity[:24]}"
    counts: dict[str, int] = {}
    prepared = []
    for event in events:
        if isinstance(event, AddEvent):
            index = counts.get(event.session_id, 0)
            counts[event.session_id] = index + 1
            event = replace(
                event, request_id=event.request_id or f"{user_id}|{event.session_id}|{index}"
            )
        prepared.append(event)
    requests = [event.request_id for event in prepared if isinstance(event, AddEvent)]
    if len(set(requests)) != len(requests):
        raise ValueError("AML plan: duplicate Add request IDs")
    policy = digest(
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path(__file__).with_name("add_shape.py"),
                Path(__file__).with_name("batching.py"),
            )
        }
    )
    result = replace(
        wire_plan,
        sample=replace(plan.sample, user_id=user_id),
        events=tuple(prepared),
        metadata=plan.metadata
        | {
            "source_user_id": plan.sample.user_id,
            "namespace": namespace,
            "execution_policy_sha256": policy,
            "batching": (
                "20 messages / 2000 whitespace words; 8000-character message cap (approximate)"
            ),
        },
    )
    result.validate()
    return result


def send_event(
    client: ServiceClient, user_id: str, event: AddEvent | SearchEvent, *, top_k: int = 100
):
    """公共 HTTP 分派；已切批事件和采集原文都直接走 add，不调用 ingest/shape。"""
    if isinstance(event, AddEvent):
        return client.add(
            request_id=event.request_id,
            user_id=user_id,
            session_id=event.session_id,
            messages=list(event.messages),
        )
    return client.search(user_id=user_id, query=event.query, top_k=top_k)


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def plans_fingerprint(plans: list[InputPlan]) -> dict:
    fingerprints = [plan.fingerprint() for plan in plans]
    return {
        "input_contract": plans[0].contract if plans else "aml-v1",
        "plans_sha256": digest(fingerprints),
        "plans": fingerprints,
        "n_samples": len(plans),
        "n_questions": sum(len(p.sample.questions) for p in plans),
        "n_adds": sum(f["n_adds"] for f in fingerprints),
        "n_messages": sum(f["n_messages"] for f in fingerprints),
        "batching": "compiled AML approximation; budgets recorded per plan",
    }


def freeze_run(out_dir: Path, plans: list[InputPlan], *, execution: dict) -> dict:
    fingerprint = plans_fingerprint(plans)
    manifest = {"data_fingerprint": fingerprint, "execution": execution}
    path = out_dir / "input-manifest.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise PlanStateError(
                "AML run input/execution changed; use a new run-id and storage snapshot"
            )
        for plan in plans:
            checkpoint = out_dir / plan.sample.user_id / "checkpoint.json"
            if not checkpoint.is_file():
                raise PlanStateError(
                    "AML run is missing an initialized checkpoint; use a fresh namespace"
                )
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise PlanStateError(
            "AML run directory has legacy outputs without an input manifest; use a new run-id"
        )
    else:
        _write_atomic(path, manifest)
        # 在首次 HTTP 前初始化所有用户。后续整目录丢失也会响亮失败，
        # 避免在服务已写入未来记忆后将游标错误地重置到第一题。
        for plan in plans:
            _write_atomic(
                out_dir / plan.sample.user_id / "checkpoint.json",
                {
                    "fingerprint": plan.fingerprint() | {"top_k": execution["top_k"]},
                    "next_event": 0,
                    "hits": {},
                },
            )
    return fingerprint


def execute_plan(
    plan: InputPlan, *, client: ServiceClient, out_dir: Path, top_k: int = 100
) -> dict[str, list[SearchHit]]:
    """每个成功 Search 先落盘才允许执行后续 Add，保证历史检索可续跑。

    Add 先成功、后保存游标；响应丢失时同 payload/ID 重投。Search 失败直接抛，
    后续 Add 不执行。与原请求回放的跳过策略分别记录。
    """
    plan.validate()
    if top_k != 100:
        raise ValueError("AML input contract requires top_k=100")
    if any(
        isinstance(e, AddEvent) and (not e.batch_ready or not e.request_id) for e in plan.events
    ):
        raise ValueError("AML plan must be compiled before HTTP execution")
    path = out_dir / "checkpoint.json"
    fingerprint = plan.fingerprint() | {"top_k": top_k}
    state = {"fingerprint": fingerprint, "next_event": 0, "hits": {}}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("fingerprint") != fingerprint:
            raise PlanStateError("AML checkpoint fingerprint changed; use a new run-id")
    elif out_dir.exists() and any(out_dir.iterdir()):
        raise PlanStateError("AML user outputs are missing their checkpoint; use a fresh namespace")
    else:
        _write_atomic(path, state)
    cursor = state.get("next_event")
    if type(cursor) is not int or not 0 <= cursor <= len(plan.events):
        raise PlanStateError("AML checkpoint contains an invalid event cursor")
    completed = {e.qid for e in plan.events[:cursor] if isinstance(e, SearchEvent)}
    if not isinstance(state.get("hits"), dict) or set(state["hits"]) != completed:
        raise PlanStateError(
            "AML checkpoint is missing historical Search snapshots; use a fresh namespace"
        )
    hits = {qid: [SearchHit(**h) for h in rows] for qid, rows in state["hits"].items()}
    for index in range(cursor, len(plan.events)):
        event = plan.events[index]
        result = send_event(client, plan.sample.user_id, event, top_k=top_k)
        if isinstance(event, SearchEvent):
            hits[event.qid] = result
            state["hits"][event.qid] = [
                {
                    "id": hit.id,
                    "content": hit.content,
                    "created_at": hit.created_at,
                    "score": hit.score,
                }
                for hit in result
            ]
        state["next_event"] = index + 1
        _write_atomic(path, state)
    return hits
