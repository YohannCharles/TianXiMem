"""共同证据执行器：条件筛选、有限连接、区间比较及来源覆盖审查。"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from tianximem.facts.conversation import SPEAKER, named_utterances, sentences
from tianximem.facts.evidence import EvidenceFact, Qualifier, key, relation_key, words_key
from tianximem.facts.query import QueryPlan


@dataclass(frozen=True, slots=True)
class EvidenceSelection:
    facts: tuple[EvidenceFact, ...]
    projection: str


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    selection: EvidenceSelection | None
    reason: str


class Source(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def user_id(self) -> str: ...

    @property
    def question(self) -> str | None: ...

    @property
    def answer(self) -> str | None: ...


def _current(facts: Sequence[EvidenceFact]) -> list[EvidenceFact]:
    groups: dict[tuple[str, str], list[EvidenceFact]] = defaultdict(list)
    for fact in facts:
        groups[(key(fact.subject), relation_key(fact.relation))].append(fact)
    selected: list[EvidenceFact] = []
    for group in groups.values():
        replacements = [f for f in group if f.get("replaces_earlier")]
        if replacements and all(f.event_time is not None for f in replacements):
            latest = max(f.event_time for f in replacements if f.event_time is not None)
            replacements = [f for f in replacements if f.event_time == latest]
            # 更正只覆盖更早的陈述；后来明确出现的普通陈述仍须保留。
            replacements.extend(
                f
                for f in group
                if not f.get("replaces_earlier")
                and f.event_time is not None
                and f.event_time >= latest
            )
        unique: dict[tuple[str, tuple[tuple[str, Qualifier], ...]], EvidenceFact] = {}
        for fact in replacements or group:
            unique.setdefault((key(fact.object), fact.qualifiers), fact)
        selected.extend(unique.values())
    return selected


def _quantities(facts: Sequence[EvidenceFact]) -> list[EvidenceFact] | None:
    """仅以可信来源时间消除同一项的旧观察，未知/同刻冲突不猜。"""
    groups: dict[tuple[str, str, str, str], list[EvidenceFact]] = defaultdict(list)
    selected = [f for f in facts if f.get("quantity") is None]
    for fact in facts:
        if fact.get("quantity") is not None:
            groups[
                (
                    key(fact.subject),
                    key(str(fact.get("container", ""))),
                    key(fact.object),
                    key(str(fact.get("name", ""))),
                )
            ].append(fact)
    for group in groups.values():
        if len({key(str(f.get("unit", ""))) for f in group}) > 1:
            # 相同数值也不能消除瓶/个等不同单位的歧义。
            return None
        amounts = {f.get("quantity") for f in group}
        if len(amounts) > 1:
            if any(f.event_time is None for f in group):
                return None
            latest = max(f.event_time for f in group if f.event_time is not None)
            group = [f for f in group if f.event_time == latest]
            if len({f.get("quantity") for f in group}) > 1:
                return None
        selected.append(group[0])
    positions = {fact.id: i for i, fact in enumerate(facts)}
    if groups:
        return sorted(
            selected,
            key=lambda fact: (fact.event_time is None, -(fact.event_time or 0), positions[fact.id]),
        )
    return sorted(selected, key=lambda fact: positions[fact.id])


def _resolve(
    facts: Sequence[EvidenceFact], roots: Sequence[EvidenceFact]
) -> list[EvidenceFact] | None:
    selected: dict[str, EvidenceFact] = {}
    for root in roots:
        selected.setdefault(root.id, root)
        relation = root.get("reference_relation")
        if relation:
            dependencies = [
                f
                for f in facts
                if key(f.subject) == key(root.subject)
                and relation_key(f.relation) == relation_key(str(relation))
            ]
            if not dependencies or (
                root.get("unique_reference") and len({key(f.object) for f in dependencies}) != 1
            ):
                return None
            for dependency in dependencies:
                selected.setdefault(dependency.id, dependency)
    return list(selected.values())


def _walk(
    facts: Sequence[EvidenceFact], plan: QueryPlan, hop_limit: int
) -> list[EvidenceFact] | None:
    relations = set(plan.patterns[0].relations)
    edges = [
        f for f in facts if relation_key(f.relation) in relations and f.get("start_month") is None
    ]
    matched = [f for f in edges if f" {words_key(f.subject)} " in f" {words_key(plan.query)} "]
    names: dict[str, set[str]] = defaultdict(set)
    for fact in matched:
        names[words_key(fact.subject)].add(key(fact.subject))
    if any(len(variants) > 1 for variants in names.values()):
        return None
    roots = sorted({key(f.subject) for f in matched}, key=lambda s: (-len(s), s))
    roots = [
        s
        for s in roots
        if not any(s != other and f" {words_key(s)} " in f" {words_key(other)} " for other in roots)
    ]
    if len(roots) != 1:
        return None
    frontier = set(roots)
    visited: set[str] = set()
    selected: dict[str, EvidenceFact] = {}
    for _ in range(min(hop_limit, len(relations))):
        frontier -= visited
        if not frontier:
            break
        visited.update(frontier)
        scoped = [f for f in edges if key(f.subject) in frontier]
        names.clear()
        for fact in scoped:
            names[words_key(fact.subject)].add(key(fact.subject))
        if any(len(variants) > 1 for variants in names.values()):
            return None
        current = _current(scoped)
        for fact in current:
            selected.setdefault(fact.parent_memory_id, fact)
        # 未理解的明确更正只能保留其原文，不能造出新关系边。
        for fact in facts:
            if fact.get("unparsed"):
                for entity in frontier:
                    if f" {words_key(entity)} " in f" {words_key(fact.source_quote)} ":
                        if key(entity) not in key(fact.source_quote):
                            return None
                        selected.setdefault(fact.parent_memory_id, fact)
        frontier = {key(f.object) for f in current}
    return list(selected.values())


def _interval(facts: Sequence[EvidenceFact], plan: QueryPlan) -> list[EvidenceFact] | None:
    intervals: list[tuple[EvidenceFact, int, int]] = []
    for fact in facts:
        start, end = fact.get("start_month"), fact.get("end_month")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and any(pattern.matches(fact) for pattern in plan.patterns)
        ):
            intervals.append((fact, start, end))
    if plan.month is not None:
        return [fact for fact, start, end in intervals if start <= plan.month <= end]
    refs = [
        interval
        for interval in intervals
        if key(interval[0].object if plan.bind_side == "subject" else interval[0].subject)
        == plan.reference
    ]
    if len(refs) != 1:
        return None
    reference = refs[0]
    if plan.direction == "before":
        eligible = [
            interval
            for interval in intervals
            if interval[2] <= reference[1] and interval[0].id != reference[0].id
        ]
        boundary = max((end for _, _, end in eligible), default=None)
        closest = [fact for fact, _, end in eligible if end == boundary]
    else:
        eligible = [
            interval
            for interval in intervals
            if interval[1] >= reference[2] and interval[0].id != reference[0].id
        ]
        boundary = min((start for _, start, _ in eligible), default=None)
        closest = [fact for fact, start, _ in eligible if start == boundary]
    return [reference[0], *closest] if closest else None


def sources_supported(
    sources: Sequence[Source], facts: Sequence[EvidenceFact], plan: QueryPlan
) -> bool:
    """物理扫描齐全不等于语义齐全；同范围的未知正向/变更声明继续交还原文。"""
    needle = plan.guard_literal
    subjects = {key(f.subject) for f in facts if any(p.matches(f) for p in plan.patterns)}
    for source in sources:
        residue = (source.question or "") + "\n" + (source.answer or "")
        for fact in facts:
            if fact.parent_memory_id == source.id:
                residue = residue.replace(fact.source_quote, "")
        if needle not in key(residue):
            continue
        handled = False
        for _, utterance in named_utterances(residue):
            speaker = SPEAKER.fullmatch(utterance.strip())
            if speaker is None:
                continue
            body, actor = speaker[2], key(speaker[1])
            if actor in {"user", "用户"}:
                actor = "i"
            relevant = needle in key(body)
            if not relevant:
                continue
            # 整段有计划/虚构标签也不能吞掉另一条未解释的正向声明。
            unexplained = re.search(
                r"\bI(?:'m| am)\s+(?!not\b|no longer\b)|"
                r"\bI\s+(?:currently\s+)?(?:work|study|volunteer|belong|live|teach|coach|train)\b|"
                r"\bMy\b[^.!?\n]*\b(?:has|contains)\b|"
                r"我(?:目前|现在|仍然|仍)?(?:是|在|的[^。！？\n]*(?:有|装着|包含|放着))",
                body,
                re.I,
            )
            negated = re.search(r"\bnot (?:in )?(?:my |the )?" + re.escape(needle), key(body))
            positive = re.search(
                r"(?<!not )\b(?:my|the same)\s+" + re.escape(needle) + r"\b", key(body)
            )
            if unexplained and (not negated or positive):
                return False
            first_sentence = sentences(body)[0]
            if (
                re.match(
                    r"(?:A fictional example|I (?:hope|plan|want|will))\b|"
                    r"我(?:计划|打算|希望|想要|准备|将会|将要)|"
                    r"(?:这是|一个|某个)?(?:虚构|示例|例子|假设)",
                    body,
                    re.I,
                )
                and needle in key(first_sentence)
                and key(body).count(needle) == 1
            ):
                handled = True
                continue
            negative = re.search(
                r"\bI (?:am not|do not|have not|no longer|left|stopped)\b|"
                r"我(?:目前|现在|仍然|已经|还)?"
                r"(?:不是|并非|没有|不在|不再|未曾|从未|已经离开|离开了|已经退出|退出了)",
                body,
                re.I,
            )
            if negative:
                if actor in subjects:
                    return False
                handled = True
                continue
            if negated and not positive:
                handled = True
                continue
        if not handled:
            return False
    return True


def evaluate_evidence(
    facts: Sequence[EvidenceFact],
    plan: QueryPlan,
    *,
    source_limit: int,
    hop_limit: int,
    sources: Sequence[Source] = (),
    audit_facts: Sequence[EvidenceFact] = (),
) -> EvidenceDecision:
    if plan.operator == "current":
        return EvidenceDecision(EvidenceSelection((), "fact"), "current_input")
    if plan.operator == "walk":
        selected = _walk(facts, plan, hop_limit)
        if selected is None:
            return EvidenceDecision(None, "unsupported_walk")
    elif plan.operator == "interval":
        selected = _interval(facts, plan)
        if selected is None:
            return EvidenceDecision(None, "unresolved_interval")
    else:
        selected = [f for f in facts if any(p.matches(f) for p in plan.patterns)]
        if plan.date_bounds:
            valid = []
            for fact in selected:
                value = str(fact.get("scope_date", fact.source_date))
                try:
                    date.fromisoformat(value)
                except ValueError:
                    return EvidenceDecision(None, "invalid_scope_date")
                if plan.date_bounds[0] <= value < plan.date_bounds[1]:
                    valid.append(fact)
            selected = valid
        selected = _quantities(selected)
        if selected is None:
            return EvidenceDecision(None, "quantity_conflict")
        if selected and all(fact.get("quantity") is None for fact in selected):
            # 呈现先放明确自述；同等级按主体/关系稳定排列。数量观察保持原文次序。
            selected.sort(
                key=lambda fact: (
                    not bool(fact.get("direct", True)),
                    key(fact.subject),
                    relation_key(fact.relation),
                    key(fact.object),
                )
            )
        if selected and plan.resolve_references:
            selected = _resolve(facts, selected)
            if selected is None:
                return EvidenceDecision(None, "unresolved_reference")
        if plan.guard_literal and not sources_supported(sources, audit_facts or facts, plan):
            return EvidenceDecision(None, "unsupported_source")
    if not selected:
        return EvidenceDecision(None, "no_match")
    if plan.projection == "source":
        selected = list({f.parent_memory_id: f for f in selected}.values())
    if len(selected) > source_limit:
        return EvidenceDecision(None, "source_limit")
    return EvidenceDecision(EvidenceSelection(tuple(selected), plan.projection), "selected")


def select_evidence(
    facts: Sequence[EvidenceFact],
    plan: QueryPlan,
    *,
    source_limit: int,
    hop_limit: int,
    sources: Sequence[Source] = (),
    audit_facts: Sequence[EvidenceFact] = (),
) -> EvidenceSelection | None:
    """兼容入口；诊断与执行共用同一次选择。"""
    return evaluate_evidence(
        facts,
        plan,
        source_limit=source_limit,
        hop_limit=hop_limit,
        sources=sources,
        audit_facts=audit_facts,
    ).selection
